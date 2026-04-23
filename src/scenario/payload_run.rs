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
//! # Stdout-primary, stderr-fallback metric extraction
//!
//! The extraction pipeline runs [`extract_metrics`](crate::test_support::extract_metrics)
//! against **stdout first**. When that returns an empty metric set
//! AND stderr is non-empty, the extractor retries against stderr.
//! This preserves the stdout-primary contract for well-behaved
//! binaries (noisy stderr never corrupts the metric stream) while
//! still handling payloads that emit their structured output only on
//! stderr — e.g. schbench's default percentile tables via
//! `show_latencies` → `fprintf(stderr, ...)`. The two streams are
//! never merged: concurrent drain threads for stdout/stderr provide
//! no ordering guarantee, so interleaving would corrupt any document
//! whose bytes span both streams.
//!
//! Stderr is still forwarded verbatim into the exit-code-mismatch
//! detail produced by [`Check::ExitCodeEq`] (see the
//! `format_exit_mismatch` path) so failing binaries surface their
//! error output directly.

use std::borrow::Cow;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use std::time::Duration;

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
    /// Optional runtime bound for the foreground `.run()` path. `None`
    /// means wait indefinitely; `Some(duration)` arms a deadline
    /// watchdog that SIGKILLs the payload's process group if it has
    /// not exited by the deadline. Set via [`timeout`](Self::timeout).
    /// Ignored by `.spawn()` — background handles manage their own
    /// lifetime via [`PayloadHandle::wait`] / `.kill()` / `.try_wait()`.
    timeout: Option<Duration>,
}

impl std::fmt::Debug for PayloadRun<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PayloadRun")
            .field("payload", &self.payload.name)
            .field("args_len", &self.args.len())
            .field("checks_len", &self.checks.len())
            .field("cgroup", &self.cgroup)
            .field("timeout", &self.timeout)
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
            timeout: None,
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

    /// Bound `.run()`'s wait for the payload to exit. `None` (the
    /// default when `.timeout` is not called) waits indefinitely —
    /// suitable for payloads whose runtime is bounded internally
    /// (schbench `-r 10`, fio `--runtime`, ...). `Some(duration)`
    /// arms a deadline watchdog inside `.run()` that SIGKILLs the
    /// payload's whole process group if it has not exited by the
    /// deadline. Ignored by `.spawn()` — background handles manage
    /// their own timing.
    ///
    /// The builder shape keeps `.run()` zero-arg so non-timeout
    /// call sites read naturally, and leaves room for future
    /// knobs (per-test environment, stdin, …) without another
    /// signature break.
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    /// Blocking foreground run. Spawns the payload binary, waits
    /// for it to exit, extracts metrics from its output per the
    /// payload's [`OutputFormat`] (stdout-primary with stderr
    /// fallback for `Json` / `LlmExtract`; no extraction for
    /// `ExitCode`), and evaluates declared [`Check`]s into an
    /// [`AssertResult`]. See the module-level
    /// `# Stdout-primary, stderr-fallback metric extraction`
    /// section for the full contract.
    ///
    /// Runtime is bounded by the value set via
    /// [`timeout`](Self::timeout). When the deadline expires,
    /// [`kill_payload_process_group`] fires and the returned
    /// `(AssertResult, PayloadMetrics)` reflects the captured
    /// output plus the killed-child exit code; `status.code()`
    /// returns `None` for a SIGKILL'd child, which
    /// [`spawn_and_wait`] surfaces as `exit_code = -1` in
    /// [`SpawnOutput`]. The timeout case is not an error — the
    /// caller can still inspect metrics collected before the kill.
    /// A post-kill drain failure is reported as `Err` (wraps the
    /// original I/O error with "drain after timeout of N"); the
    /// caller loses no output that was already captured because
    /// the partial reader-thread buffers have been consumed in
    /// the error path too.
    ///
    /// Metrics are also recorded to the per-test sidecar via the
    /// SHM ring; the returned tuple is a convenience view of the
    /// same values.
    ///
    /// Returns `Err` when the payload is not
    /// [`PayloadKind::Binary`] (schedulers are framework-launched,
    /// not test-body-launched), when the cgroup name fails
    /// validation, when the spawn itself fails, or when post-kill
    /// drain fails (see the timeout paragraph).
    pub fn run(self) -> Result<(AssertResult, PayloadMetrics)> {
        let binary = payload_binary(self.payload)?;
        let cgroup_path = resolve_cgroup_path(self.ctx, self.cgroup.as_deref())?;
        let output = spawn_and_wait(
            binary,
            &self.args,
            cgroup_path.as_deref(),
            self.timeout,
            self.payload.uses_parent_pgrp,
        )
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
        let (child, sigchld) = spawn_child(
            binary,
            &self.args,
            cgroup_path.as_deref(),
            self.payload.uses_parent_pgrp,
        )
        .with_context(|| format!("spawn payload '{}'", self.payload.name))?;
        Ok(PayloadHandle {
            child: Some(child),
            payload: self.payload,
            checks: self.checks,
            _sigchld: sigchld,
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
///
/// Metric extraction is stdout-primary, stderr-fallback:
/// [`extract_metrics`] runs first against stdout, and only when the
/// result is empty AND stderr is non-empty is it retried against
/// stderr. Well-behaved binaries keep stdout as the canonical metric
/// stream; payloads like schbench that write structured output only
/// to stderr (`show_latencies` → `fprintf(stderr, ...)`) are still
/// parsed. The streams are never concatenated — the two drain
/// threads in [`wait_and_capture`] run concurrently and provide no
/// ordering guarantee, so a merged blob would corrupt any document
/// whose bytes span both. Stderr is still passed separately to
/// [`evaluate_checks`] so the exit-code-mismatch detail renders
/// stderr without stdout prefix.
fn evaluate(
    payload: &Payload,
    checks: &[Check],
    output: SpawnOutput,
) -> (AssertResult, PayloadMetrics) {
    // `extract_metrics` returns Result specifically so an
    // `OutputFormat::LlmExtract` setup failure (model cache load,
    // not mere "no metrics extracted") can surface its reason into
    // the AssertResult rather than collapse into a vague
    // "metric 'X' not found" downstream. Non-LlmExtract formats are
    // infallible and always Ok.
    let stdout_result = extract_metrics(
        &output.stdout,
        &payload.output,
        crate::test_support::MetricStream::Stdout,
    );
    let (mut metrics, mut extract_err) = match stdout_result {
        Ok(m) => (m, None::<String>),
        Err(msg) => (Vec::new(), Some(msg)),
    };
    if metrics.is_empty() && !output.stderr.is_empty() {
        // Stderr fallback — only retry if stdout produced neither
        // metrics nor a load-failure reason (a load failure is
        // sticky across stdout/stderr — reason string is identical,
        // no point re-invoking inference).
        //
        // The fallback is deliberately GLOBAL (variant-agnostic)
        // rather than a per-`OutputFormat` opt-in. Evaluated
        // alternatives + why this is the right shape:
        //
        // * `ExitCode`: `extract_metrics` returns `Ok(vec![])` on
        //   both stdout and stderr for this variant (no parsing
        //   path), so running the fallback is a no-op — no stored
        //   state, no wasted work beyond one function call. Adding
        //   a per-variant gate would be complexity without
        //   behavioral difference.
        // * `Json` / `LlmExtract`: both BENEFIT from the fallback.
        //   The motivating case is schbench-like payloads that
        //   write structured output to stderr only (see
        //   `SchbenchPayload` in tests/common/fixtures.rs for the
        //   long-form rationale). A per-variant knob would require
        //   every new fixture declaring those variants to also
        //   remember to opt in — easy to miss, and the default
        //   should match the common case.
        // * A future "stdout-only" variant would be the one case
        //   where opt-in is appropriate. That's the trigger for
        //   adding the knob: a concrete use case, not speculative
        //   flexibility. Do NOT introduce a `stderr_fallback:
        //   bool` field on `OutputFormat` in anticipation.
        //
        // The streams are never merged — fallback replaces, not
        // concatenates — so an upstream that genuinely writes to
        // both stdout and stderr gets only the stdout metrics,
        // which matches the "well-behaved binaries keep stdout
        // canonical" language on the `OutputFormat` doc.
        if extract_err.is_none() {
            match extract_metrics(
                &output.stderr,
                &payload.output,
                crate::test_support::MetricStream::Stderr,
            ) {
                Ok(m) => metrics = m,
                Err(msg) => extract_err = Some(msg),
            }
        }
    }
    resolve_polarities(&mut metrics, payload);

    let payload_metrics = PayloadMetrics {
        metrics,
        exit_code: output.exit_code,
    };

    emit_payload_metrics_to_shm(&payload_metrics);

    // Short-circuit when LlmExtract load failed: running
    // `evaluate_checks` against an empty metrics vec would flood
    // the AssertResult with a cascade of "metric 'X' not found"
    // Other-kind details, burying the real root cause. Surface
    // the load-failure detail as the sole (and first) entry and
    // set passed=false directly.
    if let Some(reason) = extract_err {
        let mut result = AssertResult {
            passed: false,
            skipped: false,
            details: vec![crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("LlmExtract model load failed: {reason}"),
            )],
            stats: Default::default(),
        };
        // Still run the exit-code gate if a Check::ExitCodeEq is
        // set on the payload — exit code is orthogonal to the
        // metric pipeline and may still be meaningful (e.g. the
        // payload itself crashed before the model-load cache check
        // returned Err, so the user wants both signals). Delegated
        // to `exit_code_mismatch_detail` so this branch and
        // `evaluate_checks`'s pre-pass produce bit-identical
        // AssertDetails for the same (expected, actual, stderr)
        // inputs — no drift between the two call sites.
        if let Some(detail) = exit_code_mismatch_detail(checks, output.exit_code, &output.stderr) {
            result.details.push(detail);
        }
        return (result, payload_metrics);
    }

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
/// the child and every process it forked are SIGKILLed via the
/// process group headed by the child, then the child is reaped with
/// `child.wait()`, and a stderr warning is emitted so the test
/// author sees the implicit drop. The process-group kill reaches
/// every descendant of multi-process payloads (stress-ng, schbench
/// worker mode, fio `--numjobs`); without it the orphans keep
/// stdout/stderr open, block [`wait_and_capture`], and lose metrics.
///
/// When multiple handles are active, sidecar entries appear in
/// finalization order (the order `.wait()`, `.kill()`, or
/// `.try_wait()` returning `Ok(Some(..))` are called), not spawn
/// order. `.try_wait()` only records on its terminal branch; an
/// `Ok(None)` return keeps the handle live and defers the sidecar
/// write to the next terminal call.
#[must_use = "dropping a PayloadHandle SIGKILLs the child's process group; call .wait() or .kill() explicitly"]
pub struct PayloadHandle {
    /// Live child process. Wrapped in `Option` so consumers can
    /// take ownership in `wait`/`kill` without making the drop-guard
    /// reach into a `None`.
    child: Option<std::process::Child>,
    payload: &'static Payload,
    checks: Vec<Check>,
    /// `SIGCHLD` guard installed at spawn time. Kept alive until
    /// the handle is consumed (via `wait`/`kill`/Drop) so the
    /// child's eventual `waitpid` sees `SIG_DFL` instead of the
    /// guest init's `SIG_IGN`. See [`SigchldScope`] for the full
    /// rationale.
    _sigchld: SigchldScope,
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

    /// Live child's OS-level pid, or `None` once `wait`/`kill`/
    /// `try_wait` has consumed the child.
    ///
    /// Integration tests that spawn a workload and then need to
    /// target it with a second tool (for example the jemalloc-TLS
    /// probe in `tests/jemalloc_probe_tests.rs`, which passes the
    /// workload's pid to `ktstr-jemalloc-probe --pid`) read this
    /// value between `spawn` and `wait`/`kill`/`try_wait`. The
    /// internal fork-descendant reap test also uses it to probe
    /// the process group via `killpg(_, 0)` after `kill()` without
    /// reaching into the private `child` field.
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(|c| c.id())
    }

    /// Block until the child exits naturally, then extract metrics
    /// and evaluate checks, matching the foreground `.run()` return
    /// shape.
    ///
    /// Metrics are also recorded to the per-test sidecar via the
    /// SHM ring; the returned tuple is a convenience view of the
    /// same values.
    pub fn wait(mut self) -> Result<(AssertResult, PayloadMetrics)> {
        let mut child = self
            .child
            .take()
            .ok_or_else(|| already_consumed(self.payload))?;
        match wait_and_capture(&mut child) {
            Ok(output) => Ok(evaluate(self.payload, &self.checks, output)),
            Err(e) => {
                // Reader-thread panic or wait-syscall failure left
                // the child (and any fork descendants holding the
                // pipes open) alive. `kill_payload_process_group`
                // sends killpg + single-pid SIGKILL to close the
                // pipes and guarantee the leader exits; the
                // trailing `wait` reaps it so the pid slot is freed.
                kill_payload_process_group(&child, self.payload.name, self.payload.uses_parent_pgrp);
                let _ = child.wait();
                Err(e).with_context(|| format!("wait payload '{}'", self.payload.name))
            }
        }
    }

    /// SIGKILL the child **and every process it forked**, reap it,
    /// and return whatever stdout+stderr was captured along with the
    /// process exit code. Suitable for time-boxed background loads.
    ///
    /// The signal is delivered via `killpg(child_pid, SIGKILL)`
    /// rather than `child.kill()` because `build_command` places the
    /// payload at the head of its own process group. Multi-process
    /// payloads (stress-ng, schbench worker mode, fio --numjobs) fork
    /// descendants that keep stdout/stderr open; killing only the
    /// head would orphan those writers and block
    /// [`wait_and_capture`] forever, losing every metric.
    ///
    /// Metrics are also recorded to the per-test sidecar via the
    /// SHM ring; the returned tuple is a convenience view of the
    /// same values.
    pub fn kill(mut self) -> Result<(AssertResult, PayloadMetrics)> {
        let mut child = self
            .child
            .take()
            .ok_or_else(|| already_consumed(self.payload))?;
        kill_payload_process_group(&child, self.payload.name, self.payload.uses_parent_pgrp);
        match wait_and_capture(&mut child) {
            Ok(output) => Ok(evaluate(self.payload, &self.checks, output)),
            Err(e) => {
                // killpg + single-pid SIGKILL already ran at the
                // start; the reap or pipe-drain failed afterwards.
                // One more `wait` clears the zombie so the pid slot
                // is freed regardless of the capture error.
                let _ = child.wait();
                Err(e).with_context(|| format!("reap killed payload '{}'", self.payload.name))
            }
        }
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
                let mut child = self
                    .child
                    .take()
                    .expect("child still present on terminal branch");
                match wait_and_capture(&mut child) {
                    Ok(output) => Ok(Some(evaluate(self.payload, &self.checks, output))),
                    Err(e) => {
                        // Leader exited (try_wait returned Some) but
                        // pipe drain failed — descendants may still
                        // hold the pipes. Kill the group to release
                        // them, then reap the leader zombie.
                        kill_payload_process_group(&child, self.payload.name, self.payload.uses_parent_pgrp);
                        let _ = child.wait();
                        Err(e).with_context(|| format!("reap payload '{}'", self.payload.name))
                    }
                }
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

/// Drop-safety net for handles that fall out of scope without
/// going through [`PayloadHandle::wait`], [`PayloadHandle::kill`],
/// or [`PayloadHandle::try_wait`] (the three paths that
/// `.take()` the child normally). Drop routes the process group
/// through `kill_payload_process_group` — the SAME kill path the
/// explicit `kill()` method uses — so there is no redundant
/// `child.kill()` call: the killpg + single-pid SIGKILL inside
/// `kill_payload_process_group` is belt-and-suspenders-by-design
/// (see its doc for the pre-exec ESRCH race rationale), not
/// two independent kills stacked. `child.wait()` reaps the
/// zombie so the pid slot is freed even on the "dropped without
/// consume" path, and the one-shot eprintln tells the operator
/// metrics were lost.
impl Drop for PayloadHandle {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            kill_payload_process_group(&child, self.payload.name, self.payload.uses_parent_pgrp);
            let _ = child.wait();
            eprintln!(
                "ktstr: PayloadHandle for '{}' dropped without wait/kill — \
                 process group SIGKILLed, metrics not recorded.",
                self.payload.name,
            );
        }
    }
}

/// Send `SIGKILL` to the process group headed by `child` AND to the
/// leader pid directly.
///
/// `build_command` requests `CommandExt::process_group(0)` by default
/// so the child's pid becomes its own process-group leader, coordinated
/// with exec setup by the standard library. `killpg(pgid, SIGKILL)`
/// on the child's pid therefore reaches every fork descendant in
/// one shot — a single `child.kill()` would otherwise miss
/// grandchildren of multi-process payloads (stress-ng, schbench
/// worker mode, fio --numjobs) and those orphans would keep the
/// stdout/stderr pipes open, hanging `wait_and_capture` forever.
///
/// When `uses_parent_pgrp` is `true`, the child shares its parent's
/// pgrp ([`Payload::uses_parent_pgrp`] opted out of the fresh
/// process group for tty-dependent binaries). The `killpg` call is
/// skipped entirely in that case — issuing it would either hit
/// `ESRCH` (child is not a pgrp leader) in the common case or, worse,
/// target an unrelated group if the pgrp id happened to match a stale
/// value. Only the direct `kill(pid)` on the leader runs; opt-out
/// payloads accept responsibility for cleaning up their own
/// descendants.
///
/// The follow-up `kill(pid, SIGKILL)` on the leader pid is
/// belt-and-suspenders coverage for the edge case where `killpg`
/// alone is insufficient: the kernel-side pgid transition during
/// exec may not have completed yet when `killpg` fires, so
/// `killpg` returns `ESRCH` (no such group) and the leader
/// survives. A direct `kill(pid, SIGKILL)` always reaches the
/// leader, and the SIGKILL survives `execve(2)` to take effect
/// once exec completes (signal disposition is preserved across
/// exec; the pending signal is delivered once the new image
/// starts). SIGKILL is idempotent against zombies and
/// already-dead processes, so the extra signal is safe after a
/// successful `killpg` — a killpg that reached the leader has
/// already queued it for SIGKILL, and the follow-up `kill(pid)`
/// is a no-op on the terminated process.
///
/// `child.id()` returns `u32` for API ergonomics; on Linux the
/// kernel's `pid_max ≤ 2²²` guarantees the value fits in
/// [`libc::pid_t`]'s positive `i32` range, so `try_from` succeeds on
/// every live child. `debug_assert!(pgid > 0)` catches the
/// theoretically-impossible non-positive case before
/// [`nix::sys::signal::killpg`] would otherwise interpret it as a
/// broadcast target. `ESRCH` is logged as-a-no-op for both calls
/// — it means either "group/process already reaped" or "group not
/// yet set up"; the follow-up direct `kill` plus the leader's
/// eventual `waitpid` consumer handle both.
///
/// # Process-group escape (not handled here)
///
/// `killpg` reaches every process whose `getpgrp()` equals the
/// leader's pgid. A descendant that calls `setpgid(0, 0)` or
/// `setsid(2)` between fork and exit leaves the leader's process
/// group and becomes invisible to this helper — the escaping
/// descendant keeps running after SIGKILL and may continue holding
/// pipe fds that stall `wait_and_capture`. The bundled payloads
/// (stress-ng, schbench, fio) have not been audited for these
/// syscalls. `build_command` does not place an exec'd
/// child into any other group; this limitation applies only to
/// third-party payloads that deliberately re-parent themselves. The
/// mitigation path is the caller's: wrap the misbehaving payload in
/// a shell that traps SIGTERM → SIGKILL of its own descendants, or
/// register the leader as a subreaper
/// (`PR_SET_CHILD_SUBREAPER`) and reap orphans explicitly.
///
/// # Caller contract
///
/// Every caller MUST hold a live [`SigchldScope`] for the duration of
/// the `wait` / `waitpid` that reaps the leader after this call
/// returns. Without `SIG_DFL` for `SIGCHLD`, the guest init's
/// `SIG_IGN` default causes `wait` to block until the child is
/// re-reaped by init or to return `ECHILD` on an already-ignored
/// SIGCHLD. Audited caller set — every invocation of this function:
///
/// - `PayloadHandle::wait` (one site: error arm after a
///   `wait_and_capture` failure) — holds `self._sigchld`.
/// - `PayloadHandle::kill` (one site: top of the method, before
///   drain) — holds `self._sigchld`.
/// - `PayloadHandle::try_wait` (one site: error arm after a
///   terminal `try_wait` when drain fails) — holds `self._sigchld`.
/// - `impl Drop for PayloadHandle` (one site: handle dropped without
///   an explicit `wait`/`kill`/`try_wait` consume) — holds
///   `self._sigchld` for the full Drop body.
/// - `spawn_and_wait` (one site: error arm when `wait_and_capture`
///   fails on a timeout-less foreground spawn) — opens a local
///   `let _sigchld = SigchldScope::new()` at the top of the
///   function.
/// - `wait_with_deadline` (two sites: deadline-miss kill on expiry,
///   and error arm for drain failure on natural child exit) — runs
///   inside `spawn_and_wait`'s `_sigchld` scope, which is held
///   across the callee as a local binding.
///
/// Every `PayloadHandle` method is safe because `_sigchld` is
/// declared after `child` in the struct body; Rust drops fields in
/// declaration order so `_sigchld` lives longer than the child
/// `Option`, keeping the scope live for the full method body.
///
/// A future caller that skips either pattern will see
/// `waitpid` hang on some guest runtimes — add a `SigchldScope` at
/// the call site, or extend an enclosing type with a
/// `_sigchld: SigchldScope` field, before landing.
fn kill_payload_process_group(
    child: &std::process::Child,
    payload_name: &str,
    uses_parent_pgrp: bool,
) {
    let raw_pid = child.id();
    let pgid = match libc::pid_t::try_from(raw_pid) {
        Ok(p) if p > 0 => p,
        Ok(p) => {
            tracing::error!(
                payload = payload_name,
                pid = p,
                "child pid is non-positive — cannot kill process group; \
                 skipping kill (kernel's pid_max invariant violated, \
                 no safe target)"
            );
            return;
        }
        Err(_) => {
            tracing::error!(
                payload = payload_name,
                pid = raw_pid,
                "child pid exceeds pid_t range — cannot kill process group; \
                 skipping kill (Linux pid_max is 2^22 so this is only \
                 reachable on a non-Linux target or a kernel with an \
                 extended pid space)"
            );
            return;
        }
    };
    let pid = nix::unistd::Pid::from_raw(pgid);
    // `uses_parent_pgrp=true` means `build_command` did NOT request
    // `process_group(0)`, so the child shares its parent's process
    // group. A `killpg(pgid=child_pid, …)` call would target a group
    // the child does not lead — `ESRCH` in the common case, or (worse)
    // reach the ktstr process itself if a stale pid matches. Skip the
    // group kill entirely and rely on the direct `kill(pid)` below to
    // reap the leader. Multi-process tty-dependent payloads that
    // opt out of the fresh pgrp accept responsibility for their own
    // descendant cleanup (see `Payload::uses_parent_pgrp` doc).
    if !uses_parent_pgrp {
        match nix::sys::signal::killpg(pid, nix::sys::signal::Signal::SIGKILL) {
            Ok(()) => {}
            Err(nix::errno::Errno::ESRCH) => {
                tracing::debug!(
                    payload = payload_name,
                    pgid,
                    "ESRCH — payload process group already reaped",
                );
            }
            Err(e) => {
                tracing::warn!(payload = payload_name, pgid, %e, "killpg failed for payload process group");
            }
        }
    }
    match nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGKILL) {
        Ok(()) => {}
        Err(nix::errno::Errno::ESRCH) => {
            tracing::debug!(
                payload = payload_name,
                pid = pgid,
                "ESRCH — payload leader already reaped",
            );
        }
        Err(e) => {
            tracing::warn!(payload = payload_name, pid = pgid, %e, "direct kill failed for payload leader");
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
    // Pre-pass: exit-code checks first. Delegates to
    // `exit_code_mismatch_detail` so the detail's kind + message
    // stay in lockstep with the LlmExtract-failure branch in
    // `evaluate`. Short-circuit on mismatch — a bad exit probably
    // means the metric extraction found nothing useful.
    if let Some(detail) = exit_code_mismatch_detail(checks, pm.exit_code, stderr) {
        result.merge(AssertResult::fail(detail));
        return result;
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
            result.merge(AssertResult::fail(d));
        }
    }
    result
}

fn missing_metric(metric: &str) -> AssertDetail {
    AssertDetail {
        kind: DetailKind::Other,
        message: format!("metric '{metric}' not found in payload output"),
    }
}

/// Scan `checks` for the first `Check::ExitCodeEq` whose expected
/// value differs from `actual_exit_code` and return a matching
/// diagnostic [`AssertDetail`]. Returns `None` when no
/// `ExitCodeEq` check is declared, or when every declared one
/// matches the observed exit code.
///
/// Shared between [`evaluate`]'s LlmExtract-load-failure branch
/// and [`evaluate_checks`]'s pre-pass so both sites produce
/// bit-identical details for the same inputs — without this
/// helper the two branches drift on kind, message format, or
/// the "which Check wins" order.
fn exit_code_mismatch_detail(
    checks: &[Check],
    actual_exit_code: i32,
    stderr: &str,
) -> Option<AssertDetail> {
    checks.iter().find_map(|c| match c {
        Check::ExitCodeEq(expected) if actual_exit_code != *expected => Some(AssertDetail {
            kind: DetailKind::Other,
            message: format_exit_mismatch(actual_exit_code, *expected, stderr),
        }),
        _ => None,
    })
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

/// Build a [`Command`] with args, piped stdout/stderr, a `pre_exec`
/// hook that places the child at the head of its own process group
/// (so [`PayloadHandle::kill`] can signal every descendant), and an
/// optional second `pre_exec` hook that writes the child's PID into
/// the specified cgroup's `cgroup.procs` before `execve`.
///
/// Returns `Err` if the cgroup path cannot be converted to a
/// NUL-terminated C string — `resolve_cgroup_path` already rejects
/// interior NULs, but a `PathBuf` built from `OsStr` can still
/// carry one on exotic platforms, so we check explicitly.
fn build_command(
    binary: &str,
    args: &[String],
    cgroup_path: Option<&std::path::Path>,
    uses_parent_pgrp: bool,
) -> Result<std::process::Command> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(binary);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    if !uses_parent_pgrp {
        // `process_group(0)` requests a fresh process group with
        // the child as leader (pgid == child's pid). `killpg` on
        // the child's pid then reaches every fork descendant in
        // one signal — a single `child.kill()` would otherwise
        // miss grandchildren of multi-process payloads (stress-ng,
        // schbench worker mode, fio with multiple jobs), and
        // those orphans keep the stdout/stderr pipes open,
        // hanging `wait_and_capture` and discarding the metrics.
        //
        // Previously a hand-rolled `pre_exec(setpgid(0, 0))` hook
        // did the same job, but a `killpg` issued between
        // `fork(2)` and the child's `setpgid` completion could
        // return `ESRCH` (no such group) while the child and its
        // descendants survived. `CommandExt::process_group`
        // NARROWS that window: on `posix_spawn`-capable
        // platforms (and futures where `process_group` dispatches
        // to it) the pgid transition is kernel-sequenced with
        // exec and the race is eliminated. When the standard
        // library has to fall through to the fork+exec path —
        // which it does whenever a cgroup placement `pre_exec`
        // hook is also registered below, as `process_group(0)`
        // and any `pre_exec` together force the legacy path —
        // the remaining window is covered by the direct
        // `kill(pid, SIGKILL)` follow-up in
        // `kill_payload_process_group`.
        //
        // The `uses_parent_pgrp == true` branch SKIPS this call
        // so the child inherits the parent ktstr process's pgid.
        // Opt-in for tty-dependent payloads (shells, `less`,
        // anything that reads controlling-terminal foreground-
        // pgrp for job-control signalling) — a fresh pgrp reads
        // as "no job control" and breaks their signal
        // behaviour. The cost is that `killpg(child_pid, ...)`
        // no longer reaches descendants (the child isn't a
        // pgrp leader), so multi-process tty-dependent payloads
        // must react to SIGHUP / pipe close on their own or
        // risk orphaning — see the doc on `Payload::uses_parent_pgrp`.
        cmd.process_group(0);
    }

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

/// Actionable error wrapper for Command::spawn/.output failures.
/// ENOENT — the binary isn't on PATH inside the guest — gets the
/// remediation paths spelled out: `-i`/`--include-files` for CLI
/// invocations, pre-install in the initramfs for `#[ktstr_test]`
/// entries (which cannot pass `-i`). Other errors keep the minimal
/// `"spawn '<binary>'"` context so the underlying io::Error chain
/// surfaces unchanged.
///
/// **Shebang interpreter case.** `execve(2)` ALSO returns ENOENT
/// when `binary` is itself present but is a script whose `#!`
/// shebang names an interpreter that is missing in the guest
/// (e.g. `#!/usr/bin/python3` when python3 is absent from
/// initramfs). The kernel surfaces ENOENT with the script's path
/// even though the missing file is the interpreter — there is no
/// userspace signal that distinguishes "binary missing" from
/// "interpreter missing". The wrapped message therefore names
/// both the binary and the interpreter as candidate missing
/// artifacts and tells the operator to package both with `-i`
/// (CLI) or pre-install both in the initramfs
/// (`#[ktstr_test]`); the production message body carries this
/// guidance verbatim, the test
/// `spawn_error_context_enoent_attaches_remediation` pins it.
fn spawn_error_context(err: std::io::Error, binary: &str) -> anyhow::Error {
    if err.kind() == std::io::ErrorKind::NotFound {
        anyhow::Error::new(err).context(format!(
            "spawn '{binary}': binary not found on guest PATH. \
             Note: ENOENT on execve also fires when `{binary}` is a \
             script whose `#!` shebang points at an interpreter that \
             is missing in the guest (per execve(2)); in that case \
             the error names the script but the missing file is the \
             interpreter. Remediation: for CLI invocations (ktstr / \
             cargo-ktstr shell, run, …), package every missing \
             artifact with `-i {binary}` / `--include-files {binary}` \
             (and `-i <interpreter>` if the script names one) so each \
             lands on the guest PATH under `/include-files/`. For \
             `#[ktstr_test]` entries, pre-install the binary AND its \
             shebang interpreter in the base initramfs — the macro \
             surface does not expose `-i`."
        ))
    } else {
        anyhow::Error::new(err).context(format!("spawn '{binary}'"))
    }
}

/// RAII guard that saves the process's `SIGCHLD` disposition, sets
/// it to `SIG_DFL` on construction, and restores the saved value on
/// `Drop`. Required for [`spawn_and_wait`] and the background
/// [`spawn_child`] path because the guest ktstr-init sets
/// `SIGCHLD = SIG_IGN` at startup
/// (src/vmm/rust_init.rs:100, "Ignore SIGCHLD so child processes
/// don't become zombies"). Under `SIG_IGN` the kernel auto-reaps
/// children, so `waitpid(child_pid)` returns `ECHILD` and Rust
/// std's `Command::spawn()` / `.output()` / `Child::wait()`
/// internals panic with "wait() should either return Ok or panic".
///
/// The shell-exec mode at rust_init.rs:212-225 already documents
/// this exact gotcha and uses the same save/set-`SIG_DFL` /
/// restore-on-completion pattern. `PayloadRun::run` /
/// `PayloadRun::spawn` are the second dispatch site that needs it.
///
/// For background spawns, the guard lives on [`PayloadHandle`]
/// until `.wait()` / `.kill()` / `Drop` consumes the handle, so
/// the child is reap-able via `waitpid` for the entire window
/// between spawn and final disposition. Foreground spawns
/// (`spawn_and_wait`) scope the guard to the `.output()` call —
/// the child is reaped inline, no lingering state.
/// Pins the thread ID of the first `SigchldScope` constructed in
/// this process. Every subsequent construction must come from the
/// same thread: `libc::signal` is not thread-safe, and concurrent
/// installs from distinct threads would race on the process-wide
/// `SIGCHLD` disposition. The field is `AtomicUsize` carrying a
/// `ThreadId::as_u64()`-style encoding, with `0` meaning
/// "uninitialized" (no SigchldScope has been constructed yet in
/// this process).
///
/// Zero is a safe sentinel: `current_thread_id_nonzero()` (the
/// function that writes into this AtomicUsize) explicitly
/// squashes a hash result of 0 to 1 before returning — so no
/// legitimate thread-identity value written here is ever zero,
/// and the uninitialized AtomicUsize is unambiguous. (The hash
/// is produced via `DefaultHasher` on `ThreadId`, not via
/// `ThreadId::as_u64()` which is nightly-only; the squash-to-1
/// is what guarantees the non-zero invariant, not any property
/// of `ThreadId` / `NonZeroU64`.)
///
/// Multiple concurrent `SigchldScope` instances ARE allowed on
/// the same thread — each `PayloadHandle` carries one, and a
/// single-threaded caller can hold many handles simultaneously
/// without racing the libc::signal install. Drop order must
/// remain LIFO for the handler-restore chain to leave the
/// original disposition intact; this is the caller's obligation
/// (handles dropped in reverse creation order, which is the
/// default when locals go out of scope).
static SIGCHLD_SCOPE_OWNER_THREAD: AtomicUsize = AtomicUsize::new(0);

fn current_thread_id_nonzero() -> usize {
    // `ThreadId::as_u64()` is nightly-only; stable gives no public
    // integer accessor. We hash the ThreadId instead — collisions
    // across threads are astronomically unlikely within a single
    // process, and the check is a debug / soundness guard, not a
    // cryptographic boundary.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    // Squash zero to 1 so the sentinel stays reserved for
    // "uninitialized". Collision with thread 1's legitimate hash
    // is acceptable — it only means the check is slightly weaker
    // for that single thread, never falsely-positive.
    let id = h.finish() as usize;
    if id == 0 { 1 } else { id }
}

struct SigchldScope {
    prev: libc::sighandler_t,
}

impl SigchldScope {
    /// Save current `SIGCHLD` handler and install `SIG_DFL`.
    /// On host builds the init never flips SIGCHLD to SIG_IGN, so
    /// `prev` equals `SIG_DFL` and Drop is a no-op mathematically
    /// — the extra syscall is cheap and keeps behavior uniform
    /// between host and guest.
    ///
    /// # Panics
    ///
    /// Panics if called from a thread different from the one that
    /// constructed the first `SigchldScope` in this process.
    /// `libc::signal` is not thread-safe and cross-thread installs
    /// would race on the process-wide SIGCHLD disposition.
    fn new() -> Self {
        let tid = current_thread_id_nonzero();
        // Pin the first thread that ever constructs a SigchldScope
        // in this process. Subsequent threads are rejected.
        match SIGCHLD_SCOPE_OWNER_THREAD.compare_exchange(
            0,
            tid,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                // We are the first — thread pinned.
            }
            Err(pinned) if pinned == tid => {
                // Same thread as the first construction — OK.
            }
            Err(pinned) => {
                panic!(
                    "SigchldScope constructed on a different thread than the first \
                     owner (pinned thread id hash={pinned}, this thread's hash={tid}). \
                     libc::signal is not thread-safe; cross-thread installs race on \
                     the process-wide SIGCHLD disposition."
                );
            }
        }
        // SAFETY: SIGCHLD_SCOPE_OWNER_THREAD pins construction to
        // a single thread across the whole process, so no other
        // thread is concurrently installing a SIGCHLD handler.
        // Drop must run on the same thread (Rust's dropping
        // invariants hold if the handle stays !Send, which it is
        // by default since libc::sighandler_t contains a raw
        // pointer).
        let prev = unsafe { libc::signal(libc::SIGCHLD, libc::SIG_DFL) };
        SigchldScope { prev }
    }
}

impl Drop for SigchldScope {
    fn drop(&mut self) {
        // SAFETY: same rationale as `new` — the owner-thread pin
        // guarantees no concurrent installer on another thread.
        // Restoring in LIFO order across nested scopes unwinds
        // back to the original disposition; drop-order is the
        // caller's obligation.
        unsafe {
            libc::signal(libc::SIGCHLD, self.prev);
        }
    }
}

/// Foreground path: spawn + wait + capture. Used by `.run()`.
///
/// Wraps the child's lifetime in a [`SigchldScope`] so `waitpid`
/// sees `SIG_DFL` and returns the child's real exit status instead
/// of `ECHILD` under the guest init's `SIGCHLD = SIG_IGN`.
///
/// When `timeout` is `Some`, a poll loop bounds the payload's
/// runtime. Exceeding the deadline fires
/// [`kill_payload_process_group`] (killpg + single-pid SIGKILL)
/// so fork descendants die and release the pipes, then
/// [`wait_and_capture`] drains whatever output accumulated before
/// the kill. The `SpawnOutput` returned on timeout carries the
/// partial output and the post-kill exit code; the caller decides
/// whether that counts as a test failure.
fn spawn_and_wait(
    binary: &str,
    args: &[String],
    cgroup_path: Option<&std::path::Path>,
    timeout: Option<Duration>,
    uses_parent_pgrp: bool,
) -> Result<SpawnOutput> {
    let _sigchld = SigchldScope::new();
    let mut child = build_command(binary, args, cgroup_path, uses_parent_pgrp)?
        .spawn()
        .map_err(|e| spawn_error_context(e, binary))?;
    match timeout {
        Some(deadline) => wait_with_deadline(&mut child, deadline, binary, uses_parent_pgrp),
        None => match wait_and_capture(&mut child) {
            Ok(out) => Ok(out),
            Err(e) => {
                kill_payload_process_group(&child, binary, uses_parent_pgrp);
                let _ = child.wait();
                Err(e)
            }
        },
    }
}

/// Block in the kernel until the child exits or `timeout` elapses.
/// On expiry, kill the whole process group (killpg + single-pid
/// SIGKILL) and drain captured output.
///
/// Implementation uses `pidfd_open(2)` + `epoll_wait` so the waiter
/// is kernel-blocked instead of spinning on a 10ms `try_wait` loop.
/// The earlier poll burned one wake per 10ms for the entire payload
/// runtime (typically multi-second schbench / fio runs), producing a
/// small but measurable CPU spike on every timed payload; pidfd
/// parks the thread until the kernel signals child exit, so idle
/// waiters contribute zero CPU. Minimum kernel: Linux 5.3.
///
/// Deadline honoring: the `epoll_wait` timeout is re-derived from
/// `saturating_duration_since` each iteration so `EINTR` restarts
/// narrow the remaining window rather than extending it.
fn wait_with_deadline(
    child: &mut std::process::Child,
    timeout: Duration,
    payload_name: &str,
    uses_parent_pgrp: bool,
) -> Result<SpawnOutput> {
    use nix::sys::epoll::{Epoll, EpollCreateFlags, EpollEvent, EpollFlags, EpollTimeout};
    use std::os::fd::{AsFd, FromRawFd, OwnedFd};

    let deadline = std::time::Instant::now() + timeout;

    let pid = libc::pid_t::try_from(child.id())
        .expect("child pid fits in pid_t (Linux pid_max <= 2^22)");
    // `pidfd_open(pid, 0)`: returns an fd that becomes readable when
    // the pid exits. No `PIDFD_NONBLOCK` flag — epoll is the gate.
    let pidfd_raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0i32) };
    if pidfd_raw < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("pidfd_open({pid})"));
    }
    // SAFETY: the syscall succeeded and returned a fresh fd.
    let pidfd: OwnedFd = unsafe { OwnedFd::from_raw_fd(pidfd_raw as i32) };

    let epoll = Epoll::new(EpollCreateFlags::EPOLL_CLOEXEC)
        .with_context(|| "epoll_create1 for pidfd wait")?;
    // `data` field is unused — we only ever watch one fd. The add()
    // syscall still needs an `EpollEvent` with populated events.
    let event = EpollEvent::new(EpollFlags::EPOLLIN, 0);
    epoll
        .add(pidfd.as_fd(), event)
        .with_context(|| "epoll_ctl ADD pidfd")?;

    let mut events = [EpollEvent::empty()];
    loop {
        // Race-safe reap attempt first: if the child exited between
        // spawn and pidfd_open, or between iterations while we were
        // outside epoll_wait, `try_wait` catches it without a needless
        // syscall.
        if child
            .try_wait()
            .with_context(|| "try_wait child")?
            .is_some()
        {
            return match wait_and_capture(child) {
                Ok(out) => Ok(out),
                Err(e) => {
                    kill_payload_process_group(child, payload_name, uses_parent_pgrp);
                    let _ = child.wait();
                    Err(e)
                }
            };
        }

        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            kill_payload_process_group(child, payload_name, uses_parent_pgrp);
            return match wait_and_capture(child) {
                Ok(out) => Ok(out),
                Err(e) => {
                    let _ = child.wait();
                    Err(e).with_context(|| format!("drain after timeout of {timeout:?}"))
                }
            };
        }

        // `PollTimeout` (aliased as `EpollTimeout`) stores the value
        // as `i32`, so `TryFrom<u32>` rejects any input larger than
        // `i32::MAX` (~24.8 days of milliseconds). Clamp both casts —
        // `u128 → u32` and then `u32 → i32`-range — so a
        // `Duration::MAX`-shaped remainder saturates to the max
        // accepted value instead of bubbling up a conversion error.
        let ms_u32 = u32::try_from(remaining.as_millis()).unwrap_or(u32::MAX);
        let ms_u32 = std::cmp::min(ms_u32, i32::MAX as u32);
        let timeout_param = EpollTimeout::try_from(ms_u32)
            .with_context(|| "epoll timeout conversion")?;

        match epoll.wait(&mut events, timeout_param) {
            Ok(_) => {
                // Either the pidfd went readable (child exit) OR the
                // timeout fired (ready_count == 0). Loop back: the
                // `try_wait` at top handles the exit path, the
                // `remaining.is_zero()` branch handles the deadline.
            }
            Err(nix::errno::Errno::EINTR) => {
                // Signal interrupted the wait; loop and re-compute
                // the remaining window.
            }
            Err(e) => {
                return Err(anyhow::anyhow!("epoll_wait: {e}"));
            }
        }
    }
}

/// Background path: spawn without waiting. Returns the live
/// [`Child`] plus a [`SigchldScope`] that must be held for the
/// child's lifetime — [`PayloadHandle`] keeps it alive until
/// `.wait()` / `.kill()` / `Drop` so `waitpid` during reap sees
/// `SIG_DFL` and observes the child's real exit.
fn spawn_child(
    binary: &str,
    args: &[String],
    cgroup_path: Option<&std::path::Path>,
    uses_parent_pgrp: bool,
) -> Result<(std::process::Child, SigchldScope)> {
    let sigchld = SigchldScope::new();
    let child = build_command(binary, args, cgroup_path, uses_parent_pgrp)?
        .spawn()
        .map_err(|e| spawn_error_context(e, binary))?;
    Ok((child, sigchld))
}

/// Per-stream cap on captured child output. 16 MiB covers every
/// realistic benchmark stdout in the crate (typical schbench /
/// stress-ng / LLM-extract flows emit kilobytes to low-hundreds-of-KB)
/// with multiple orders of magnitude of slack, while cutting off
/// OOM pressure from a pathological payload that prints unbounded
/// GBs. Output past the cap is truncated, not errored, so downstream
/// (metric extraction, sidecar) still sees a prefix — the only loss
/// is the tail, which is rarely load-bearing. Each truncation emits
/// a paired `eprintln!` + `tracing::warn!` notice naming the stream
/// and the cap byte count.
pub(crate) const MAX_CAPTURED_STREAM_BYTES: u64 = 16 * 1024 * 1024;

/// Reap a (possibly already-killed) [`Child`]: wait for it to
/// exit, drain stdout + stderr, return the captured output.
///
/// Takes `&mut Child` so callers retain ownership and can
/// `kill_payload_process_group` + `wait` to clean up descendants
/// when this function returns `Err` (e.g. a reader thread panicked
/// or the wait syscall itself failed). An owned-child signature
/// would lose the handle inside this function and leave descendants
/// running because [`std::process::Child::drop`] is a no-op.
///
/// Sequential stdout-then-stderr reads deadlock when the child
/// fills one pipe buffer (typically 64KiB) while the other is
/// unread — the child blocks on write, the parent blocks on read
/// of the empty pipe. Drain both pipes concurrently via helper
/// threads, mirroring what `std::process::Command::output` does
/// for the foreground path.
///
/// Each reader thread wraps its source in
/// `Read::take(MAX_CAPTURED_STREAM_BYTES)` — see the constant's
/// rationale — so a runaway child cannot OOM the host. The tail
/// past the cap is discarded; `compose_prompt` / metric pipelines
/// always receive a bounded buffer.
fn wait_and_capture(child: &mut std::process::Child) -> Result<SpawnOutput> {
    let stdout_handle = child.stdout.take().map(|out| {
        std::thread::spawn(move || -> std::io::Result<(String, bool)> {
            drain_capped(out, "stdout")
        })
    });
    let stderr_handle = child.stderr.take().map(|err| {
        std::thread::spawn(move || -> std::io::Result<(String, bool)> {
            drain_capped(err, "stderr")
        })
    });
    let status = child.wait().with_context(|| "wait child")?;
    // `.join().unwrap()` below is NOT a bug: the workspace builds
    // with `panic = "abort"` in release (see Cargo.toml
    // `[profile.release]`), so a panicked reader thread aborts the
    // whole process and `join()` never returns an
    // `Err(Box<dyn Any + Send>)` (`std::thread::Result::Err`). The
    // historic `.map_err(|_| anyhow!("...panicked"))` arm could not
    // fire and misled readers into expecting a recoverable error.
    //
    // Under cargo's default `panic = "unwind"` (dev + test profiles
    // — only `[profile.release]` overrides to abort in this crate),
    // a reader-thread panic DOES unwind into `thread::Result::Err`;
    // `.unwrap()` re-panics on the main thread so the test harness
    // sees the failure instead of hanging on a half-drained stream.
    // The panic=abort caller contract holds in release; debug/test
    // callers get a loud re-panic rather than a silent wrong answer.
    // Either way no `Err` reaches the `?`.
    let (stdout, _stdout_truncated) = match stdout_handle {
        Some(h) => h.join().unwrap().with_context(|| "read child stdout")?,
        None => (String::new(), false),
    };
    let (stderr, _stderr_truncated) = match stderr_handle {
        Some(h) => h.join().unwrap().with_context(|| "read child stderr")?,
        None => (String::new(), false),
    };
    Ok(SpawnOutput {
        stdout,
        stderr,
        exit_code: status.code().unwrap_or(-1),
    })
}

/// Read `src` into a `String` with `MAX_CAPTURED_STREAM_BYTES` cap.
/// Returns `(buf, truncated)`. Emits a paired `eprintln!` +
/// `tracing::warn!` notice with the stream label (e.g. "stdout" /
/// "stderr") and cap byte count when the cap is hit.
///
/// Truncation is performed at the byte level on a `Vec<u8>` so a
/// split multi-byte UTF-8 char at the cap boundary cannot panic.
/// The final `String::from_utf8_lossy` replaces any invalid UTF-8
/// bytes with U+FFFD — including the partial-char split that byte
/// truncation can introduce. Non-truncated output preserves the
/// original bytes verbatim when it is already valid UTF-8; the
/// only behavioral delta vs the pre-cap `read_to_string` path is
/// that invalid UTF-8 in the child's full output now produces
/// replacement chars instead of an `io::ErrorKind::InvalidData`
/// upstream error. That trade is deliberate: past the cap there is
/// no way to report "invalid UTF-8" meaningfully since the tail is
/// gone, and making the pre-cap path lossy keeps semantics uniform.
fn drain_capped(
    src: impl std::io::Read,
    label: &'static str,
) -> std::io::Result<(String, bool)> {
    use std::io::Read;
    // One extra byte probes whether the source had more to offer —
    // `Take` returns EOF at exactly the cap, indistinguishable from
    // a child that emitted exactly `cap` bytes. We cap our own buffer
    // at MAX + 1 and check the read count.
    let mut raw: Vec<u8> = Vec::new();
    let n = src
        .take(MAX_CAPTURED_STREAM_BYTES + 1)
        .read_to_end(&mut raw)?;
    let truncated = n as u64 > MAX_CAPTURED_STREAM_BYTES;
    if truncated {
        raw.truncate(MAX_CAPTURED_STREAM_BYTES as usize);
        // Dual-emit: stderr for nextest-direct test runs (no
        // tracing subscriber installed in the default test-support
        // dispatch path), tracing for cargo-ktstr-wrapped runs and
        // structured-log consumers. Same rationale as the prefetch
        // notices — a silent-truncation warn that only reaches the
        // no-op dispatcher fails the visibility goal of this check.
        eprintln!(
            "ktstr: payload {label} exceeded {MAX_CAPTURED_STREAM_BYTES} bytes; tail discarded"
        );
        tracing::warn!(
            stream = label,
            cap_bytes = MAX_CAPTURED_STREAM_BYTES,
            "payload {label} exceeded capture cap; tail discarded",
        );
    }
    Ok((String::from_utf8_lossy(&raw).into_owned(), truncated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgroup::CgroupManager;
    use crate::test_support::{MetricSource, MetricStream, OutputFormat, Polarity, Scheduler};
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
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
    };

    const EEVDF_SCHED_PAYLOAD: Payload = Payload {
        name: "eevdf",
        kind: PayloadKind::Scheduler(&Scheduler::EEVDF),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
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
        use crate::test_support::{Metric, MetricSource, MetricStream, Polarity};
        let reversed = Check::range("iops", 100.0, 50.0);
        for actual in &[0.0, 50.0, 75.0, 100.0, 200.0, -1000.0, 1e9] {
            let pm = PayloadMetrics {
                metrics: vec![Metric {
                    name: "iops".to_string(),
                    value: *actual,
                    polarity: Polarity::HigherBetter,
                    unit: String::new(),
                    source: MetricSource::Json,
                    stream: MetricStream::Stdout,
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
            // `stderr_tail` returns a `String`, which Rust guarantees
            // is valid UTF-8 by construction. Calling it across every
            // byte offset proves the helper never panics on
            // multi-byte codepoint boundaries — the failure mode
            // that motivated this test is a panic from slicing at
            // mid-codepoint, not a corrupt string.
            let _ = stderr_tail(s, max);
        }
    }

    /// Production-scale counterpart to the boundary tests above. The
    /// existing small-string cases use ~20 bytes, well below the
    /// production [`STDERR_TAIL_BYTES`] threshold of 1024. This test
    /// lands a multi-byte character's interior byte on the truncation
    /// offset of a >1 KiB string, matching the actual shape of an
    /// overflowing stderr from a real payload. The snap-forward must
    /// advance past the interior byte so `stderr_tail` does not panic
    /// on mid-codepoint slicing.
    #[test]
    fn stderr_tail_snaps_forward_at_production_threshold() {
        // Layout: "A"*100 + "é" (2B) + "B"*1023 = 1125 bytes.
        // start = 1125 - 1024 = 101 — the interior byte of "é"
        // (whose boundary bytes are at 100 and 102). The snap-forward
        // advances start to 102, so the tail begins with the "B"
        // suffix rather than a corrupt split "é".
        let mut s = "A".repeat(100);
        s.push('é');
        s.push_str(&"B".repeat(1023));
        assert!(
            s.len() > STDERR_TAIL_BYTES,
            "fixture must exceed STDERR_TAIL_BYTES to exercise the truncation path",
        );
        let tail = stderr_tail(&s, STDERR_TAIL_BYTES);
        assert!(tail.starts_with("..."));
        assert!(
            tail[3..].starts_with('B'),
            "expected snap-forward past 'é' interior byte at >1 KiB, got prefix: {:?}",
            &tail[..20.min(tail.len())],
        );
    }

    /// Production-scale complement: when the truncation offset lands
    /// exactly on a multi-byte character's first byte (a boundary),
    /// the character survives — no off-by-one that would drop it.
    /// Covers the is_char_boundary-true branch of the snap-forward
    /// loop at the real [`STDERR_TAIL_BYTES`] size.
    #[test]
    fn stderr_tail_preserves_multibyte_at_production_boundary() {
        // Layout: "A"*100 + "é" (2B) + "B"*1022 = 1124 bytes.
        // start = 1124 - 1024 = 100 — the first byte of "é" (which
        // IS a char boundary). No snap runs; "é" is included whole.
        let mut s = "A".repeat(100);
        s.push('é');
        s.push_str(&"B".repeat(1022));
        assert!(
            s.len() > STDERR_TAIL_BYTES,
            "fixture must exceed STDERR_TAIL_BYTES to exercise the truncation path",
        );
        let tail = stderr_tail(&s, STDERR_TAIL_BYTES);
        assert!(tail.starts_with("..."));
        assert!(
            tail.contains('é'),
            "boundary-aligned 'é' at the >1 KiB truncation offset must survive, got prefix: {:?}",
            &tail[..40.min(tail.len())],
        );
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
                stream: MetricStream::Stdout,
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
                stream: MetricStream::Stdout,
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
                stream: MetricStream::Stdout,
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
                stream: MetricStream::Stdout,
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
                stream: MetricStream::Stdout,
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
                stream: MetricStream::Stdout,
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
                stream: MetricStream::Stdout,
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
                stream: MetricStream::Stdout,
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
            stream: MetricStream::Stdout,
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
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        resolve_polarities(&mut metrics, &HINTED);
        assert_eq!(metrics[0].polarity, Polarity::HigherBetter);
        assert_eq!(metrics[0].unit, "iops");
    }

    // -- PayloadHandle + .spawn() tests --

    const TRUE_BIN: Payload = Payload::binary("true_bin", "/bin/true");
    const FALSE_BIN: Payload = Payload::binary("false_bin", "/bin/false");

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
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
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
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
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
            stream: MetricStream::Stdout,
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
                    stream: MetricStream::Stdout,
                },
                Metric {
                    name: "lat".to_string(),
                    value: 900.0,
                    polarity: Polarity::LowerBetter,
                    unit: String::new(),
                    source: MetricSource::Json,
                    stream: MetricStream::Stdout,
                },
                Metric {
                    name: "cpu".to_string(),
                    value: 200.0,
                    polarity: Polarity::Unknown,
                    unit: String::new(),
                    source: MetricSource::Json,
                    stream: MetricStream::Stdout,
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
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
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
                    stream: MetricStream::Stdout,
                },
                Metric {
                    name: "lat_ns".to_string(),
                    value: 500.0,
                    polarity: Polarity::LowerBetter,
                    unit: "ns".to_string(),
                    source: MetricSource::LlmExtract,
                    stream: MetricStream::Stdout,
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
        use crate::test_support::{Metric, MetricHint, MetricSource, MetricStream, Polarity};
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
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        let mut ms = vec![
            Metric {
                name: "iops".into(),
                value: 1.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            },
            Metric {
                name: "unhinted".into(),
                value: 2.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            },
            Metric {
                name: "lat_ns".into(),
                value: 3.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
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
        use crate::test_support::{Metric, MetricHint, MetricSource, MetricStream, Polarity};
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
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        let mut ms = vec![Metric {
            name: "iops".into(),
            value: 1.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
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
        use crate::test_support::{Metric, MetricHint, MetricSource, MetricStream, Polarity};
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
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        let mut ms = vec![
            Metric {
                name: "iops".into(),
                value: 1.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            },
            Metric {
                name: "iops".into(),
                value: 2.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
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
        use crate::test_support::{Metric, MetricSource, MetricStream, Polarity};
        static NO_HINTS: crate::test_support::Payload = crate::test_support::Payload {
            name: "no_hints",
            kind: crate::test_support::PayloadKind::Binary("x"),
            output: crate::test_support::OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        let mut ms = vec![Metric {
            name: "anything".into(),
            value: 1.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
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

    // -- stdout-primary / stderr-fallback evaluation --

    const JSON_PAYLOAD: Payload = Payload {
        name: "json_payload",
        kind: PayloadKind::Binary("json_payload"),
        output: OutputFormat::Json,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
    };

    /// Well-behaved case: stdout carries the JSON document; stderr
    /// carries banner noise the extractor must NOT see. Merging the
    /// streams would pull the banner into the metric blob; the
    /// fallback contract keeps stdout canonical.
    #[test]
    fn evaluate_prefers_stdout_when_stdout_yields_metrics() {
        let output = SpawnOutput {
            stdout: r#"{"iops": 500}"#.to_string(),
            stderr: "unrelated banner: open fd error (ignore)".to_string(),
            exit_code: 0,
        };
        let (_, pm) = evaluate(&JSON_PAYLOAD, &[], output);
        assert_eq!(pm.metrics.len(), 1, "stdout JSON must win");
        assert_eq!(pm.metrics[0].name, "iops");
        assert_eq!(pm.metrics[0].value, 500.0);
    }

    /// schbench-style: the payload emits JSON percentiles on stderr,
    /// leaves stdout empty. Stdout-primary extraction returns an
    /// empty Vec, then the stderr fallback runs and produces the
    /// real metrics.
    #[test]
    fn evaluate_falls_back_to_stderr_when_stdout_empty() {
        let output = SpawnOutput {
            stdout: String::new(),
            stderr: r#"{"latency_ns": 1234}"#.to_string(),
            exit_code: 0,
        };
        let (_, pm) = evaluate(&JSON_PAYLOAD, &[], output);
        assert_eq!(pm.metrics.len(), 1, "stderr fallback must fire");
        assert_eq!(pm.metrics[0].name, "latency_ns");
        assert_eq!(pm.metrics[0].value, 1234.0);
    }

    /// Stdout present but unparseable (not-JSON prose); stderr
    /// carries the real document. `extract_metrics` returns `Vec`
    /// empty for malformed stdout, so the fallback runs against
    /// stderr and recovers the metrics. Pins that "non-empty stdout
    /// that yields no metrics" still triggers the retry — the
    /// stdout-primary contract gates on the result, not on emptiness.
    #[test]
    fn evaluate_falls_back_to_stderr_when_stdout_yields_no_metrics() {
        let output = SpawnOutput {
            stdout: "no json here, just prose from a banner line\n".to_string(),
            stderr: r#"{"throughput": 42}"#.to_string(),
            exit_code: 0,
        };
        let (_, pm) = evaluate(&JSON_PAYLOAD, &[], output);
        assert_eq!(pm.metrics.len(), 1, "stderr fallback must fire on empty result");
        assert_eq!(pm.metrics[0].name, "throughput");
        assert_eq!(pm.metrics[0].value, 42.0);
    }

    /// Stdout is valid JSON but contains only non-numeric leaves
    /// (strings, bools, nulls). `walk_json_leaves` at
    /// src/test_support/metrics.rs skips non-numeric leaves, so
    /// `extract_metrics` returns `Ok(vec![])` — a SUCCESSFUL parse
    /// with zero metrics. This is distinct from the
    /// "unparseable prose" case (`evaluate_falls_back_to_stderr_when_stdout_yields_no_metrics`
    /// above): that path fails to find any JSON document at all.
    /// The fallback condition at src/scenario/payload_run.rs:298
    /// gates on `metrics.is_empty()`, not on parse success, so both
    /// paths must fall back to stderr. This test pins that: the
    /// fallback must not surface the empty stdout set as the
    /// result, and the string/bool/null leaves from stdout must
    /// not leak into the returned metrics (they can't — the walker
    /// never emitted them — but a future refactor that concatenated
    /// streams or merged results could regress this).
    #[test]
    fn evaluate_falls_back_when_stdout_json_has_no_numeric_leaves() {
        let output = SpawnOutput {
            stdout: r#"{"status": "ok", "ready": true, "note": null}"#
                .to_string(),
            stderr: r#"{"iops": 9001}"#.to_string(),
            exit_code: 0,
        };
        let (_, pm) = evaluate(&JSON_PAYLOAD, &[], output);
        assert_eq!(
            pm.metrics.len(),
            1,
            "stderr fallback must fire when stdout parses but has \
             no numeric leaves; got metrics: {:?}",
            pm.metrics,
        );
        assert_eq!(pm.metrics[0].name, "iops");
        assert_eq!(pm.metrics[0].value, 9001.0);
        // No stray string/bool/null names leaked in from stdout.
        for m in &pm.metrics {
            assert!(
                !matches!(m.name.as_str(), "status" | "ready" | "note"),
                "non-numeric stdout leaf {:?} leaked into metrics",
                m.name,
            );
        }
    }

    /// Inverse of the above: both streams parse to JSON with no
    /// numeric leaves. Stdout extracts to `Ok(vec![])`, fallback
    /// fires, stderr also extracts to `Ok(vec![])`. Final metric
    /// set must be empty — not a synthetic pseudo-metric, not a
    /// silent merge of the two empty results with added string
    /// keys. Guards against a fallback refactor that might
    /// misinterpret "both empty" as "degenerate, emit a sentinel".
    #[test]
    fn evaluate_returns_empty_when_both_streams_have_no_numeric_leaves() {
        let output = SpawnOutput {
            stdout: r#"{"phase": "warmup"}"#.to_string(),
            stderr: r#"{"phase": "shutdown"}"#.to_string(),
            exit_code: 0,
        };
        let (_, pm) = evaluate(&JSON_PAYLOAD, &[], output);
        assert!(
            pm.metrics.is_empty(),
            "both-streams-non-numeric must produce no metrics; \
             got: {:?}",
            pm.metrics,
        );
    }

    /// Both streams empty ⇒ no metrics; the fallback guard
    /// (`!output.stderr.is_empty()`) skips the second call and the
    /// extractor is invoked exactly once against empty stdout.
    #[test]
    fn evaluate_returns_empty_metrics_on_empty_stdout_and_stderr() {
        let output = SpawnOutput {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
        };
        let (_, pm) = evaluate(&JSON_PAYLOAD, &[], output);
        assert!(pm.metrics.is_empty(), "both-empty must produce no metrics");
        assert_eq!(pm.exit_code, 0);
    }

    /// Multi-process payloads (schbench worker mode, stress-ng, fio)
    /// fork descendants that keep stdout/stderr open past the head
    /// process. Without a process-group kill, `wait_and_capture`
    /// would block on a pipe that never EOFs and the test would
    /// either hang or time out without metrics.
    ///
    /// The payload `/bin/sh -c 'sleep 60 & exec sleep 60'` uses the
    /// shell's head process to exec into `sleep 60` (pid == pgid)
    /// while a background `sleep 60` descendant inherits the pgid.
    /// A single-process SIGKILL would leave the background sleeper
    /// alive; `killpg` must reach it.
    ///
    /// The existence probe reaps may lag the SIGKILL delivery — the
    /// loop waits up to 30s, which covers slow CI runners, a
    /// heavily-loaded host, and the `waitpid` race where the child
    /// is dying but not yet reaped.
    #[cfg(unix)]
    #[test]
    fn kill_reaps_fork_descendants_via_process_group() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        const MULTI_SLEEPER: Payload = Payload {
            name: "multi_sleeper",
            kind: PayloadKind::Binary("/bin/sh"),
            output: crate::test_support::OutputFormat::ExitCode,
            default_args: &["-c", "sleep 60 & exec sleep 60"],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        let handle = PayloadRun::new(&ctx, &MULTI_SLEEPER)
            .spawn()
            .expect("spawn multi-sleeper");
        // The pgid equals the head child's pid. Capture it via the
        // public `pid()` accessor so the test does not reach into the
        // private `child` field.
        let pgid = libc::pid_t::try_from(handle.pid().expect("child still present"))
            .expect("child pid fits in pid_t");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (_, _) = handle.kill().expect("kill+reap");
        // After kill+reap the whole process group must be gone.
        // Poll `killpg(pgid, 0)` (existence probe) until ESRCH;
        // SIGKILL delivery + reap can lag the caller.
        loop {
            // SAFETY: killpg with signal 0 is a pure existence query
            // with no side effects beyond errno.
            let rc = unsafe { libc::killpg(pgid, 0) };
            if rc != 0 {
                let err = std::io::Error::last_os_error();
                assert_eq!(
                    err.raw_os_error(),
                    Some(libc::ESRCH),
                    "unexpected errno from killpg probe: {err}",
                );
                break;
            }
            if std::time::Instant::now() >= deadline {
                panic!("process group {pgid} still alive after kill+reap");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// Drop path of [`PayloadHandle`]: a handle that falls out of
    /// scope WITHOUT any consuming call (no `wait`, no `kill`, no
    /// `try_wait`) must still SIGKILL the whole process group via
    /// `kill_payload_process_group`. Without the Drop sweep,
    /// multi-process payloads whose head exits while descendants
    /// linger would leak their leader pid and keep descendants
    /// alive on init, polluting later tests with stray children
    /// holding file descriptors.
    ///
    /// Mirrors `kill_reaps_fork_descendants_via_process_group`
    /// (the explicit-`kill()` counterpart) but drops the handle
    /// instead of calling kill — pins the Drop implementation's
    /// killpg route against the same backgrounded-sleeper shape.
    #[cfg(unix)]
    #[test]
    fn drop_kills_fork_descendants_via_process_group() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        const MULTI_SLEEPER: Payload = Payload {
            name: "multi_sleeper_drop",
            kind: PayloadKind::Binary("/bin/sh"),
            output: crate::test_support::OutputFormat::ExitCode,
            default_args: &["-c", "sleep 60 & exec sleep 60"],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };
        let handle = PayloadRun::new(&ctx, &MULTI_SLEEPER)
            .spawn()
            .expect("spawn multi-sleeper");
        // Capture the pgid via the public `pid()` accessor before
        // dropping, so we can probe the group after the handle
        // goes out of scope.
        let pgid = libc::pid_t::try_from(handle.pid().expect("child still present"))
            .expect("child pid fits in pid_t");
        // Drop (no wait/kill/try_wait). The Drop impl at
        // src/scenario/payload_run.rs routes through
        // `kill_payload_process_group` + `child.wait()`.
        drop(handle);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            // SAFETY: killpg with signal 0 is a pure existence
            // query with no side effects beyond errno.
            let rc = unsafe { libc::killpg(pgid, 0) };
            if rc != 0 {
                let err = std::io::Error::last_os_error();
                assert_eq!(
                    err.raw_os_error(),
                    Some(libc::ESRCH),
                    "unexpected errno from killpg probe after drop: {err}",
                );
                break;
            }
            if std::time::Instant::now() >= deadline {
                panic!(
                    "process group {pgid} still alive 30 s after \
                     PayloadHandle drop — Drop-path killpg sweep \
                     failed to reach every member",
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// `uses_parent_pgrp = true` SKIPS the `process_group(0)` call
    /// in `build_command`, so the child inherits the test
    /// process's pgid instead of becoming its own pgrp leader.
    /// Spawn a sleeping binary via a Payload with the flag set,
    /// `getpgid` the child's pid, and assert it equals the
    /// parent's pgid — that pairs the "opt-out" directive with
    /// the observable behaviour.
    #[cfg(unix)]
    #[test]
    fn payload_uses_parent_pgrp_opts_out_of_process_group() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        const PARENT_PGRP_SLEEPER: Payload = Payload {
            name: "parent_pgrp_sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: crate::test_support::OutputFormat::ExitCode,
            default_args: &["60"],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: true,
            known_flags: None,
        };
        let handle = PayloadRun::new(&ctx, &PARENT_PGRP_SLEEPER)
            .spawn()
            .expect("spawn opt-out sleeper");
        let child_pid = libc::pid_t::try_from(handle.pid().expect("child alive"))
            .expect("child pid fits in pid_t");
        // SAFETY: getpgid(pid) is a pure lookup with no side
        // effects beyond returning the queried pid's pgid (or -1
        // + errno on failure).
        let child_pgid = unsafe { libc::getpgid(child_pid) };
        // SAFETY: getpgid(0) returns the CURRENT process's pgid
        // and cannot fail.
        let parent_pgid = unsafe { libc::getpgid(0) };
        assert!(child_pgid > 0, "getpgid(child) failed: {child_pgid}");
        assert_eq!(
            child_pgid, parent_pgid,
            "uses_parent_pgrp=true payload must inherit the \
             parent's pgid (child_pgid={child_pgid}, \
             parent_pgid={parent_pgid}); a mismatch means \
             `build_command` still called `process_group(0)` \
             despite the opt-out",
        );
        // kill() on a handle whose child is not a pgrp leader
        // still reaps normally — kill_payload_process_group
        // falls back to single-pid SIGKILL. Consume the handle
        // so the sleeper doesn't outlive the test; a silent
        // failure here would mask the test's own regression
        // (e.g. a broken kill path that leaks sleepers).
        let _ = handle.kill().expect("kill opt-out sleeper");
    }

    /// `wait_with_deadline` timeout kills the whole process group
    /// via killpg + single-pid SIGKILL. Spawn a multi-process
    /// shell, drive `wait_with_deadline` with a 500 ms budget
    /// (so the whole test fits inside the 30s-slack nextest
    /// budget without standing up a whole scenario) and probes the
    /// pgid with `killpg(pgid, 0)` after the deadline fires —
    /// ESRCH proves the sweep reached every member.
    #[cfg(unix)]
    #[test]
    fn wait_with_deadline_timeout_kills_process_group() {
        use std::os::unix::process::CommandExt;
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "sleep 60 & exec sleep 60"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0)
            .spawn()
            .expect("spawn multi-sleeper");
        let pgid = libc::pid_t::try_from(child.id())
            .expect("child pid fits in pid_t");
        let start = std::time::Instant::now();
        let out = wait_with_deadline(
            &mut child,
            std::time::Duration::from_millis(500),
            "multi_sleeper_timeout",
            false,
        )
        .expect("wait_with_deadline returns Ok on timeout");
        let elapsed = start.elapsed();
        // Timeout must actually have elapsed — if the function
        // returns almost instantly, the pidfd/epoll loop is
        // short-circuiting on an unrelated signal rather than
        // waiting for the 500 ms deadline.
        assert!(
            elapsed >= std::time::Duration::from_millis(400),
            "wait_with_deadline returned after only {elapsed:?}; \
             deadline was 500 ms — check the epoll loop is honoring \
             the timeout rather than unblocking on an unrelated event",
        );
        // The drain result must be captured even on timeout.
        // After SIGKILL the child's std::process::ExitStatus has
        // no numeric code (killed by signal, `status.code()`
        // returns None), so `wait_and_capture` defaults to -1 at
        // src/scenario/payload_run.rs per the `unwrap_or(-1)`
        // fallback in its status-code read. Pin that contract —
        // a future refactor that surfaces the signal number as
        // the exit_code would regress this.
        assert_eq!(out.exit_code, -1);
        // After timeout-driven kill+reap, the whole process group
        // must be gone. Poll `killpg(pgid, 0)` (existence probe)
        // until ESRCH — SIGKILL delivery + reap of the backgrounded
        // sleeper can lag the caller, so allow up to 30 s (matches
        // kill_reaps_fork_descendants_via_process_group's budget).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            // SAFETY: killpg with signal 0 is a pure existence
            // query with no side effects beyond errno.
            let rc = unsafe { libc::killpg(pgid, 0) };
            if rc != 0 {
                let err = std::io::Error::last_os_error();
                assert_eq!(
                    err.raw_os_error(),
                    Some(libc::ESRCH),
                    "unexpected errno from killpg probe after \
                     timeout: {err}",
                );
                break;
            }
            if std::time::Instant::now() >= deadline {
                panic!(
                    "process group {pgid} still alive 30 s after \
                     wait_with_deadline timeout fired — killpg sweep \
                     in the timeout branch failed to reach every \
                     member",
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// [`spawn_error_context`] is the sole place the spawn-error
    /// surface is shaped. An `ErrorKind::NotFound` must grow the full
    /// remediation chain (include-files for CLI invocations,
    /// pre-install for `#[ktstr_test]` entries); every other errno
    /// MUST keep the minimal `"spawn '<binary>'"` context so the
    /// underlying `io::Error` chain surfaces unchanged. Pin both
    /// directions so a regression that (a) swallows the NotFound
    /// remediation or (b) sprays the remediation across unrelated
    /// errno paths surfaces here.
    #[test]
    fn spawn_error_context_enoent_attaches_remediation() {
        let err =
            std::io::Error::from_raw_os_error(libc::ENOENT);
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let wrapped = super::spawn_error_context(err, "fio");
        let rendered = format!("{wrapped:#}");
        // Binary name still appears so `grep fio` still finds the error.
        assert!(rendered.contains("spawn 'fio'"), "got: {rendered}");
        // Remediation text must surface both mitigation paths.
        assert!(
            rendered.contains("not found on guest PATH"),
            "ENOENT branch must name the PATH miss: {rendered}"
        );
        assert!(
            rendered.contains("-i fio") || rendered.contains("--include-files fio"),
            "ENOENT branch must name the `-i <binary>` remediation: {rendered}"
        );
        assert!(
            rendered.contains("#[ktstr_test]"),
            "ENOENT branch must name the ktstr_test pre-install remediation: {rendered}"
        );
    }

    #[test]
    fn spawn_error_context_non_enoent_keeps_minimal_context() {
        // EACCES is a representative non-NotFound errno. Any
        // remediation text leaking onto this path would mislead
        // users who e.g. hit a permission problem — the remediation
        // paths above (include-files, pre-install) are orthogonal
        // to the failure mode. Pin the absence.
        let err =
            std::io::Error::from_raw_os_error(libc::EACCES);
        assert_ne!(err.kind(), std::io::ErrorKind::NotFound);
        let wrapped = super::spawn_error_context(err, "fio");
        let rendered = format!("{wrapped:#}");
        assert!(rendered.contains("spawn 'fio'"), "got: {rendered}");
        assert!(
            !rendered.contains("-i fio"),
            "non-ENOENT must not leak the `-i` remediation: {rendered}"
        );
        assert!(
            !rendered.contains("--include-files"),
            "non-ENOENT must not leak the --include-files remediation: {rendered}"
        );
        assert!(
            !rendered.contains("#[ktstr_test]"),
            "non-ENOENT must not leak the ktstr_test remediation: {rendered}"
        );
        assert!(
            !rendered.contains("not found on guest PATH"),
            "non-ENOENT must not claim 'not found on PATH': {rendered}"
        );
    }
}
