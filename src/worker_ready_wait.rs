//! Test-side poll helper for the worker ready marker. Separated from
//! [`crate::worker_ready`] because this helper references
//! [`crate::scenario::payload_run::PayloadHandle`]; the bin crate
//! `ktstr-jemalloc-alloc-worker` pulls `worker_ready.rs` in via
//! `#[path]` and must stay dependency-free (see that module's doc for
//! why). This module is library-only.

use crate::scenario::payload_run::PayloadHandle;
use crate::worker_ready::worker_ready_marker_path;

/// Poll for the worker's ready marker with a deadline, returning
/// early if the worker exits before writing the marker or after
/// writing but before the caller's subsequent dispatch.
///
/// The caller supplies `role` (e.g. `"worker"`, `"churn worker"`) and
/// `exit_code_legend` (a variant-specific decoder for the
/// worker-binary exit codes the caller wants printed in the error
/// message). Worker cleanup on timeout happens via
/// [`PayloadHandle::drop`] when the caller's `Result` error
/// propagates — calling `PayloadHandle::kill(self)` here would take
/// the handle by value, which we can't do behind an `&mut` borrow.
///
/// Consolidates what used to be two near-identical 20-line poll
/// loops in `tests/jemalloc_probe_tests.rs` — a rename of the marker
/// path, a change in poll interval, or a new early-exit shape now
/// edits one site instead of two.
pub fn wait_for_worker_ready(
    worker: &mut PayloadHandle,
    worker_pid: u32,
    timeout: std::time::Duration,
    role: &str,
    exit_code_legend: &str,
) -> anyhow::Result<()> {
    let ready_path = worker_ready_marker_path(worker_pid);
    let deadline = std::time::Instant::now() + timeout;
    while !std::path::Path::new(&ready_path).exists() {
        if let Some((_, metrics)) = worker.try_wait()? {
            anyhow::bail!(
                "{role} pid={worker_pid} exited before creating ready marker \
                 {ready_path} (exit_code={} — see stderr; worker exit codes: \
                 {exit_code_legend})",
                metrics.exit_code,
            );
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!(
                "{role} pid={worker_pid} did not create ready marker {ready_path} \
                 within {timeout:?}",
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Narrow-race close: the worker may have written the marker and
    // then died between the write and the caller's next probe
    // dispatch (unusual — the worker is supposed to park — but a
    // fatal Drop or kernel SIGKILL could still fire). One more
    // try_wait surfaces that case with an actionable error instead
    // of letting the caller burn wall-time on a dead pid.
    if let Some((_, metrics)) = worker.try_wait()? {
        anyhow::bail!(
            "{role} pid={worker_pid} exited after writing ready marker but \
             before the caller's next dispatch (exit_code={} — see stderr)",
            metrics.exit_code,
        );
    }
    Ok(())
}
