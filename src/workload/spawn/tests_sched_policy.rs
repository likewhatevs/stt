//! Spawn-pipeline tests — sched_policy group.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::types::*;
use super::super::worker::*;
use super::testing::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Deadline path: file never appears, `liveness_pid` stays alive
/// (use self), helper panics with "did not write ready file" once
/// the timeout elapses. Short timeout (50ms) to keep the test
/// fast.
#[test]
fn wait_for_file_or_panic_panics_on_deadline_miss() {
    let nonexistent = std::env::temp_dir().join(format!(
        "ktstr-wfp-deadline-never-exists-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&nonexistent);
    let self_pid = unsafe { libc::getpid() };
    let result = std::panic::catch_unwind(|| {
        wait_for_file_or_panic(
            &nonexistent,
            Duration::from_millis(50),
            self_pid,
            "deadline path",
        );
    });
    let err = result.expect_err("must panic when deadline expires");
    let msg = crate::test_support::test_helpers::panic_payload_to_string(err);
    assert!(
        msg.contains("did not write ready file"),
        "panic must name the deadline-miss path, got: {msg}"
    );
}
/// Deadline-elapse path: `stop` stays `false`, so
/// [`wait_for_deadline`] runs until `timeout` elapses. Uses a
/// 1-second deadline; asserts the call returned no earlier than
/// ~900ms (granularity slop from the 10ms sleep cadence).
#[test]
fn wait_for_deadline_waits_full_duration_when_stop_stays_false() {
    let stop = AtomicBool::new(false);
    let start = Instant::now();
    wait_for_deadline(&stop, Duration::from_secs(1));
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(900),
        "wait_for_deadline must hold for ~full duration; elapsed={elapsed:?}",
    );
    assert!(
        elapsed < Duration::from_millis(2_000),
        "wait_for_deadline must not massively overshoot; elapsed={elapsed:?}",
    );
}
/// Stop-flip path: another thread flips `stop` to `true` ~50ms in,
/// and [`wait_for_deadline`] returns shortly after. Asserts the
/// call returned well before the 10s deadline.
#[test]
fn wait_for_deadline_returns_early_when_stop_is_set() {
    use std::sync::Arc;
    let stop = Arc::new(AtomicBool::new(false));
    let stop_setter = Arc::clone(&stop);
    let flipper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        stop_setter.store(true, Ordering::Relaxed);
    });
    let start = Instant::now();
    wait_for_deadline(&stop, Duration::from_secs(10)); // 10s deadline — should never hit
    let elapsed = start.elapsed();
    flipper.join().unwrap();
    assert!(
        elapsed < Duration::from_secs(1),
        "wait_for_deadline must return promptly after stop flips; elapsed={elapsed:?}",
    );
}
