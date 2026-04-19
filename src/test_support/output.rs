//! Guest-output and console parsing for ktstr test results.
//!
//! Every VM-hosted `#[ktstr_test]` run emits three distinguishable
//! streams that the host must parse before it can judge pass/fail or
//! surface useful diagnostics:
//!
//! - **AssertResult JSON** on stdout/COM2, bracketed by
//!   [`RESULT_START`] / [`RESULT_END`] — and also on the SHM ring with
//!   `MSG_TYPE_TEST_RESULT` for the primary transport. See
//!   [`print_assert_result`] (guest emit), [`parse_assert_result`]
//!   (host parse from COM2 string), and [`parse_assert_result_shm`]
//!   (host parse from SHM drain).
//! - **Scheduler log** on COM2, bracketed by [`SCHED_OUTPUT_START`] /
//!   [`SCHED_OUTPUT_END`]. [`parse_sched_output`] extracts the block;
//!   [`sched_log_fingerprint`] returns the last non-empty line as a
//!   failure fingerprint so duplicate failures cluster visually in
//!   nextest output.
//! - **sched_ext dump** on COM1 (kernel trace_pipe). Parsed by
//!   [`extract_sched_ext_dump`] — it filters lines containing
//!   `sched_ext_dump` out of the raw trace stream.
//!
//! Supporting helpers:
//! - [`extract_kernel_version`] reads the `Linux version X.Y.Z ...`
//!   line from boot output.
//! - [`extract_panic_message`] pulls the guest's `PANIC:` line (the
//!   Rust panic hook in `rust_init.rs` writes these to COM2).
//! - [`classify_init_stage`] reads [`SENTINEL_INIT_STARTED`] and
//!   [`SENTINEL_PAYLOAD_STARTING`] to pinpoint where in the init
//!   lifecycle a silent failure happened.
//! - [`format_console_diagnostics`] composes the `--- diagnostics ---`
//!   block appended to failed-test error output.
//! - [`ensure_kvm`] is the pre-flight check that aborts early if
//!   `/dev/kvm` isn't readable/writable — no point constructing a VM
//!   builder if the kernel won't let us open the device.

use anyhow::{Context, Result};

use crate::assert::AssertResult;
use crate::vmm;

/// Delimiters for the AssertResult JSON in guest output.
pub(crate) const RESULT_START: &str = "===KTSTR_TEST_RESULT_START===";
pub(crate) const RESULT_END: &str = "===KTSTR_TEST_RESULT_END===";

/// Write AssertResult to SHM (primary) and stdout/COM2 (fallback).
pub(crate) fn print_assert_result(r: &AssertResult) {
    if let Ok(json) = serde_json::to_string(r) {
        vmm::shm_ring::write_msg(vmm::shm_ring::MSG_TYPE_TEST_RESULT, json.as_bytes());
        println!("{RESULT_START}");
        println!("{json}");
        println!("{RESULT_END}");
    }
}

/// Extract AssertResult from SHM drain entries.
pub(crate) fn parse_assert_result_shm(
    shm: Option<&vmm::shm_ring::ShmDrainResult>,
) -> Result<AssertResult> {
    let shm = shm.ok_or_else(|| anyhow::anyhow!("no SHM data"))?;
    let entry = shm
        .entries
        .iter()
        .rev()
        .find(|e| e.msg_type == vmm::shm_ring::MSG_TYPE_TEST_RESULT && e.crc_ok)
        .ok_or_else(|| anyhow::anyhow!("no test result in SHM"))?;
    serde_json::from_slice(&entry.payload).context("parse AssertResult from SHM")
}

/// Parse AssertResult from guest COM2 output between delimiters.
pub(crate) fn parse_assert_result(output: &str) -> Result<AssertResult> {
    let json = crate::probe::output::extract_section(output, RESULT_START, RESULT_END);
    anyhow::ensure!(!json.is_empty(), "missing result delimiters");
    serde_json::from_str(&json).context("parse AssertResult JSON")
}

/// Delimiters for the scheduler log in guest output (written by init script).
pub(crate) const SCHED_OUTPUT_START: &str = "===SCHED_OUTPUT_START===";
pub(crate) const SCHED_OUTPUT_END: &str = "===SCHED_OUTPUT_END===";

/// Extract the last non-empty line from the scheduler log.
///
/// This serves as a failure fingerprint: when many tests fail with the
/// same scheduler error, the fingerprint makes identical failures
/// visually obvious in nextest output.
pub(crate) fn sched_log_fingerprint(output: &str) -> Option<&str> {
    let log = parse_sched_output(output)?;
    log.lines().rev().find(|l| !l.trim().is_empty())
}

/// Extract the scheduler log from guest output between delimiters.
/// Returns `None` if the delimiters are absent or the content is empty.
pub(crate) fn parse_sched_output(output: &str) -> Option<&str> {
    // Cannot use extract_section here: it returns an owned String,
    // but callers need a borrowed &str tied to `output`'s lifetime.
    //
    // `find` on the start marker and `rfind` on the end marker: if the
    // scheduler's own output happens to contain the end sentinel string
    // (e.g. a stack trace that quotes the marker), `find` would
    // truncate the section early. `rfind` anchors on the last
    // occurrence, which is the real terminator emitted by the guest's
    // post-scenario shutdown path.
    let start = output.find(SCHED_OUTPUT_START)?;
    let end = output.rfind(SCHED_OUTPUT_END)?;
    let after_marker = start + SCHED_OUTPUT_START.len();
    if after_marker >= end {
        return None;
    }
    let content = output[after_marker..end].trim();
    if content.is_empty() {
        return None;
    }
    Some(content)
}

/// Extract sched_ext_dump lines from COM1 kernel console (trace_pipe output).
///
/// The trace_pipe stream contains lines with `sched_ext_dump:` prefixes when
/// a SysRq-D dump is triggered. Collects all such lines into a single string.
/// Returns `None` if no dump lines are present.
pub(crate) fn extract_sched_ext_dump(output: &str) -> Option<String> {
    let lines: Vec<&str> = output
        .lines()
        .filter(|l| l.contains("sched_ext_dump"))
        .collect();
    if lines.is_empty() {
        return None;
    }
    Some(lines.join("\n"))
}

/// Extract kernel version from console output (COM1/stderr).
///
/// Looks for "Linux version X.Y.Z..." in boot messages.
pub(crate) fn extract_kernel_version(console: &str) -> Option<String> {
    for line in console.lines() {
        if let Some(rest) = line.split("Linux version ").nth(1) {
            return Some(rest.split_whitespace().next().unwrap_or("").to_string());
        }
    }
    None
}

/// Extract the panic message from guest COM2 output.
///
/// Looks for a line containing "PANIC:" (written by the guest panic hook
/// in `rust_init.rs`). Returns the trimmed text after the "PANIC:" prefix,
/// or `None` if no panic line is present.
pub(crate) fn extract_panic_message(output: &str) -> Option<&str> {
    output.lines().find(|l| l.contains("PANIC:")).map(|l| {
        l.trim()
            .strip_prefix("PANIC:")
            .map(|s| s.trim_start())
            .unwrap_or(l.trim())
    })
}

/// Written to COM2 by Rust init after filesystem mounts complete.
pub(crate) const SENTINEL_INIT_STARTED: &str = "KTSTR_INIT_STARTED";

/// Written to COM2 by guest dispatch immediately before the test
/// function is called.
pub(crate) const SENTINEL_PAYLOAD_STARTING: &str = "KTSTR_PAYLOAD_STARTING";

/// Classify the failure stage based on which sentinels appear in COM2 output.
pub(crate) fn classify_init_stage(output: &str) -> &'static str {
    if output.contains(SENTINEL_PAYLOAD_STARTING) {
        "payload started but produced no test result"
    } else if output.contains(SENTINEL_INIT_STARTED) {
        "init started but payload never ran (cgroup/scheduler setup failed)"
    } else {
        "init script never started (kernel or mount failure)"
    }
}

/// Format diagnostic info from COM1 kernel console output, VM exit code,
/// and init stage classification.
///
/// Returns an empty string when there is nothing useful to show.
/// Otherwise returns a section starting with a blank line, containing the
/// init stage, exit code, and the last few lines of kernel console output.
pub(crate) fn format_console_diagnostics(
    console: &str,
    exit_code: i32,
    init_stage: &str,
) -> String {
    const TAIL_LINES: usize = 20;
    let trimmed = console.trim();
    if trimmed.is_empty() && exit_code == 0 {
        return String::new();
    }
    let mut parts = Vec::with_capacity(3);
    parts.push(format!("stage: {init_stage}"));
    let exit_label = if exit_code < 0 {
        // Negative exit codes are typically negated errno values.
        crate::errno_name(-exit_code)
            .map(|name| format!("exit_code={exit_code} ({name})"))
            .unwrap_or_else(|| format!("exit_code={exit_code}"))
    } else {
        format!("exit_code={exit_code}")
    };
    parts.push(exit_label);
    if !trimmed.is_empty() {
        let lines: Vec<&str> = trimmed.lines().collect();
        // Show all lines when a crash is detected (PANIC: in output),
        // otherwise show only the last TAIL_LINES.
        let has_crash = lines.iter().any(|l| l.contains("PANIC:"));
        let limit = if has_crash { lines.len() } else { TAIL_LINES };
        let start = lines.len().saturating_sub(limit);
        let tail = &lines[start..];
        let truncated = !console.ends_with('\n');
        parts.push(format!(
            "console ({} lines{}):\n{}{}",
            tail.len(),
            if truncated { ", truncated" } else { "" },
            tail.join("\n"),
            if truncated { " [truncated]" } else { "" },
        ));
    }
    format!("\n\n--- diagnostics ---\n{}", parts.join("\n"))
}

/// Verify that `/dev/kvm` is accessible for read+write.
pub(crate) fn ensure_kvm() -> Result<()> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
        .context(
            "/dev/kvm not accessible — KVM is required for ktstr_test. \
             Check that KVM is enabled and your user is in the kvm group.",
        )?;
    Ok(())
}
