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
//! - **Scheduler log** on COM2, bracketed by
//!   [`SCHED_OUTPUT_START`](crate::verifier::SCHED_OUTPUT_START) /
//!   [`SCHED_OUTPUT_END`](crate::verifier::SCHED_OUTPUT_END) — both
//!   live in [`crate::verifier`] since that is the primary host-side
//!   consumer that also parses the BPF verifier log carried inside the
//!   block.
//!   [`parse_sched_output`](crate::verifier::parse_sched_output)
//!   extracts the block; [`sched_log_fingerprint`] returns the last
//!   non-empty line as a failure fingerprint so duplicate failures
//!   cluster visually in nextest output.
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

use anyhow::{Context, Result};

use crate::assert::AssertResult;
use crate::verifier::parse_sched_output;
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

/// Extract the last non-empty line from the scheduler log.
///
/// This serves as a failure fingerprint: when many tests fail with the
/// same scheduler error, the fingerprint makes identical failures
/// visually obvious in nextest output.
pub(crate) fn sched_log_fingerprint(output: &str) -> Option<&str> {
    let log = parse_sched_output(output)?;
    log.lines().rev().find(|l| !l.trim().is_empty())
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

/// Prefix written by guest init on final exit. The full marker is
/// `KTSTR_EXIT=<code>`; callers that need the code parse the suffix.
pub(crate) const SENTINEL_EXIT_PREFIX: &str = "KTSTR_EXIT=";

/// Prefix written by `shell --exec` after the user command returns.
/// Carries the exec'd process's exit code (`KTSTR_EXEC_EXIT=<code>`).
pub(crate) const SENTINEL_EXEC_EXIT_PREFIX: &str = "KTSTR_EXEC_EXIT=";

/// Written by guest init when the scheduler process dies during
/// startup. Paired with `KTSTR_EXIT=1` on the surrounding lines.
pub(crate) const SENTINEL_SCHEDULER_DIED: &str = "SCHEDULER_DIED";

/// Written by guest init when the scheduler process stays alive but
/// never attaches to sched_ext (BPF verifier reject, ops mismatch,
/// sysfs absent). Emitted as `SCHEDULER_NOT_ATTACHED: <reason>`; the
/// reason suffix is appended by the caller. Paired with `KTSTR_EXIT=1`
/// on the surrounding lines.
pub(crate) const SENTINEL_SCHEDULER_NOT_ATTACHED: &str = "SCHEDULER_NOT_ATTACHED";

// ---------------------------------------------------------------------------
// Init-stage classification labels
// ---------------------------------------------------------------------------
//
// Returned by `classify_init_stage` and asserted by eval.rs tests via
// substring match. Shared constants keep the production label and the
// test pins from drifting silently.

/// Stage label when no init sentinel appears in COM2 — indicates the
/// guest kernel or initramfs never reached Rust init. Pinned by
/// `classify_no_sentinels` (output.rs) and `eval_no_sentinels_shows_initramfs_failure`
/// (eval.rs).
pub(crate) const STAGE_INIT_NOT_STARTED: &str =
    "init script never started (kernel or mount failure)";

/// Stage label when `KTSTR_INIT_STARTED` was written but the payload
/// sentinel never appeared — cgroup or scheduler setup failed after
/// filesystem mounts. Pinned by `classify_init_started_only` (output.rs)
/// and `eval_init_started_but_no_payload` (eval.rs).
pub(crate) const STAGE_INIT_STARTED_NO_PAYLOAD: &str =
    "init started but payload never ran (cgroup/scheduler setup failed)";

/// Stage label when `KTSTR_PAYLOAD_STARTING` was written but no
/// AssertResult JSON followed — the test function entered and then
/// crashed, hung, or produced no output. Pinned by
/// `classify_payload_starting` / `classify_payload_starting_without_init`
/// (output.rs) and `eval_payload_started_no_result` (eval.rs).
pub(crate) const STAGE_PAYLOAD_STARTED_NO_RESULT: &str =
    "payload started but produced no test result";

/// Classify the failure stage based on which sentinels appear in COM2 output.
pub(crate) fn classify_init_stage(output: &str) -> &'static str {
    if output.contains(SENTINEL_PAYLOAD_STARTING) {
        STAGE_PAYLOAD_STARTED_NO_RESULT
    } else if output.contains(SENTINEL_INIT_STARTED) {
        STAGE_INIT_STARTED_NO_PAYLOAD
    } else {
        STAGE_INIT_NOT_STARTED
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::build_assert_result_json;
    use super::*;
    use crate::assert::{AssertDetail, DetailKind};
    use crate::verifier::{SCHED_OUTPUT_END, SCHED_OUTPUT_START};

    // -- parse_assert_result --

    #[test]
    fn parse_assert_result_valid() {
        let json = build_assert_result_json(true, vec![]);
        let output = format!("noise\n{RESULT_START}\n{json}\n{RESULT_END}\nmore");
        let r = parse_assert_result(&output).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn parse_assert_result_missing_start() {
        let output = format!("no start\n{RESULT_END}\n");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_missing_end() {
        let output = format!("{RESULT_START}\n{{}}");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_failed() {
        let json = build_assert_result_json(
            false,
            vec![AssertDetail::new(DetailKind::Stuck, "stuck 3000ms")],
        );
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let r = parse_assert_result(&output).unwrap();
        assert!(!r.passed);
        assert_eq!(r.details, vec!["stuck 3000ms"]);
    }

    #[test]
    fn parse_assert_result_malformed_json() {
        let output = format!("{RESULT_START}\nnot valid json\n{RESULT_END}");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_empty_json_between_delimiters() {
        let output = format!("{RESULT_START}\n\n{RESULT_END}");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_with_details() {
        let json = build_assert_result_json(
            false,
            vec![
                AssertDetail::new(DetailKind::Other, "err1"),
                AssertDetail::new(DetailKind::Other, "err2"),
            ],
        );
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let r = parse_assert_result(&output).unwrap();
        assert!(!r.passed);
        assert_eq!(r.details.len(), 2);
        assert_eq!(r.details[0], "err1");
        assert_eq!(r.details[1], "err2");
    }

    // -- sched_log_fingerprint --

    #[test]
    fn sched_log_fingerprint_last_line() {
        let output = format!(
            "{SCHED_OUTPUT_START}\nstarting scheduler\nError: apply_cell_config BPF program returned error -2\n{SCHED_OUTPUT_END}",
        );
        assert_eq!(
            sched_log_fingerprint(&output),
            Some("Error: apply_cell_config BPF program returned error -2"),
        );
    }

    #[test]
    fn sched_log_fingerprint_skips_trailing_blanks() {
        let output = format!("{SCHED_OUTPUT_START}\nfatal error here\n\n\n{SCHED_OUTPUT_END}",);
        assert_eq!(sched_log_fingerprint(&output), Some("fatal error here"));
    }

    #[test]
    fn sched_log_fingerprint_none_without_markers() {
        assert!(sched_log_fingerprint("no markers").is_none());
    }

    #[test]
    fn sched_log_fingerprint_none_empty_content() {
        let output = format!("{SCHED_OUTPUT_START}\n\n{SCHED_OUTPUT_END}");
        assert!(sched_log_fingerprint(&output).is_none());
    }

    // -- extract_sched_ext_dump --

    #[test]
    fn extract_sched_ext_dump_present() {
        let output = "noise\n  ktstr-0  [001]  0.500: sched_ext_dump: Debug dump\n  ktstr-0  [001]  0.501: sched_ext_dump: scheduler state\nmore";
        let parsed = extract_sched_ext_dump(output);
        assert!(parsed.is_some());
        let dump = parsed.unwrap();
        assert!(dump.contains("sched_ext_dump: Debug dump"));
        assert!(dump.contains("sched_ext_dump: scheduler state"));
    }

    #[test]
    fn extract_sched_ext_dump_absent() {
        assert!(extract_sched_ext_dump("no dump lines here").is_none());
    }

    #[test]
    fn extract_sched_ext_dump_empty_output() {
        assert!(extract_sched_ext_dump("").is_none());
    }

    // -- extract_kernel_version --

    #[test]
    fn extract_kernel_version_from_boot() {
        let console = "[    0.000000] Linux version 6.14.0-rc3+ (user@host) (gcc) #1 SMP\n\
                        [    0.001000] Command line: console=ttyS0";
        assert_eq!(
            extract_kernel_version(console),
            Some("6.14.0-rc3+".to_string()),
        );
    }

    #[test]
    fn extract_kernel_version_none() {
        assert_eq!(extract_kernel_version("no kernel here"), None);
    }

    #[test]
    fn extract_kernel_version_bare() {
        let console = "Linux version 6.12.0";
        assert_eq!(extract_kernel_version(console), Some("6.12.0".to_string()),);
    }

    // -- format_console_diagnostics --

    #[test]
    fn format_console_diagnostics_empty_ok() {
        assert_eq!(format_console_diagnostics("", 0, "test stage"), "");
    }

    #[test]
    fn format_console_diagnostics_empty_nonzero_exit() {
        let s = format_console_diagnostics("", 1, "test stage");
        assert!(s.contains("exit_code=1"));
        assert!(s.contains("--- diagnostics ---"));
        assert!(s.contains("stage: test stage"));
        assert!(!s.contains("console ("));
    }

    #[test]
    fn format_console_diagnostics_with_console() {
        let console = "line1\nline2\nKernel panic - not syncing\n";
        let s = format_console_diagnostics(console, -1, "payload started");
        assert!(s.contains("exit_code=-1"));
        assert!(s.contains("console (3 lines)"));
        assert!(s.contains("Kernel panic"));
        assert!(s.contains("stage: payload started"));
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn format_console_diagnostics_truncates_long() {
        let lines: Vec<String> = (0..50).map(|i| format!("boot line {i}")).collect();
        let console = format!("{}\n", lines.join("\n"));
        let s = format_console_diagnostics(&console, 0, "test");
        assert!(s.contains("console (20 lines)"));
        assert!(s.contains("boot line 49"));
        assert!(!s.contains("boot line 29"));
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn format_console_diagnostics_short_console() {
        let console = "Linux version 6.14.0\nbooted ok\n";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("console (2 lines)"));
        assert!(s.contains("Linux version 6.14.0"));
        assert!(s.contains("booted ok"));
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn format_console_diagnostics_no_truncation_with_trailing_newline() {
        let console = "line1\nline2\nline3\n";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("console (3 lines)"));
        assert!(!s.contains("truncated"));
        assert!(!s.contains("[truncated]"));
    }

    #[test]
    fn format_console_diagnostics_truncation_without_trailing_newline() {
        let console = "line1\nline2\npartial li";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains(", truncated)"));
        assert!(s.contains("partial li [truncated]"));
    }

    // -- classify_init_stage --

    #[test]
    fn classify_no_sentinels() {
        assert_eq!(classify_init_stage(""), STAGE_INIT_NOT_STARTED);
    }

    #[test]
    fn classify_init_started_only() {
        assert_eq!(
            classify_init_stage("KTSTR_INIT_STARTED\nsome noise"),
            STAGE_INIT_STARTED_NO_PAYLOAD,
        );
    }

    #[test]
    fn classify_payload_starting() {
        let output = "KTSTR_INIT_STARTED\nKTSTR_PAYLOAD_STARTING\nsome output";
        assert_eq!(classify_init_stage(output), STAGE_PAYLOAD_STARTED_NO_RESULT);
    }

    #[test]
    fn classify_payload_starting_without_init() {
        // Edge case: payload sentinel present but init sentinel
        // missing. payload_starting implies init ran, so classify as
        // payload started.
        assert_eq!(
            classify_init_stage("KTSTR_PAYLOAD_STARTING"),
            STAGE_PAYLOAD_STARTED_NO_RESULT,
        );
    }

    // -- extract_panic_message --

    #[test]
    fn extract_panic_message_found() {
        let output = "noise\nPANIC: panicked at src/main.rs:5: oh no\nmore";
        assert_eq!(
            extract_panic_message(output),
            Some("panicked at src/main.rs:5: oh no"),
        );
    }

    #[test]
    fn extract_panic_message_absent() {
        assert!(extract_panic_message("no panic here").is_none());
    }

    #[test]
    fn extract_panic_message_empty() {
        assert!(extract_panic_message("").is_none());
    }
}
