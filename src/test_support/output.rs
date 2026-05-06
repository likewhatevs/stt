//! Guest-output and console parsing for ktstr test results.
//!
//! Every VM-hosted `#[ktstr_test]` run emits these distinguishable
//! streams that the host must parse before it can judge pass/fail or
//! surface useful diagnostics:
//!
//! - **AssertResult bincode** on the bulk data channel under
//!   `MSG_TYPE_TEST_RESULT`. See [`print_assert_result`] (guest
//!   emit) and [`parse_assert_result_from_drain`] (host parse).
//!   The wire format is bincode v2 with `bincode::config::standard()`
//!   so guest and host stay in lock-step on the encoding choice.
//! - **Scheduler log** in `MSG_TYPE_SCHED_LOG` chunks on the bulk
//!   data channel. The chunks carry the
//!   [`SCHED_OUTPUT_START`](crate::verifier::SCHED_OUTPUT_START) /
//!   [`SCHED_OUTPUT_END`](crate::verifier::SCHED_OUTPUT_END) markers
//!   verbatim (defined in [`crate::verifier`] since that is the
//!   primary host-side consumer that also parses the BPF verifier
//!   log carried inside the block).
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
//!   Rust panic hook in `rust_init.rs` still writes these to COM2
//!   because the panic hook cannot block on virtio backpressure;
//!   every other guest stream now travels over the bulk port).
//! - [`classify_init_stage`] walks the bucketed lifecycle phase
//!   vec to pinpoint where in the init lifecycle a silent failure
//!   happened.
//! - [`format_console_diagnostics`] composes the `--- diagnostics ---`
//!   block appended to failed-test error output.

use anyhow::{Context, Result};

use crate::assert::AssertResult;
use crate::verifier::parse_sched_output;
use crate::vmm;

/// Emit AssertResult to the host over the bulk data channel
/// (`MSG_TYPE_TEST_RESULT`) using bincode v2 with
/// `bincode::config::standard()`. The encoding choice is paired
/// with the host's [`parse_assert_result_from_drain`] decoder so
/// layout never diverges.
///
/// Pre-1.0: the legacy COM2 `RESULT_START` / `RESULT_END` JSON
/// fallback is gone — bulk port is the only transport.
pub(crate) fn print_assert_result(r: &AssertResult) {
    vmm::guest_comms::send_test_result(r);
}

/// Extract AssertResult from a bulk-drain entries.
///
/// Walks entries in reverse so the last-emitted (most recent)
/// `MSG_TYPE_TEST_RESULT` frame wins — matches the existing
/// "latest wins" semantics for the primary transport.
pub(crate) fn parse_assert_result_from_drain(
    drain: Option<&vmm::host_comms::BulkDrainResult>,
) -> Result<AssertResult> {
    let drain = drain.ok_or_else(|| anyhow::anyhow!("no guest messages"))?;
    let entry = drain
        .entries
        .iter()
        .rev()
        .find(|e| e.msg_type == vmm::wire::MSG_TYPE_TEST_RESULT && e.crc_ok)
        .ok_or_else(|| anyhow::anyhow!("no test result in guest messages"))?;
    let (result, _) = bincode::serde::decode_from_slice::<AssertResult, _>(
        &entry.payload,
        bincode::config::standard(),
    )
    .context("decode AssertResult bincode payload from drain")?;
    Ok(result)
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
/// Looks for a line whose trimmed form starts with `PANIC:` (the
/// prefix the guest panic hook in `rust_init.rs` writes verbatim).
/// Returns the text after the prefix with leading whitespace
/// trimmed; returns `None` when no panic line is present.
///
/// The match deliberately requires the prefix to anchor at the
/// start of a (trimmed) line. A `.contains("PANIC:")` would also
/// match unrelated mid-line occurrences — a console log that
/// happened to mention the literal text "PANIC:" inside an info
/// message ("expected PANIC: from this test") would be
/// misclassified as the panic line. Guest panic-hook output is
/// always emitted at the start of a line in `rust_init.rs`, so
/// the prefix anchor is always satisfied for genuine panics.
pub(crate) fn extract_panic_message(output: &str) -> Option<&str> {
    output
        .lines()
        .map(|l| l.trim())
        .find_map(|l| l.strip_prefix("PANIC:").map(str::trim_start))
}

// Pre-bincode-migration the guest emitted COM2 sentinel strings
// (`KTSTR_INIT_STARTED`, `KTSTR_PAYLOAD_STARTING`,
// `KTSTR_EXIT=<code>`, `KTSTR_EXEC_EXIT=<code>`, `SCHEDULER_DIED`,
// `SCHEDULER_NOT_ATTACHED: <reason>`) that the host scraped to
// classify boot phase, exit code, and scheduler-attach failures.
// All five now travel as typed `MSG_TYPE_LIFECYCLE` /
// `MSG_TYPE_EXIT` / `MSG_TYPE_EXEC_EXIT` frames on the bulk data
// port (see `crate::vmm::guest_comms::send_lifecycle`,
// `send_exit`, and `send_exec_exit`). The `SENTINEL_*` const
// strings are gone — host code walks
// `result.guest_messages.entries` and matches on
// `crate::vmm::wire::LifecyclePhase` / `MSG_TYPE_*` discriminants.

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

/// Classify the failure stage based on which `MSG_TYPE_LIFECYCLE`
/// phase events appear in the bulk-port drain.
///
/// Pre-bincode-migration: read `KTSTR_INIT_STARTED` /
/// `KTSTR_PAYLOAD_STARTING` substrings from COM2 output. Now the
/// guest emits each phase as a typed
/// [`crate::vmm::wire::LifecyclePhase`] frame; the classifier walks
/// the entries in arrival order and picks the latest known phase,
/// which is the deepest stage the guest reached before failing.
///
/// Returns the matching `STAGE_*` label. `None` drain (the bulk
/// port produced no entries at all) maps to
/// [`STAGE_INIT_NOT_STARTED`] — same as the prior "no sentinel
/// seen" branch.
pub(crate) fn classify_init_stage(
    drain: Option<&vmm::host_comms::BulkDrainResult>,
) -> &'static str {
    use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE};
    let Some(drain) = drain else {
        return STAGE_INIT_NOT_STARTED;
    };
    let mut latest: Option<LifecyclePhase> = None;
    for e in &drain.entries {
        if e.msg_type != MSG_TYPE_LIFECYCLE || !e.crc_ok || e.payload.is_empty() {
            continue;
        }
        if let Some(phase) = LifecyclePhase::from_wire(e.payload[0]) {
            latest = Some(phase);
        }
    }
    match latest {
        Some(LifecyclePhase::PayloadStarting)
        | Some(LifecyclePhase::SchedulerDied)
        | Some(LifecyclePhase::SchedulerNotAttached) => STAGE_PAYLOAD_STARTED_NO_RESULT,
        Some(LifecyclePhase::InitStarted) => STAGE_INIT_STARTED_NO_PAYLOAD,
        None => STAGE_INIT_NOT_STARTED,
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
        // Two independent truncation conditions:
        //   - `window_dropped`: the tail window dropped earlier
        //     lines (more lines existed than fit in TAIL_LINES).
        //   - `last_line_incomplete`: the captured stream ends
        //     without a newline, so the final line is partial.
        // Conflating these — as a single `truncated` flag — would
        // claim "truncated" on a complete short console that just
        // happens to lack a trailing newline (e.g. a line buffer
        // flushed without `\n`), or hide window truncation behind
        // the same label as a partial last line. Track and report
        // them separately so the operator knows whether earlier
        // boot output went missing or only the final line was cut.
        let window_dropped = start > 0;
        let last_line_incomplete = !console.ends_with('\n');
        let header_suffix = match (window_dropped, last_line_incomplete) {
            (true, true) => ", window-truncated, last line incomplete",
            (true, false) => ", window-truncated",
            (false, true) => ", last line incomplete",
            (false, false) => "",
        };
        let body_suffix = if last_line_incomplete {
            " [partial]"
        } else {
            ""
        };
        parts.push(format!(
            "console ({} lines{}):\n{}{}",
            tail.len(),
            header_suffix,
            tail.join("\n"),
            body_suffix,
        ));
    }
    format!("\n\n--- diagnostics ---\n{}", parts.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::{assert_result_tlv_entry, build_assert_result};
    use super::*;
    use crate::assert::{AssertDetail, DetailKind};
    use crate::verifier::{SCHED_OUTPUT_END, SCHED_OUTPUT_START};
    use crate::vmm::host_comms::BulkDrainResult;
    use crate::vmm::wire::{MSG_TYPE_TEST_RESULT, ShmEntry};

    fn drain_with_assert(r: &AssertResult) -> BulkDrainResult {
        BulkDrainResult {
            entries: vec![assert_result_tlv_entry(r)],
        }
    }

    // -- parse_assert_result_from_drain --

    /// A bincode-encoded `MSG_TYPE_TEST_RESULT` entry decodes back to
    /// the same `AssertResult`. Pin the round-trip so a future
    /// encoding tweak (e.g. config swap to legacy()) trips this test
    /// before reaching the real wire.
    #[test]
    fn parse_assert_result_from_drain_round_trips() {
        let original = build_assert_result(true, vec![]);
        let drain = drain_with_assert(&original);
        let r = parse_assert_result_from_drain(Some(&drain)).unwrap();
        assert!(r.passed);
    }

    /// A failing `AssertResult` round-trips its details verbatim.
    #[test]
    fn parse_assert_result_from_drain_failed_preserves_details() {
        let original = build_assert_result(
            false,
            vec![AssertDetail::new(DetailKind::Stuck, "stuck 3000ms")],
        );
        let drain = drain_with_assert(&original);
        let r = parse_assert_result_from_drain(Some(&drain)).unwrap();
        assert!(!r.passed);
        assert_eq!(r.details, vec!["stuck 3000ms"]);
    }

    /// `None` drain → "no guest messages" error. Mirrors the host
    /// path where `result.guest_messages` is `None` because the bulk
    /// port never produced a single byte.
    #[test]
    fn parse_assert_result_from_drain_none_returns_error() {
        assert!(parse_assert_result_from_drain(None).is_err());
    }

    /// Empty `BulkDrainResult` → "no test result in guest messages"
    /// error. Mirrors the host path where the bulk port produced
    /// other entries (PROFRAW, STIMULUS, EXIT) but never a
    /// MSG_TYPE_TEST_RESULT frame.
    #[test]
    fn parse_assert_result_from_drain_empty_returns_error() {
        let drain = BulkDrainResult { entries: vec![] };
        assert!(parse_assert_result_from_drain(Some(&drain)).is_err());
    }

    /// CRC-bad MSG_TYPE_TEST_RESULT entry is ignored; the helper
    /// returns "no test result" rather than feeding a corrupt
    /// payload into the bincode decoder.
    #[test]
    fn parse_assert_result_from_drain_skips_crc_bad_entries() {
        let drain = BulkDrainResult {
            entries: vec![ShmEntry {
                msg_type: MSG_TYPE_TEST_RESULT,
                payload: vec![0xff; 4],
                crc_ok: false,
            }],
        };
        assert!(parse_assert_result_from_drain(Some(&drain)).is_err());
    }

    /// Multiple MSG_TYPE_TEST_RESULT entries — the helper picks the
    /// last (latest-emitted) one. Pins "latest wins" semantics.
    #[test]
    fn parse_assert_result_from_drain_picks_latest() {
        let early = build_assert_result(false, vec![AssertDetail::new(DetailKind::Other, "early")]);
        let late = build_assert_result(true, vec![]);
        let drain = BulkDrainResult {
            entries: vec![assert_result_tlv_entry(&early), assert_result_tlv_entry(&late)],
        };
        let r = parse_assert_result_from_drain(Some(&drain)).unwrap();
        assert!(r.passed, "latest entry must win");
    }

    /// Malformed bincode payload (right msg_type, right CRC, wrong
    /// bytes) surfaces the bincode decode error rather than silently
    /// returning a default `AssertResult`.
    #[test]
    fn parse_assert_result_from_drain_rejects_garbage_payload() {
        let payload = vec![0xab; 8];
        let drain = BulkDrainResult {
            entries: vec![ShmEntry {
                msg_type: MSG_TYPE_TEST_RESULT,
                payload,
                crc_ok: true,
            }],
        };
        assert!(parse_assert_result_from_drain(Some(&drain)).is_err());
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

    /// E2E expectation: when scx-ktstr's ops.dump / dump_cpu /
    /// dump_task callbacks fire under a real stall, every line they
    /// emit via `scx_bpf_dump` reaches userspace through the kernel
    /// `sched_ext_dump` tracepoint (kernel/sched/ext.c:dump_line)
    /// and lands in the captured trace_pipe stream as
    /// `<task>  [<cpu>]  <ts>: sched_ext_dump: <ops.dump-line>`.
    ///
    /// `scx-ktstr/src/bpf/main.bpf.c` declares three ops callbacks:
    ///   - `ktstr_dump`: emits `ktstr scheduler state:` and three
    ///     follow-up lines naming `stall`, `crash`, `degrade_rt`,
    ///     etc.;
    ///   - `ktstr_dump_cpu`: emits `ktstr cpu N: no per-cpu state`
    ///     for every NON-idle CPU (idle CPUs are skipped so the
    ///     kernel's `if (idle && used == seq_buf_used(&ns))` gate
    ///     suppresses the per-CPU section, see
    ///     kernel/sched/ext.c:6127-6283);
    ///   - `ktstr_dump_task`: emits `ktstr task: magic=0x... counter=N`
    ///     for every runnable task whose `scx_task_data(p)` is
    ///     non-null.
    ///
    /// This test pins those three string shapes against the
    /// trace_pipe parser. A regression that renames a marker, drops
    /// the `\n`, or changes the prefix scheme breaks the host-side
    /// failure-dump rendering — the parser would silently fall
    /// through to "no dump lines" and the operator would lose every
    /// scheduler-author hint that ops.dump exists to surface. The
    /// fixture is the same wire format the kernel produces; the
    /// test asserts the parser can slice it cleanly without losing
    /// any of the three layers.
    #[test]
    fn extract_sched_ext_dump_recovers_ktstr_ops_dump_layers() {
        // Synthetic trace_pipe stream emitted by the kernel's
        // `dump_line` -> `trace_sched_ext_dump(line_buf)` path when
        // scx-ktstr's ops.dump callbacks fire. Real lines carry
        // a leading task/CPU/timestamp prefix and the
        // `sched_ext_dump:` tag injected by the tracepoint.
        let trace_pipe = "\
   scx_-1234  [002]   100.000000: sched_ext_dump: Debug dump triggered by error\n\
   scx_-1234  [002]   100.000001: sched_ext_dump: ktstr scheduler state:\n\
   scx_-1234  [002]   100.000002: sched_ext_dump:   stall=1 crash=0 degrade_rt=0\n\
   scx_-1234  [002]   100.000003: sched_ext_dump:   rodata: degrade=0 slow=0 scattershot=0 verify_loop=0 fail_verify=0\n\
   scx_-1234  [002]   100.000004: sched_ext_dump:   ktstr_alloc_count=42 degrade_cnt=0 slow_cnt=0\n\
   scx_-1234  [002]   100.000005: sched_ext_dump: CPU states\n\
   scx_-1234  [002]   100.000006: sched_ext_dump: ----------\n\
   scx_-1234  [002]   100.000007: sched_ext_dump: ktstr cpu 1: no per-cpu state\n\
   scx_-1234  [002]   100.000008: sched_ext_dump: ktstr cpu 3: no per-cpu state\n\
   scx_-1234  [002]   100.000009: sched_ext_dump:   ktstr task: magic=0xdeadbeefcafebabe counter=42\n\
   unrelated noise that must NOT match\n";

        let parsed = extract_sched_ext_dump(trace_pipe)
            .expect("parser must surface every sched_ext_dump line");

        // ops.dump layer.
        assert!(
            parsed.contains("ktstr scheduler state:"),
            "ops.dump header missing; parser dropped the `ktstr scheduler state:` \
             line emitted by scx-ktstr's ktstr_dump callback. parsed: {parsed}"
        );
        assert!(
            parsed.contains("stall=1 crash=0 degrade_rt=0"),
            "ops.dump body missing; parser dropped the runtime-flag line. \
             parsed: {parsed}"
        );
        assert!(
            parsed.contains("ktstr_alloc_count=42"),
            "ops.dump body missing; parser dropped the alloc-count line. \
             parsed: {parsed}"
        );

        // ops.dump_cpu layer (NON-idle CPUs only — idle CPUs emit
        // nothing per the kernel's idle-suppression gate).
        assert!(
            parsed.contains("ktstr cpu 1: no per-cpu state"),
            "ops.dump_cpu output missing; parser dropped the CPU 1 \
             marker emitted by ktstr_dump_cpu for non-idle CPUs. \
             parsed: {parsed}"
        );
        assert!(
            parsed.contains("ktstr cpu 3: no per-cpu state"),
            "ops.dump_cpu output missing; parser dropped the CPU 3 \
             marker. parsed: {parsed}"
        );

        // ops.dump_task layer — magic reads as the LE u64 of
        // KTSTR_ARENA_MAGIC verbatim (the BPF format string uses
        // %llx so it appears in the dump as a hex literal).
        assert!(
            parsed.contains("ktstr task: magic=0xdeadbeefcafebabe counter=42"),
            "ops.dump_task output missing; parser dropped the per-task \
             magic/counter line emitted by ktstr_dump_task. parsed: \
             {parsed}"
        );

        // Negative: the non-prefixed noise line MUST NOT slip into
        // the dump string — extract_sched_ext_dump filters by the
        // `sched_ext_dump` substring, so unrelated trace_pipe lines
        // are dropped.
        assert!(
            !parsed.contains("unrelated noise"),
            "parser leaked a non-sched_ext_dump line: {parsed}"
        );
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
        assert!(!s.contains("window-truncated"));
        assert!(!s.contains("last line incomplete"));
        assert!(!s.contains("[partial]"));
    }

    #[test]
    fn format_console_diagnostics_truncates_long() {
        // Window truncation: 50 lines reduces to TAIL_LINES (20)
        // tail. The header announces `window-truncated`; the body
        // does NOT carry the per-line `[partial]` marker because
        // the input ends with `\n` (last line is complete).
        let lines: Vec<String> = (0..50).map(|i| format!("boot line {i}")).collect();
        let console = format!("{}\n", lines.join("\n"));
        let s = format_console_diagnostics(&console, 0, "test");
        assert!(s.contains("console (20 lines, window-truncated)"));
        assert!(s.contains("boot line 49"));
        assert!(!s.contains("boot line 29"));
        assert!(!s.contains("last line incomplete"));
        assert!(!s.contains("[partial]"));
    }

    #[test]
    fn format_console_diagnostics_short_console() {
        let console = "Linux version 6.14.0\nbooted ok\n";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("console (2 lines)"));
        assert!(s.contains("Linux version 6.14.0"));
        assert!(s.contains("booted ok"));
        assert!(!s.contains("window-truncated"));
        assert!(!s.contains("last line incomplete"));
        assert!(!s.contains("[partial]"));
    }

    #[test]
    fn format_console_diagnostics_no_truncation_with_trailing_newline() {
        let console = "line1\nline2\nline3\n";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("console (3 lines)"));
        assert!(!s.contains("window-truncated"));
        assert!(!s.contains("last line incomplete"));
        assert!(!s.contains("[partial]"));
    }

    #[test]
    fn format_console_diagnostics_last_line_incomplete_without_window_drop() {
        // Three short lines, last one missing trailing newline.
        // The window does NOT drop anything (3 < TAIL_LINES); only
        // the last line is partial.
        let console = "line1\nline2\npartial li";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("console (3 lines, last line incomplete)"));
        assert!(s.contains("partial li [partial]"));
        assert!(!s.contains("window-truncated"));
    }

    #[test]
    fn format_console_diagnostics_window_drop_and_last_line_incomplete() {
        // Both conditions: 50 lines (window drops earlier ones)
        // AND the input lacks a trailing newline (final line is
        // partial). Header carries both labels in canonical order;
        // body marks the partial last line.
        let lines: Vec<String> = (0..50).map(|i| format!("boot line {i}")).collect();
        let console = lines.join("\n");
        let s = format_console_diagnostics(&console, 0, "test");
        assert!(s.contains("console (20 lines, window-truncated, last line incomplete)"));
        assert!(s.contains("boot line 49 [partial]"));
        assert!(!s.contains("boot line 29"));
    }

    // -- classify_init_stage --

    fn lifecycle_only_drain(
        phases: &[crate::vmm::wire::LifecyclePhase],
    ) -> crate::vmm::host_comms::BulkDrainResult {
        let entries = phases
            .iter()
            .map(|p| ShmEntry {
                msg_type: crate::vmm::wire::MSG_TYPE_LIFECYCLE,
                payload: vec![p.wire_value()],
                crc_ok: true,
            })
            .collect();
        crate::vmm::host_comms::BulkDrainResult { entries }
    }

    #[test]
    fn classify_no_lifecycle_frames() {
        // None drain — bulk port produced no entries. Maps to the
        // "init did not even start" stage.
        assert_eq!(classify_init_stage(None), STAGE_INIT_NOT_STARTED);
        // Empty drain — same outcome.
        let drain = crate::vmm::host_comms::BulkDrainResult { entries: vec![] };
        assert_eq!(classify_init_stage(Some(&drain)), STAGE_INIT_NOT_STARTED);
    }

    #[test]
    fn classify_init_started_only() {
        let drain = lifecycle_only_drain(&[crate::vmm::wire::LifecyclePhase::InitStarted]);
        assert_eq!(
            classify_init_stage(Some(&drain)),
            STAGE_INIT_STARTED_NO_PAYLOAD,
        );
    }

    #[test]
    fn classify_payload_starting() {
        let drain = lifecycle_only_drain(&[
            crate::vmm::wire::LifecyclePhase::InitStarted,
            crate::vmm::wire::LifecyclePhase::PayloadStarting,
        ]);
        assert_eq!(
            classify_init_stage(Some(&drain)),
            STAGE_PAYLOAD_STARTED_NO_RESULT,
        );
    }

    #[test]
    fn classify_payload_starting_without_init() {
        // Edge case: PayloadStarting present but InitStarted
        // missing. PayloadStarting implies init ran, so classify
        // as payload started — same semantics the old string-walk
        // pinned.
        let drain = lifecycle_only_drain(&[crate::vmm::wire::LifecyclePhase::PayloadStarting]);
        assert_eq!(
            classify_init_stage(Some(&drain)),
            STAGE_PAYLOAD_STARTED_NO_RESULT,
        );
    }

    #[test]
    fn classify_scheduler_died_after_init() {
        // SchedulerDied / SchedulerNotAttached fire AFTER init has
        // started — both map to the "payload started but no
        // result" stage so the operator sees the deepest stage
        // reached. Pin both so a future regression that flipped
        // either to a shallower stage would trip here.
        let died = lifecycle_only_drain(&[
            crate::vmm::wire::LifecyclePhase::InitStarted,
            crate::vmm::wire::LifecyclePhase::SchedulerDied,
        ]);
        assert_eq!(
            classify_init_stage(Some(&died)),
            STAGE_PAYLOAD_STARTED_NO_RESULT,
        );
        let not_attached = lifecycle_only_drain(&[
            crate::vmm::wire::LifecyclePhase::InitStarted,
            crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
        ]);
        assert_eq!(
            classify_init_stage(Some(&not_attached)),
            STAGE_PAYLOAD_STARTED_NO_RESULT,
        );
    }

    #[test]
    fn classify_init_stage_skips_crc_bad_lifecycle_frames() {
        // CRC-bad lifecycle frames are ignored. With only a
        // CRC-bad InitStarted in the drain, the classifier sees no
        // valid phase and returns NOT_STARTED.
        let mut drain = lifecycle_only_drain(&[crate::vmm::wire::LifecyclePhase::InitStarted]);
        drain.entries[0].crc_ok = false;
        assert_eq!(classify_init_stage(Some(&drain)), STAGE_INIT_NOT_STARTED);
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

    /// Mid-line `PANIC:` occurrences must NOT match. The guest's
    /// panic hook in `rust_init.rs` always emits the prefix at the
    /// start of a line; a console log that incidentally contains
    /// the literal `PANIC:` somewhere inside a longer info message
    /// must not be misclassified as the panic. The previous
    /// `.contains("PANIC:")` form would have surfaced this fixture
    /// as a panic by stripping nothing and returning the trimmed
    /// raw line.
    #[test]
    fn extract_panic_message_ignores_midline_occurrences() {
        let output = "info: expected PANIC: somewhere in test\nmore noise";
        assert!(
            extract_panic_message(output).is_none(),
            "mid-line `PANIC:` must not be matched as a panic prefix",
        );
    }

    /// Whitespace-prefixed panic lines DO match — the guest panic
    /// hook is anchored at column 0 in `rust_init.rs`, but COM2
    /// transports can insert framing whitespace; the trim before
    /// `strip_prefix` keeps matching robust against that.
    #[test]
    fn extract_panic_message_matches_whitespace_prefixed_line() {
        let output = "noise\n   PANIC: indented panic\nmore";
        assert_eq!(
            extract_panic_message(output),
            Some("indented panic"),
            "leading whitespace before `PANIC:` must be trimmed before the prefix check",
        );
    }

    // -- Verdict API integration coverage -------------------------------
    //
    // The host-side runner decodes the guest's bincode-encoded
    // AssertResult via `parse_assert_result_from_drain`, then a
    // scenario's verifier folds that result into a Verdict via
    // `Verdict::merge`. This integration shape is the single most
    // important assertion path for any test running inside a VM —
    // pin it here so a regression that broke either the decode OR the
    // merge surface trips at the seam.

    /// Round-trip: failing AssertResult through bincode TLV →
    /// `parse_assert_result_from_drain`, then merge into a Verdict
    /// that records a pointwise claim from the host side. The
    /// combined result must fail (the parsed AssertResult was
    /// failing) AND carry both the guest-side detail AND the
    /// host-side claim's failure detail — pinning that
    /// `Verdict::merge` preserves details from both sides.
    #[test]
    fn parse_assert_result_threads_into_verdict_merge() {
        use crate::assert::Verdict;

        // Guest produced a failing AssertResult with one Stuck detail
        // and a wake-latency measurement.
        let original = build_assert_result(
            false,
            vec![AssertDetail::new(DetailKind::Stuck, "tid 42 stuck 3000ms")],
        );
        let drain = drain_with_assert(&original);
        let parsed = parse_assert_result_from_drain(Some(&drain)).unwrap();
        assert!(!parsed.passed, "guest result must be failing");

        // Host-side scenario adds its own claim — say, a deadline
        // budget the host can verify post-VM (a pseudo-value here
        // for the test).
        let observed_runtime_us: u64 = 9000;
        let mut v = Verdict::new();
        crate::claim!(v, observed_runtime_us).at_most(5000);
        v.merge(parsed);

        let r = v.into_result();
        assert!(
            !r.passed,
            "merge of failing parsed result + failing host claim must fail",
        );
        // Both failures must be visible in the merged details: the
        // host-side at_most claim AND the guest-side Stuck.
        assert!(
            r.details.iter().any(|d| d.message.contains("at most 5000")),
            "host claim failure missing: {:?}",
            r.details,
        );
        assert!(
            r.details.iter().any(|d| d.kind == DetailKind::Stuck),
            "guest Stuck detail missing: {:?}",
            r.details,
        );
    }

    /// Round-trip: a passing guest AssertResult merged into a
    /// Verdict with passing host claims keeps the verdict passing.
    /// Sibling of the failing-merge test — pins the happy path so
    /// a regression that always-fails on merge (e.g. flipping the
    /// passed-conjunction direction) trips here.
    #[test]
    fn parse_assert_result_passing_merge_keeps_verdict_passing() {
        use crate::assert::Verdict;

        let original = build_assert_result(true, vec![]);
        let drain = drain_with_assert(&original);
        let parsed = parse_assert_result_from_drain(Some(&drain)).unwrap();

        let observed: u64 = 100;
        let mut v = Verdict::new();
        crate::claim!(v, observed).at_most(1000);
        v.merge(parsed);

        let r = v.into_result();
        assert!(
            r.passed,
            "passing merge must keep verdict passing: {:?}",
            r.details
        );
    }
}
