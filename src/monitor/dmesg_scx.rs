//! Parse `sched_ext` disable events from kernel-message buffer
//! (dmesg / `/dev/kmsg`) output.
//!
//! Companion to [`super::live_host_kernel`] (kernel discovery) and
//! [`super::debug_capture`] (capture record format). The live-host
//! pipeline tails `/dev/kmsg`, anchors on the kernel's
//! `sched_ext: BPF scheduler "X" disabled (...)` line, and extracts
//! the stack-trace symbols the kernel printed via `%pS`. Those
//! symbol names feed the auto-repro instrumentation: the live-host
//! producer attaches kprobes / fentry probes to the discovered
//! functions to capture their arguments + timing on the next failure.
//!
//! # Why parse dmesg
//!
//! The kernel ALWAYS emits stack traces for error-class scx exits
//! via `pr_err` (kernel/sched/ext.c around the `dump_stack` call in
//! the disable path). The `%pS` format renders kernel addresses as
//! `funcname+0xoff/0xsz` — readable AND symbolic, so the live-host
//! pipeline can extract function names without doing its own
//! kallsyms walk against raw addresses (the alternative path).
//!
//! For STALL-class exits the stack is the WATCHDOG KTHREAD's stack
//! (`check_rq_for_timeouts` → `scx_watchdog_workfn`), NOT the BPF
//! scheduler's path. The parser surfaces the `kind` field so callers
//! can distinguish "stack tells us where the BPF prog hung" vs
//! "stack tells us the watchdog noticed a stuck task — go probe the
//! BPF ops callbacks via the fallback path".
//!
//! # Async timing
//!
//! dmesg lines arrive 100-500ms after the actual `scx_exit` call
//! (the kernel buffers prints through klogd). The library is purely
//! a parser — it doesn't do timing or polling. The capture-mode
//! binary (separate task) tails `/dev/kmsg` and feeds new lines
//! into [`parse_kmsg_window`] when an scx anchor appears.

use serde::{Deserialize, Serialize};

/// Kind of scx exit event extracted from dmesg.
///
/// Distinguishes the source of the printed stack so the auto-repro
/// pipeline knows whether to trust the stack as "where the BPF
/// scheduler hung" or "where the watchdog noticed the hang".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
#[allow(dead_code)] // wired by the live-host capture-mode binary;
                    // library lands the data shape.
pub enum ScxExitKind {
    /// Default — used by `ScxExitEvent::default()` in test fixtures
    /// and as the post-anchor placeholder before classification
    /// runs. Real parsed events overwrite this with one of the
    /// classified variants below.
    #[default]
    Unclassified,
    /// Operator-error exit (`scx_error()` called from BPF program
    /// or from kernel-side validation). Stack trace is the BPF
    /// caller's path — useful for direct probe placement.
    Error,
    /// Watchdog-detected stall (`check_rq_for_timeouts` fired).
    /// Stack trace is the watchdog kthread, NOT the BPF scheduler.
    /// Auto-repro fallback: probe all BPF ops callbacks since the
    /// causal callback is not directly recoverable from the watchdog
    /// stack.
    Stall,
    /// Normal disable — ops.exit() called cleanly. No error stack.
    Normal,
    /// A scx event line was detected but the kind couldn't be
    /// classified from the surrounding text. Treat as unknown
    /// classification rather than dropping the event.
    Other,
}

/// One parsed scx exit event from a kernel-message buffer window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
#[allow(dead_code)]
pub struct ScxExitEvent {
    /// Scheduler name extracted from `sched_ext: BPF scheduler "<name>" disabled (...)`.
    pub scheduler_name: String,
    /// Exit-kind classification.
    #[serde(default = "default_exit_kind")]
    pub kind: ScxExitKind,
    /// Operator-supplied exit message (`ei->msg`) when present —
    /// the parenthesized text in the anchor line plus any
    /// follow-on `pr_err` lines that look like `<scheduler>:`-
    /// prefixed diagnostic. Empty when the kernel emitted no
    /// message body.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub message: String,
    /// Stuck task COMM (16-byte limit per `TASK_COMM_LEN`)
    /// extracted from the message body when the parser detects
    /// `task <COMM>:<pid>` or `pid <pid>` patterns. `None` when
    /// the kernel didn't print a stuck-task identifier (typical
    /// for normal exits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stuck_task_comm: Option<String>,
    /// Stack-trace symbol frames in dmesg order (top of stack
    /// first). Each frame holds the function name plus the
    /// `funcname+0xoff/0xsz` raw text so the consumer can either
    /// use the structured form or recreate the original line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stack: Vec<StackSymbol>,
}

fn default_exit_kind() -> ScxExitKind {
    ScxExitKind::Unclassified
}

/// One frame of a `%pS`-formatted stack trace.
///
/// `funcname+0xoff/0xsz` is the canonical kernel format
/// (`include/linux/printk.h::printk_format` / `lib/vsprintf.c`'s
/// `pointer_string`). The parser captures the full original token
/// alongside the structured fields so the producer can render either.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StackSymbol {
    /// Symbol name (the part before `+0x...`).
    pub name: String,
    /// Byte offset within the function (the `+0xNNN` value).
    pub offset: u64,
    /// Total function size when present (the `/0xNNN` value).
    /// `None` when the kernel rendered the offset without a size
    /// (older kernels and some configs omit `/<size>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Original text token from dmesg, for reference.
    pub raw: String,
}

/// Anchor pattern that marks the start of an scx exit event in
/// kernel-message output.
///
/// Source: `kernel/sched/ext.c` — `pr_info("sched_ext: BPF scheduler
/// \"%s\" disabled (...)", ei->name, ...)`. The format string has
/// shifted slightly across kernel versions but the leading `sched_ext:`
/// prefix and the `BPF scheduler "..."` shape have been stable since
/// 6.12.
const ANCHOR_PREFIX: &str = "sched_ext: BPF scheduler \"";

/// Parse a window of `/dev/kmsg` (or `dmesg` text) and return every
/// scx exit event found in it.
///
/// Looks for [`ANCHOR_PREFIX`] anchor lines, then collects
/// follow-on lines (typically `<N>` or `[ts]` prefixed kernel print
/// continuation) until the next non-stack-looking line or the next
/// anchor. Stack-trace `%pS` tokens are extracted from the
/// collected lines via [`extract_stack_symbols`].
///
/// Multiple events in one window produce multiple records — the
/// kernel can emit several `disable` events back-to-back (especially
/// when a scheduler load+disable cycles rapidly during a test).
///
/// Returns an empty vec when the window contains no anchor — that
/// is "no scx events in this slice", NOT an error.
#[allow(dead_code)]
pub fn parse_kmsg_window(text: &str) -> Vec<ScxExitEvent> {
    let mut events = Vec::new();
    let lines: Vec<&str> = text.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(anchor_pos) = line.find(ANCHOR_PREFIX) {
            // Parse the anchor line for the scheduler name + the
            // parenthesized exit-context expression.
            let after = &line[anchor_pos + ANCHOR_PREFIX.len()..];
            let scheduler_name = after.split('"').next().unwrap_or("").to_string();
            let message_body = after
                .split_once('(')
                .map(|(_, m)| m.trim_end_matches(')').trim().to_string())
                .unwrap_or_default();

            // Collect follow-on lines until we either hit another
            // anchor or run out of stack-looking content.
            let mut frames: Vec<StackSymbol> = Vec::new();
            let mut full_message = message_body.clone();
            let mut j = i + 1;
            while j < lines.len() {
                let next = lines[j];
                if next.contains(ANCHOR_PREFIX) {
                    break;
                }
                // Try to extract %pS frames from this follow-on
                // line. Lines that are pure narrative (no symbol
                // tokens) just contribute to the message text.
                let mut new_frames = extract_stack_symbols(next);
                if !new_frames.is_empty() {
                    frames.append(&mut new_frames);
                } else if !next.trim().is_empty() {
                    // Skip standard kernel stack-trace formatting lines
                    // (`Call Trace:`, `<TASK>`, `</TASK>`) that appear
                    // between the message body and the actual %pS frames.
                    // These carry no symbol tokens but mark the start of
                    // a stack section that follows — breaking here would
                    // miss the frames the kernel emits next.
                    let trimmed = next.trim();
                    if trimmed.ends_with("Call Trace:")
                        || trimmed == "<TASK>"
                        || trimmed == "</TASK>"
                        || trimmed.ends_with("<TASK>")
                        || trimmed.ends_with("</TASK>")
                    {
                        j += 1;
                        continue;
                    }
                    // Stop accumulating message text once we leave
                    // the printk continuation block — heuristic:
                    // lines that don't carry a kmsg priority/timestamp
                    // prefix and aren't blank are probably from a
                    // different subsystem. Conservative early exit
                    // when the line contains no `sched_ext` or
                    // `BPF` / `scx_` token.
                    let lower = next.to_ascii_lowercase();
                    if !(lower.contains("sched_ext")
                        || lower.contains("scx_")
                        || lower.contains("bpf"))
                    {
                        break;
                    }
                    if !full_message.is_empty() {
                        full_message.push(' ');
                    }
                    full_message.push_str(next.trim());
                }
                j += 1;
            }

            let stuck_task_comm = extract_stuck_task_comm(&full_message);
            let kind = classify_exit_kind(&full_message, &frames);

            events.push(ScxExitEvent {
                scheduler_name,
                kind,
                message: full_message,
                stuck_task_comm,
                stack: frames,
            });
            i = j;
            continue;
        }
        i += 1;
    }

    events
}

/// Extract `funcname+0xoff/0xsz` tokens from one line of kernel
/// output.
///
/// Recognized shapes:
/// - `funcname+0xNN/0xMM` — standard `%pS` with function size
/// - `funcname+0xNN` — `%pS` without function size (older kernels)
/// - `funcname+0xNN/0xMM [module]` — same with module suffix; the
///   module name is currently dropped (the live-host pipeline
///   resolves to the same function regardless of containing module)
///
/// Returns the structured frames in encounter order.
pub fn extract_stack_symbols(line: &str) -> Vec<StackSymbol> {
    let mut frames = Vec::new();

    // Tokenize on whitespace; for each token containing `+0x`,
    // attempt to parse it as a stack frame.
    for token in line.split_whitespace() {
        let Some(plus) = token.find("+0x") else {
            continue;
        };
        // The function name must be a non-empty sequence of valid
        // identifier characters before the `+`.
        let name_part = &token[..plus];
        if name_part.is_empty() {
            continue;
        }
        if !name_part.chars().all(is_kernel_symbol_char) {
            continue;
        }

        let after_plus = &token[plus + 3..]; // skip "+0x"
        let (off_str, size_str) = match after_plus.split_once('/') {
            Some((off, rest)) => {
                // rest may start with "0x"; strip if present
                let s = rest.strip_prefix("0x").unwrap_or(rest);
                // Strip trailing punctuation / module suffix
                let s = s.trim_end_matches(|c: char| !c.is_ascii_hexdigit());
                (off, Some(s))
            }
            None => {
                // No '/size' part; off_str runs until end-of-token
                // or until a non-hex char (some kernels print
                // trailing punctuation like ',' between frames).
                let off = after_plus.trim_end_matches(|c: char| !c.is_ascii_hexdigit());
                (off, None)
            }
        };

        let Ok(offset) = u64::from_str_radix(off_str, 16) else {
            continue;
        };
        let size = size_str.and_then(|s| u64::from_str_radix(s, 16).ok());

        frames.push(StackSymbol {
            name: name_part.to_string(),
            offset,
            size,
            raw: token.to_string(),
        });
    }

    frames
}

/// Heuristic: classify an scx exit event based on its message body
/// + the function names in its stack trace.
///
/// - `Stall` when the message mentions "watchdog" / "stall" / "stuck"
///   OR when the stack contains `check_rq_for_timeouts` /
///   `scx_watchdog_workfn`.
/// - `Error` when the message starts with the scheduler's
///   error-prefix (`scx_error()` callers print `<scheduler>: <msg>`)
///   or the message contains `aborting` / `error` keywords.
/// - `Normal` when the message indicates `unloaded` or `removed`
///   without error keywords AND the stack is empty.
/// - `Other` otherwise.
fn classify_exit_kind(message: &str, stack: &[StackSymbol]) -> ScxExitKind {
    let lower = message.to_ascii_lowercase();
    if lower.contains("watchdog")
        || lower.contains("stall")
        || lower.contains("stuck")
        || stack.iter().any(|f| {
            f.name == "check_rq_for_timeouts" || f.name == "scx_watchdog_workfn"
        })
    {
        return ScxExitKind::Stall;
    }
    if lower.contains("aborting")
        || lower.contains("error")
        || lower.contains("ebpf")
        || lower.contains("enabled")
            && lower.contains("disabled") // weak — see Other below
    {
        return ScxExitKind::Error;
    }
    if (lower.contains("unloaded") || lower.contains("removed") || lower.contains("done"))
        && stack.is_empty()
    {
        return ScxExitKind::Normal;
    }
    if !stack.is_empty() {
        // Stack present but no obvious watchdog / error keyword;
        // treat as Error since normal exits don't carry a stack.
        return ScxExitKind::Error;
    }
    ScxExitKind::Other
}

/// Extract a stuck-task COMM from an exit message body.
///
/// Scans for the patterns ktstr-aware schedulers (and the
/// upstream watchdog) tend to emit:
/// - `task <COMM>:<pid>` (mainline watchdog: "...stalled task <COMM>:<pid>")
/// - `comm=<COMM>` (some custom schedulers)
///
/// The `task <COMM>` pattern requires the COMM to be followed by
/// `:<digits>` (the watchdog always prints the pid). This avoids
/// matching prose like "runnable task stall" in the anchor's
/// parenthesized body, where "stall" is a classification keyword
/// rather than a process name.
///
/// Returns the first matching COMM truncated to TASK_COMM_LEN bytes,
/// or `None` when no pattern matches.
fn extract_stuck_task_comm(message: &str) -> Option<String> {
    const TASK_COMM_LEN: usize = 16;
    // Kernel watchdog format: `task <COMM>:<pid>`. The bare `find("task ")`
    // matches phrases like "task stall" before "task hot_path:1234", so we
    // walk every "task " occurrence and accept the first whose next
    // whitespace-delimited token has the `<COMM>:<digits>` shape.
    let mut search_from = 0;
    while let Some(rel) = message[search_from..].find("task ") {
        let idx = search_from + rel;
        let after = &message[idx + 5..];
        let token = after
            .split_whitespace()
            .next()
            .unwrap_or("");
        if let Some((comm, pid_part)) = token.split_once(':') {
            let pid_digits: String = pid_part
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if !pid_digits.is_empty() {
                let comm = comm.trim_matches(
                    |c: char| !c.is_alphanumeric() && c != '_' && c != '-',
                );
                if !comm.is_empty() {
                    let bounded: String = comm.chars().take(TASK_COMM_LEN).collect();
                    return Some(bounded);
                }
            }
        }
        search_from = idx + 5;
    }
    if let Some(idx) = message.find("comm=") {
        let after = &message[idx + 5..];
        let token = after
            .split(|c: char| c.is_whitespace() || c == ',' || c == ')')
            .next()?
            .trim_matches('"');
        if !token.is_empty() {
            let bounded: String = token.chars().take(TASK_COMM_LEN).collect();
            return Some(bounded);
        }
    }
    None
}

/// True for characters valid in a Linux kernel symbol name. Kernel
/// symbols use C identifier rules plus `.` (compiler-emitted local
/// labels like `func.cold` and `func.constprop.0`).
fn is_kernel_symbol_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.'
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `extract_stack_symbols` recovers a single frame from the
    /// canonical `funcname+0xoff/0xsz` shape.
    #[test]
    fn extract_single_frame_with_size() {
        let frames = extract_stack_symbols("? scx_watchdog_workfn+0x123/0x456");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].name, "scx_watchdog_workfn");
        assert_eq!(frames[0].offset, 0x123);
        assert_eq!(frames[0].size, Some(0x456));
        assert_eq!(frames[0].raw, "scx_watchdog_workfn+0x123/0x456");
    }

    /// Frame without `/size` — older kernel shape — still parses.
    #[test]
    fn extract_frame_without_size() {
        let frames = extract_stack_symbols("scx_disable_workfn+0x42");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].name, "scx_disable_workfn");
        assert_eq!(frames[0].offset, 0x42);
        assert_eq!(frames[0].size, None);
    }

    /// Multiple frames on one line all extract.
    #[test]
    fn extract_multiple_frames_one_line() {
        let frames = extract_stack_symbols(
            "? func_a+0x10/0x20 func_b+0x30/0x40 func_c+0x50",
        );
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].name, "func_a");
        assert_eq!(frames[1].name, "func_b");
        assert_eq!(frames[2].name, "func_c");
        assert_eq!(frames[2].size, None);
    }

    /// Symbol names with `.` (cold / constprop variants) parse.
    #[test]
    fn extract_frame_with_dot_in_name() {
        let frames = extract_stack_symbols("scx_dispatch_q.cold+0x10/0x40");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].name, "scx_dispatch_q.cold");
        assert_eq!(frames[0].offset, 0x10);
    }

    /// Lines without `+0x` produce no frames.
    #[test]
    fn extract_no_frames_from_plain_text() {
        let frames = extract_stack_symbols(
            "[12345.678] sched_ext: BPF scheduler \"foo\" disabled (operator request)",
        );
        assert!(frames.is_empty());
    }

    /// The canonical anchor line + a follow-on stack trace produces
    /// one event with the right scheduler name + extracted frames.
    /// Verdict-routed so a multi-field parser regression (anchor
    /// shape change, classifier rename, stack extractor regression)
    /// surfaces every drift in one run.
    #[test]
    fn parse_kmsg_window_simple_error() {
        use crate::assert::Verdict;

        let text = "\
[12345.678] sched_ext: BPF scheduler \"scx_test\" disabled (BPF runtime error)
[12345.679] scx_test: aborting due to BPF runtime error
[12345.680] Call Trace:
[12345.681]  ? scx_disable_workfn+0x100/0x200
[12345.682]  ? scx_internal_disable+0x50/0x100
[12345.683]  ? scx_error+0x30/0x80
";
        let events = parse_kmsg_window(text);
        let event_count = events.len();
        assert_eq!(event_count, 1, "expected exactly one event");
        let ev = &events[0];
        let scheduler_name = ev.scheduler_name.clone();
        // ScxExitKind doesn't Display; match-against-shape and claim
        // on the resulting bool so the verdict carries a labeled detail.
        let kind_is_error = matches!(ev.kind, ScxExitKind::Error);
        let message_has_runtime_error = ev.message.contains("BPF runtime error");
        let stack_len = ev.stack.len();
        let first_frame_name = ev.stack[0].name.clone();

        let mut v = Verdict::new();
        crate::claim!(v, scheduler_name).eq("scx_test".to_string());
        crate::claim!(v, kind_is_error).eq(true);
        crate::claim!(v, message_has_runtime_error).eq(true);
        crate::claim!(v, stack_len).eq(3usize);
        crate::claim!(v, first_frame_name).eq("scx_disable_workfn".to_string());
        let r = v.into_result();
        assert!(
            r.passed,
            "kmsg parse drift on canonical error event: {:?}",
            r.details,
        );
    }

    /// Watchdog / stall classification fires when the message
    /// mentions stall keywords OR the stack carries
    /// check_rq_for_timeouts.
    ///
    /// Fixture mirrors production `dump_stack` output: a `Call Trace:`
    /// header followed by `<TASK>` / `</TASK>` brackets around the
    /// actual `%pS` frames. The parser must skip those formatting
    /// lines (they carry no symbol tokens) instead of treating them
    /// as end-of-stack.
    #[test]
    fn parse_kmsg_window_stall_classification() {
        let text = "\
[1.0] sched_ext: BPF scheduler \"scx_test\" disabled (runnable task stall)
[1.1] scx_test: stalled task hot_path:1234 not dispatched
[1.2] Call Trace:
[1.3]  <TASK>
[1.4]  ? check_rq_for_timeouts+0x50/0x100
[1.5]  ? scx_watchdog_workfn+0x10/0x80
[1.6]  </TASK>
";
        let events = parse_kmsg_window(text);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, ScxExitKind::Stall);
        assert_eq!(events[0].stack.len(), 2);
        // stuck-task COMM extracted from "stalled task hot_path:1234"
        // — pattern matched on "task hot_path".
        assert_eq!(events[0].stuck_task_comm.as_deref(), Some("hot_path"));
    }

    /// Stuck-task COMM extraction handles the `task <COMM>` and
    /// `comm=<COMM>` patterns plus 16-byte truncation.
    #[test]
    fn extract_stuck_task_comm_patterns() {
        assert_eq!(
            extract_stuck_task_comm("stalled task foo:1234 stuck"),
            Some("foo".to_string())
        );
        assert_eq!(
            extract_stuck_task_comm("operator complaint about comm=bar)"),
            Some("bar".to_string())
        );
        assert_eq!(
            extract_stuck_task_comm("task this_is_a_very_long_task_name_too_long:1"),
            // Truncated to 16 bytes (TASK_COMM_LEN).
            Some("this_is_a_very_l".to_string())
        );
        assert_eq!(extract_stuck_task_comm("no patterns here"), None);
    }

    /// Multiple events in one window produce multiple records.
    #[test]
    fn parse_kmsg_window_multiple_events() {
        let text = "\
[1.0] sched_ext: BPF scheduler \"scx_a\" disabled (manual unload)
[2.0] sched_ext: BPF scheduler \"scx_b\" disabled (BPF runtime error)
[2.1]  ? scx_disable_workfn+0x100/0x200
";
        let events = parse_kmsg_window(text);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].scheduler_name, "scx_a");
        assert_eq!(events[1].scheduler_name, "scx_b");
        assert_eq!(events[1].stack.len(), 1);
    }

    /// Window with no anchor produces no events (NOT an error).
    #[test]
    fn parse_kmsg_window_no_anchor() {
        let text = "\
[1.0] kernel: random unrelated message
[1.1] systemd: started service
";
        let events = parse_kmsg_window(text).len();
        assert_eq!(events, 0);
    }

    /// ScxExitEvent serializes with empty fields suppressed.
    #[test]
    fn scx_exit_event_serde_skips_empty() {
        let ev = ScxExitEvent {
            scheduler_name: "scx_test".into(),
            kind: ScxExitKind::Normal,
            message: String::new(),
            stuck_task_comm: None,
            stack: Vec::new(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(!json.contains("message"));
        assert!(!json.contains("stuck_task_comm"));
        assert!(!json.contains("stack"));
        assert!(json.contains("scx_test"));
        assert!(json.contains("Normal"));
    }
}
