//! BPF verifier log parsing, cycle detection, and output formatting.
//!
//! Provides:
//! - [`VerifierStats`] / [`ProgStats`] / [`DiffRow`] — data types
//! - [`parse_verifier_stats`] — extract insn/state counts from verifier log
//! - [`parse_vm_verifier_output`] — parse structured `STT_VERIFIER_*` lines
//! - [`normalize_verifier_line`] — strip variable register state annotations
//! - [`detect_cycle`] / [`collapse_cycles`] — loop iteration compression
//! - [`format_brief_line`] — single-line program summary
//! - [`build_b_map`] / [`build_diff_rows`] — A/B comparison helpers

use std::collections::HashMap;

/// Parsed verifier stats from the kernel log line:
/// `processed N insns (limit M) max_states_per_insn X total_states Y peak_states Z mark_read W`
pub struct VerifierStats {
    pub processed_insns: u64,
    pub total_states: u64,
    pub peak_states: u64,
    pub time_usec: Option<u64>,
    pub stack_depth: Option<String>,
}

/// Per-program verifier statistics parsed from VM output.
pub struct ProgStats {
    pub name: String,
    /// Pre-verification program size (BPF insns).
    pub insn_cnt: usize,
    /// Verifier log (stats-only or full, depending on log level).
    pub log: String,
}

/// A single row in the A/B diff output.
pub struct DiffRow {
    pub name: String,
    pub a: u64,
    pub b: u64,
    pub delta: i64,
}

/// Parse verifier stats from the log output.
///
/// The kernel always emits a "processed N insns ..." line. When
/// BPF_LOG_STATS is set, it also emits "verification time" and
/// "stack depth" lines.
pub fn parse_verifier_stats(log: &str) -> VerifierStats {
    let mut stats = VerifierStats {
        processed_insns: 0,
        total_states: 0,
        peak_states: 0,
        time_usec: None,
        stack_depth: None,
    };

    let mut found_insns = false;
    let mut found_time = false;
    let mut found_stack = false;

    for line in log.lines().rev() {
        if !found_insns && line.starts_with("processed ") {
            found_insns = true;
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.len() >= 2 {
                stats.processed_insns = words[1].parse().unwrap_or(0);
            }
            for (i, &w) in words.iter().enumerate() {
                if w == "total_states"
                    && let Some(v) = words.get(i + 1)
                {
                    stats.total_states = v.parse().unwrap_or(0);
                }
                if w == "peak_states"
                    && let Some(v) = words.get(i + 1)
                {
                    stats.peak_states = v.parse().unwrap_or(0);
                }
            }
        }
        if !found_time && line.contains("verification time") {
            found_time = true;
            for word in line.split_whitespace() {
                if let Ok(n) = word.parse::<u64>() {
                    stats.time_usec = Some(n);
                    break;
                }
            }
        }
        if !found_stack && line.contains("stack depth") {
            found_stack = true;
            if let Some(pos) = line.find("stack depth") {
                let after = &line[pos + "stack depth".len()..];
                let depth_str = after.trim();
                if !depth_str.is_empty() {
                    stats.stack_depth = Some(depth_str.to_string());
                }
            }
        }
        if found_insns && found_time && found_stack {
            break;
        }
    }

    stats
}

/// Normalize a BPF verifier log line by stripping variable register-state
/// annotations so that lines from different loop iterations compare equal.
///
/// Handles:
/// - Instruction with `; frame` annotation: `3006: (07) r9 += 1  ; frame1: R9_w=2`
/// - Instruction with `; R` + digit annotation: `9: (15) if r7 == 0x0 goto pc+1  ; R7=scalar(...)`
/// - Branch with inline target state: `3026: (b5) if r6 <= 0x11dc0 goto pc+2 3029: frame1: R0=1`
/// - Standalone register dump with frame: `3041: frame1: R0_w=scalar()`
/// - Standalone register dump without frame: `3029: R0=1 R6=scalar()`
///
/// Preserves source comments (`; for (int j = 0; ...)`) and non-annotation
/// semicolons (`; Return value`) -- these serve as cycle anchors.
pub fn normalize_verifier_line(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.is_empty() || !trimmed.as_bytes()[0].is_ascii_digit() {
        return trimmed;
    }
    // "3041: frame1: ..." or "3041: R0_w=scalar()" — standalone register dump.
    // State-only lines; keep just the instruction index.
    if let Some(colon) = trimmed.find(": ") {
        let after = &trimmed[colon + 2..];
        if after.starts_with("frame")
            || (after.starts_with('R')
                && after.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit()))
        {
            return &trimmed[..colon + 1];
        }
    }
    // "; frame" annotation on instruction line
    if let Some(pos) = trimmed.find("; frame") {
        return trimmed[..pos].trim_end();
    }
    // "; R" followed by digit — register annotation without frame prefix
    if let Some(pos) = trimmed.find("; R")
        && trimmed
            .as_bytes()
            .get(pos + 3)
            .is_some_and(|b| b.is_ascii_digit())
    {
        return trimmed[..pos].trim_end();
    }
    // Inline branch-target state: "goto pc+2 3029: frame1: ..."
    if let Some(goto_pos) = trimmed.find("goto pc") {
        let after_goto = &trimmed[goto_pos + 7..];
        let end = after_goto
            .find(|c: char| c != '+' && c != '-' && !c.is_ascii_digit())
            .unwrap_or(after_goto.len());
        let insn_end = goto_pos + 7 + end;
        if insn_end < trimmed.len() {
            return trimmed[..insn_end].trim_end();
        }
    }
    trimmed
}

/// Normalize for cycle detection: strip register annotations (via
/// `normalize_verifier_line`) then strip the leading instruction address
/// (`NNN: `). Unrolled loops place each copy at different addresses, so
/// the address must be removed for block comparison to find repeats.
fn normalize_for_cycle_detection(line: &str) -> &str {
    let n = normalize_verifier_line(line);
    // Strip leading digits + ": " prefix (e.g. "42: (07) r1 += 8" -> "(07) r1 += 8").
    if let Some(colon) = n.find(": ") {
        let before = &n[..colon];
        if !before.is_empty() && before.bytes().all(|b| b.is_ascii_digit()) {
            return &n[colon + 2..];
        }
    }
    n
}

/// Detect a single repeating cycle in a slice of lines.
///
/// Returns `Some((start, period, count))` where the cycle begins at
/// `start`, each iteration is `period` lines, and it repeats `count` times.
pub fn detect_cycle(lines: &[&str]) -> Option<(usize, usize, usize)> {
    const MIN_PERIOD: usize = 5;
    const MIN_REPS: usize = 3;

    if lines.len() < MIN_PERIOD * MIN_REPS {
        return None;
    }

    let normalized: Vec<&str> = lines
        .iter()
        .map(|l| normalize_for_cycle_detection(l))
        .collect();

    // Find most frequent non-trivial normalized line (the "anchor").
    let mut sorted_norms: Vec<&str> = normalized
        .iter()
        .filter(|l| l.len() >= 10)
        .copied()
        .collect();
    sorted_norms.sort_unstable();

    let mut best_anchor: Option<(&str, usize)> = None;
    let mut i = 0;
    while i < sorted_norms.len() {
        let mut j = i + 1;
        while j < sorted_norms.len() && sorted_norms[j] == sorted_norms[i] {
            j += 1;
        }
        let count = j - i;
        if count >= MIN_REPS && best_anchor.is_none_or(|(_, best)| count > best) {
            best_anchor = Some((sorted_norms[i], count));
        }
        i = j;
    }

    let (anchor, _) = best_anchor?;

    let positions: Vec<usize> = normalized
        .iter()
        .enumerate()
        .filter(|(_, l)| **l == anchor)
        .map(|(i, _)| i)
        .collect();

    // Try strides 1..3 to handle anchors appearing K times per cycle.
    for stride in 1..=3usize {
        if positions.len() <= stride {
            continue;
        }

        let mut gaps: Vec<usize> = positions
            .windows(stride + 1)
            .map(|w| w[stride] - w[0])
            .filter(|g| *g >= MIN_PERIOD)
            .collect();
        gaps.sort_unstable();

        let mut best_period = 0;
        let mut best_gap_count = 0;
        let mut gi = 0;
        while gi < gaps.len() {
            let mut gj = gi + 1;
            while gj < gaps.len() && gaps[gj] == gaps[gi] {
                gj += 1;
            }
            let count = gj - gi;
            if count > best_gap_count {
                best_gap_count = count;
                best_period = gaps[gi];
            }
            gi = gj;
        }
        if best_period == 0 || best_gap_count < MIN_REPS - 1 {
            continue;
        }
        let period = best_period;

        for &pos in &positions {
            if pos + 2 * period > lines.len() {
                break;
            }
            if normalized[pos..pos + period] == normalized[pos + period..pos + 2 * period] {
                let first_block = &normalized[pos..pos + period];
                let mut count = 1;
                while pos + (count + 1) * period <= lines.len() {
                    if normalized[pos + count * period..pos + (count + 1) * period] != *first_block
                    {
                        break;
                    }
                    count += 1;
                }
                // Try earlier starts to find best alignment.
                let mut best_start = pos;
                let mut best_count = count;
                for offset in 1..period {
                    let Some(cand) = pos.checked_sub(offset) else {
                        break;
                    };
                    if cand + 2 * period > lines.len() {
                        continue;
                    }
                    if normalized[cand..cand + period]
                        != normalized[cand + period..cand + 2 * period]
                    {
                        continue;
                    }
                    let mut c = 2;
                    while cand + (c + 1) * period <= lines.len()
                        && normalized[cand + c * period..cand + (c + 1) * period]
                            == normalized[cand..cand + period]
                    {
                        c += 1;
                    }
                    if c > best_count {
                        best_start = cand;
                        best_count = c;
                    }
                }
                if best_count >= MIN_REPS {
                    return Some((best_start, period, best_count));
                }
            }
        }
    }

    None
}

/// Collapse repeating cycles in a verifier log.
///
/// Runs cycle detection iteratively (up to 5 passes for nested loops).
/// Each cycle is replaced with a header (`--- Nx of the following M lines ---`),
/// the first iteration, an omission marker (`--- N identical iterations omitted ---`),
/// the last iteration, and an end marker (`--- end repeat ---`).
pub fn collapse_cycles(log: &str) -> String {
    const MAX_PASSES: usize = 5;
    let mut text = log.to_string();

    for _ in 0..MAX_PASSES {
        let lines: Vec<&str> = text.lines().collect();
        let (start, period, count) = match detect_cycle(&lines) {
            Some(c) => c,
            None => break,
        };

        let mut out = String::new();
        for line in &lines[..start] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!(
            "--- {}x of the following {} lines ---\n",
            count, period
        ));
        for line in &lines[start..start + period] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!(
            "--- {} identical iterations omitted ---\n",
            count - 2
        ));
        let last_start = start + (count - 1) * period;
        for line in &lines[last_start..last_start + period] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("--- end repeat ---\n");
        let suffix_start = start + count * period;
        for line in &lines[suffix_start..] {
            out.push_str(line);
            out.push('\n');
        }
        text = out;
    }

    text
}

/// Format a single program's brief output line (without ANSI color).
pub fn format_brief_line(name: &str, insn_cnt: usize, vs: &VerifierStats) -> String {
    let mut extra = String::new();
    if vs.total_states > 0 {
        extra.push_str(&format!("  states={}/{}", vs.peak_states, vs.total_states));
    }
    if let Some(t) = vs.time_usec {
        extra.push_str(&format!("  time={t}us"));
    }
    if let Some(ref s) = vs.stack_depth {
        extra.push_str(&format!("  stack={s}"));
    }
    format!(
        "  {:<40} insns={:<6} processed={:<6}{}",
        name, insn_cnt, vs.processed_insns, extra
    )
}

/// Parse structured verifier output from a VM run.
///
/// The scheduler binary emits lines when invoked with `--dump-verifier`:
///   STT_VERIFIER_PROG <name> insn_cnt=<N>
///   STT_VERIFIER_LOG <name> <log line>
///   STT_VERIFIER_DONE
pub fn parse_vm_verifier_output(output: &str) -> Vec<ProgStats> {
    let mut stats: Vec<ProgStats> = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_insn_cnt = 0usize;
    let mut current_log = String::new();

    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("STT_VERIFIER_PROG ") {
            if let Some(name) = current_name.take() {
                stats.push(ProgStats {
                    name,
                    insn_cnt: current_insn_cnt,
                    log: std::mem::take(&mut current_log),
                });
            }
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            current_name = Some(parts[0].to_string());
            current_insn_cnt = parts
                .get(1)
                .and_then(|s| s.strip_prefix("insn_cnt="))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            current_log.clear();
        } else if let Some(rest) = line.strip_prefix("STT_VERIFIER_LOG ") {
            if let Some((_, log_line)) = rest.split_once(' ') {
                if !current_log.is_empty() {
                    current_log.push('\n');
                }
                current_log.push_str(log_line);
            }
        } else if line.starts_with("STT_VERIFIER_DONE")
            && let Some(name) = current_name.take()
        {
            stats.push(ProgStats {
                name,
                insn_cnt: current_insn_cnt,
                log: std::mem::take(&mut current_log),
            });
        }
    }
    if let Some(name) = current_name {
        stats.push(ProgStats {
            name,
            insn_cnt: current_insn_cnt,
            log: current_log,
        });
    }
    stats
}

/// Build diff rows from A stats and B lookup map.
pub fn build_diff_rows(stats_a: &[ProgStats], b_map: &HashMap<String, u64>) -> Vec<DiffRow> {
    let mut rows = Vec::new();
    for ps in stats_a {
        let a = parse_verifier_stats(&ps.log).processed_insns;
        let b = b_map.get(&ps.name).copied().unwrap_or(0);
        rows.push(DiffRow {
            name: ps.name.clone(),
            a,
            b,
            delta: a as i64 - b as i64,
        });
    }
    rows
}

/// Build the B-side lookup map from collected stats.
pub fn build_b_map(stats_b: &[ProgStats]) -> HashMap<String, u64> {
    stats_b
        .iter()
        .map(|ps| {
            let vs = parse_verifier_stats(&ps.log);
            (ps.name.clone(), vs.processed_insns)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// VM-based verifier collection
// ---------------------------------------------------------------------------

/// Result of collecting verifier output from a VM run.
pub struct VerifierVmResult {
    /// Per-program verifier statistics.
    pub stats: Vec<ProgStats>,
    /// Scheduler output (stdout+stderr) from the VM, extracted between
    /// ===SCHED_OUTPUT_START=== and ===SCHED_OUTPUT_END=== markers.
    /// Contains libbpf's verifier instruction traces when BPF load fails.
    pub scheduler_log: String,
}

/// Boot a VM with `--dump-verifier` and parse the structured verifier
/// output. Callers provide pre-built binary paths.
pub fn collect_verifier_output(
    sched_bin: &std::path::Path,
    stt_bin: &std::path::Path,
    kernel: &std::path::Path,
    extra_sched_args: &[String],
) -> anyhow::Result<VerifierVmResult> {
    use anyhow::Context;

    let mut sched_args = vec!["--dump-verifier".to_string()];
    sched_args.extend(extra_sched_args.iter().cloned());

    let vm = crate::vmm::SttVm::builder()
        .kernel(kernel)
        .init_binary(stt_bin)
        .scheduler_binary(sched_bin)
        .sched_args(&sched_args)
        .topology(1, 1, 1)
        .memory_mb(2048)
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .context("build verifier VM")?;

    let result = vm.run().context("run verifier VM")?;

    if !result.output.contains("STT_VERIFIER_DONE") {
        anyhow::bail!(
            "verifier VM exited with code {} (timed_out={})\n{}",
            result.exit_code,
            result.timed_out,
            result.output,
        );
    }

    let scheduler_log = result
        .output
        .split("===SCHED_OUTPUT_START===")
        .nth(1)
        .and_then(|s| s.split("===SCHED_OUTPUT_END===").next())
        .unwrap_or("")
        .to_string();

    Ok(VerifierVmResult {
        stats: parse_vm_verifier_output(&result.output),
        scheduler_log,
    })
}

/// Format verifier results as text: brief lines per program, collapsed
/// logs, and optional A/B diff table.
pub fn format_verifier_output(label: &str, result: &VerifierVmResult, raw: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{label}\n"));
    for ps in &result.stats {
        let vs = parse_verifier_stats(&ps.log);
        out.push_str(&format!(
            "{}\n",
            format_brief_line(&ps.name, ps.insn_cnt, &vs)
        ));
    }

    for ps in &result.stats {
        if !ps.log.is_empty() {
            out.push_str(&format!("\n{label}  {}\n", ps.name));
            if raw {
                out.push_str(&ps.log);
            } else {
                out.push_str(&collapse_cycles(&ps.log));
            }
        }
    }

    if !result.scheduler_log.is_empty() {
        out.push_str(&format!("\n{label} --- scheduler log ---\n"));
        if raw {
            out.push_str(&result.scheduler_log);
        } else {
            out.push_str(&collapse_cycles(&result.scheduler_log));
        }
    }

    out
}

/// Format an A/B diff table comparing two sets of verifier stats.
pub fn format_verifier_diff(
    label_a: &str,
    stats_a: &[ProgStats],
    label_b: &str,
    stats_b: &[ProgStats],
) -> String {
    let b_map = build_b_map(stats_b);
    let diff_rows = build_diff_rows(stats_a, &b_map);

    let mut out = String::new();
    out.push_str(&format!("\ndelta A/B diff: {label_a} vs {label_b}\n"));
    out.push_str(&format!(
        "  {:<40} {:>10} {:>10} {:>10}\n",
        "program", "A", "B", "delta"
    ));
    out.push_str(&format!("  {}\n", "-".repeat(72)));

    for row in &diff_rows {
        out.push_str(&format!(
            "  {:<40} {:>10} {:>10} {:>+10}\n",
            row.name, row.a, row.b, row.delta
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // parse_verifier_stats
    // -----------------------------------------------------------------------

    #[test]
    fn parse_verifier_stats_full_line() {
        let log = "processed 1234 insns (limit 1000000) max_states_per_insn 5 total_states 200 peak_states 50 mark_read 10\nverification time 42 usec\nstack depth 32+0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 1234);
        assert_eq!(vs.total_states, 200);
        assert_eq!(vs.peak_states, 50);
        assert_eq!(vs.time_usec, Some(42));
        assert_eq!(vs.stack_depth.as_deref(), Some("32+0"));
    }

    #[test]
    fn parse_verifier_stats_insns_only() {
        let log = "processed 500 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states 3 mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 500);
        assert_eq!(vs.total_states, 10);
        assert_eq!(vs.peak_states, 3);
        assert!(vs.time_usec.is_none());
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_empty() {
        let vs = parse_verifier_stats("");
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
        assert!(vs.time_usec.is_none());
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_garbage_lines() {
        let log = "some random output\nnot a stats line\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert!(vs.time_usec.is_none());
    }

    #[test]
    fn parse_verifier_stats_time_without_insns() {
        let log = "verification time 100 usec\nstack depth 64\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.time_usec, Some(100));
        assert_eq!(vs.stack_depth.as_deref(), Some("64"));
    }

    #[test]
    fn parse_verifier_stats_multi_subprogram_stack() {
        let log = "processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\nstack depth 32+16+8\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 42);
        assert_eq!(vs.stack_depth.as_deref(), Some("32+16+8"));
    }

    #[test]
    fn parse_verifier_stats_noise_between_lines() {
        let log = "\
libbpf: loading something
processed 999 insns (limit 1000000) max_states_per_insn 3 total_states 77 peak_states 20 mark_read 5
libbpf: prog 'dispatch': attached
verification time 7 usec
stack depth 48+0
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 999);
        assert_eq!(vs.total_states, 77);
        assert_eq!(vs.peak_states, 20);
        assert_eq!(vs.time_usec, Some(7));
        assert_eq!(vs.stack_depth.as_deref(), Some("48+0"));
    }

    #[test]
    fn parse_verifier_stats_partial_insns_line() {
        let log = "processed 123\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 123);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_only_stack_depth() {
        let log = "stack depth 128\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("128"));
        assert_eq!(vs.processed_insns, 0);
    }

    #[test]
    fn parse_verifier_stats_zero_insns() {
        let log = "processed 0 insns (limit 1000000) max_states_per_insn 0 total_states 0 peak_states 0 mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_large_values() {
        let log = "processed 999999 insns (limit 1000000) max_states_per_insn 100 total_states 50000 peak_states 12345 mark_read 9999\nverification time 123456 usec\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 999999);
        assert_eq!(vs.total_states, 50000);
        assert_eq!(vs.peak_states, 12345);
        assert_eq!(vs.time_usec, Some(123456));
    }

    #[test]
    fn parse_verifier_stats_stack_depth_single() {
        let log = "stack depth 64\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("64"));
    }

    #[test]
    fn parse_verifier_stats_stack_depth_many_subprograms() {
        let log = "stack depth 32+16+8+0+0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("32+16+8+0+0"));
    }

    #[test]
    fn parse_verifier_stats_multiple_processed_lines_takes_last() {
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\nprocessed 200 insns (limit 1000000) max_states_per_insn 2 total_states 10 peak_states 4 mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 200);
        assert_eq!(vs.total_states, 10);
    }

    #[test]
    fn parse_verifier_stats_complexity_error_with_stats() {
        let log = "\
func#0 @0
0: R1=ctx() R10=fp0
1: (bf) r6 = r1                       ; R1=ctx() R6_w=ctx()
back-edge from insn 42 to 10
BPF program is too complex
processed 131071 insns (limit 131072) max_states_per_insn 12 total_states 9999 peak_states 5000 mark_read 800
verification time 250000 usec
stack depth 96+32
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 131071);
        assert_eq!(vs.total_states, 9999);
        assert_eq!(vs.peak_states, 5000);
        assert_eq!(vs.time_usec, Some(250000));
        assert_eq!(vs.stack_depth.as_deref(), Some("96+32"));
    }

    #[test]
    fn parse_verifier_stats_complexity_error_no_stats() {
        let log = "\
func#0 @0
0: R1=ctx() R10=fp0
R1 type=ctx expected=fp
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert!(vs.time_usec.is_none());
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_loop_warning_with_stats() {
        let log = "\
infinite loop detected at insn 15
back-edge from insn 30 to 15
processed 500 insns (limit 1000000) max_states_per_insn 3 total_states 40 peak_states 15 mark_read 5
verification time 100 usec
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 500);
        assert_eq!(vs.total_states, 40);
        assert_eq!(vs.peak_states, 15);
        assert_eq!(vs.time_usec, Some(100));
    }

    #[test]
    fn parse_verifier_stats_processed_no_number() {
        let log = "processed\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
    }

    #[test]
    fn parse_verifier_stats_keyword_at_end_no_value() {
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 100);
        assert_eq!(vs.total_states, 0);
    }

    #[test]
    fn parse_verifier_stats_non_numeric_values() {
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states abc peak_states xyz mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 100);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_verification_time_no_number() {
        let log = "verification time unknown usec\n";
        let vs = parse_verifier_stats(log);
        assert!(vs.time_usec.is_none());
    }

    #[test]
    fn parse_verifier_stats_stack_depth_empty() {
        let log = "stack depth   \n";
        let vs = parse_verifier_stats(log);
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_peak_states_at_end() {
        let log = "processed 50 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 50);
        assert_eq!(vs.total_states, 10);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_windows_line_endings() {
        let log = "processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\r\nverification time 10 usec\r\nstack depth 16\r\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 42);
        assert_eq!(vs.time_usec, Some(10));
        assert!(vs.stack_depth.is_some());
    }

    // -----------------------------------------------------------------------
    // format_brief_line
    // -----------------------------------------------------------------------

    #[test]
    fn format_brief_line_full_stats() {
        let vs = VerifierStats {
            processed_insns: 1234,
            total_states: 200,
            peak_states: 50,
            time_usec: Some(42),
            stack_depth: Some("32+0".into()),
        };
        let line = format_brief_line("dispatch", 100, &vs);
        assert!(line.contains("insns=100"), "insns: {line}");
        assert!(line.contains("processed=1234"), "processed: {line}");
        assert!(line.contains("states=50/200"), "states: {line}");
        assert!(line.contains("time=42us"), "time: {line}");
        assert!(line.contains("stack=32+0"), "stack: {line}");
    }

    #[test]
    fn format_brief_line_insns_only() {
        let vs = VerifierStats {
            processed_insns: 500,
            total_states: 0,
            peak_states: 0,
            time_usec: None,
            stack_depth: None,
        };
        let line = format_brief_line("init", 20, &vs);
        assert!(line.contains("insns=20"), "insns: {line}");
        assert!(line.contains("processed=500"), "processed: {line}");
        assert!(!line.contains("states="), "no states: {line}");
        assert!(!line.contains("time="), "no time: {line}");
        assert!(!line.contains("stack="), "no stack: {line}");
    }

    #[test]
    fn format_brief_line_zero_processed() {
        let vs = VerifierStats {
            processed_insns: 0,
            total_states: 0,
            peak_states: 0,
            time_usec: None,
            stack_depth: None,
        };
        let line = format_brief_line("broken", 0, &vs);
        assert!(line.contains("insns=0"), "insns: {line}");
        assert!(line.contains("processed=0"), "processed: {line}");
    }

    #[test]
    fn format_brief_line_states_without_time() {
        let vs = VerifierStats {
            processed_insns: 100,
            total_states: 10,
            peak_states: 5,
            time_usec: None,
            stack_depth: None,
        };
        let line = format_brief_line("prog", 50, &vs);
        assert!(line.contains("states=5/10"), "states: {line}");
        assert!(!line.contains("time="), "no time: {line}");
    }

    #[test]
    fn format_brief_line_long_name_alignment() {
        let vs = VerifierStats {
            processed_insns: 42,
            total_states: 0,
            peak_states: 0,
            time_usec: None,
            stack_depth: None,
        };
        let short = format_brief_line("x", 1, &vs);
        let long = format_brief_line("a_very_long_program_name_here", 1, &vs);
        assert!(short.contains("processed=42"));
        assert!(long.contains("processed=42"));
    }

    // -----------------------------------------------------------------------
    // normalize_verifier_line
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_plain_instruction() {
        assert_eq!(
            normalize_verifier_line("100: (07) r1 += 8"),
            "100: (07) r1 += 8"
        );
    }

    #[test]
    fn normalize_strips_frame_annotation() {
        assert_eq!(
            normalize_verifier_line("3006: (07) r9 += 1  ; frame1: R9_w=2"),
            "3006: (07) r9 += 1"
        );
    }

    #[test]
    fn normalize_strips_register_annotation() {
        assert_eq!(
            normalize_verifier_line("42: (bf) r6 = r1 ; R1=ctx() R6_w=ctx()"),
            "42: (bf) r6 = r1"
        );
    }

    #[test]
    fn normalize_standalone_register_dump() {
        assert_eq!(
            normalize_verifier_line("3041: frame1: R0_w=scalar()"),
            "3041:"
        );
    }

    #[test]
    fn normalize_goto_inline_state() {
        assert_eq!(
            normalize_verifier_line(
                "3026: (b5) if r6 <= 0x11dc0 goto pc+2 3029: frame1: R0=1 R6=scalar()"
            ),
            "3026: (b5) if r6 <= 0x11dc0 goto pc+2"
        );
    }

    #[test]
    fn normalize_goto_no_inline_state() {
        assert_eq!(
            normalize_verifier_line("50: (05) goto pc+10"),
            "50: (05) goto pc+10"
        );
    }

    #[test]
    fn normalize_non_instruction_line() {
        assert_eq!(normalize_verifier_line("func#0 @0"), "func#0 @0");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize_verifier_line(""), "");
    }

    #[test]
    fn normalize_goto_negative_offset() {
        assert_eq!(
            normalize_verifier_line("50: (05) goto pc-10 60: frame1: R0=1"),
            "50: (05) goto pc-10"
        );
    }

    #[test]
    fn normalize_semicolon_source_comment() {
        let line = "100: (07) r1 += 8 ; for (int j = 0; j < n; j++)";
        assert_eq!(normalize_verifier_line(line), line);
    }

    #[test]
    fn normalize_semicolon_return_value_comment() {
        let line = "200: (b7) r0 = 0 ; Return value";
        assert_eq!(normalize_verifier_line(line), line);
    }

    #[test]
    fn normalize_standalone_bare_register_dump() {
        assert_eq!(
            normalize_verifier_line("3029: R0=1 R6=scalar(id=1)"),
            "3029:"
        );
    }

    #[test]
    fn normalize_standalone_r10_dump() {
        assert_eq!(normalize_verifier_line("42: R10=fp0"), "42:");
    }

    // -----------------------------------------------------------------------
    // detect_cycle / collapse_cycles
    // -----------------------------------------------------------------------

    fn repeating_log(prefix: usize, period: usize, reps: usize, suffix: usize) -> String {
        let mut lines = Vec::new();
        for i in 0..prefix {
            lines.push(format!("{}: (07) r1 += {i}", 1000 + i));
        }
        for rep in 0..reps {
            for j in 0..period {
                let insn = 100 + j;
                lines.push(format!(
                    "{insn}: (bf) r{} = r{} ; frame1: R{}_w={}",
                    j % 10,
                    (j + 1) % 10,
                    j % 10,
                    rep * 100 + j
                ));
            }
        }
        for i in 0..suffix {
            lines.push(format!("{}: (95) exit_{i}", 2000 + i));
        }
        lines.join("\n")
    }

    #[test]
    fn detect_cycle_basic() {
        let log = repeating_log(0, 10, 8, 0);
        let lines: Vec<&str> = log.lines().collect();
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle");
        let (start, period, count) = result.unwrap();
        assert_eq!(period, 10);
        assert!(count >= 6, "count={count}");
        assert_eq!(start, 0);
    }

    #[test]
    fn detect_cycle_with_prefix_suffix() {
        let log = repeating_log(5, 10, 8, 5);
        let lines: Vec<&str> = log.lines().collect();
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle with prefix/suffix");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 10);
        assert!(count >= 6);
    }

    #[test]
    fn detect_cycle_too_few_reps() {
        let log = repeating_log(0, 10, 2, 0);
        let lines: Vec<&str> = log.lines().collect();
        assert!(detect_cycle(&lines).is_none());
    }

    #[test]
    fn detect_cycle_too_few_lines() {
        let lines: Vec<String> = (0..20)
            .map(|i| format!("{}: (07) r1 += {i}", 100 + i % 3))
            .collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        assert!(detect_cycle(&refs).is_none());
    }

    #[test]
    fn detect_cycle_no_cycle() {
        let lines: Vec<String> = (0..100).map(|i| format!("{i}: unique_insn_{i}")).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        assert!(detect_cycle(&refs).is_none());
    }

    #[test]
    fn detect_cycle_empty() {
        let empty: Vec<&str> = vec![];
        assert!(detect_cycle(&empty).is_none());
    }

    #[test]
    fn detect_cycle_exact_boundary() {
        let log = repeating_log(0, 5, 6, 0);
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 30);
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "boundary case should detect cycle");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 5);
        assert_eq!(count, 6);
    }

    #[test]
    fn collapse_cycles_empty_string() {
        assert_eq!(collapse_cycles(""), "");
    }

    #[test]
    fn collapse_cycles_basic() {
        let log = repeating_log(2, 10, 8, 2);
        let collapsed = collapse_cycles(&log);
        assert!(collapsed.contains("identical iterations omitted"));
        assert!(collapsed.contains("8x of the following 10 lines"));
        assert!(collapsed.contains("end repeat"));
        assert!(collapsed.lines().count() < log.lines().count());
    }

    #[test]
    fn collapse_cycles_no_cycle() {
        let log = "line 1\nline 2\nline 3\n";
        let collapsed = collapse_cycles(log);
        assert_eq!(collapsed, log);
    }

    #[test]
    fn collapse_cycles_preserves_stats() {
        let mut log = repeating_log(0, 10, 8, 0);
        log.push_str("\nprocessed 1000 insns (limit 1000000) max_states_per_insn 5 total_states 100 peak_states 30 mark_read 10\n");
        let collapsed = collapse_cycles(&log);
        assert!(collapsed.contains("processed 1000 insns"));
    }

    #[test]
    fn collapse_cycles_with_register_annotations() {
        let mut lines = Vec::new();
        lines.push("0: (07) r1 += 1".to_string());
        for rep in 0..8 {
            for j in 0..6 {
                let insn = 100 + j;
                lines.push(format!(
                    "{insn}: (bf) r{} = r{} ; frame1: R{}_w={}",
                    j % 10,
                    (j + 1) % 10,
                    j % 10,
                    rep * 100 + j
                ));
            }
        }
        lines.push("200: (95) exit".to_string());
        let log = lines.join("\n");
        let collapsed = collapse_cycles(&log);
        assert!(collapsed.contains("identical iterations omitted"));
    }

    // -----------------------------------------------------------------------
    // parse_vm_verifier_output
    // -----------------------------------------------------------------------

    #[test]
    fn parse_vm_verifier_output_basic() {
        let output = "\
some boot noise
STT_VERIFIER_PROG stt_init insn_cnt=42
STT_VERIFIER_LOG stt_init processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0
STT_VERIFIER_LOG stt_init verification time 10 usec
STT_VERIFIER_PROG stt_dispatch insn_cnt=200
STT_VERIFIER_LOG stt_dispatch processed 500 insns (limit 1000000) max_states_per_insn 3 total_states 50 peak_states 20 mark_read 5
STT_VERIFIER_DONE
more noise
";
        let stats = parse_vm_verifier_output(output);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].name, "stt_init");
        assert_eq!(stats[0].insn_cnt, 42);
        assert!(stats[0].log.contains("processed 100 insns"));
        assert_eq!(stats[1].name, "stt_dispatch");
        assert_eq!(stats[1].insn_cnt, 200);
        assert!(stats[1].log.contains("processed 500 insns"));
    }

    #[test]
    fn parse_vm_verifier_output_empty() {
        let stats = parse_vm_verifier_output("no markers here\n");
        assert!(stats.is_empty());
    }

    #[test]
    fn parse_vm_verifier_output_no_done_marker() {
        let output = "\
STT_VERIFIER_PROG stt_init insn_cnt=10
STT_VERIFIER_LOG stt_init processed 50 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
";
        let stats = parse_vm_verifier_output(output);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "stt_init");
    }

    #[test]
    fn parse_vm_verifier_output_fail_program() {
        let output = "\
STT_VERIFIER_PROG broken_prog insn_cnt=0
STT_VERIFIER_LOG broken_prog FAIL: verification failed
STT_VERIFIER_DONE
";
        let stats = parse_vm_verifier_output(output);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "broken_prog");
        assert!(stats[0].log.contains("FAIL"));
    }

    // -----------------------------------------------------------------------
    // build_b_map / build_diff_rows
    // -----------------------------------------------------------------------

    fn prog(name: &str, insn_cnt: usize, log: &str) -> ProgStats {
        ProgStats {
            name: name.to_string(),
            insn_cnt,
            log: log.to_string(),
        }
    }

    #[test]
    fn build_b_map_basic() {
        let stats_b = vec![prog(
            "dispatch",
            100,
            "processed 500 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\n",
        )];
        let map = build_b_map(&stats_b);
        assert_eq!(map.get("dispatch"), Some(&500));
    }

    #[test]
    fn build_b_map_empty() {
        let map = build_b_map(&[]);
        assert!(map.is_empty());
    }

    #[test]
    fn build_diff_rows_matching_programs() {
        let stats_a = vec![prog(
            "dispatch",
            100,
            "processed 500 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\n",
        )];
        let mut b_map = HashMap::new();
        b_map.insert("dispatch".to_string(), 300u64);
        let rows = build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "dispatch");
        assert_eq!(rows[0].a, 500);
        assert_eq!(rows[0].b, 300);
        assert_eq!(rows[0].delta, 200);
    }

    #[test]
    fn build_diff_rows_program_missing_from_b() {
        let stats_a = vec![prog(
            "new_prog",
            50,
            "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 2 peak_states 1 mark_read 0\n",
        )];
        let b_map = HashMap::new();
        let rows = build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].a, 100);
        assert_eq!(rows[0].b, 0);
        assert_eq!(rows[0].delta, 100);
    }

    #[test]
    fn build_diff_rows_negative_delta() {
        let stats_a = vec![prog(
            "dispatch",
            100,
            "processed 200 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0\n",
        )];
        let mut b_map = HashMap::new();
        b_map.insert("dispatch".to_string(), 500u64);
        let rows = build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows[0].delta, -300);
    }

    #[test]
    fn build_diff_rows_empty_a() {
        let b_map = HashMap::new();
        let rows = build_diff_rows(&[], &b_map);
        assert!(rows.is_empty());
    }
}
