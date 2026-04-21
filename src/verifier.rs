//! BPF verifier log parsing, cycle detection, and output formatting.
//!
//! Provides:
//! - [`VerifierStats`] / [`ProgStats`] / [`DiffRow`] — data types
//! - [`collect_verifier_output`] — boot VM, collect stats via host introspection
//! - [`format_verifier_output`] / [`format_verifier_diff`] — text formatting
//! - [`extract_verifier_log`] — extract verifier trace from libbpf log blob
//! - [`parse_verifier_stats`] — extract insn/state counts from verifier log
//! - [`normalize_verifier_line`] — strip variable register state annotations
//! - [`detect_cycle`] / [`collapse_cycles`] — loop iteration compression
//! - [`build_b_map`] / [`build_diff_rows`] — A/B comparison helpers
//! - [`SCHED_OUTPUT_START`] / [`SCHED_OUTPUT_END`] — COM2 delimiters
//!   written by the guest's rust_init around the scheduler log region;
//!   [`parse_sched_output`] extracts the enclosed block

use std::collections::HashMap;

/// Delimiter written to COM2 by the guest's rust_init immediately
/// before the scheduler log block. Paired with [`SCHED_OUTPUT_END`].
pub(crate) const SCHED_OUTPUT_START: &str = "===SCHED_OUTPUT_START===";
/// Delimiter written to COM2 by the guest's rust_init immediately
/// after the scheduler log block. Paired with [`SCHED_OUTPUT_START`].
pub(crate) const SCHED_OUTPUT_END: &str = "===SCHED_OUTPUT_END===";

/// Extract the scheduler log from guest output between
/// [`SCHED_OUTPUT_START`] and [`SCHED_OUTPUT_END`]. Returns `None` if
/// the delimiters are absent or the enclosed content is empty after
/// trimming.
///
/// Uses `find` on the start marker and `rfind` on the end marker: if
/// the scheduler log itself contains the end sentinel string (e.g. a
/// stack trace that quotes the marker), `rfind` anchors on the last
/// occurrence, which is the real terminator emitted by the guest's
/// post-scenario shutdown path.
pub(crate) fn parse_sched_output(output: &str) -> Option<&str> {
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

/// Parsed verifier stats from the kernel log line:
/// `processed N insns (limit M) max_states_per_insn X total_states Y peak_states Z mark_read W`
pub struct VerifierStats {
    /// Instructions processed during verification.
    pub processed_insns: u64,
    /// Total explored verifier states.
    pub total_states: u64,
    /// Peak concurrent explored states.
    pub peak_states: u64,
    /// Total verification wall time in microseconds, when
    /// BPF_LOG_STATS emitted a "verification time" line.
    pub time_usec: Option<u64>,
    /// Stack depth in the format "<prog>+<subprog>+<main>" (e.g.
    /// `"32+16+8"`) when BPF_LOG_STATS emitted a "stack depth" line.
    pub stack_depth: Option<String>,
}

/// Per-program verifier statistics collected from a VM run.
pub struct ProgStats {
    /// Program name as registered with the kernel.
    pub name: String,
    /// Instructions processed by the verifier (from host-side
    /// `bpf_prog_aux->verified_insns`).
    pub verified_insns: u32,
}

/// A single row in the A/B diff output.
pub struct DiffRow {
    /// Program name present in both A and B runs.
    pub name: String,
    /// `verified_insns` from the A run.
    pub a: u64,
    /// `verified_insns` from the B run.
    pub b: u64,
    /// Signed delta (`b - a`); positive means B's verifier cost grew
    /// relative to A.
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

    // Two normalization levels:
    // - anchor_norms: keeps addresses, strips register annotations. Used for
    //   anchor frequency counting — prevents within-period duplicates at
    //   different addresses from inflating frequency.
    // - block_norms: also strips addresses. Used for block equality comparison
    //   so unrolled loops (same instructions at different addresses) can match.
    let anchor_norms: Vec<&str> = lines.iter().map(|l| normalize_verifier_line(l)).collect();
    let block_norms: Vec<&str> = lines
        .iter()
        .map(|l| normalize_for_cycle_detection(l))
        .collect();

    // Find most frequent non-trivial anchor-normalized line.
    let mut sorted_norms: Vec<&str> = anchor_norms
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

    // If address-preserving anchor search found nothing (unrolled loops
    // where every address is unique), fall back to address-stripped norms.
    let (anchor, use_block_norms_for_positions) = match best_anchor {
        Some((a, _)) => (a, false),
        None => {
            let mut sorted_block: Vec<&str> = block_norms
                .iter()
                .filter(|l| l.len() >= 10)
                .copied()
                .collect();
            sorted_block.sort_unstable();
            let mut ba: Option<(&str, usize)> = None;
            let mut bi = 0;
            while bi < sorted_block.len() {
                let mut bj = bi + 1;
                while bj < sorted_block.len() && sorted_block[bj] == sorted_block[bi] {
                    bj += 1;
                }
                let c = bj - bi;
                if c >= MIN_REPS && ba.is_none_or(|(_, best)| c > best) {
                    ba = Some((sorted_block[bi], c));
                }
                bi = bj;
            }
            match ba {
                Some((a, _)) => (a, true),
                None => return None,
            }
        }
    };

    let norms_for_pos = if use_block_norms_for_positions {
        &block_norms
    } else {
        &anchor_norms
    };
    let positions: Vec<usize> = norms_for_pos
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
            if block_norms[pos..pos + period] == block_norms[pos + period..pos + 2 * period] {
                let first_block = &block_norms[pos..pos + period];
                let mut count = 1;
                while pos + (count + 1) * period <= lines.len() {
                    if block_norms[pos + count * period..pos + (count + 1) * period] != *first_block
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
                    if block_norms[cand..cand + period]
                        != block_norms[cand + period..cand + 2 * period]
                    {
                        continue;
                    }
                    let mut c = 2;
                    while cand + (c + 1) * period <= lines.len()
                        && block_norms[cand + c * period..cand + (c + 1) * period]
                            == block_norms[cand..cand + period]
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
/// Each cycle is replaced with:
/// - `--- Nx of the following M lines ---` (count header, no closing marker)
/// - first iteration (with original register annotations)
/// - `--- K identical iterations omitted ---` (omission marker)
/// - last iteration (with original register annotations)
/// - `--- end repeat ---` (closes the omission)
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

/// Build diff rows from A stats and B lookup map.
pub fn build_diff_rows(stats_a: &[ProgStats], b_map: &HashMap<String, u64>) -> Vec<DiffRow> {
    let mut rows = Vec::new();
    for ps in stats_a {
        let a = ps.verified_insns as u64;
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
        .map(|ps| (ps.name.clone(), ps.verified_insns as u64))
        .collect()
}

// ---------------------------------------------------------------------------
// VM-based verifier collection
// ---------------------------------------------------------------------------

/// Result of collecting verifier output from a VM run.
pub struct VerifierVmResult {
    /// Per-program verifier statistics from host-side memory
    /// introspection (`bpf_prog_aux->verified_insns`).
    pub stats: Vec<ProgStats>,
    /// Scheduler log (stdout+stderr) from the VM. Contains libbpf's
    /// verifier instruction traces when BPF load fails.
    pub scheduler_log: String,
}

/// Boot a VM and collect verifier statistics via host-side memory
/// introspection. Per-program `verified_insns` comes from
/// `bpf_prog_aux->verified_insns` read through the guest's physical
/// memory. On load failure, libbpf prints the verifier log to stderr;
/// the returned `scheduler_log` field contains the scheduler's captured
/// output from the VM.
pub fn collect_verifier_output(
    sched_bin: &std::path::Path,
    ktstr_bin: &std::path::Path,
    kernel: &std::path::Path,
    extra_sched_args: &[String],
) -> anyhow::Result<VerifierVmResult> {
    use anyhow::Context;

    let sched_args: Vec<String> = extra_sched_args.to_vec();

    let no_perf_mode = std::env::var("KTSTR_NO_PERF_MODE").is_ok();
    let vm = crate::vmm::KtstrVm::builder()
        .kernel(kernel)
        .init_binary(ktstr_bin)
        .scheduler_binary(sched_bin)
        .sched_args(&sched_args)
        .topology(1, 1, 1, 1)
        .memory_mb(2048)
        .timeout(std::time::Duration::from_secs(120))
        .no_perf_mode(no_perf_mode)
        .build()
        .context("build verifier VM")?;

    let result = vm.run().context("run verifier VM")?;

    let scheduler_log = parse_sched_output(&result.output).unwrap_or("").to_string();

    // Build ProgStats from host-side ProgVerifierStats. Each program
    // that loaded successfully is visible in prog_idr with its
    // verified_insns count.
    let stats: Vec<ProgStats> = result
        .verifier_stats
        .iter()
        .map(|pvs| ProgStats {
            name: pvs.name.clone(),
            verified_insns: pvs.verified_insns,
        })
        .collect();

    Ok(VerifierVmResult {
        stats,
        scheduler_log,
    })
}

/// Extract the verifier instruction trace from a scheduler log blob.
///
/// libbpf wraps the kernel verifier log between marker lines:
///   `-- BEGIN PROG LOAD LOG --`
///   `-- END PROG LOAD LOG --`
///
/// Returns the content between the first pair of markers, or `None` if
/// no markers are found (backward compat with logs that contain only
/// raw verifier output).
pub fn extract_verifier_log(scheduler_log: &str) -> Option<&str> {
    const BEGIN: &str = "-- BEGIN PROG LOAD LOG --";
    const END: &str = "-- END PROG LOAD LOG --";

    let begin_pos = scheduler_log.find(BEGIN)?;
    let content_start = begin_pos + BEGIN.len();
    // Skip the newline after the BEGIN marker if present.
    let content_start = if scheduler_log.as_bytes().get(content_start) == Some(&b'\n') {
        content_start + 1
    } else {
        content_start
    };
    let end_pos = scheduler_log[content_start..].find(END)?;
    let content = &scheduler_log[content_start..content_start + end_pos];
    // The END marker may appear mid-line (e.g. "libbpf: -- END ...").
    // Trim back to the last newline to drop the partial prefix.
    let content = content
        .rfind('\n')
        .map(|p| &content[..p])
        .unwrap_or(content);
    Some(content.trim_end_matches('\n'))
}

/// Format verifier results as text: brief lines per program and collapsed
/// logs.
pub fn format_verifier_output(label: &str, result: &VerifierVmResult, raw: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{label}\n"));
    for ps in &result.stats {
        out.push_str(&format!(
            "  {:<40} verified_insns={}\n",
            ps.name, ps.verified_insns
        ));
    }

    if !result.scheduler_log.is_empty() {
        // Extract the verifier log from between libbpf's markers.
        // Falls back to the full scheduler_log when no markers exist.
        let verifier_log =
            extract_verifier_log(&result.scheduler_log).unwrap_or(&result.scheduler_log);

        let vs = parse_verifier_stats(verifier_log);
        if vs.processed_insns > 0 {
            out.push_str(&format!("\n{label} --- verifier stats ---\n"));
            out.push_str(&format!(
                "  processed={}  states={}/{}",
                vs.processed_insns, vs.peak_states, vs.total_states
            ));
            if let Some(t) = vs.time_usec {
                out.push_str(&format!("  time={t}us"));
            }
            if let Some(ref s) = vs.stack_depth {
                out.push_str(&format!("  stack={s}"));
            }
            out.push('\n');
        }

        out.push_str(&format!("\n{label} --- scheduler log ---\n"));
        if raw {
            out.push_str(&result.scheduler_log);
        } else {
            out.push_str(&collapse_cycles(verifier_log));
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
    let mut table = crate::cli::new_table();
    table.set_header(vec!["program", "A", "B", "delta"]);
    for row in &diff_rows {
        table.add_row(vec![
            row.name.clone(),
            row.a.to_string(),
            row.b.to_string(),
            format!("{:+}", row.delta),
        ]);
    }
    out.push_str(&table.to_string());
    out.push('\n');
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
    // build_b_map / build_diff_rows
    // -----------------------------------------------------------------------

    fn prog(name: &str, verified_insns: u32) -> ProgStats {
        ProgStats {
            name: name.to_string(),
            verified_insns,
        }
    }

    #[test]
    fn build_b_map_basic() {
        let stats_b = vec![prog("dispatch", 500)];
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
        let stats_a = vec![prog("dispatch", 500)];
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
        let stats_a = vec![prog("new_prog", 100)];
        let b_map = HashMap::new();
        let rows = build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].a, 100);
        assert_eq!(rows[0].b, 0);
        assert_eq!(rows[0].delta, 100);
    }

    #[test]
    fn build_diff_rows_negative_delta() {
        let stats_a = vec![prog("dispatch", 200)];
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

    /// Simulates the verifier trace produced by #pragma unroll loops.
    /// Each copy is at a different base address but has the same
    /// instruction sequence. After normalize_for_cycle_detection strips
    /// addresses and register annotations, all copies look identical.
    fn unrolled_verifier_log(copies: usize, body_len: usize) -> String {
        let ops = [
            "(85) call bpf_ktime_get_ns#5",
            "(bf) r2 = r0",
            "(77) r0 >>= 16",
            "(af) r1 ^= r0",
            "(77) r2 >>= 32",
            "(0f) r1 += r2",
            "(24) w1 *= 7",
            "(04) w1 += 1",
        ];
        let mut lines = Vec::new();
        lines.push("func#0 @0".to_string());
        lines.push("0: R1=ctx() R10=fp0".to_string());
        let mut addr = 10;
        for copy in 0..copies {
            for (j, op) in ops.iter().enumerate().take(body_len) {
                lines.push(format!(
                    "{}: {op} ; R0_w=scalar(id={})",
                    addr,
                    copy * 100 + j
                ));
                addr += 1;
            }
        }
        lines.push(format!("{addr}: (05) goto pc-1"));
        lines.push(
            "processed 1000 insns (limit 1000000) max_states_per_insn 3 \
             total_states 50 peak_states 20 mark_read 5"
                .to_string(),
        );
        lines.join("\n")
    }

    #[test]
    fn detect_cycle_unrolled_loop() {
        let log = unrolled_verifier_log(8, 6);
        let lines: Vec<&str> = log.lines().collect();
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle in unrolled loop");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 6);
        assert!(count >= 6, "count={count}");
    }

    #[test]
    fn collapse_cycles_unrolled_loop() {
        let log = unrolled_verifier_log(8, 6);
        let collapsed = collapse_cycles(&log);
        assert!(
            collapsed.contains("identical iterations omitted"),
            "should collapse unrolled loop"
        );
        assert!(collapsed.lines().count() < log.lines().count());
    }

    // -----------------------------------------------------------------------
    // extract_verifier_log
    // -----------------------------------------------------------------------

    #[test]
    fn extract_verifier_log_basic() {
        let log = "\
libbpf: prog 'dispatch': BPF program load failed: -22
-- BEGIN PROG LOAD LOG --
func#0 @0
0: R1=ctx() R10=fp0
processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0
-- END PROG LOAD LOG --
libbpf: failed to load object 'ktstr_ops'
";
        let extracted = extract_verifier_log(log);
        assert!(extracted.is_some());
        let v = extracted.unwrap();
        assert!(v.starts_with("func#0 @0"));
        assert!(v.contains("processed 100 insns"));
        assert!(!v.contains("BEGIN PROG LOAD LOG"));
        assert!(!v.contains("END PROG LOAD LOG"));
        assert!(!v.contains("libbpf:"));
    }

    #[test]
    fn extract_verifier_log_none_without_markers() {
        let log = "func#0 @0\n0: R1=ctx()\nprocessed 50 insns\n";
        assert!(extract_verifier_log(log).is_none());
    }

    #[test]
    fn extract_verifier_log_empty() {
        assert!(extract_verifier_log("").is_none());
    }

    /// Attack 1: libbpf wraps verifier output with "libbpf: " prefix lines.
    /// `parse_verifier_stats` looks for `starts_with("processed ")` which
    /// won't match `libbpf: processed ...`. Without extraction, stats
    /// parsing fails on blobs where the `processed` line is only inside
    /// the markers.
    #[test]
    fn extract_verifier_log_attack1_stats_parse() {
        let blob = "\
libbpf: prog 'ktstr_ops_dispatch': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#0 @0
0: R1=ctx() R10=fp0
1: (bf) r6 = r1 ; R1=ctx() R6_w=ctx()
back-edge from insn 42 to 10
BPF program is too complex
processed 131071 insns (limit 131072) max_states_per_insn 12 total_states 9999 peak_states 5000 mark_read 800
verification time 250000 usec
stack depth 96+32
libbpf: -- END PROG LOAD LOG --
libbpf: failed to load BPF skeleton 'ktstr_ops': -22
";
        let extracted = extract_verifier_log(blob);
        assert!(extracted.is_some(), "should find markers");
        let v = extracted.unwrap();
        let vs = parse_verifier_stats(v);
        assert_eq!(vs.processed_insns, 131071);
        assert_eq!(vs.total_states, 9999);
        assert_eq!(vs.peak_states, 5000);
        assert_eq!(vs.time_usec, Some(250000));
        assert_eq!(vs.stack_depth.as_deref(), Some("96+32"));

        // Without extraction, parsing the full blob must also work
        // because the "processed" line doesn't have a "libbpf: " prefix
        // inside the markers. But verify extraction gives cleaner input.
        let vs_raw = parse_verifier_stats(blob);
        assert_eq!(vs_raw.processed_insns, 131071);
    }

    /// Attack 3: three distinct program load logs in a single blob.
    /// Each has different instructions. `collapse_cycles` must NOT treat
    /// them as a repeating cycle.
    #[test]
    fn extract_verifier_log_attack3_no_false_collapse() {
        let blob = "\
libbpf: prog 'init': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#0 @0
0: R1=ctx() R10=fp0
1: (bf) r6 = r1
2: (07) r6 += 8
3: (61) r0 = *(u32 *)(r6 + 0)
4: (95) exit
processed 5 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
libbpf: -- END PROG LOAD LOG --
libbpf: prog 'dispatch': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#1 @10
10: R1=ctx() R10=fp0
11: (bf) r7 = r1
12: (85) call bpf_ktime_get_ns#5
13: (77) r0 >>= 32
14: (95) exit
processed 5 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
libbpf: -- END PROG LOAD LOG --
libbpf: prog 'enqueue': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#2 @20
20: R1=ctx() R10=fp0
21: (b7) r0 = 0
22: (63) *(u32 *)(r10 - 4) = r0
23: (61) r1 = *(u32 *)(r10 - 4)
24: (95) exit
processed 5 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
libbpf: -- END PROG LOAD LOG --
libbpf: failed to load BPF skeleton 'ktstr_ops': -22
";
        // extract_verifier_log returns the FIRST log section.
        let extracted = extract_verifier_log(blob);
        assert!(extracted.is_some());
        let v = extracted.unwrap();
        assert!(v.contains("func#0 @0"), "should get first program's log");
        assert!(!v.contains("func#1"), "should not include second program");

        // collapse_cycles on the extracted first section must not
        // collapse — it's only 7 lines total.
        let collapsed = collapse_cycles(v);
        assert!(
            !collapsed.contains("identical iterations omitted"),
            "must not false-collapse distinct program logs"
        );
    }

    // -- insta snapshot tests --

    #[test]
    fn snapshot_format_verifier_output_no_log() {
        let result = VerifierVmResult {
            stats: vec![
                ProgStats {
                    name: "enqueue".into(),
                    verified_insns: 500,
                },
                ProgStats {
                    name: "dispatch".into(),
                    verified_insns: 1200,
                },
                ProgStats {
                    name: "init".into(),
                    verified_insns: 300,
                },
            ],
            scheduler_log: String::new(),
        };
        insta::assert_snapshot!(format_verifier_output("default", &result, false));
    }

    #[test]
    fn snapshot_format_verifier_output_with_log() {
        let log = "\
-- BEGIN PROG LOAD LOG --\n\
func#0 @0\n\
0: R1=ctx() R10=fp0\n\
processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states 8 mark_read 5\n\
-- END PROG LOAD LOG --";
        let result = VerifierVmResult {
            stats: vec![ProgStats {
                name: "enqueue".into(),
                verified_insns: 42,
            }],
            scheduler_log: log.into(),
        };
        insta::assert_snapshot!(format_verifier_output("llc+steal", &result, false));
    }

    #[test]
    fn snapshot_format_verifier_diff() {
        let stats_a = vec![
            ProgStats {
                name: "enqueue".into(),
                verified_insns: 500,
            },
            ProgStats {
                name: "dispatch".into(),
                verified_insns: 1200,
            },
            ProgStats {
                name: "init".into(),
                verified_insns: 300,
            },
        ];
        let stats_b = vec![
            ProgStats {
                name: "enqueue".into(),
                verified_insns: 480,
            },
            ProgStats {
                name: "dispatch".into(),
                verified_insns: 1350,
            },
            ProgStats {
                name: "init".into(),
                verified_insns: 300,
            },
        ];
        insta::assert_snapshot!(format_verifier_diff("default", &stats_a, "llc", &stats_b));
    }

    #[test]
    fn snapshot_format_verifier_diff_missing_program() {
        let stats_a = vec![
            ProgStats {
                name: "enqueue".into(),
                verified_insns: 500,
            },
            ProgStats {
                name: "new_prog".into(),
                verified_insns: 100,
            },
        ];
        let stats_b = vec![ProgStats {
            name: "enqueue".into(),
            verified_insns: 500,
        }];
        insta::assert_snapshot!(format_verifier_diff("A", &stats_a, "B", &stats_b));
    }

    // -----------------------------------------------------------------------
    // extract_verifier_log — log extraction + cross-check against
    // parse_sched_output so the two slicers stay consistent on shared input.
    // -----------------------------------------------------------------------

    #[test]
    fn extract_verifier_log_between_begin_end_markers() {
        // libbpf wraps the verifier log between explicit marker lines;
        // the extractor returns the content between them, trimmed of
        // the BEGIN newline and the trailing libbpf END prefix.
        let blob = "\
            unrelated preamble\n\
            libbpf: -- BEGIN PROG LOAD LOG --\n\
            processed 1234 insns (limit 1000000) max_states_per_insn 5 total_states 200 peak_states 50 mark_read 10\n\
            libbpf: -- END PROG LOAD LOG --\n\
            trailing diagnostics\n";
        let log = extract_verifier_log(blob).expect("markers present");
        assert!(log.contains("processed 1234 insns"));
        assert!(!log.contains("BEGIN PROG LOAD LOG"));
        assert!(!log.contains("END PROG LOAD LOG"));
    }

    #[test]
    fn extract_verifier_log_returns_none_when_markers_absent() {
        // Backward compat: logs without the libbpf markers are treated
        // as "no markers" — the caller falls back to using the raw blob.
        assert!(extract_verifier_log("no markers in here").is_none());
        assert!(extract_verifier_log("only BEGIN marker -- BEGIN PROG LOAD LOG --").is_none());
    }

    #[test]
    fn extract_verifier_log_consistent_with_parse_sched_output() {
        // `collect_verifier_output` chains parse_sched_output →
        // extract_verifier_log on the VM stdout blob. Both slicers
        // operate on the same input without duplicating work, so a
        // single SCHED_OUTPUT block that wraps a libbpf-marked verifier
        // log must produce the same verifier text when extracted in
        // that order.
        let sched_inner = "\
            libbpf: -- BEGIN PROG LOAD LOG --\n\
            processed 7 insns (limit 1000000) max_states_per_insn 1 total_states 1 peak_states 1 mark_read 0\n\
            libbpf: -- END PROG LOAD LOG --\n";
        let vm_output = format!(
            "kernel boot junk\n{SCHED_OUTPUT_START}\n{sched_inner}{SCHED_OUTPUT_END}\nafterward\n",
        );
        let sched = parse_sched_output(&vm_output).expect("SCHED_OUTPUT block");
        let verifier_log = extract_verifier_log(sched).expect("verifier markers");
        assert!(verifier_log.contains("processed 7 insns"));
        assert!(!verifier_log.contains("SCHED_OUTPUT"));
        assert!(!verifier_log.contains("BEGIN PROG LOAD LOG"));
    }

    #[test]
    fn parse_sched_output_valid() {
        let output = format!(
            "noise\n{SCHED_OUTPUT_START}\nscheduler log line 1\nline 2\n{SCHED_OUTPUT_END}\nmore"
        );
        let parsed = parse_sched_output(&output);
        assert!(parsed.is_some());
        let content = parsed.unwrap();
        assert!(content.contains("scheduler log line 1"));
        assert!(content.contains("line 2"));
    }

    #[test]
    fn parse_sched_output_missing_start() {
        let output = format!("no start\n{SCHED_OUTPUT_END}\n");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_missing_end() {
        let output = format!("{SCHED_OUTPUT_START}\nsome content");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_empty_content() {
        let output = format!("{SCHED_OUTPUT_START}\n\n{SCHED_OUTPUT_END}");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_with_stack_traces() {
        let stack = "do_enqueue_task+0x1a0/0x380\nbalance_one+0x50/0x100\n";
        let output = format!("{SCHED_OUTPUT_START}\n{stack}\n{SCHED_OUTPUT_END}");
        let parsed = parse_sched_output(&output).unwrap();
        assert!(parsed.contains("do_enqueue_task"));
        assert!(parsed.contains("balance_one"));
    }

    #[test]
    fn parse_sched_output_rfind_survives_end_marker_in_content() {
        // Regression: if the scheduler log echoes the END marker
        // inside its own content (e.g. a shell heredoc, a diagnostic
        // that quotes the sentinel), `find` truncated the section at
        // the first occurrence — which was inside the content, not
        // at the terminator. `rfind` anchors on the last occurrence,
        // which is the real terminator.
        let content = format!("line1\nfake {SCHED_OUTPUT_END} inside\nline3");
        let output = format!("{SCHED_OUTPUT_START}\n{content}\n{SCHED_OUTPUT_END}\n");
        let parsed = parse_sched_output(&output).unwrap();
        assert!(
            parsed.contains("line3"),
            "rfind must keep content after an embedded END marker: {parsed:?}"
        );
        assert!(
            parsed.contains("fake"),
            "content before the embedded marker must also survive: {parsed:?}"
        );
    }
}
