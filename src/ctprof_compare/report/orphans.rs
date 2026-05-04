//! Fudge-pair report and one-sided group lists for
//! [`super::write_diff`].
//!
//! Three rendered blocks:
//!
//! - **Fudged cgroup matches** — N:1 cgroup merges the operator
//!   asked the fudge stage to produce. Rendered FIRST so they
//!   surface above the orphan lists; they are the most
//!   informative output and putting them after the orphans
//!   buries them under noise.
//! - **only-baseline group list** — keys present in baseline
//!   but not candidate.
//! - **only-candidate group list** — keys present in candidate
//!   but not baseline.
//!
//! Under [`GroupBy::All`] the orphan lists render with the
//! same cgroup tree shape as the primary table so the operator
//! can map an orphan back to its cgroup hierarchy without
//! parsing the NUL-separated compound key.

use std::fmt;
use std::path::Path;

use super::super::diff_types::CtprofDiff;
use super::super::options::GroupBy;

pub(super) fn write_orphans_section<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    baseline_path: &Path,
    candidate_path: &Path,
    group_by: GroupBy,
) -> fmt::Result {
    write_fudged_pairs(w, diff)?;
    write_only_list(w, "baseline", baseline_path, &diff.only_baseline, group_by)?;
    write_only_list(
        w,
        "candidate",
        candidate_path,
        &diff.only_candidate,
        group_by,
    )?;
    Ok(())
}

fn write_fudged_pairs<W: fmt::Write>(w: &mut W, diff: &CtprofDiff) -> fmt::Result {
    if diff.fudged_pairs.is_empty() {
        return Ok(());
    }
    writeln!(
        w,
        "\n\x1b[1;33m## Fudged cgroup matches ({} pair(s))\x1b[0m",
        diff.fudged_pairs.len()
    )?;
    for fp in &diff.fudged_pairs {
        writeln!(w, "\n  \x1b[36mbaseline:\x1b[0m {}", fp.baseline_cgroup)?;
        writeln!(w, "  \x1b[36mcandidate:\x1b[0m {}", fp.candidate_cgroup)?;
        // Surface cascade roots when they differ from the
        // matched baseline / candidate paths — operators
        // need to see the longest-common-suffix root that
        // governs how cascaded children get joined.
        if fp.baseline_root != fp.baseline_cgroup || fp.candidate_root != fp.candidate_cgroup {
            writeln!(
                w,
                "  cascade roots: baseline={} candidate={}",
                fp.baseline_root, fp.candidate_root,
            )?;
        }
        writeln!(
            w,
            "  overlap: {} thread types, Jaccard: {:.1}%, cascaded children: {}",
            fp.overlap,
            fp.jaccard * 100.0,
            fp.cascaded_children
        )?;
        if !fp.baseline_residual.is_empty() {
            writeln!(
                w,
                "  residual (baseline only): {}",
                fp.baseline_residual.join(", ")
            )?;
        }
        if !fp.candidate_residual.is_empty() {
            writeln!(
                w,
                "  residual (candidate only): {}",
                fp.candidate_residual.join(", ")
            )?;
        }
    }
    Ok(())
}

fn write_only_list<W: fmt::Write>(
    w: &mut W,
    label: &str,
    path: &Path,
    keys: &[String],
    group_by: GroupBy,
) -> fmt::Result {
    if keys.is_empty() {
        return Ok(());
    }
    writeln!(
        w,
        "\n{} group(s) only in {label} ({}):",
        keys.len(),
        path.display()
    )?;
    if group_by == GroupBy::All {
        let mut sorted: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        sorted.sort();
        let mut last_segs: Vec<&str> = Vec::new();
        for k in &sorted {
            let (cg, pc) = k.split_once('\x00').unwrap_or(("", k));
            let segs: Vec<&str> = cg.split('/').filter(|s| !s.is_empty()).collect();
            let common = segs
                .iter()
                .zip(last_segs.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if common < last_segs.len() || segs.len() > last_segs.len() {
                for (depth, seg) in segs.iter().enumerate().skip(common) {
                    let indent = "  ".repeat(depth + 1);
                    writeln!(w, "{indent}{seg}")?;
                }
                last_segs = segs;
            }
            let indent = "  ".repeat(last_segs.len() + 1);
            writeln!(w, "{indent}{pc}")?;
        }
    } else {
        for k in keys {
            writeln!(w, "  {k}")?;
        }
    }
    Ok(())
}
