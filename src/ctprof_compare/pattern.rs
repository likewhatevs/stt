//! Token-based name normalization used by [`crate::ctprof_compare`]
//! to fold ephemeral digit/hex suffixes into pattern skeletons.
//!
//! Two callers feed this module:
//! - thread / process axes ([`super::GroupBy::Comm`] /
//!   [`super::GroupBy::Pcomm`]) and the smaps_rollup keying use
//!   [`pattern_key`] / [`pattern_counts_union`] /
//!   [`pattern_display_label`].
//! - the cgroup axis applies a three-layer pipeline
//!   ([`apply_systemd_template`] →
//!   [`cgroup_skeleton_tokens`] → [`tighten_group`]) wrapped by
//!   [`cgroup_normalize_skeleton`].
//!
//! The rules are pure (no kernel introspection, no IO) so they
//! sit in their own module without pulling the data-type or
//! compute layers along.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::ctprof::{CtprofSnapshot, ThreadState};

/// Placeholder for a pure-digit token (rule 1 of the token-based
/// normalizer). Replaces a token of all ASCII digits.
const TOKEN_DIGIT_PLACEHOLDER: &str = "{N}";

/// Placeholder for a hex-like token (rule 2 of the token-based
/// normalizer). Replaces a token whose chars are all in `[0-9a-f]`,
/// length ≥ 2, and contain at least one digit.
const TOKEN_HEX_PLACEHOLDER: &str = "{H}";

/// Placeholder for a systemd template instance whose value is an
/// opaque ID (rule applied by [`apply_systemd_template`] in cgroup
/// layer 1). For example, `user@0.service` and `user@1001.service`
/// both normalize to `user@{I}.service` because their instances
/// (`0`, `1001`) carry no `[._-]` separators that would suggest a
/// structured service name.
const TOKEN_INSTANCE_PLACEHOLDER: &str = "{I}";

/// Rule 1 pattern: pure ASCII digits.
static TOKEN_RULE_PURE_DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[0-9]+$").unwrap());

/// Rule 2 pattern: hex-like (all chars in `[0-9a-f]`, length ≥ 2).
/// The "must contain at least one digit" check is applied
/// separately because anchored character-class repetition does
/// not natively express that constraint.
static TOKEN_RULE_HEX_LIKE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[0-9a-f]{2,}$").unwrap());

/// Rule 3 pattern: alpha prefix (length ≥ 1) followed by
/// trailing digits. Capture group 1 is the alpha prefix.
static TOKEN_RULE_ALPHA_PREFIX_DIGITS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Za-z]+)[0-9]+$").unwrap());

/// Rule 4 pattern: leading digits followed by an alpha suffix
/// (length ≥ 1). Capture group 1 is the alpha suffix. Catches
/// kworker high-priority bound workers (`1H`, `0H`, `2H` etc. —
/// the `H` suffix added by `format_worker_id` in
/// `kernel/workqueue.c` when the worker pool's nice value is
/// negative).
static TOKEN_RULE_DIGITS_ALPHA_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[0-9]+([A-Za-z]+)$").unwrap());

/// Token-classification rule. The token-based normalizer
/// ([`pattern_key`]) walks segments produced by
/// [`split_into_segments`] and applies the first rule that
/// matches each token. Rules are checked in order; the first
/// match wins. Rule patterns are direct regex translations of
/// the thread-name normalization rules.
pub(super) fn classify_token(t: &str) -> String {
    if t.is_empty() {
        return String::new();
    }
    // Rule 1: pure digits → `{N}`.
    if TOKEN_RULE_PURE_DIGITS.is_match(t) {
        return TOKEN_DIGIT_PLACEHOLDER.to_string();
    }
    // Rule 2: hex-like (all chars in [0-9a-f], length ≥ 2,
    // contains at least one digit) → `{H}`. The regex enforces
    // the character set + length; the `.contains` check enforces
    // the "must have at least one digit" gate that the spec
    // requires. Pure-alpha tokens like `abc` fail the digit check;
    // pure-digit tokens fall through to rule 1 first.
    if TOKEN_RULE_HEX_LIKE.is_match(t) && t.chars().any(|c| c.is_ascii_digit()) {
        return TOKEN_HEX_PLACEHOLDER.to_string();
    }
    // Rule 3: alpha prefix + trailing digits → `prefix{N}`. The
    // captured group is the alpha prefix; the trailing digit run
    // is replaced with the placeholder. Single-letter alpha
    // prefixes like `u8` (`kworker/u8:7`) qualify because the
    // spec sets the prefix lower bound at 1.
    if let Some(caps) = TOKEN_RULE_ALPHA_PREFIX_DIGITS.captures(t) {
        return format!("{}{}", &caps[1], TOKEN_DIGIT_PLACEHOLDER);
    }
    // Rule 4: leading digits + alpha suffix → `{N}suffix`. The
    // captured group is the alpha suffix. Comes AFTER rule 2 so
    // hex-like tokens (`1a`, `0f`) take precedence over the
    // leading-digit-suffix interpretation.
    if let Some(caps) = TOKEN_RULE_DIGITS_ALPHA_SUFFIX.captures(t) {
        return format!("{}{}", TOKEN_DIGIT_PLACEHOLDER, &caps[1]);
    }
    // Otherwise: keep literal.
    t.to_string()
}

/// One segment of a tokenized string: either a non-separator run
/// (a token) or a run of separator characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Segment<'a> {
    Token(&'a str),
    Separator(&'a str),
}

/// Returns true for any character treated as a token separator by
/// the token-based normalizer. The set is `[.\-_/:@+\[\]\s]+` —
/// ASCII punctuation `.`, `-`, `_`, `/`, `:`, `@`, `+`, `[`, `]`
/// plus any Unicode whitespace. The `+` decoration kworker uses on
/// active workers (`kworker/<cpu>:<id>+<wq>` per `wq_worker_comm`
/// in `kernel/workqueue.c`) is a separator so the digit tokens on
/// either side normalize independently. Brackets appear in
/// process names set via `prctl(PR_SET_NAME)` (kernel threads in
/// userspace tooling render as `[ksoftirqd/0]`, etc.) AND in the
/// literal-mode smaps key shape `pcomm[tgid]` produced by
/// [`crate::ctprof_compare::collect_smaps_rollup`] under
/// [`crate::ctprof_compare::CompareOptions::no_thread_normalize`];
/// treating brackets as separators allows the digit / hex tokens
/// inside them to normalize independently from the surrounding
/// alpha tokens.
pub(super) fn is_token_separator(c: char) -> bool {
    matches!(c, '.' | '-' | '_' | '/' | ':' | '@' | '+' | '[' | ']') || c.is_whitespace()
}

/// Walk the input and emit alternating token / separator runs.
/// Empty input yields zero segments. Maximal runs are emitted —
/// `a..b` produces `[Token("a"), Separator(".."), Token("b")]`.
pub(super) fn split_into_segments(s: &str) -> Vec<Segment<'_>> {
    let mut out = Vec::new();
    if s.is_empty() {
        return out;
    }
    let mut chars = s.char_indices().peekable();
    while let Some(&(start, first_c)) = chars.peek() {
        let is_sep = is_token_separator(first_c);
        let mut end = start;
        while let Some(&(idx, c)) = chars.peek() {
            if is_token_separator(c) != is_sep {
                break;
            }
            end = idx + c.len_utf8();
            chars.next();
        }
        let slice = &s[start..end];
        if is_sep {
            out.push(Segment::Separator(slice));
        } else {
            out.push(Segment::Token(slice));
        }
    }
    out
}

/// Compute the token-normalized skeleton for a name string.
///
/// Consumed by [`crate::ctprof_compare::GroupBy::Comm`]
/// (thread-name grouping),
/// [`crate::ctprof_compare::GroupBy::Pcomm`] (process-name
/// grouping), and
/// [`crate::ctprof_compare::collect_smaps_rollup`] (per-pcomm
/// smaps aggregation) — each callsite passes a different field
/// (`t.comm`, `t.pcomm`, `t.pcomm` respectively) and applies its
/// own callsite-level policy on top of the skeleton this function
/// returns.
///
/// Splits the input on a separator class (`[.\-_/:@+\[\]\s]+`),
/// classifies each non-separator token by [`classify_token`], and
/// rejoins with the original separator runs preserved verbatim.
/// The first matching rule wins per token:
///
/// 1. Pure digits → `{N}` (e.g. `42` → `{N}`).
/// 2. Hex-like (all chars `[0-9a-f]`, length ≥ 2, contains at
///    least one digit) → `{H}` (e.g. `abc123def` → `{H}`).
/// 3. Alpha prefix + trailing digits (`^[A-Za-z]+\d+$`, alpha
///    prefix length ≥ 1) → `prefix{N}` (e.g. `worker7` →
///    `worker{N}`, `u8` → `u{N}`).
/// 4. Leading digits + alpha suffix (`^\d+[A-Za-z]+$`) →
///    `{N}suffix` (e.g. `1H` → `{N}H`, `100Hz` → `{N}Hz`).
/// 5. Otherwise: keep literal.
///
/// Two names that produce the same skeleton group together at
/// the bucket layer. The singleton-revert policy ("if only one
/// thread / process matches a skeleton, revert to literal") is a
/// callsite policy enforced by
/// [`crate::ctprof_compare::build_groups`] — `pattern_key` itself
/// always returns the skeleton, leaving callsites free to override
/// (and indeed [`crate::ctprof_compare::collect_smaps_rollup`]
/// does NOT singleton-revert; see its doc for why).
///
/// Examples:
/// - `whirly-gig-15` → `whirly-gig-{N}`.
/// - `kworker/0:0-wq_reclaim` → `kworker/{N}:{N}-wq_reclaim`.
/// - `kworker/u8:7` → `kworker/u{N}:{N}` (single-letter alpha
///   prefix `u` qualifies under rule 3).
/// - `session-a1234` → `session-{H}` (hex-like).
/// - `BPF_CUBIC` → `BPF_CUBIC` (pure alpha, no digits).
/// - `bloop-tangler` → `bloop-tangler` (pure alpha).
pub fn pattern_key(name: &str) -> String {
    let segments = split_into_segments(name);
    let mut out = String::new();
    for seg in segments {
        match seg {
            Segment::Token(t) => out.push_str(&classify_token(t)),
            Segment::Separator(s) => out.push_str(s),
        }
    }
    out
}

/// Cgroup layer 1: systemd `template@instance.service`
/// normalization. Walks the path, finding each
/// `@<instance>.service` segment (bounded by `/` or end-of-string).
/// If the instance contains any of `[._-]`, it is a structured
/// service name and the segment is kept verbatim. Otherwise, the
/// instance is treated as an opaque ID and the segment is rewritten
/// to `@{I}.service`.
///
/// Examples:
/// - `/user.slice/user-0.slice/user@0.service/boot.scope`
///   → `/user.slice/user-0.slice/user@{I}.service/boot.scope`
///   (`0` has no `[._-]`).
/// - `/critical.slice/launcher@foo.bar.baz.service`
///   → unchanged (instance `foo.bar.baz` has `.`).
pub(super) fn apply_systemd_template(path: &str) -> String {
    let mut out = String::new();
    let mut rest = path;
    while let Some(at_idx) = rest.find('@') {
        out.push_str(&rest[..at_idx]);
        out.push('@');
        let after_at = &rest[at_idx + 1..];
        // Bound the instance segment by the next `/` (or end-of-input).
        let segment_end = after_at.find('/').unwrap_or(after_at.len());
        let segment = &after_at[..segment_end];
        if let Some(instance) = segment.strip_suffix(".service") {
            if instance.is_empty() || instance.contains(['.', '_', '-']) {
                // Structured instance — keep verbatim.
                out.push_str(segment);
            } else {
                // Opaque ID — normalize.
                out.push_str(TOKEN_INSTANCE_PLACEHOLDER);
                out.push_str(".service");
            }
            rest = &after_at[segment_end..];
        } else {
            // No `.service` suffix on this segment — leave the `@`
            // and continue scanning after it.
            rest = after_at;
        }
    }
    out.push_str(rest);
    out
}

/// Cgroup layer 2: token-based normalization. Identical to
/// [`pattern_key`] but operates on a cgroup path string. Returns
/// the post-Layer-1 token list alongside the normalized skeleton —
/// the token list is consumed by [`tighten_group`] to revert
/// constant-across-members positions to literals (Layer 3).
pub(super) fn cgroup_skeleton_tokens(post_l1: &str) -> (String, Vec<String>) {
    let segments = split_into_segments(post_l1);
    let mut skeleton = String::new();
    let mut tokens = Vec::new();
    for seg in segments {
        match seg {
            Segment::Token(t) => {
                tokens.push(t.to_string());
                skeleton.push_str(&classify_token(t));
            }
            Segment::Separator(s) => {
                skeleton.push_str(s);
            }
        }
    }
    (skeleton, tokens)
}

/// Cgroup layer 3 (tighten): for a multi-member group sharing the
/// same Layer-2 skeleton, revert any token position whose value is
/// identical across every member to its literal form. Positions
/// that vary across members keep their Layer-2 placeholder.
///
/// Members carry both their post-Layer-1 path (used to recover
/// separator runs verbatim from a representative member) and their
/// per-position token list (compared across members for the
/// position-by-position equality check). All members share the
/// same number of tokens and the same separator structure by
/// construction — they share a Layer-2 skeleton.
///
/// Returns the tightened skeleton; if every position varies
/// (nothing to tighten), the result equals the input skeleton.
pub(super) fn tighten_group(post_l1_paths: &[String], member_tokens: &[Vec<String>]) -> String {
    let representative = match post_l1_paths.first() {
        Some(p) => p,
        None => return String::new(),
    };
    let segments = split_into_segments(representative);
    let mut out = String::new();
    let mut token_pos = 0;
    for seg in segments {
        match seg {
            Segment::Token(_) => {
                let first = &member_tokens[0][token_pos];
                let classified = classify_token(first);
                let all_equal = member_tokens
                    .iter()
                    .all(|tokens| &tokens[token_pos] == first);
                if all_equal && classified == *first {
                    out.push_str(first);
                } else {
                    out.push_str(&classified);
                }
                token_pos += 1;
            }
            Segment::Separator(s) => {
                out.push_str(s);
            }
        }
    }
    out
}

/// Compute the cgroup grouping key for a path under
/// [`crate::ctprof_compare::GroupBy::Cgroup`] aggregation. Applies
/// Layer 1 (systemd template) and Layer 2 (token normalization).
/// Layer 3 (tighten) runs separately on multi-member groups inside
/// [`crate::ctprof_compare::build_groups`].
///
/// Returns `(layer2_skeleton, post_l1_path, post_l1_tokens)`. The
/// skeleton is the join key; the post-L1 path and tokens feed
/// [`tighten_group`] for groups with ≥ 2 members.
pub(super) fn cgroup_normalize_skeleton(path: &str) -> (String, String, Vec<String>) {
    let post_l1 = apply_systemd_template(path);
    let (skeleton, tokens) = cgroup_skeleton_tokens(&post_l1);
    (skeleton, post_l1, tokens)
}

/// Compute the operator-facing display label for a pattern-aware
/// group, given the union of baseline+candidate member comms. For
/// buckets with ≥ 2 distinct member names, runs grex over the
/// sorted union to emit a regex that exactly matches the
/// constituent thread names. For singleton or all-identical
/// buckets, returns the join key unchanged so the rendered label
/// equals what would have shown under literal grouping.
///
/// Empty `members` returns `key` — defensive against synthetic
/// inputs; production builds populate `members` for every
/// bucket.
pub fn pattern_display_label(key: &str, members: &[String]) -> String {
    if members.len() < 2 {
        return key.to_string();
    }
    let regex = grex::RegExpBuilder::from(members).build();
    if regex.len() <= key.len() {
        regex
    } else {
        key.to_string()
    }
}

/// Build the union frequency map for pattern-aware grouping
/// ([`crate::ctprof_compare::GroupBy::Comm`] or
/// [`crate::ctprof_compare::GroupBy::Pcomm`]) across the
/// baseline + candidate snapshots. The frequency gate that
/// promotes a `pattern_key` from per-thread literal to a clustered
/// bucket must be evaluated against the UNION of both
/// snapshots' threads — otherwise a pattern that has 1 thread
/// in baseline + 3 threads in candidate would join under a
/// `worker-{N}` key in candidate but a literal `worker-7` key in
/// baseline, and `compare()` would surface the row as
/// only-in-candidate. Computing the count from the union ensures
/// the same key is used on both sides.
///
/// `field` selects which [`ThreadState`] string feeds the count:
/// `|t| t.comm.as_str()` for `Comm`, `|t| t.pcomm.as_str()` for
/// `Pcomm`. The two axes share the same union-frequency contract
/// so one helper covers both.
pub(super) fn pattern_counts_union(
    baseline: &CtprofSnapshot,
    candidate: &CtprofSnapshot,
    field: fn(&ThreadState) -> &str,
) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for t in baseline.threads.iter().chain(candidate.threads.iter()) {
        *counts.entry(pattern_key(field(t))).or_insert(0) += 1;
    }
    counts
}
