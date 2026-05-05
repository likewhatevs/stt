//! Source-file size guard.
//!
//! Asserts that no `src/**/*.rs` file exceeds 3000 lines, with a
//! grandfathered-exceptions list naming each file currently above
//! the limit and pinning its exact line count. The list drains as
//! split tasks land: when a previously-grandfathered file falls back
//! below 3000 lines (or is removed), the test fails with a "remove
//! `<path>` from EXCEPTIONS — the file is now under 3000 lines"
//! message that forces the next person to delete the now-irrelevant
//! entry. This makes the exceptions list a working ratchet that
//! tightens monotonically toward zero entries.
//!
//! # Failure modes
//!
//! Three independent regressions surface here:
//!
//! 1. A file NOT in `EXCEPTIONS` exceeds 3000 lines — a brand-new
//!    monolith was introduced or an existing small file grew past
//!    the threshold. Either split it or add it to `EXCEPTIONS` with
//!    its current line count and a note explaining the deferral.
//! 2. A file IN `EXCEPTIONS` grew past its grandfathered line count.
//!    The exception is a ceiling, not a license — entries are
//!    expected to shrink over time, never grow.
//! 3. A file IN `EXCEPTIONS` dropped to ≤ 3000 lines (or was
//!    removed from the source tree). The `EXCEPTIONS` entry is now
//!    stale and must be deleted so future regressions are caught
//!    by the default 3000-line gate rather than masked by a stale
//!    grandfather entry.
//!
//! Each failure mode names the exact file and the corrective action
//! so the operator does not have to reverse-engineer the intent.
//!
//! # Why a hard line-count limit
//!
//! Files past 3000 lines are effectively un-navigable: a reviewer's
//! "read every line of every changed file" obligation (per the
//! project's convergence protocol) becomes a multi-pass effort
//! with high risk of skimmed sections. Splitting into submodules
//! restores the per-pass cost to something a reviewer can actually
//! cover. The 3000-line threshold matches the precedent set by the
//! recent split commits ("split 8 monolithic source files into
//! submodules for navigability", "split cache, ctprof, ctprof_compare,
//! and workload into submodules") — every file split in those
//! commits was past 3000 lines.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Default soft-limit for `src/**/*.rs` files. Any file that exceeds
/// this AND is not in `EXCEPTIONS` fails the guard.
const DEFAULT_MAX_LINES: usize = 3000;

/// Grandfathered files that exceed `DEFAULT_MAX_LINES` today. Each
/// entry records the path (relative to `<repo>/src/`) and the file's
/// current line count at the time the exception was added. The
/// guard treats the recorded count as the file's individual ceiling
/// — the file may shrink (and must, to drain its entry) but must
/// never grow past the ceiling.
///
/// **Drain protocol.** When a file in this list is split or
/// otherwise reduced below `DEFAULT_MAX_LINES`, the guard fails with
/// a "remove `<path>` from EXCEPTIONS" message. The fix is to delete
/// the entry from this list — the default 3000-line gate then
/// guards the file going forward without the grandfathered
/// loosening. Do NOT lower the recorded count to track a partial
/// reduction; the entry exists to acknowledge the file is over the
/// limit, and any reduction below 3000 means the entry's purpose is
/// served and it should be removed entirely.
///
/// **Adding a new entry** is allowed only when the file is being
/// genuinely deferred (e.g. a split task is queued but not yet
/// landed). New entries should not be used to paper over an
/// uncoordinated growth; the standard remediation for a NEW file
/// past 3000 lines is to split it before the change lands. A new
/// entry must reference the queued split task in a `// queued: ...`
/// comment so the deferral is auditable.
///
/// Counts pinned 2026-05-04; refresh per the drain protocol.
const EXCEPTIONS: &[(&str, usize)] = &[
    ("stats.rs", 9541),
    ("assert.rs", 8232),
    ("test_support/model.rs", 7841),
    ("test_support/sidecar.rs", 7411),
    ("scenario/ops/mod.rs", 7051),
    ("test_support/eval.rs", 5327),
    ("scenario/payload_run.rs", 5200),
    ("vmm/virtio_blk/device.rs", 4988),
    ("vmm/host_topology.rs", 4906),
    ("monitor/mod.rs", 4361),
    ("monitor/dump/tests.rs", 4131),
    ("monitor/reader.rs", 3655),
    ("cache/cache_dir.rs", 3650),
    ("ctprof/mod.rs", 3485),
    ("bin/cargo_ktstr/parse_tests.rs", 3440),
    ("monitor/btf_render.rs", 3332),
    ("workload/worker/mod.rs", 3274),
    ("vmm/initramfs.rs", 3058),
    ("workload/spawn/mod.rs", 3044),
];

/// Resolve `<repo>/src` from `CARGO_MANIFEST_DIR`. Cargo always
/// sets this env var when running tests; running the integration
/// test binary outside cargo without this env var would fail at
/// the env-var read with a clear missing-env message rather than
/// an obscure path-not-found later.
fn src_root() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect(
        "CARGO_MANIFEST_DIR must be set (cargo sets it for cargo-test / cargo-nextest \
         invocations; running this test outside of cargo is unsupported)",
    );
    PathBuf::from(manifest_dir).join("src")
}

/// Render a source file's path relative to `<repo>/src`. The
/// `EXCEPTIONS` list keys off this relative form so platform
/// path-separator differences do not affect lookup; the helper
/// normalises to forward-slash form on every platform so the
/// `EXCEPTIONS` literal stays portable.
fn rel_path(file: &Path, src_root: &Path) -> String {
    let rel = file
        .strip_prefix(src_root)
        .expect("file must live under src_root");
    // Forward-slash form on every platform — matches EXCEPTIONS keys.
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Count newline-terminated lines in `path` by reading the file as
/// bytes and counting `\n` occurrences. A trailing-newline-less file
/// is counted with the same line count `wc -l` reports (no trailing
/// `\n` → final fragment is not counted as a line). The integration
/// test does not need exact `wc -l` parity — it needs a stable
/// proxy for "how many lines a reviewer must read" — so the bytewise
/// `\n` count is sufficient.
fn count_lines(path: &Path) -> usize {
    let bytes = std::fs::read(path).expect("read source file");
    bytes.iter().filter(|&&b| b == b'\n').count()
}

#[test]
fn no_src_file_exceeds_3000_lines_unless_grandfathered() {
    let src = src_root();
    assert!(
        src.is_dir(),
        "src directory does not exist at {src:?}; CARGO_MANIFEST_DIR may be wrong",
    );

    // Build a name → ceiling map from EXCEPTIONS so the per-file
    // check is O(1). BTreeMap (rather than HashMap) gives
    // deterministic iteration when reporting unused entries.
    let exceptions: BTreeMap<&str, usize> = EXCEPTIONS.iter().copied().collect();
    assert_eq!(
        exceptions.len(),
        EXCEPTIONS.len(),
        "EXCEPTIONS contains a duplicate path key — each file must \
         appear exactly once",
    );

    // Walk src/**/*.rs and aggregate findings. Collecting all
    // findings before failing surfaces every regression in one
    // shot rather than aborting on the first.
    let mut new_overflows: Vec<(String, usize)> = Vec::new();
    let mut grew_past_ceiling: Vec<(String, usize, usize)> = Vec::new();
    let mut seen_exceptions: BTreeMap<&str, bool> =
        exceptions.keys().map(|k| (*k, false)).collect();

    for entry in walkdir::WalkDir::new(&src).into_iter() {
        let entry = entry.expect("walkdir must succeed under src/");
        let path = entry.path();
        if !entry.file_type().is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let rel = rel_path(path, &src);
        let lines = count_lines(path);

        match exceptions.get(rel.as_str()) {
            Some(&ceiling) => {
                if let Some(seen) = seen_exceptions.get_mut(rel.as_str()) {
                    *seen = true;
                }
                if lines > ceiling {
                    grew_past_ceiling.push((rel, lines, ceiling));
                }
                // Drain signal (lines <= DEFAULT_MAX_LINES on a
                // grandfathered file) is handled in the post-walk
                // pass below — it covers BOTH the "shrank past the
                // limit" case and the "file was deleted" case under
                // a single stale-entry banner.
            }
            None => {
                if lines > DEFAULT_MAX_LINES {
                    new_overflows.push((rel, lines));
                }
            }
        }
    }

    // Build the drain-or-remove list: every EXCEPTIONS entry whose
    // file (a) does not exist OR (b) shrank to ≤ DEFAULT_MAX_LINES
    // is now stale and must be removed from the literal.
    let mut stale_exceptions: Vec<(String, Option<usize>)> = Vec::new();
    for key in exceptions.keys() {
        let path = src.join(key);
        if !path.is_file() {
            // The file was removed (e.g. split into submodules and
            // the original `<name>.rs` deleted) — entry is stale.
            stale_exceptions.push(((*key).to_string(), None));
            continue;
        }
        let lines = count_lines(&path);
        if lines <= DEFAULT_MAX_LINES {
            stale_exceptions.push(((*key).to_string(), Some(lines)));
        }
    }

    // Sanity: every EXCEPTIONS key must correspond to an existing
    // file under src/ (otherwise the entry has no meaning). The
    // walk above only sets `seen_exceptions[...] = true` when the
    // walker visits the file, so an unreached key signals either a
    // typo in the literal OR the file was removed without dropping
    // the entry.
    let unreached: Vec<&str> = seen_exceptions
        .iter()
        .filter_map(|(k, seen)| if *seen { None } else { Some(*k) })
        .collect();

    let any_failure = !new_overflows.is_empty()
        || !grew_past_ceiling.is_empty()
        || !stale_exceptions.is_empty()
        || !unreached.is_empty();

    if any_failure {
        let mut msg = String::from("src-file size guard failed:\n\n");
        if !new_overflows.is_empty() {
            msg.push_str("(1) Files NOT in EXCEPTIONS that exceed 3000 lines:\n");
            for (path, lines) in &new_overflows {
                msg.push_str(&format!(
                    "    src/{path} = {lines} lines (limit {DEFAULT_MAX_LINES})\n"
                ));
            }
            msg.push_str(
                "    Fix: split the file into submodules, or add it to \
                 EXCEPTIONS with its current line count and a `// queued: \
                 <task>` comment naming the queued split task.\n\n",
            );
        }
        if !grew_past_ceiling.is_empty() {
            msg.push_str("(2) Grandfathered files that grew past their pinned ceiling:\n");
            for (path, lines, ceiling) in &grew_past_ceiling {
                msg.push_str(&format!(
                    "    src/{path} = {lines} lines (grandfathered ceiling {ceiling})\n"
                ));
            }
            msg.push_str(
                "    Fix: shrink the file (preferred) OR refresh the \
                 EXCEPTIONS entry's count if the growth is genuinely \
                 unavoidable. Refreshing requires a reviewer sign-off — \
                 the entry is a ratchet, not a free pass.\n\n",
            );
        }
        if !stale_exceptions.is_empty() {
            msg.push_str(
                "(3) EXCEPTIONS entries that are now stale (file ≤ 3000 \
                 lines or removed) — remove these entries:\n",
            );
            for (path, lines) in &stale_exceptions {
                match lines {
                    Some(n) => msg.push_str(&format!(
                        "    src/{path} now {n} lines (≤ {DEFAULT_MAX_LINES}); remove from EXCEPTIONS\n"
                    )),
                    None => msg.push_str(&format!(
                        "    src/{path} no longer exists; remove from EXCEPTIONS\n"
                    )),
                }
            }
            msg.push_str(
                "    Fix: delete the listed entries from EXCEPTIONS. \
                 The default 3000-line gate guards the file going forward.\n\n",
            );
        }
        if !unreached.is_empty() {
            msg.push_str(
                "(4) EXCEPTIONS keys not visited by the walk (typo or \
                 removed file):\n",
            );
            for key in &unreached {
                msg.push_str(&format!("    src/{key}\n"));
            }
            msg.push_str(
                "    Fix: correct the path string or delete the entry. \
                 Every EXCEPTIONS key must resolve to a real file under src/.\n\n",
            );
        }
        panic!("{msg}");
    }
}
