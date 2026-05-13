//! Shared test helpers for the `cli/` test modules.
//!
//! Tests across `kernel_list.rs`, `stats_cmds/`, and elsewhere
//! follow the same on-disk layout boilerplate: open a tempdir,
//! create a `<tmp>/<run-name>/` directory, write one or more
//! `.ktstr.json` files into it, call `explain_sidecar(<run-name>,
//! Some(tmp.path()), ...)`. The helpers below collapse that
//! boilerplate so:
//!
//! - new tests do not have to re-derive the directory layout
//!   (parent test code is the only place gauntlet-job nesting
//!   has to land);
//! - a regression that drifts the `.ktstr.json` filename
//!   convention surfaces in one place rather than across
//!   every test that hand-rolls the path.
//!
//! Each helper holds the `tempfile::TempDir` alive via the
//! returned tuple — dropping the helper drops the directory.

#![cfg(test)]

/// Create a tempdir + a named run directory inside it. Returns
/// `(TempDir, run_dir_path)`. The TempDir guard MUST be kept
/// alive in the test scope so the directory survives until
/// the test asserts; Rust drops it at the end of the function
/// scope, which deletes the directory tree.
pub(super) fn make_test_run(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().expect("tempdir must succeed");
    let run_dir = tmp.path().join(name);
    std::fs::create_dir(&run_dir).expect("create run dir");
    (tmp, run_dir)
}

/// Write a serialized [`crate::test_support::SidecarResult`] to
/// `<dir>/<key>.ktstr.json`. `key` is the variant-hash-shaped
/// prefix used by the production writer (see
/// `sidecar_variant_hash`); tests typically use `"a-0000…0"`
/// or a per-test `t-…` for filename-sort determinism.
pub(super) fn write_sidecar(
    dir: &std::path::Path,
    key: &str,
    sc: &crate::test_support::SidecarResult,
) -> std::path::PathBuf {
    let path = dir.join(format!("{key}.ktstr.json"));
    let json = serde_json::to_string(sc).expect("fixture must serialize");
    std::fs::write(&path, json).expect("write sidecar");
    path
}

/// Write raw bytes (intended to be unparseable JSON or an
/// alternate serialization of `SidecarResult` with mutated
/// keys) to `<dir>/<key>.ktstr.json`. Used by parse-failure
/// and old-key-archive tests. Returns the resolved path so
/// callers can assert against `path.display().to_string()`.
pub(super) fn write_corrupt_sidecar(
    dir: &std::path::Path,
    key: &str,
    body: &str,
) -> std::path::PathBuf {
    let path = dir.join(format!("{key}.ktstr.json"));
    std::fs::write(&path, body).expect("write corrupt sidecar");
    path
}

/// `Vec<T>` field names on [`crate::test_support::SidecarResult`].
/// These fields are hard-required (serde fails deserialize on
/// absence) and serialize as `[]` when empty — distinct from
/// the 10 `Option<T>` fields the diagnostic surface enumerates.
/// The catalog and projection helper MUST never surface these
/// names, since "missing Option" and "empty Vec" are different
/// invariants.
///
/// Pinned as a constant so the
/// `explain_sidecar_does_not_flag_empty_vec_fields_as_none`
/// test sources the same list. The check is one-directional:
/// it asserts that none of these names appear in the
/// explain-sidecar output. A schema change that REMOVES or
/// RENAMES a Vec field must drop the corresponding entry here
/// — leaving a stale name keeps the check enforcing absence
/// of a string that the renderer cannot emit, which still
/// passes but conveys nothing. A schema change that ADDS a
/// Vec field is not caught by this guard (it would have to
/// appear in output to fail, and the renderer only emits the
/// Option-field catalog).
pub(super) const SIDECAR_VEC_FIELDS: &[&str] = &[
    "metrics",
    "stimulus_events",
    "verifier_stats",
    "sysctls",
    "kargs",
];
