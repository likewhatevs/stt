//! `cargo ktstr export` — package a registered test as a self-
//! extracting `.run` reproducer.
//!
//! The exporter cannot run inside cargo-ktstr because the test
//! registry it needs (the `#[ktstr_test]` distributed-slice) lives
//! in user-crate test binaries, not here. [`run_export`] therefore
//! builds every workspace test binary via
//! [`build_test_binaries`] and exec's each in turn with
//! `--ktstr-export-test=NAME`, surfacing the first binary that
//! exits 0 as the winner. All-fail surfaces the most informative
//! per-binary stderr (exit-2 "rejected" preferred over exit-1
//! "not registered"), so the operator sees the actual rejection
//! reason rather than N×"missing" lines.

use std::path::PathBuf;
use std::process::Command;

/// Route `cargo ktstr export <NAME>` to the test binary that owns
/// the named `#[ktstr_test]` registration. cargo-ktstr cannot embed
/// itself into the .run file because it has no `#[ktstr_test]`
/// entries from the user's crate — only the test binary that
/// links against the user's code carries the registry. The router
/// builds every workspace test binary, then exec's each in turn
/// with `--ktstr-export-test=NAME`. The first binary that exits 0
/// wins; all-fail surfaces the per-binary stderrs so the operator
/// can see why the lookup missed (typically: typoed test name).
///
/// `package` (when `Some`) restricts the build via
/// `cargo build --tests --package <NAME>` — necessary in
/// multi-package workspaces where a test name might exist in
/// multiple packages and the operator wants a deterministic
/// resolution.
///
/// `release: true` builds with `--release` so the embedded test
/// binary matches the profile the operator is running. Mismatched
/// profiles can produce a `.run` whose embedded binary's threshold
/// behavior differs from the test runs the operator is reproducing
/// from.
pub(crate) fn run_export(
    test: String,
    output: Option<PathBuf>,
    package: Option<String>,
    release: bool,
) -> Result<(), String> {
    let bins = build_test_binaries(package.as_deref(), release)?;
    if bins.is_empty() {
        return Err("cargo build --tests produced no executable artifacts; \
             ensure the workspace has at least one [[test]] target or \
             a [lib]/[bin] with #[cfg(test)] tests"
            .to_string());
    }

    // Track per-candidate stderr by exit-code category so we can
    // surface the most useful diagnostic on full miss. Exit 2 means
    // "the test exists in this binary but was rejected by export"
    // (host_only, bpf_map_write, KernelBuiltin, or I/O failure) —
    // ALWAYS the most informative outcome, since every other
    // candidate's exit-1 "not registered" message is uninformative
    // when the test actually exists somewhere.
    let mut rejection_stderr: Option<String> = None;
    let mut last_miss_stderr = String::new();
    for bin in &bins {
        let mut cmd = Command::new(bin);
        cmd.arg(format!("--ktstr-export-test={test}"));
        if let Some(o) = output.as_deref() {
            // Resolve relative paths against cwd before forwarding so
            // the test binary writes to the operator's pwd, not its
            // own (the binary lives under target/debug/deps/...).
            let abs = if o.is_absolute() {
                o.to_path_buf()
            } else {
                std::env::current_dir()
                    .map_err(|e| format!("resolve cwd for --output: {e}"))?
                    .join(o)
            };
            cmd.arg(format!("--ktstr-export-output={}", abs.display()));
        }
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::inherit())
            // Capture stderr so per-candidate "no registered test
            // named X" diagnostics don't spam the operator's terminal
            // for every binary we try. We forward the WINNING
            // binary's stderr (carrying export_test's "wrote ..."
            // confirmation) below; on full miss we surface the
            // exit-2 candidate's stderr (rejection reason) when one
            // exists, falling back to the last exit-1 stderr.
            .stderr(std::process::Stdio::piped());

        let out = cmd
            .output()
            .map_err(|e| format!("exec {}: {e}", bin.display()))?;
        if out.status.success() {
            // Forward the winner's stderr so the "wrote ..." line
            // (and any operator-visible diagnostics) reach the
            // user's terminal.
            std::io::Write::write_all(&mut std::io::stderr(), &out.stderr)
                .map_err(|e| format!("forward winner stderr: {e}"))?;
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        // Exit 2 = "registered but rejected." Save the FIRST one and
        // keep going (other candidates might still succeed if
        // multiple binaries register the same test name and one of
        // them admits it). Exit 1 = "not registered here" — record
        // as last_miss_stderr for the fallback diagnostic.
        if out.status.code() == Some(2) {
            if rejection_stderr.is_none() {
                rejection_stderr = Some(stderr);
            }
        } else {
            last_miss_stderr = stderr;
        }
    }

    if let Some(reason) = rejection_stderr {
        return Err(format!(
            "test '{test}' is registered but cannot be exported:\n{}",
            reason.trim_end(),
        ));
    }
    Err(format!(
        "test '{test}' not found in any workspace test binary ({} candidates tried). \
         Last stderr from a candidate:\n{}",
        bins.len(),
        last_miss_stderr.trim_end(),
    ))
}

/// Compile the workspace's test binaries via
/// `cargo build --tests --message-format=json` and collect the
/// resulting executable paths.
///
/// Filters to artifacts where `executable != null` AND either
/// `target.kind` contains `"test"` (integration tests under
/// `tests/`) or `profile.test == true` (unit-test binaries built
/// from `[lib]` / `[bin]` targets). Both shapes carry the
/// `#[ktstr_test]` distributed-slice registry that the export
/// dispatcher reads, so both are valid candidates.
fn build_test_binaries(package: Option<&str>, release: bool) -> Result<Vec<PathBuf>, String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--tests", "--message-format=json"]);
    if let Some(p) = package {
        cmd.args(["--package", p]);
    }
    if release {
        cmd.arg("--release");
    }
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());

    let out = cmd
        .output()
        .map_err(|e| format!("spawn cargo build --tests: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo build --tests failed (exit {})",
            out.status.code().unwrap_or(-1),
        ));
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut bins: Vec<PathBuf> = Vec::new();
    for line in stdout.lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let Some(exe) = msg.get("executable").and_then(|e| e.as_str()) else {
            continue;
        };
        let kinds: Vec<&str> = msg
            .get("target")
            .and_then(|t| t.get("kind"))
            .and_then(|k| k.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let is_test_target = kinds.contains(&"test");
        let is_unit_test = msg
            .get("profile")
            .and_then(|p| p.get("test"))
            .and_then(|t| t.as_bool())
            == Some(true);
        if is_test_target || is_unit_test {
            bins.push(PathBuf::from(exe));
        }
    }
    bins.sort();
    bins.dedup();
    Ok(bins)
}
