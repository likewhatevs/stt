use assert_cmd::Command;
use predicates::prelude::*;

fn cargo_ktstr() -> Command {
    let mut cmd = Command::cargo_bin("cargo-ktstr").unwrap();
    cmd.arg("ktstr");
    cmd
}

// -- help output --

#[test]
fn help_lists_subcommands() {
    cargo_ktstr()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("test"))
        .stdout(predicate::str::contains("shell"))
        .stdout(predicate::str::contains("kernel"))
        .stdout(predicate::str::contains("verifier"))
        .stdout(predicate::str::contains("completions"))
        // `LlvmCov` variant renders as `llvm-cov` (clap derive
        // kebab-case default). Pinned with the two-space leading
        // indent that `HelpTemplate::subcmd` emits before every
        // subcommand name (clap_builder-4.6.0/src/output/mod.rs:21
        // `TAB = "  "` + help_template.rs:1070-1071 which pushes
        // TAB then the name). This discriminates the subcommand
        // list entry from incidental doc-text occurrences of
        // "llvm-cov" that would satisfy a bare substring check.
        .stdout(predicate::str::contains("  llvm-cov"))
        // `visible_alias = "nextest"` on the Test variant makes
        // the alias user-facing. Pinned by the literal
        // `[aliases: nextest]` tag emitted by
        // `HelpTemplate::sc_spec_vals` at clap_builder-4.6.0/src/
        // output/help_template.rs:1043 — the styled-ANSI wrappers
        // collapse to empty strings under `assert_cmd`'s non-TTY
        // capture so the plain tag appears verbatim. A regression
        // that dropped `visible_alias` (or switched to the
        // non-visible `alias` form, which `sc_spec_vals` ignores
        // at :1026 where it calls `get_visible_aliases`) would
        // strip the tag and fail this assertion.
        .stdout(predicate::str::contains("[aliases: nextest]"));
}

#[test]
fn help_test() {
    cargo_ktstr()
        .args(["test", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--no-perf-mode"))
        .stdout(predicate::str::contains("cargo nextest"));
}

/// `cargo ktstr nextest --help` reaches the same help page as
/// `cargo ktstr test --help` via the `visible_alias = "nextest"`
/// on the Test variant. Pins that the alias is wired as an alias
/// (not a separate variant) — the help page inherits `--kernel`,
/// `--no-perf-mode`, and the "cargo nextest" passthrough doc.
#[test]
fn help_nextest_alias() {
    cargo_ktstr()
        .args(["nextest", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--no-perf-mode"));
}

/// `cargo ktstr llvm-cov --help` renders the LlvmCov variant's
/// help page. The variant's about text advertises `cargo llvm-cov`
/// passthrough, and both `--kernel` + `--no-perf-mode` are
/// declared on the variant — any of the three would fail if a
/// clap regression re-generated the subcommand with drifted
/// metadata.
#[test]
fn help_llvm_cov() {
    cargo_ktstr()
        .args(["llvm-cov", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--no-perf-mode"))
        .stdout(predicate::str::contains("cargo llvm-cov"));
}

#[test]
fn help_shell() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--topology"))
        .stdout(predicate::str::contains("--memory-mb"))
        .stdout(predicate::str::contains("--no-perf-mode"));
}

/// `cargo ktstr export --help` exposes the four flags the router
/// dispatches on: `<TEST>` positional, `--output`/-o, `--package`/-p
/// (workspace disambiguation), and `--release` (profile pin).
/// Pins the router CLI surface so a future clap regression
/// that drops one of these flags is caught at the help-text level
/// before it surfaces as a misleading "test not found" error in the
/// router.
#[test]
fn help_export() {
    cargo_ktstr()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--package"))
        .stdout(predicate::str::contains("--release"))
        .stdout(predicate::str::contains("<TEST>"));
}

/// `cargo ktstr export <missing>` exits non-zero with a router
/// diagnostic. Pins the "test not found in any workspace test
/// binary" error path: the router builds tests, exec's each, sees
/// every candidate fail with "no registered test named X", and
/// surfaces a bundled error mentioning the candidate count and the
/// last per-binary stderr.
///
/// `#[ignore]`-d because the router executes a full
/// `cargo build --tests` over the entire workspace, compiling
/// every integration test binary — minutes of build time, too
/// heavy for the default `cargo nextest run` pass. Run via
/// `cargo nextest run --include-ignored -E 'test(export_unknown_test_errors)'`
/// to opt in locally.
#[test]
#[ignore = "runs cargo build --tests over the full workspace; minutes of compile time"]
fn export_unknown_test_errors() {
    cargo_ktstr()
        .args(["export", "definitely_not_a_real_ktstr_test_xyzzy_987"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found in any workspace test binary").or(
                predicate::str::contains("definitely_not_a_real_ktstr_test_xyzzy_987"),
            ),
        );
}

#[test]
fn help_kernel() {
    cargo_ktstr()
        .args(["kernel", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("build"))
        .stdout(predicate::str::contains("clean"));
}

#[test]
fn help_kernel_list() {
    cargo_ktstr()
        .args(["kernel", "list", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--json"));
}

#[test]
fn help_kernel_build() {
    cargo_ktstr()
        .args(["kernel", "build", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--source"))
        .stdout(predicate::str::contains("--git"))
        .stdout(predicate::str::contains("--ref"))
        .stdout(predicate::str::contains("--force"))
        .stdout(predicate::str::contains("--clean"))
        .stdout(predicate::str::contains("--extra-kconfig"))
        // `--extra-kconfig` doc must explain that `make olddefconfig`
        // resolves dependencies — the help is the discoverability
        // surface for the merge pipeline. A regression that dropped
        // the explanation would leave operators guessing why a
        // fragment line silently disappeared from the final
        // `.config`.
        .stdout(predicate::str::contains("olddefconfig"));
}

/// `kernel build --extra-kconfig <nonexistent>` must surface an
/// actionable error containing the user's input path verbatim, so a
/// typo names the exact string they passed. Pin the diagnostic
/// shape `--extra-kconfig {path}: {fs error}` produced by
/// `kernel_build`'s up-front file read.
///
/// `KTSTR_CACHE_DIR` is pointed at a tempdir so this test does not
/// touch the developer's real cache root, and `--source` is set to
/// a clearly-nonexistent path so even if the extra-kconfig check
/// were skipped (and the source-tree validation fired instead), the
/// command would still bail before any network or build work.
#[test]
fn kernel_build_extra_kconfig_nonexistent_path_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-extra-kconfig-source-test",
            "--extra-kconfig",
            "/definitely/not/a/real/file/ktstr-extra-kconfig-test.kconfig",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains(
            "/definitely/not/a/real/file/ktstr-extra-kconfig-test.kconfig",
        ));
}

/// Coordinator item 25: a directory passed to `--extra-kconfig`
/// must surface a clear "is a directory" error. The 4-arm error
/// classification in [`ktstr::cli::read_extra_kconfig`] maps the
/// kernel's EISDIR to "is a directory; pass a file" — pin that
/// the operator-facing message names BOTH `--extra-kconfig` and
/// the offending path.
#[test]
fn kernel_build_extra_kconfig_directory_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().join("not-a-file");
    std::fs::create_dir(&dir).unwrap();
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-source-test-dir-arg",
            "--extra-kconfig",
        ])
        .arg(&dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains("is a directory"));
}

/// Coordinator item 24: a non-UTF-8 file passed to
/// `--extra-kconfig` must surface a clear "not valid UTF-8"
/// error. `read_extra_kconfig` rejects with a message that names
/// `--extra-kconfig` + the path so the operator can fix the file.
/// kconfig fragments are required to be ASCII text per kbuild's
/// own parser.
#[test]
fn kernel_build_extra_kconfig_invalid_utf8_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("invalid.kconfig");
    // Lone 0xff is invalid UTF-8 — Vec<u8> with a single 0xff byte
    // fails String::from_utf8 with `Utf8Error`.
    std::fs::write(&path, [0xffu8]).unwrap();
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-source-test-utf8-arg",
            "--extra-kconfig",
        ])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains("not valid UTF-8"));
}

/// Coordinator item 19: an empty file passed to `--extra-kconfig`
/// is NOT an error — `read_extra_kconfig` warns but proceeds. The
/// build then bails when the source-tree check fails (we point
/// `--source` at a nonexistent path), proving the empty-file
/// branch passed through without aborting on the fragment read.
/// stderr carries both the empty-file warning AND the source-tree
/// failure, confirming sequence: empty fragment → warn → continue
/// → source-tree fail.
#[test]
fn kernel_build_extra_kconfig_empty_file_warns_but_proceeds() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("empty.kconfig");
    std::fs::write(&path, b"").unwrap();
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        // RUST_LOG ensures the tracing::warn! emission lands on
        // stderr where the integration test can observe it.
        .env("RUST_LOG", "warn")
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-source-test-empty-arg",
            "--extra-kconfig",
        ])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig file is empty"));
}

/// Coordinator item 26: symlink chain resolution. A
/// `--extra-kconfig` argument that points at a symlink chain
/// (link → link → file) must resolve transparently — the
/// `read_extra_kconfig` helper uses `std::fs::read` which goes
/// through `open(2)` and follows symlinks per kernel default
/// (the same way kbuild reads `KCONFIG_CONFIG`). Pin that a
/// chain of two symlinks resolves to the underlying file's
/// contents without manual canonicalization.
///
/// Test passes when the build proceeds past the fragment-read
/// stage (we point `--source` at a nonexistent path so the
/// command bails on source-tree validation, AFTER the fragment
/// is successfully read). If symlink resolution were broken,
/// `read_extra_kconfig` would error before reaching the source
/// stage and stderr would carry the "--extra-kconfig …" error
/// instead of the source-tree error.
#[test]
fn kernel_build_extra_kconfig_symlink_chain_resolves() {
    let tmp = tempfile::TempDir::new().unwrap();
    let real = tmp.path().join("real.kconfig");
    std::fs::write(&real, b"CONFIG_KTSTR_SYMLINK_TEST=y\n").unwrap();
    let link1 = tmp.path().join("link1.kconfig");
    let link2 = tmp.path().join("link2.kconfig");
    // Build link1 → real, link2 → link1 (two-hop chain).
    std::os::unix::fs::symlink(&real, &link1).unwrap();
    std::os::unix::fs::symlink(&link1, &link2).unwrap();
    let assert = cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-source-symlink-test",
            "--extra-kconfig",
        ])
        .arg(&link2)
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    // Must NOT carry the `--extra-kconfig` error string — the
    // fragment was read successfully through the chain. The
    // failure that surfaces is the source-tree validation
    // (since --source points at nothing), proving the read
    // completed before that next stage.
    assert!(
        !stderr.contains("--extra-kconfig"),
        "symlink chain must resolve transparently — read_extra_kconfig \
         should not surface a `--extra-kconfig` error when the chain \
         resolves to a readable file. stderr={stderr:?}"
    );
}

/// Coordinator item 27: the `--extra-kconfig` validation fires
/// BEFORE source acquisition. A nonexistent extra-kconfig path
/// MUST produce the `--extra-kconfig`-named error even when
/// `--source` is also nonexistent — proving the error precedence.
/// If the order were reversed the test would see the
/// source-tree error instead.
#[test]
fn kernel_build_extra_kconfig_validation_fires_before_source_acquire() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-source-precedence-test",
            "--extra-kconfig",
            "/nonexistent/ktstr-extra-precedence-test.kconfig",
        ])
        .assert()
        .failure()
        // The error MUST name --extra-kconfig (not source-tree
        // failure). `read_extra_kconfig` runs first in
        // `kernel_build`, so its 4-arm classifier surfaces the
        // ENOENT before `kernel_build_one`'s source-acquire branch
        // would have fired.
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains(
            "/nonexistent/ktstr-extra-precedence-test.kconfig",
        ));
}

#[test]
fn help_kernel_clean() {
    cargo_ktstr()
        .args(["kernel", "clean", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--keep"))
        .stdout(predicate::str::contains("--force"));
}

#[test]
fn help_verifier() {
    cargo_ktstr()
        .args(["verifier", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--scheduler"))
        .stdout(predicate::str::contains("--scheduler-bin"))
        .stdout(predicate::str::contains("--all-profiles"))
        .stdout(predicate::str::contains("--profiles"));
}

#[test]
fn help_completions() {
    cargo_ktstr()
        .args(["completions", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("<SHELL>"))
        .stdout(predicate::str::contains("possible values: bash"));
}

// -- error cases --

#[test]
fn verifier_no_scheduler_fails() {
    cargo_ktstr()
        .arg("verifier")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--scheduler"));
}

#[test]
fn no_subcommand_fails() {
    cargo_ktstr().assert().failure();
}

// -- completions --

#[test]
fn completions_bash_produces_output() {
    cargo_ktstr()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_zsh_produces_output() {
    cargo_ktstr()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_fish_produces_output() {
    cargo_ktstr()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_invalid_shell() {
    cargo_ktstr()
        .args(["completions", "noshell"])
        .assert()
        .failure();
}

// -- shell flags in help --

#[test]
fn help_shell_shows_exec() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--exec"));
}

#[test]
fn help_shell_shows_dmesg() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--dmesg"));
}

#[test]
fn help_shell_shows_include_files() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--include-files"));
}

// -- error cases --

#[test]
fn include_files_nonexistent_path() {
    cargo_ktstr()
        .args(["shell", "-i", "/nonexistent/path/to/file"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn shell_invalid_topology() {
    cargo_ktstr()
        .args(["shell", "--topology", "abc"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid topology"));
}

// -- stats --

#[test]
fn stats_no_data() {
    // Pin the read path to an empty directory via KTSTR_SIDECAR_DIR
    // so the test is independent of whatever sits under the
    // developer's target/ktstr/. Bare `cargo ktstr stats` honors
    // KTSTR_SIDECAR_DIR (cli.rs print_stats_report). With nothing
    // there to read the empty-state notice goes to stderr and
    // stdout stays clean.
    let tmp = tempfile::tempdir().unwrap();
    cargo_ktstr()
        .env("KTSTR_SIDECAR_DIR", tmp.path())
        .args(["stats"])
        .assert()
        .success()
        .stderr(predicate::str::contains("no sidecar data found"))
        .stdout(predicate::str::is_empty());
}

// -- kernel list --

#[test]
fn kernel_list_runs() {
    // Isolate from the user's real kernel cache so the assertion is
    // deterministic. With an empty cache directory, `kernel list`
    // prints the cache path header on stderr and a "no cached
    // kernels" hint on stdout.
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .args(["kernel", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("no cached kernels"))
        .stderr(predicate::str::contains("cache:"));
}

#[test]
fn kernel_list_json() {
    cargo_ktstr()
        .args(["kernel", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("entries"));
}

// -- --cpu-cap vs KTSTR_BYPASS_LLC_LOCKS conflict — cargo-ktstr sites --
//
// Pins the parse-time rejection when both the --cpu-cap resource
// contract and the KTSTR_BYPASS_LLC_LOCKS=1 escape hatch are
// active simultaneously. Both sites (cargo-ktstr shell and
// cargo-ktstr kernel build) must bail with "resource contract" in
// the error text so the operator sees the contradiction before a
// pipeline deep-bail.

/// `cargo ktstr shell --no-perf-mode --cpu-cap N` under
/// KTSTR_BYPASS_LLC_LOCKS=1 must fail with the "resource contract"
/// substring. Pins the rejection at bin/cargo-ktstr.rs:851.
#[test]
fn cargo_ktstr_shell_cpu_cap_with_bypass_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .env("KTSTR_BYPASS_LLC_LOCKS", "1")
        .args(["shell", "--no-perf-mode", "--cpu-cap", "2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource contract"));
}

/// `cargo ktstr kernel build --cpu-cap N` under
/// KTSTR_BYPASS_LLC_LOCKS=1 must fail with the "resource contract"
/// substring. Pins the rejection at bin/cargo-ktstr.rs:729.
#[test]
fn cargo_ktstr_kernel_build_cpu_cap_with_bypass_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Pass a clearly-nonexistent --source so if the conflict check
    // were somehow skipped, we'd get a source-acquire failure (not
    // a network fetch hanging forever in CI).
    cargo_ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .env("KTSTR_BYPASS_LLC_LOCKS", "1")
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-cargo-ktstr-cpu-cap-bypass-test",
            "--cpu-cap",
            "2",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource contract"));
}
