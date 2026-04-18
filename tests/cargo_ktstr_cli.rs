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
        .stdout(predicate::str::contains("completions"));
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

#[test]
fn help_shell() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--topology"))
        .stdout(predicate::str::contains("--memory-mb"));
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
        .stdout(predicate::str::contains("--clean"));
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
