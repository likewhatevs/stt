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
    cargo_ktstr().args(["test", "--help"]).assert().success();
}

#[test]
fn help_shell() {
    cargo_ktstr().args(["shell", "--help"]).assert().success();
}

#[test]
fn help_kernel() {
    cargo_ktstr().args(["kernel", "--help"]).assert().success();
}

#[test]
fn help_kernel_list() {
    cargo_ktstr()
        .args(["kernel", "list", "--help"])
        .assert()
        .success();
}

#[test]
fn help_kernel_build() {
    cargo_ktstr()
        .args(["kernel", "build", "--help"])
        .assert()
        .success();
}

#[test]
fn help_kernel_clean() {
    cargo_ktstr()
        .args(["kernel", "clean", "--help"])
        .assert()
        .success();
}

#[test]
fn help_verifier() {
    cargo_ktstr()
        .args(["verifier", "--help"])
        .assert()
        .success();
}

#[test]
fn help_completions() {
    cargo_ktstr()
        .args(["completions", "--help"])
        .assert()
        .success();
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
