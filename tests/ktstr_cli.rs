use assert_cmd::Command;
use predicates::prelude::*;

fn ktstr() -> Command {
    Command::cargo_bin("ktstr").unwrap()
}

// -- help output --

#[test]
fn help_lists_subcommands() {
    ktstr()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("shell"))
        .stdout(predicate::str::contains("kernel"))
        .stdout(predicate::str::contains("completions"));
}

#[test]
fn help_shell() {
    ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--topology"))
        .stdout(predicate::str::contains("--memory-mb"));
}

#[test]
fn help_shell_shows_exec() {
    ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--exec"));
}

#[test]
fn help_shell_shows_dmesg() {
    ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--dmesg"));
}

#[test]
fn help_shell_shows_include_files() {
    ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--include-files"));
}

#[test]
fn help_kernel() {
    ktstr()
        .args(["kernel", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("build"))
        .stdout(predicate::str::contains("clean"));
}

#[test]
fn help_kernel_list() {
    ktstr()
        .args(["kernel", "list", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--json"));
}

#[test]
fn help_kernel_build() {
    ktstr()
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
    ktstr()
        .args(["kernel", "clean", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--keep"))
        .stdout(predicate::str::contains("--force"));
}

// -- error cases --

#[test]
fn no_subcommand_fails() {
    ktstr().assert().failure();
}

#[test]
fn include_files_nonexistent_path() {
    ktstr()
        .args(["shell", "-i", "/nonexistent/path/to/file"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn shell_invalid_topology() {
    ktstr()
        .args(["shell", "--topology", "abc"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid topology"));
}

#[test]
fn shell_zero_topology() {
    ktstr()
        .args(["shell", "--topology", "0,1,1,1"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must be >= 1"));
}

// -- completions --

#[test]
fn completions_bash() {
    ktstr()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_zsh() {
    ktstr()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

// -- include-files directory support --

#[test]
fn include_files_empty_dir_warns() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Empty dir should warn but not fail (the shell command will fail
    // for other reasons like no KVM, but the include resolution succeeds).
    // We test via the resolve function rather than the full shell command.
    let result = ktstr::cli::resolve_include_files(&[tmp.path().to_path_buf()]);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn include_files_dir_walks_recursively() {
    let tmp = tempfile::TempDir::new().unwrap();
    let sub = tmp.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("file.txt"), "hello").unwrap();
    std::fs::write(tmp.path().join("root.txt"), "world").unwrap();

    let result = ktstr::cli::resolve_include_files(&[tmp.path().to_path_buf()]).unwrap();
    assert_eq!(result.len(), 2);
    // Archive paths should preserve directory structure.
    let paths: Vec<&str> = result.iter().map(|(a, _)| a.as_str()).collect();
    assert!(paths.iter().any(|p| p.contains("root.txt")));
    assert!(paths.iter().any(|p| p.contains("sub/file.txt")));
}

// -- virtio-console end-to-end via --exec --

/// Full data path test: host → virtio RX → guest hvc0 → busybox sh -c →
/// virtio TX → host stdout. Requires /dev/kvm and a cached kernel.
/// Skips when either is unavailable.
#[test]
fn shell_exec_echo() {
    // Skip if no /dev/kvm.
    if !std::path::Path::new("/dev/kvm").exists() {
        eprintln!("skipping shell_exec_echo: /dev/kvm not found");
        return;
    }
    // Skip if no kernel available (don't trigger auto-download in tests).
    if ktstr::find_kernel().ok().flatten().is_none() {
        eprintln!("skipping shell_exec_echo: no cached kernel");
        return;
    }
    ktstr()
        .args(["shell", "--exec", "echo hello-from-guest"])
        .timeout(std::time::Duration::from_secs(120))
        .assert()
        .success()
        .stdout(predicate::str::contains("hello-from-guest"));
}

#[test]
fn include_files_duplicate_archive_path_errors() {
    let tmp1 = tempfile::TempDir::new().unwrap();
    let tmp2 = tempfile::TempDir::new().unwrap();
    // Create files with the same name in both dirs.
    let dir1 = tmp1.path().join("data");
    let dir2 = tmp2.path().join("data");
    std::fs::create_dir(&dir1).unwrap();
    std::fs::create_dir(&dir2).unwrap();
    std::fs::write(dir1.join("file.txt"), "a").unwrap();
    std::fs::write(dir2.join("file.txt"), "b").unwrap();

    let result = ktstr::cli::resolve_include_files(&[dir1, dir2]);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("duplicate"), "{err}");
}

// -- list scenarios --

#[test]
fn list_shows_scenarios() {
    ktstr()
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("cgroup_steady"));
}

#[test]
fn list_json() {
    ktstr()
        .args(["list", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\""));
}

#[test]
fn list_filter() {
    ktstr()
        .args(["list", "--filter", "cpuset"])
        .assert()
        .success()
        .stdout(predicate::str::contains("cpuset"));
}

#[test]
fn list_filter_no_match() {
    ktstr()
        .args(["list", "--filter", "nonexistent_scenario_xyz"])
        .assert()
        .success()
        .stdout(predicate::str::contains("0 scenarios"));
}

// -- topo --

#[test]
fn topo_shows_cpus() {
    ktstr()
        .arg("topo")
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

// -- completions (additional shells) --

#[test]
fn completions_fish() {
    ktstr()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_invalid_shell() {
    ktstr().args(["completions", "noshell"]).assert().failure();
}

// -- run --flags / --work-type drift detection --

/// `ktstr run --help` hardcodes the valid `--flags` list. This test
/// fails the build when that list drifts from `scenario::flags::ALL`,
/// catching new flag additions or renames before they ship.
#[test]
fn run_help_flags_lists_match_flags_all() {
    use ktstr::scenario::flags;
    let assert = ktstr().args(["run", "--help"]).assert().success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    for &name in flags::ALL {
        assert!(
            stdout.contains(name),
            "ktstr run --help missing flag '{name}' that is in scenario::flags::ALL; \
             update the --flags doc-comment in src/bin/ktstr.rs to include it",
        );
    }
}

/// `ktstr run --help` hardcodes the valid `--work-type` list. This
/// test fails the build when that list drifts from
/// `WorkType::ALL_NAMES` (excluding the two non-CLI-constructible
/// variants `Sequence` and `Custom`).
#[test]
fn run_help_work_type_lists_match_all_names() {
    use ktstr::workload::WorkType;
    let assert = ktstr().args(["run", "--help"]).assert().success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    for &name in WorkType::ALL_NAMES {
        if name == "Sequence" || name == "Custom" {
            continue;
        }
        assert!(
            stdout.contains(name),
            "ktstr run --help missing work_type '{name}' that is in WorkType::ALL_NAMES; \
             update the --work-type doc-comment in src/bin/ktstr.rs to include it",
        );
    }
}

// -- kernel list --

#[test]
fn kernel_list_runs() {
    // Isolate from the user's real kernel cache so the assertion is
    // deterministic. With an empty cache directory, `kernel list`
    // prints the cache path header on stderr and a "no cached
    // kernels" hint on stdout.
    let tmp = tempfile::TempDir::new().unwrap();
    ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .args(["kernel", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("no cached kernels"))
        .stderr(predicate::str::contains("cache:"));
}

#[test]
fn kernel_list_json() {
    ktstr()
        .args(["kernel", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("entries"));
}
