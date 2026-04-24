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
fn help_shell_shows_no_perf_mode() {
    ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--no-perf-mode"));
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

// -- kernel list legend channel + ordering --
//
// Replaces two source-scanning tests (`eol_legend_emits_via_eprintln`
// and `kernel_list_footer_ordering_pin`) that previously used
// `include_str!("cli.rs")` + a hand-rolled brace-balanced matcher to
// static-analyze the cli.rs source for legend emit sites. The new
// tests exercise the real binary against a fixture cache and assert
// against captured stdout/stderr — the actual behaviour operators
// observe, not the source-form of the code that produces it.
//
// Fixture shape (shared between both tests):
//
//   <tmp_cache>/
//     valid-untracked/           # metadata.json with ktstr_kconfig_hash: null
//       metadata.json            # → KconfigStatus::Untracked → "(untracked kconfig)" tag
//       Image                    # empty sentinel, just needs to exist per cache::list()
//     valid-stale/               # metadata.json with wrong ktstr_kconfig_hash
//       metadata.json            # → KconfigStatus::Stale → "(stale kconfig)" tag
//       Image
//     corrupt-malformed/         # metadata.json unparseable → ListedEntry::Corrupt
//       metadata.json            # "{" — invalid JSON
//
// `cache::list` sorts Valid entries by built_at descending; Corrupt
// entries sort last. Untracked and stale are both Valid, so they
// render in the rows first; corrupt renders last. Legend footers
// then emit in the documented order (untracked → stale → corrupt).
// EOL coverage: not fixtured — the `(EOL)` tag requires a non-empty
// active-prefixes list from kernel.org, which needs network access
// and a version string that survives the active-prefixes filter. The
// old source-pattern test pinned all four via static analysis; these
// integration tests pin the three we can fixture deterministically
// offline, and the in-source block comment + per-helper unit tests
// in cli.rs continue to cover EOL's design.

/// Helper: write a valid-shape metadata.json to `dir` with the given
/// `ktstr_kconfig_hash` (None = untracked, Some(non-matching) = stale).
/// Also creates an empty `Image` file so `cache::list()` classifies
/// the directory as [`ListedEntry::Valid`] rather than
/// image-missing-corrupt. Mirrors the on-disk shape
/// `cache::CacheDir::store` writes, specified by JSON directly so
/// the test stays at the CLI boundary (no dependency on the crate's
/// internal constructors).
fn write_valid_entry(dir: &std::path::Path, ktstr_kconfig_hash: Option<&str>) {
    std::fs::create_dir_all(dir).expect("create fixture entry dir");
    let kconfig = match ktstr_kconfig_hash {
        Some(h) => format!("\"{h}\""),
        None => "null".to_string(),
    };
    let metadata = format!(
        "{{\
         \"version\":\"6.99.0\",\
         \"source\":{{\"type\":\"tarball\"}},\
         \"arch\":\"x86_64\",\
         \"image_name\":\"Image\",\
         \"config_hash\":null,\
         \"built_at\":\"2025-01-01T00:00:00Z\",\
         \"ktstr_kconfig_hash\":{kconfig},\
         \"has_vmlinux\":false\
         }}",
    );
    std::fs::write(dir.join("metadata.json"), metadata.as_bytes()).expect("write metadata.json");
    std::fs::write(dir.join("Image"), b"").expect("write Image");
}

/// Helper: write a deliberately malformed metadata.json so
/// `cache::list()` surfaces the directory as
/// [`ListedEntry::Corrupt`]. The body is an incomplete JSON object;
/// serde_json classifies it as `Category::Syntax`, which the list
/// path wraps into a `"metadata.json malformed: ..."` reason
/// string. The tag rendered in the row is `(corrupt)`.
fn write_corrupt_entry(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).expect("create fixture corrupt dir");
    std::fs::write(dir.join("metadata.json"), b"{").expect("write malformed metadata.json");
}

/// Build the shared fixture cache for the legend tests. Returns the
/// temp dir guard so the caller keeps it alive for the duration of
/// the spawned binary — dropping it earlier would remove the cache
/// while `ktstr` is mid-`list`.
fn build_legend_fixture_cache() -> tempfile::TempDir {
    let tmp = tempfile::TempDir::new().expect("tempdir for fixture cache");
    let root = tmp.path();
    write_valid_entry(&root.join("valid-untracked"), None);
    // A 7-char hex-looking string that cannot collide with the
    // real ktstr.kconfig hash — the CRC32 the cache uses is 8 hex
    // chars; a 7-char literal guarantees mismatch without assuming
    // any specific current hash value.
    write_valid_entry(&root.join("valid-stale"), Some("deadbe7"));
    write_corrupt_entry(&root.join("corrupt-malformed"));
    tmp
}

/// Channel-routing pin: every legend / footer must land on STDERR,
/// never STDOUT. Downstream scripts redirect stdout to machine-
/// parseable data (`kernel list > kernels.txt`) and stderr to an
/// interactive channel; a legend leaked onto stdout would corrupt
/// the row data for those consumers. Previously pinned by a source-
/// pattern test scanning cli.rs for `eprintln!`; now pinned by
/// reading the real binary's streams.
#[test]
fn kernel_list_legends_emit_on_stderr() {
    let cache = build_legend_fixture_cache();
    let out = ktstr()
        .env("KTSTR_CACHE_DIR", cache.path())
        .args(["kernel", "list"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("stdout utf-8");
    let stderr = String::from_utf8(out.stderr).expect("stderr utf-8");

    // Each of the three offline-fixturable legends must appear in
    // stderr. The exact wording comes from the *_EXPLANATION consts
    // / format_corrupt_footer body in cli.rs; pinning a stable
    // substring from each catches a reword at the CLI boundary
    // without over-specifying the full string.
    for needle in [
        "(untracked kconfig) marks entries",
        "warning: entries marked (stale kconfig)",
        "warning: entries marked (corrupt)",
    ] {
        assert!(
            stderr.contains(needle),
            "stderr must contain legend fragment {needle:?}; got:\n{stderr}",
        );
        // Same fragment must NOT leak to stdout — the row data
        // channel stays legend-free for script consumers.
        assert!(
            !stdout.contains(needle),
            "stdout must NOT contain legend fragment {needle:?}; got:\n{stdout}",
        );
    }
}

/// Ordering pin: the four annotation footers emit in a fixed order
/// (EOL → untracked → stale → corrupt) documented in `kernel_list`'s
/// emission block. Previously pinned by source-pattern scan of the
/// function body; now pinned by checking byte offsets of each
/// legend's fingerprint in captured stderr.
///
/// Offline guarantee: the three fixturable legends (untracked,
/// stale, corrupt) are always present in the output from the
/// fixture, so their relative ordering is always pinnable. EOL
/// coverage is conditional — when the kernel.org active-prefixes
/// fetch succeeds AND the fixture version "6.99.0" does not appear
/// in the active list, EOL fires and must precede untracked. When
/// the fetch fails (offline CI, DNS outage) `active_prefixes` is
/// empty and EOL is silently disabled per `fetch_active_prefixes`'s
/// error arm; the test still passes on the three we CAN guarantee.
#[test]
fn kernel_list_legend_ordering_pins_untracked_stale_corrupt() {
    let cache = build_legend_fixture_cache();
    let out = ktstr()
        .env("KTSTR_CACHE_DIR", cache.path())
        .args(["kernel", "list"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("stderr utf-8");

    let i_untracked = stderr
        .find("(untracked kconfig) marks entries")
        .expect("untracked legend must appear in stderr");
    let i_stale = stderr
        .find("warning: entries marked (stale kconfig)")
        .expect("stale legend must appear in stderr");
    let i_corrupt = stderr
        .find("warning: entries marked (corrupt)")
        .expect("corrupt footer must appear in stderr");

    assert!(
        i_untracked < i_stale,
        "untracked legend must precede stale legend in stderr — \
         kconfig-tag rebuild recipes are kept adjacent so operators \
         see both remediation shapes together. \
         untracked at byte {i_untracked}, stale at {i_stale}:\n{stderr}",
    );
    assert!(
        i_stale < i_corrupt,
        "stale legend must precede corrupt footer — informational \
         trio (EOL/untracked/stale) comes before the operationally-\
         disruptive corrupt entry per the emission block comment in \
         cli.rs. stale at byte {i_stale}, corrupt at {i_corrupt}:\n{stderr}",
    );

    // EOL ordering: only enforceable when the fixture's version
    // was actually tagged EOL by the fetched active-prefixes list.
    // When the fetch succeeds and classifies 6.99.0 as EOL, the
    // legend appears in stderr and must precede untracked. When
    // the fetch fails (offline runner) the legend is absent and
    // the guard short-circuits to Ok — which is correct since
    // there's nothing to order.
    if let Some(i_eol) = stderr.find("(EOL) marks entries") {
        assert!(
            i_eol < i_untracked,
            "EOL legend must precede untracked legend — EOL is \
             informational-first (upstream-release state, not a \
             cache pathology) per the emission block comment. \
             eol at byte {i_eol}, untracked at {i_untracked}:\n{stderr}",
        );
    }
}

// -- --cpu-cap vs KTSTR_BYPASS_LLC_LOCKS conflict — ktstr binary sites --
//
// Pins the parse-time rejection when both the --cpu-cap resource
// contract and the KTSTR_BYPASS_LLC_LOCKS=1 escape hatch are
// active simultaneously. All three ktstr-binary sites (shell,
// kernel build, library fallback via shell --no-perf-mode) must
// bail with "resource contract" in the error text so the operator
// sees the contradiction before a pipeline deep-bail.

/// `ktstr shell --no-perf-mode --cpu-cap N` under
/// KTSTR_BYPASS_LLC_LOCKS=1 must fail with "resource contract" in
/// the error. Pins the rejection at bin/ktstr.rs:577.
#[test]
fn ktstr_shell_cpu_cap_with_bypass_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .env("KTSTR_BYPASS_LLC_LOCKS", "1")
        .args(["shell", "--no-perf-mode", "--cpu-cap", "2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource contract"));
}

/// `ktstr kernel build --cpu-cap N` under
/// KTSTR_BYPASS_LLC_LOCKS=1 must fail with "resource contract" in
/// the error. Pins the rejection at bin/ktstr.rs:298.
#[test]
fn ktstr_kernel_build_cpu_cap_with_bypass_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .env("KTSTR_BYPASS_LLC_LOCKS", "1")
        .args([
            "kernel",
            "build",
            "--source",
            "/nonexistent/ktstr-ktstr-cpu-cap-bypass-test",
            "--cpu-cap",
            "2",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource contract"));
}

/// Library-layer fallback via `ktstr shell --no-perf-mode` under
/// `KTSTR_CPU_CAP` + `KTSTR_BYPASS_LLC_LOCKS`. This exercises the
/// KtstrVmBuilder::build site at vmm/mod.rs:3866 — the CLI binary
/// checks KTSTR_BYPASS against the --cpu-cap FLAG, but a consumer
/// setting KTSTR_CPU_CAP env + KTSTR_BYPASS_LLC_LOCKS env (no
/// --cpu-cap flag) flows through the library-layer check instead.
/// Both paths must emit "resource contract" so the operator sees
/// a consistent message regardless of which layer detects the
/// conflict.
#[test]
fn ktstr_library_cpu_cap_env_with_bypass_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    ktstr()
        .env("KTSTR_CACHE_DIR", tmp.path())
        .env("KTSTR_CPU_CAP", "2")
        .env("KTSTR_BYPASS_LLC_LOCKS", "1")
        .args(["shell", "--no-perf-mode"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource contract"));
}
