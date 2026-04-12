use std::io::IsTerminal;
use std::path::Path;

use comfy_table::presets::ASCII_FULL_CONDENSED;
use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table};
use serde::Deserialize;

/// Check if a kernel .config contains CONFIG_SCHED_CLASS_EXT=y.
pub fn has_sched_ext(kernel_dir: &Path) -> bool {
    let config = kernel_dir.join(".config");
    std::fs::read_to_string(config)
        .map(|s| s.lines().any(|l| l == "CONFIG_SCHED_CLASS_EXT=y"))
        .unwrap_or(false)
}

/// Build the make arguments for a kernel build.
///
/// Returns the argument list that would be passed to `make` for a
/// parallel kernel build: `["-jN", "KCFLAGS=-Wno-error"]`.
pub fn build_make_args(nproc: usize) -> Vec<String> {
    vec![format!("-j{nproc}"), "KCFLAGS=-Wno-error".into()]
}

// ---------------------------------------------------------------------------
// JUnit XML deserialization types (nextest output)
// ---------------------------------------------------------------------------

/// Root `<testsuites>` element.
#[derive(Debug, Deserialize)]
pub struct TestSuites {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@tests")]
    pub tests: u32,
    #[serde(rename = "@failures")]
    pub failures: u32,
    #[serde(rename = "@errors")]
    pub errors: u32,
    #[serde(rename = "@time")]
    pub time: f64,
    #[serde(rename = "testsuite", default)]
    pub suites: Vec<TestSuite>,
}

/// `<testsuite>` element (one per test binary).
#[derive(Debug, Deserialize)]
pub struct TestSuite {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@tests")]
    pub tests: u32,
    #[serde(rename = "@disabled")]
    pub disabled: u32,
    #[serde(rename = "@errors")]
    pub errors: u32,
    #[serde(rename = "@failures")]
    pub failures: u32,
    #[serde(rename = "testcase", default)]
    pub cases: Vec<TestCase>,
}

/// `<testcase>` element.
#[derive(Debug, Deserialize)]
pub struct TestCase {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@classname")]
    pub classname: String,
    #[serde(rename = "@time")]
    pub time: f64,
    #[serde(default)]
    pub failure: Option<Failure>,
    #[serde(rename = "flakyFailure", default)]
    pub flaky_failures: Vec<FlakyFailure>,
    #[serde(rename = "rerunFailure", default)]
    pub rerun_failures: Vec<RerunFailure>,
}

/// `<failure>` element on a test case that ultimately failed.
#[derive(Debug, Deserialize)]
pub struct Failure {
    #[serde(rename = "@type", default)]
    pub failure_type: String,
    #[serde(rename = "@message", default)]
    pub message: Option<String>,
}

/// `<flakyFailure>` element: a retry that failed but the test
/// eventually passed on a later attempt.
#[derive(Debug, Deserialize)]
pub struct FlakyFailure {
    #[serde(rename = "@time", default)]
    pub time: f64,
    #[serde(rename = "system-err", default)]
    pub system_err: Option<String>,
}

/// `<rerunFailure>` element: a retry that failed on a test that
/// ultimately never passed.
#[derive(Debug, Deserialize)]
pub struct RerunFailure {
    #[serde(rename = "@time", default)]
    pub time: f64,
    #[serde(rename = "system-err", default)]
    pub system_err: Option<String>,
}

/// Parse nextest JUnit XML from a string.
pub fn parse_junit_xml(xml: &str) -> Result<TestSuites, String> {
    quick_xml::de::from_str(xml).map_err(|e| format!("parse JUnit XML: {e}"))
}

// ---------------------------------------------------------------------------
// Stats aggregation
// ---------------------------------------------------------------------------

/// Aggregated statistics from a parsed JUnit XML report.
#[derive(Debug)]
pub struct TestStats {
    pub total: u32,
    pub passed: u32,
    pub failed: u32,
    pub flaky: u32,
    pub skipped: u32,
    pub errors: u32,
    pub total_retries: u32,
    pub wall_clock_s: f64,
    pub suites: Vec<SuiteStats>,
    pub slowest: Vec<SlowTest>,
    pub failed_tests: Vec<FailedTest>,
    pub flaky_tests: Vec<FlakyTest>,
}

/// Per-suite aggregate.
#[derive(Debug)]
pub struct SuiteStats {
    pub name: String,
    pub tests: u32,
    pub passed: u32,
    pub failed: u32,
    pub flaky: u32,
    pub duration_s: f64,
}

/// A slow test entry.
#[derive(Debug)]
pub struct SlowTest {
    pub name: String,
    pub suite: String,
    pub duration_s: f64,
}

/// A failed test entry.
#[derive(Debug)]
pub struct FailedTest {
    pub name: String,
    pub suite: String,
    pub failure_type: String,
    pub message: Option<String>,
    pub retries: u32,
}

/// A flaky test entry (passed after retries).
#[derive(Debug)]
pub struct FlakyTest {
    pub name: String,
    pub suite: String,
    pub retries: u32,
}

/// Compute aggregate statistics from parsed JUnit XML.
pub fn compute_stats(report: &TestSuites) -> TestStats {
    let mut total = 0u32;
    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut flaky = 0u32;
    let mut total_retries = 0u32;
    let mut suites = Vec::new();
    let mut all_tests: Vec<(&TestCase, &str)> = Vec::new();
    let mut failed_tests = Vec::new();
    let mut flaky_tests = Vec::new();

    for suite in &report.suites {
        let mut s_passed = 0u32;
        let mut s_failed = 0u32;
        let mut s_flaky = 0u32;
        let mut s_duration = 0.0f64;

        for tc in &suite.cases {
            total += 1;
            s_duration += tc.time;
            all_tests.push((tc, &suite.name));

            let retries = tc.flaky_failures.len() as u32 + tc.rerun_failures.len() as u32;
            total_retries += retries;

            if tc.failure.is_some() {
                failed += 1;
                s_failed += 1;
                failed_tests.push(FailedTest {
                    name: tc.name.clone(),
                    suite: suite.name.clone(),
                    failure_type: tc
                        .failure
                        .as_ref()
                        .map(|f| f.failure_type.clone())
                        .unwrap_or_default(),
                    message: tc.failure.as_ref().and_then(|f| f.message.clone()),
                    retries,
                });
            } else if !tc.flaky_failures.is_empty() {
                passed += 1;
                s_passed += 1;
                flaky += 1;
                s_flaky += 1;
                flaky_tests.push(FlakyTest {
                    name: tc.name.clone(),
                    suite: suite.name.clone(),
                    retries: tc.flaky_failures.len() as u32,
                });
            } else {
                passed += 1;
                s_passed += 1;
            }
        }

        suites.push(SuiteStats {
            name: suite.name.clone(),
            tests: suite.cases.len() as u32,
            passed: s_passed,
            failed: s_failed,
            flaky: s_flaky,
            duration_s: s_duration,
        });
    }

    // Top 10 slowest tests.
    all_tests.sort_by(|a, b| {
        b.0.time
            .partial_cmp(&a.0.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let slowest: Vec<SlowTest> = all_tests
        .iter()
        .take(10)
        .map(|(tc, suite)| SlowTest {
            name: tc.name.clone(),
            suite: suite.to_string(),
            duration_s: tc.time,
        })
        .collect();

    TestStats {
        total,
        passed,
        failed,
        flaky,
        skipped: report.suites.iter().map(|s| s.disabled).sum(),
        errors: report.errors,
        total_retries,
        wall_clock_s: report.time,
        suites,
        slowest,
        failed_tests,
        flaky_tests,
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format duration as human-readable string.
fn fmt_duration(seconds: f64) -> String {
    if seconds < 1.0 {
        format!("{:.0}ms", seconds * 1000.0)
    } else if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else {
        let mins = (seconds / 60.0).floor() as u64;
        let secs = seconds - (mins as f64 * 60.0);
        format!("{mins}m{secs:.0}s")
    }
}

/// Whether colored output should be used.
///
/// Returns `true` when stdout is a terminal and `NO_COLOR` is not set.
fn use_color() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

/// Wrap `text` in ANSI SGR codes if `color` is true.
fn ansi(text: &str, code: &str, color: bool) -> String {
    if color {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

/// Format test stats into a pretty report string.
pub fn format_stats(stats: &TestStats) -> String {
    let mut out = String::new();
    let color = use_color();

    // Summary line.
    out.push_str(&format!(
        "\n  {} tests | {} | {} | {} | {} skipped | {} retries | {}\n",
        stats.total,
        ansi(
            &format!("{} passed", stats.passed),
            "32",
            color && stats.passed > 0,
        ),
        ansi(
            &format!("{} failed", stats.failed),
            "31",
            color && stats.failed > 0,
        ),
        ansi(
            &format!("{} flaky", stats.flaky),
            "33",
            color && stats.flaky > 0,
        ),
        stats.skipped,
        stats.total_retries,
        ansi(&fmt_duration(stats.wall_clock_s), "2", color),
    ));

    // Per-suite table.
    if stats.suites.len() > 1 {
        out.push('\n');
        let mut table = Table::new();
        table
            .load_preset(ASCII_FULL_CONDENSED)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Suite", "Tests", "Pass", "Fail", "Flaky", "Test Time"]);
        if color {
            table.enforce_styling();
        } else {
            table.force_no_tty();
        }
        for s in &stats.suites {
            let pass_cell = Cell::new(s.passed.to_string());
            let pass_cell = if color && s.passed > 0 {
                pass_cell.fg(Color::Green)
            } else {
                pass_cell
            };
            let fail_cell = Cell::new(s.failed.to_string());
            let fail_cell = if color && s.failed > 0 {
                fail_cell.fg(Color::Red).add_attribute(Attribute::Bold)
            } else {
                fail_cell
            };
            let flaky_cell = Cell::new(s.flaky.to_string());
            let flaky_cell = if color && s.flaky > 0 {
                flaky_cell.fg(Color::Yellow)
            } else {
                flaky_cell
            };
            let duration_cell = Cell::new(fmt_duration(s.duration_s));
            let duration_cell = if color {
                duration_cell.add_attribute(Attribute::Dim)
            } else {
                duration_cell
            };
            table.add_row(vec![
                Cell::new(&s.name),
                Cell::new(s.tests.to_string()),
                pass_cell,
                fail_cell,
                flaky_cell,
                duration_cell,
            ]);
        }
        out.push_str(&table.to_string());
        out.push('\n');
    }

    // Failed tests.
    if !stats.failed_tests.is_empty() {
        out.push_str(&format!("\n  {}:\n", ansi("FAILED", "31;1", color)));
        for ft in &stats.failed_tests {
            let retry_info = if ft.retries > 0 {
                format!(" ({} retries)", ft.retries)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "    {} [{}]{}\n",
                ansi(&ft.name, "31", color),
                ansi(&ft.suite, "2", color),
                retry_info,
            ));
            if let Some(ref msg) = ft.message {
                out.push_str(&format!("      {}\n", msg));
            }
        }
    }

    // Flaky tests.
    if !stats.flaky_tests.is_empty() {
        out.push_str(&format!(
            "\n  {} (passed after retries):\n",
            ansi("FLAKY", "33;1", color),
        ));
        for ft in &stats.flaky_tests {
            out.push_str(&format!(
                "    {} [{}] ({} retries)\n",
                ansi(&ft.name, "33", color),
                ansi(&ft.suite, "2", color),
                ft.retries,
            ));
        }
    }

    // Slowest tests.
    if !stats.slowest.is_empty() {
        out.push_str("\n  SLOWEST:\n");
        for st in &stats.slowest {
            out.push_str(&format!(
                "    {}  {} [{}]\n",
                ansi(&format!("{:>7}", fmt_duration(st.duration_s)), "2", color),
                st.name,
                ansi(&st.suite, "2", color),
            ));
        }
    }

    out
}

/// Read a nextest JUnit XML file, compute stats, and return a
/// formatted report.
///
/// Resolves the XML path from `junit` (explicit path) or `profile`
/// (looks at `target/nextest/{profile}/junit.xml`).
pub fn run_test_stats(junit: Option<&Path>, profile: &str) -> Result<String, String> {
    let xml_path = match junit {
        Some(p) => p.to_path_buf(),
        None => std::path::PathBuf::from(format!("target/nextest/{profile}/junit.xml")),
    };

    let xml = std::fs::read_to_string(&xml_path).map_err(|e| {
        format!(
            "read {}: {e}\nhint: run tests with --profile {profile} to generate JUnit XML",
            xml_path.display()
        )
    })?;

    let report = parse_junit_xml(&xml)?;
    let stats = compute_stats(&report);
    Ok(format_stats(&stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -- has_sched_ext --

    #[test]
    fn cargo_ktstr_has_sched_ext_present() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_SCHED_CLASS_EXT=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(has_sched_ext(tmp.path()));
    }

    #[test]
    fn cargo_ktstr_has_sched_ext_absent() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cargo_ktstr_has_sched_ext_module_not_builtin() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "CONFIG_SCHED_CLASS_EXT=m\n").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cargo_ktstr_has_sched_ext_commented_out() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "# CONFIG_SCHED_CLASS_EXT is not set\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cargo_ktstr_has_sched_ext_no_config_file() {
        let tmp = TempDir::new().unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cargo_ktstr_has_sched_ext_empty_config() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    // -- build_make_args --

    #[test]
    fn cargo_ktstr_build_make_args_single_core() {
        let args = build_make_args(1);
        assert_eq!(args, vec!["-j1", "KCFLAGS=-Wno-error"]);
    }

    #[test]
    fn cargo_ktstr_build_make_args_multi_core() {
        let args = build_make_args(16);
        assert_eq!(args, vec!["-j16", "KCFLAGS=-Wno-error"]);
    }

    // -- JUnit XML parsing --

    #[test]
    fn parse_junit_minimal() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="2" failures="0" errors="0" time="1.5">
    <testsuite name="my-crate" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="test_a" classname="my-crate" time="0.5">
        </testcase>
        <testcase name="test_b" classname="my-crate" time="1.0">
        </testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        assert_eq!(report.tests, 2);
        assert_eq!(report.failures, 0);
        assert_eq!(report.suites.len(), 1);
        assert_eq!(report.suites[0].cases.len(), 2);
    }

    #[test]
    fn parse_junit_with_failure() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="1" failures="1" errors="0" time="5.0">
    <testsuite name="my-crate" tests="1" disabled="0" errors="0" failures="1">
        <testcase name="test_fail" classname="my-crate" time="5.0">
            <failure type="test failure with exit code 1"/>
        </testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        assert_eq!(report.failures, 1);
        let tc = &report.suites[0].cases[0];
        assert!(tc.failure.is_some());
        assert_eq!(
            tc.failure.as_ref().unwrap().failure_type,
            "test failure with exit code 1"
        );
    }

    #[test]
    fn parse_junit_with_flaky_failure() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="1" failures="0" errors="0" time="10.0">
    <testsuite name="my-crate" tests="1" disabled="0" errors="0" failures="0">
        <testcase name="test_flaky" classname="my-crate" time="5.0">
            <flakyFailure timestamp="2026-01-01T00:00:00Z" time="2.0" type="test failure">
                <system-err>resource contention</system-err>
            </flakyFailure>
            <flakyFailure timestamp="2026-01-01T00:00:03Z" time="3.0" type="test failure">
                <system-err>resource contention again</system-err>
            </flakyFailure>
        </testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let tc = &report.suites[0].cases[0];
        assert!(tc.failure.is_none());
        assert_eq!(tc.flaky_failures.len(), 2);
    }

    #[test]
    fn parse_junit_with_rerun_failure() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="1" failures="1" errors="0" time="10.0">
    <testsuite name="my-crate" tests="1" disabled="0" errors="0" failures="1">
        <testcase name="test_rerun" classname="my-crate" time="5.0">
            <failure type="test failure with exit code 1"/>
            <rerunFailure timestamp="2026-01-01T00:00:00Z" time="5.0" type="test failure">
                <system-err>still failing</system-err>
            </rerunFailure>
        </testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let tc = &report.suites[0].cases[0];
        assert!(tc.failure.is_some());
        assert_eq!(tc.rerun_failures.len(), 1);
    }

    #[test]
    fn parse_junit_empty_suites() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="0" failures="0" errors="0" time="0.0">
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        assert_eq!(report.tests, 0);
        assert!(report.suites.is_empty());
    }

    #[test]
    fn parse_junit_invalid_xml() {
        let xml = "not xml at all";
        assert!(parse_junit_xml(xml).is_err());
    }

    // -- Stats computation --

    #[test]
    fn compute_stats_all_pass() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="3" failures="0" errors="0" time="2.0">
    <testsuite name="crate-a" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="a1" classname="crate-a" time="0.5"></testcase>
        <testcase name="a2" classname="crate-a" time="0.3"></testcase>
    </testsuite>
    <testsuite name="crate-b" tests="1" disabled="0" errors="0" failures="0">
        <testcase name="b1" classname="crate-b" time="1.2"></testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let stats = compute_stats(&report);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.passed, 3);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.flaky, 0);
        assert_eq!(stats.total_retries, 0);
        assert!(stats.failed_tests.is_empty());
        assert!(stats.flaky_tests.is_empty());
        assert_eq!(stats.suites.len(), 2);
        assert_eq!(stats.suites[0].passed, 2);
        assert_eq!(stats.suites[1].passed, 1);
    }

    #[test]
    fn compute_stats_mixed() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="3" failures="1" errors="0" time="10.0">
    <testsuite name="crate-a" tests="3" disabled="0" errors="0" failures="1">
        <testcase name="pass_test" classname="crate-a" time="1.0"></testcase>
        <testcase name="fail_test" classname="crate-a" time="3.0">
            <failure type="exit code 1"/>
            <rerunFailure timestamp="2026-01-01T00:00:00Z" time="3.0" type="retry">
                <system-err>err</system-err>
            </rerunFailure>
        </testcase>
        <testcase name="flaky_test" classname="crate-a" time="2.0">
            <flakyFailure timestamp="2026-01-01T00:00:00Z" time="1.0" type="retry">
                <system-err>transient</system-err>
            </flakyFailure>
        </testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let stats = compute_stats(&report);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.passed, 2);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.flaky, 1);
        assert_eq!(stats.total_retries, 2);
        assert_eq!(stats.failed_tests.len(), 1);
        assert_eq!(stats.failed_tests[0].name, "fail_test");
        assert_eq!(stats.failed_tests[0].retries, 1);
        assert_eq!(stats.flaky_tests.len(), 1);
        assert_eq!(stats.flaky_tests[0].name, "flaky_test");
        assert_eq!(stats.flaky_tests[0].retries, 1);
    }

    #[test]
    fn compute_stats_slowest_ordering() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="3" failures="0" errors="0" time="6.0">
    <testsuite name="s" tests="3" disabled="0" errors="0" failures="0">
        <testcase name="fast" classname="s" time="1.0"></testcase>
        <testcase name="slow" classname="s" time="3.0"></testcase>
        <testcase name="medium" classname="s" time="2.0"></testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let stats = compute_stats(&report);
        assert_eq!(stats.slowest[0].name, "slow");
        assert_eq!(stats.slowest[1].name, "medium");
        assert_eq!(stats.slowest[2].name, "fast");
    }

    #[test]
    fn compute_stats_suite_duration() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="2" failures="0" errors="0" time="3.0">
    <testsuite name="s" tests="2" disabled="0" errors="0" failures="0">
        <testcase name="a" classname="s" time="1.5"></testcase>
        <testcase name="b" classname="s" time="2.5"></testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let stats = compute_stats(&report);
        assert!((stats.suites[0].duration_s - 4.0).abs() < 0.01);
    }

    #[test]
    fn compute_stats_flaky_subset_of_passed() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="1" failures="0" errors="0" time="1.0">
    <testsuite name="s" tests="1" disabled="0" errors="0" failures="0">
        <testcase name="t" classname="s" time="1.0">
            <flakyFailure timestamp="2026-01-01T00:00:00Z" time="0.5" type="retry">
                <system-err>err</system-err>
            </flakyFailure>
        </testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let stats = compute_stats(&report);
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.flaky, 1);
        assert_eq!(stats.suites[0].passed, 1);
        assert_eq!(stats.suites[0].flaky, 1);
    }

    // -- Formatting --

    #[test]
    fn fmt_duration_millis() {
        assert_eq!(fmt_duration(0.005), "5ms");
        assert_eq!(fmt_duration(0.123), "123ms");
    }

    #[test]
    fn fmt_duration_seconds() {
        assert_eq!(fmt_duration(5.5), "5.5s");
        assert_eq!(fmt_duration(59.0), "59.0s");
    }

    #[test]
    fn fmt_duration_minutes() {
        assert_eq!(fmt_duration(65.0), "1m5s");
        assert_eq!(fmt_duration(335.9), "5m36s");
    }

    #[test]
    fn format_stats_summary_line() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="2" failures="1" errors="0" time="10.0">
    <testsuite name="s" tests="2" disabled="0" errors="0" failures="1">
        <testcase name="ok" classname="s" time="1.0"></testcase>
        <testcase name="bad" classname="s" time="9.0">
            <failure type="exit code 1"/>
        </testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let stats = compute_stats(&report);
        let out = format_stats(&stats);
        assert!(out.contains("2 tests"), "total, got:\n{out}");
        assert!(out.contains("1 passed"), "passed, got:\n{out}");
        assert!(out.contains("1 failed"), "failed, got:\n{out}");
        assert!(out.contains("FAILED:"), "failed section, got:\n{out}");
        assert!(out.contains("bad [s]"), "failed test name, got:\n{out}");
    }

    #[test]
    fn format_stats_no_failed_section_when_all_pass() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="nextest-run" tests="1" failures="0" errors="0" time="1.0">
    <testsuite name="s" tests="1" disabled="0" errors="0" failures="0">
        <testcase name="ok" classname="s" time="1.0"></testcase>
    </testsuite>
</testsuites>"#;
        let report = parse_junit_xml(xml).unwrap();
        let stats = compute_stats(&report);
        let out = format_stats(&stats);
        assert!(!out.contains("FAILED:"), "no failed section, got:\n{out}");
        assert!(!out.contains("FLAKY"), "no flaky section, got:\n{out}");
    }
}
