//! CLI support functions for `ktstr`.
//!
//! Pure validation and configuration logic extracted from the binary
//! so tests are nextest-discoverable.

use std::time::Duration;

use anyhow::{Result, bail};

use crate::runner::RunConfig;
use crate::scenario::{Scenario, flags};
use crate::workload::WorkType;

/// Resolve flag names, erroring on unknown flags.
pub fn resolve_flags(flag_arg: Option<Vec<String>>) -> Result<Option<Vec<&'static str>>> {
    match flag_arg {
        Some(fs) => {
            let mut resolved = Vec::new();
            for f in &fs {
                match flags::from_short_name(f) {
                    Some(name) => resolved.push(name),
                    None => bail!(
                        "unknown flag: '{f}'. valid flags: {}",
                        flags::ALL.join(", "),
                    ),
                }
            }
            Ok(Some(resolved))
        }
        None => Ok(None),
    }
}

/// Parse and validate a work type name.
pub fn parse_work_type(name: Option<&str>) -> Result<Option<WorkType>> {
    match name {
        Some(name) => match WorkType::from_name(name) {
            Some(wt) => Ok(Some(wt)),
            None => bail!(
                "unknown work type: '{name}'. valid types: {}",
                WorkType::ALL_NAMES.join(", "),
            ),
        },
        None => Ok(None),
    }
}

/// Filter scenarios by name substring.
pub fn filter_scenarios<'a>(
    scenarios: &'a [Scenario],
    filter: Option<&str>,
) -> Result<Vec<&'a Scenario>> {
    let refs: Vec<&Scenario> = scenarios
        .iter()
        .filter(|s| filter.is_none_or(|f| s.name.contains(f)))
        .collect();
    if refs.is_empty() {
        bail!("no scenarios matched filter. run 'ktstr list' to see available scenarios");
    }
    Ok(refs)
}

/// Build a RunConfig from parsed CLI arguments.
#[allow(clippy::too_many_arguments)]
pub fn build_run_config(
    parent_cgroup: String,
    duration: u64,
    workers: usize,
    active_flags: Option<Vec<&'static str>>,
    repro: bool,
    probe_stack: Option<String>,
    auto_repro: bool,
    kernel_dir: Option<String>,
    work_type_override: Option<WorkType>,
) -> RunConfig {
    RunConfig {
        parent_cgroup,
        duration: Duration::from_secs(duration),
        workers_per_cgroup: workers,
        active_flags,
        repro,
        probe_stack,
        auto_repro,
        kernel_dir,
        work_type_override,
        ..Default::default()
    }
}

/// Check if a kernel .config contains CONFIG_SCHED_CLASS_EXT=y.
pub fn has_sched_ext(kernel_dir: &std::path::Path) -> bool {
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

/// Read sidecar JSON files and return the gauntlet analysis report.
///
/// When `dir` is `Some`, reads from that directory. Otherwise uses
/// the default sidecar directory (KTSTR_SIDECAR_DIR or
/// `target/ktstr/{branch}-{hash}/`).
///
/// Returns an empty report with a warning on stderr when no sidecars
/// are found. This is not an error -- regular test runs that skip
/// gauntlet tests produce no sidecar files.
pub fn run_test_stats(dir: Option<&std::path::Path>) -> String {
    let report = crate::test_support::analyze_sidecars(dir);
    if report.is_empty() {
        eprintln!("cargo-ktstr: no sidecar data found (skipped)");
        return String::new();
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario;

    // -- resolve_flags --

    #[test]
    fn cli_resolve_flags_none_returns_none() {
        assert!(resolve_flags(None).unwrap().is_none());
    }

    #[test]
    fn cli_resolve_flags_valid_single() {
        let result = resolve_flags(Some(vec!["llc".into()])).unwrap().unwrap();
        assert_eq!(result, vec!["llc"]);
    }

    #[test]
    fn cli_resolve_flags_valid_multiple() {
        let result = resolve_flags(Some(vec!["llc".into(), "borrow".into()]))
            .unwrap()
            .unwrap();
        assert_eq!(result, vec!["llc", "borrow"]);
    }

    #[test]
    fn cli_resolve_flags_all_valid() {
        let all: Vec<String> = flags::ALL.iter().map(|s| s.to_string()).collect();
        let result = resolve_flags(Some(all)).unwrap().unwrap();
        assert_eq!(result.len(), flags::ALL.len());
    }

    #[test]
    fn cli_resolve_flags_unknown_errors() {
        let err = resolve_flags(Some(vec!["nonexistent".into()])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown flag: 'nonexistent'"), "{msg}");
        assert!(msg.contains("valid flags:"), "{msg}");
    }

    #[test]
    fn cli_resolve_flags_mixed_valid_and_unknown_errors() {
        let err = resolve_flags(Some(vec!["llc".into(), "bogus".into()])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown flag: 'bogus'"), "{msg}");
    }

    // -- parse_work_type --

    #[test]
    fn cli_parse_work_type_none_returns_none() {
        assert!(parse_work_type(None).unwrap().is_none());
    }

    #[test]
    fn cli_parse_work_type_cpu_spin() {
        let wt = parse_work_type(Some("CpuSpin")).unwrap().unwrap();
        assert_eq!(wt.name(), "CpuSpin");
    }

    #[test]
    fn cli_parse_work_type_yield_heavy() {
        let wt = parse_work_type(Some("YieldHeavy")).unwrap().unwrap();
        assert_eq!(wt.name(), "YieldHeavy");
    }

    #[test]
    fn cli_parse_work_type_all_valid() {
        for &name in WorkType::ALL_NAMES {
            if name == "Sequence" {
                continue;
            }
            let wt = parse_work_type(Some(name)).unwrap().unwrap();
            assert_eq!(wt.name(), name);
        }
    }

    #[test]
    fn cli_parse_work_type_unknown_errors() {
        let err = parse_work_type(Some("Nonexistent")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown work type: 'Nonexistent'"), "{msg}");
        assert!(msg.contains("valid types:"), "{msg}");
    }

    #[test]
    fn cli_parse_work_type_sequence_errors() {
        let err = parse_work_type(Some("Sequence")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown work type: 'Sequence'"), "{msg}");
    }

    #[test]
    fn cli_parse_work_type_case_sensitive() {
        let err = parse_work_type(Some("cpuspin")).unwrap_err();
        assert!(format!("{err}").contains("unknown work type:"));
    }

    // -- filter_scenarios --

    #[test]
    fn cli_filter_scenarios_no_filter_returns_all() {
        let scenarios = scenario::all_scenarios();
        let result = filter_scenarios(&scenarios, None).unwrap();
        assert_eq!(result.len(), scenarios.len());
    }

    #[test]
    fn cli_filter_scenarios_matching_filter() {
        let scenarios = scenario::all_scenarios();
        let first_name = scenarios[0].name;
        let result = filter_scenarios(&scenarios, Some(first_name)).unwrap();
        assert!(!result.is_empty());
        for s in &result {
            assert!(s.name.contains(first_name));
        }
    }

    #[test]
    fn cli_filter_scenarios_no_match_errors() {
        let scenarios = scenario::all_scenarios();
        let err = filter_scenarios(&scenarios, Some("__nonexistent_scenario_xyz__")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no scenarios matched"), "{msg}");
        assert!(msg.contains("ktstr list"), "{msg}");
    }

    #[test]
    fn cli_filter_scenarios_partial_match() {
        let scenarios = scenario::all_scenarios();
        let result = filter_scenarios(&scenarios, Some("steady")).unwrap();
        assert!(!result.is_empty());
    }

    // -- build_run_config --

    #[test]
    fn cli_build_run_config_defaults() {
        let config = build_run_config(
            "/sys/fs/cgroup/ktstr".into(),
            20,
            4,
            None,
            false,
            None,
            false,
            None,
            None,
        );
        assert_eq!(config.parent_cgroup, "/sys/fs/cgroup/ktstr");
        assert_eq!(config.duration, Duration::from_secs(20));
        assert_eq!(config.workers_per_cgroup, 4);
        assert!(config.active_flags.is_none());
        assert!(!config.repro);
        assert!(config.probe_stack.is_none());
        assert!(!config.auto_repro);
        assert!(config.kernel_dir.is_none());
        assert!(config.work_type_override.is_none());
    }

    #[test]
    fn cli_build_run_config_all_fields() {
        let config = build_run_config(
            "/sys/fs/cgroup/test".into(),
            30,
            8,
            Some(vec!["llc", "borrow"]),
            true,
            Some("do_enqueue_task".into()),
            true,
            Some("/usr/src/linux".into()),
            Some(WorkType::Mixed),
        );
        assert_eq!(config.parent_cgroup, "/sys/fs/cgroup/test");
        assert_eq!(config.duration, Duration::from_secs(30));
        assert_eq!(config.workers_per_cgroup, 8);
        let af = config.active_flags.unwrap();
        assert_eq!(af, vec!["llc", "borrow"]);
        assert!(config.repro);
        assert_eq!(config.probe_stack.as_deref(), Some("do_enqueue_task"));
        assert!(config.auto_repro);
        assert_eq!(config.kernel_dir.as_deref(), Some("/usr/src/linux"));
        assert!(config.work_type_override.is_some());
    }

    #[test]
    fn cli_build_run_config_duration_converts() {
        let config = build_run_config("cg".into(), 60, 1, None, false, None, false, None, None);
        assert_eq!(config.duration, Duration::from_secs(60));
    }

    // -- scenario catalog --

    #[test]
    fn cli_all_scenarios_non_empty() {
        let scenarios = scenario::all_scenarios();
        assert!(!scenarios.is_empty());
    }

    #[test]
    fn cli_all_scenarios_have_names() {
        for s in &scenario::all_scenarios() {
            assert!(!s.name.is_empty());
            assert!(!s.category.is_empty());
        }
    }

    // -- has_sched_ext --

    #[test]
    fn cli_has_sched_ext_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_SCHED_CLASS_EXT=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_absent() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_module_not_builtin() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "CONFIG_SCHED_CLASS_EXT=m\n").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_commented_out() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "# CONFIG_SCHED_CLASS_EXT is not set\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_no_config_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_empty_config() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    // -- build_make_args --

    #[test]
    fn cli_build_make_args_single_core() {
        let args = build_make_args(1);
        assert_eq!(args, vec!["-j1", "KCFLAGS=-Wno-error"]);
    }

    #[test]
    fn cli_build_make_args_multi_core() {
        let args = build_make_args(16);
        assert_eq!(args, vec!["-j16", "KCFLAGS=-Wno-error"]);
    }

    // -- run_test_stats --

    #[test]
    fn cli_test_stats_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = run_test_stats(Some(tmp.path()));
        assert!(result.is_empty());
    }

    #[test]
    fn cli_test_stats_nonexistent_dir() {
        let result = run_test_stats(Some(std::path::Path::new("/nonexistent/path")));
        assert!(result.is_empty());
    }
}
