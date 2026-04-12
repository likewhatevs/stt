use std::path::Path;

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

/// Read sidecar JSON files and return the gauntlet analysis report.
///
/// When `dir` is `Some`, reads from that directory. Otherwise uses
/// the default sidecar directory (KTSTR_SIDECAR_DIR or
/// `target/ktstr/{branch}-{hash}/`).
///
/// Returns an empty report with a warning on stderr when no sidecars
/// are found. This is not an error -- regular test runs that skip
/// gauntlet tests produce no sidecar files.
pub fn run_test_stats(dir: Option<&Path>) -> String {
    let report = ktstr::test_support::analyze_sidecars(dir);
    if report.is_empty() {
        eprintln!("cargo-ktstr: no sidecar data found (skipped)");
        return String::new();
    }
    report
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

    // -- run_test_stats --

    #[test]
    fn cargo_ktstr_test_stats_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let result = run_test_stats(Some(tmp.path()));
        assert!(result.is_empty());
    }

    #[test]
    fn cargo_ktstr_test_stats_nonexistent_dir() {
        let result = run_test_stats(Some(std::path::Path::new("/nonexistent/path")));
        assert!(result.is_empty());
    }
}
