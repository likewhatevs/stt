use std::path::{Path, PathBuf};

/// Check if a kernel .config contains CONFIG_SCHED_CLASS_EXT=y.
pub fn has_sched_ext(kernel_dir: &Path) -> bool {
    let config = kernel_dir.join(".config");
    std::fs::read_to_string(config)
        .map(|s| s.lines().any(|l| l == "CONFIG_SCHED_CLASS_EXT=y"))
        .unwrap_or(false)
}

/// Locate ktstr.kconfig by walking up from a starting directory.
///
/// Walks up from `start` looking for ktstr.kconfig. Returns the first
/// match found, or None if the file doesn't exist in any ancestor.
pub fn find_kconfig_from(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start.to_path_buf());
    while let Some(d) = dir {
        let candidate = d.join("ktstr.kconfig");
        if candidate.exists() {
            return Some(candidate);
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    None
}

/// Build the make arguments for a kernel build.
///
/// Returns the argument list that would be passed to `make` for a
/// parallel kernel build: `["-jN", "KCFLAGS=-Wno-error"]`.
pub fn build_make_args(nproc: usize) -> Vec<String> {
    vec![format!("-j{nproc}"), "KCFLAGS=-Wno-error".into()]
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

    // -- find_kconfig_from --

    #[test]
    fn cargo_ktstr_find_kconfig_in_start_dir() {
        let tmp = TempDir::new().unwrap();
        let kconfig = tmp.path().join("ktstr.kconfig");
        std::fs::write(&kconfig, "# kconfig fragment\n").unwrap();
        let result = find_kconfig_from(tmp.path());
        assert_eq!(result, Some(kconfig));
    }

    #[test]
    fn cargo_ktstr_find_kconfig_in_parent() {
        let tmp = TempDir::new().unwrap();
        let child = tmp.path().join("subdir/nested");
        std::fs::create_dir_all(&child).unwrap();
        let kconfig = tmp.path().join("ktstr.kconfig");
        std::fs::write(&kconfig, "# kconfig fragment\n").unwrap();
        let result = find_kconfig_from(&child);
        assert_eq!(result, Some(kconfig));
    }

    #[test]
    fn cargo_ktstr_find_kconfig_not_found() {
        let tmp = TempDir::new().unwrap();
        // No ktstr.kconfig anywhere in this tree. Will walk up to / and
        // only find it if the host root happens to have one. Either way
        // the function must not panic.
        let result = find_kconfig_from(tmp.path());
        if let Some(ref p) = result {
            assert!(p.exists());
        }
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
}
