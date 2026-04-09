//! Cgroup v2 filesystem operations for test cgroup management.
//!
//! Creates, configures, and removes cgroups under a parent path
//! (default `/sys/fs/cgroup/stt`). Provides cpuset assignment,
//! task migration, and cleanup.

use crate::topology::TestTopology;
use anyhow::{Context, Result, bail};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

/// Default timeout for cgroup filesystem writes. Normally <1ms; 2s catches
/// real hangs without waiting so long the test result is meaningless.
const CGROUP_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Write `data` to `path` with a timeout. Spawns a thread for the blocking
/// `fs::write` and waits on a channel. If the write does not complete within
/// `timeout`, returns an error (the spawned thread may still be blocked in
/// the kernel but will not prevent the caller from making progress).
fn write_with_timeout(path: &Path, data: &str, timeout: Duration) -> Result<()> {
    let display = path.display().to_string();
    let path = path.to_owned();
    let data = data.to_owned();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = fs::write(&path, &data);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(anyhow::anyhow!("{e}")).with_context(|| format!("write {display}")),
        Err(_) => bail!(
            "cgroup write to {display} timed out after {}ms",
            timeout.as_millis()
        ),
    }
}

/// RAII manager for cgroup v2 filesystem operations.
///
/// Creates, configures, and removes cgroups under a parent directory.
/// Provides cpuset assignment and task migration.
#[derive(Debug)]
pub struct CgroupManager {
    parent: PathBuf,
}

impl CgroupManager {
    /// Create a manager rooted at the given cgroup v2 path.
    pub fn new(parent: &str) -> Self {
        Self {
            parent: PathBuf::from(parent),
        }
    }
    /// Path to the parent cgroup directory.
    pub fn parent_path(&self) -> &std::path::Path {
        &self.parent
    }

    /// Create the parent directory and enable cgroup controllers (cpuset, optionally cpu).
    pub fn setup(&self, enable_cpu_controller: bool) -> Result<()> {
        if !self.parent.exists() {
            fs::create_dir_all(&self.parent)
                .with_context(|| format!("mkdir {}", self.parent.display()))?;
        }
        let controllers = if enable_cpu_controller {
            "+cpuset +cpu"
        } else {
            "+cpuset"
        };
        let root = PathBuf::from("/sys/fs/cgroup");
        if let Ok(rel) = self.parent.strip_prefix(&root) {
            let mut cur = root.clone();
            for c in rel.components() {
                let sc = cur.join("cgroup.subtree_control");
                if sc.exists()
                    && let Err(e) = write_with_timeout(&sc, controllers, CGROUP_WRITE_TIMEOUT)
                {
                    tracing::warn!(path = %sc.display(), err = %e, "failed to enable controllers");
                }
                cur = cur.join(c);
            }
            let sc = self.parent.join("cgroup.subtree_control");
            if sc.exists()
                && let Err(e) = write_with_timeout(&sc, controllers, CGROUP_WRITE_TIMEOUT)
            {
                tracing::warn!(path = %sc.display(), err = %e, "failed to enable controllers at parent");
            }
        }
        Ok(())
    }

    /// Create a child cgroup directory.
    pub fn create_cgroup(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name);
        if !p.exists() {
            fs::create_dir_all(&p).with_context(|| format!("mkdir {}", p.display()))?;
        }
        Ok(())
    }

    /// Drain tasks from a child cgroup and remove it.
    pub fn remove_cgroup(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name);
        if !p.exists() {
            return Ok(());
        }
        self.drain_tasks(name)?;
        std::thread::sleep(std::time::Duration::from_millis(50));
        fs::remove_dir(&p).with_context(|| format!("rmdir {}", p.display()))
    }

    /// Write `cpuset.cpus` for a child cgroup.
    pub fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
        let p = self.parent.join(name).join("cpuset.cpus");
        write_with_timeout(&p, &TestTopology::cpuset_string(cpus), CGROUP_WRITE_TIMEOUT)
    }

    /// Clear `cpuset.cpus` for a child cgroup (empty string = inherit parent).
    pub fn clear_cpuset(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name).join("cpuset.cpus");
        write_with_timeout(&p, "", CGROUP_WRITE_TIMEOUT)
    }

    /// Move a single task into a child cgroup via `cgroup.procs`.
    pub fn move_task(&self, name: &str, tid: u32) -> Result<()> {
        let p = self.parent.join(name).join("cgroup.procs");
        write_with_timeout(&p, &tid.to_string(), CGROUP_WRITE_TIMEOUT)
    }

    /// Move all tasks from a workload handle into a cgroup.
    pub fn move_tasks(&self, name: &str, tids: &[u32]) -> Result<()> {
        for &t in tids {
            self.move_task(name, t)?;
        }
        Ok(())
    }

    /// Move all tasks from a child cgroup back to the parent.
    pub fn drain_tasks(&self, name: &str) -> Result<()> {
        let src = self.parent.join(name).join("cgroup.procs");
        let dst = self.parent.join("cgroup.procs");
        if !src.exists() {
            return Ok(());
        }
        if let Ok(content) = fs::read_to_string(&src) {
            for line in content.lines() {
                if let Ok(pid) = line.trim().parse::<u32>() {
                    let _ = write_with_timeout(&dst, &pid.to_string(), CGROUP_WRITE_TIMEOUT);
                }
            }
        }
        Ok(())
    }

    /// Remove all child cgroups under the parent (keeps the parent itself).
    pub fn cleanup_all(&self) -> Result<()> {
        if !self.parent.exists() {
            return Ok(());
        }
        // Remove all child cgroups but keep the parent
        if let Ok(entries) = fs::read_dir(&self.parent) {
            for e in entries.flatten() {
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    cleanup_recursive(&e.path());
                }
            }
        }
        Ok(())
    }
}

fn cleanup_recursive(path: &std::path::Path) {
    // Depth-first: clean children before parent
    if let Ok(entries) = fs::read_dir(path) {
        for e in entries.flatten() {
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                cleanup_recursive(&e.path());
            }
        }
    }
    // Drain tasks to parent
    let procs = path.join("cgroup.procs");
    if let (Some(parent), Ok(content)) = (path.parent(), fs::read_to_string(&procs)) {
        let dst = parent.join("cgroup.procs");
        for l in content.lines() {
            if let Ok(pid) = l.trim().parse::<u32>() {
                let _ = write_with_timeout(&dst, &pid.to_string(), CGROUP_WRITE_TIMEOUT);
            }
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(10));
    let _ = fs::remove_dir(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgroup_manager_path() {
        let cg = CgroupManager::new("/sys/fs/cgroup/test");
        assert_eq!(
            cg.parent_path(),
            std::path::Path::new("/sys/fs/cgroup/test")
        );
    }

    #[test]
    fn create_cgroup_in_tmpdir() {
        let dir = std::env::temp_dir().join(format!("stt-cg-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("test_cg").unwrap();
        assert!(dir.join("test_cg").exists());
        cg.create_cgroup("nested/deep").unwrap();
        assert!(dir.join("nested/deep").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_cgroup_idempotent() {
        let dir = std::env::temp_dir().join(format!("stt-cg-idem-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("cg_0").unwrap();
        cg.create_cgroup("cg_0").unwrap(); // should not error
        assert!(dir.join("cg_0").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_all_on_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/stt-test-path");
        assert!(cg.cleanup_all().is_ok());
    }

    #[test]
    fn remove_cgroup_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/stt-test-path");
        assert!(cg.remove_cgroup("no_such_cgroup").is_ok());
    }

    #[test]
    fn cleanup_removes_child_dirs() {
        let dir = std::env::temp_dir().join(format!("stt-cg-clean-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("a").unwrap();
        cg.create_cgroup("b").unwrap();
        cg.create_cgroup("c/deep").unwrap();
        assert!(dir.join("a").exists());
        assert!(dir.join("c/deep").exists());
        // cleanup_all removes child dirs (not real cgroups, so drain_tasks is a no-op)
        cg.cleanup_all().unwrap();
        assert!(!dir.join("a").exists());
        assert!(!dir.join("b").exists());
        assert!(!dir.join("c").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn drain_tasks_nonexistent_source() {
        let cg = CgroupManager::new("/nonexistent/stt-drain-test");
        assert!(cg.drain_tasks("missing_cgroup").is_ok());
    }

    #[test]
    fn setup_non_cgroup_path() {
        // setup() on a non-cgroup path should still create the dir
        let dir = std::env::temp_dir().join(format!("stt-setup-{}", std::process::id()));
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.setup(true).unwrap();
        assert!(dir.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_timeout_success() {
        let dir = std::env::temp_dir().join(format!("stt-wt-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("test_write");
        write_with_timeout(&f, "hello", Duration::from_secs(5)).unwrap();
        assert_eq!(fs::read_to_string(&f).unwrap(), "hello");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_timeout_bad_path() {
        let f = Path::new("/nonexistent/dir/file");
        assert!(write_with_timeout(f, "data", Duration::from_secs(5)).is_err());
    }

    #[test]
    fn move_task_nonexistent_cgroup() {
        let cg = CgroupManager::new("/nonexistent/stt-move-test");
        assert!(cg.move_task("no_cgroup", 1).is_err());
    }

    #[test]
    fn set_cpuset_empty() {
        let dir = std::env::temp_dir().join(format!("stt-cg-cpuset-{}", std::process::id()));
        let dir_a = dir.join("cg_a");
        fs::create_dir_all(&dir_a).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // Empty BTreeSet → writes empty string via cpuset_string
        cg.set_cpuset("cg_a", &BTreeSet::new()).unwrap();
        assert_eq!(fs::read_to_string(dir_a.join("cpuset.cpus")).unwrap(), "");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn move_tasks_partial_failure() {
        // move_tasks iterates and returns early on first error
        let cg = CgroupManager::new("/nonexistent/stt-partial");
        let err = cg.move_tasks("cg", &[1, 2, 3]).unwrap_err();
        // The error comes from the first tid (write to nonexistent path)
        let msg = format!("{err:#}");
        assert!(msg.contains("cgroup.procs"), "unexpected error: {msg}");
    }

    #[test]
    fn drain_tasks_empty_cgroup() {
        let dir = std::env::temp_dir().join(format!("stt-cg-drain-{}", std::process::id()));
        let dir_d = dir.join("cg_d");
        fs::create_dir_all(&dir_d).unwrap();
        // Create an empty cgroup.procs file
        fs::write(dir_d.join("cgroup.procs"), "").unwrap();
        // Parent also needs cgroup.procs
        fs::write(dir.join("cgroup.procs"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // drain_tasks on a cgroup with empty procs file should succeed
        assert!(cg.drain_tasks("cg_d").is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_timeout_blocks_on_fifo() {
        use std::ffi::CString;
        let dir = std::env::temp_dir().join(format!("stt-cg-fifo-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let fifo_path = dir.join("blocked_write");
        let c_path = CString::new(fifo_path.to_str().unwrap()).unwrap();
        let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o700) };
        assert_eq!(rc, 0, "mkfifo failed: {}", std::io::Error::last_os_error());
        // Very short timeout — write blocks until a reader opens the FIFO
        let err = write_with_timeout(&fifo_path, "data", Duration::from_millis(50)).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("timed out"), "unexpected error: {msg}");
        let _ = fs::remove_dir_all(&dir);
    }
}
