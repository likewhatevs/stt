//! Cgroup v2 filesystem operations for test cgroup management.
//!
//! Creates, configures, and removes cgroups under a parent path
//! (default `/sys/fs/cgroup/ktstr`). Provides cpuset assignment,
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
        Ok(Err(e)) => {
            let errno_suffix = e
                .raw_os_error()
                .and_then(crate::errno_name)
                .map(|name| format!(" ({name})"))
                .unwrap_or_default();
            Err(e).with_context(|| format!("write {display}{errno_suffix}"))
        }
        Err(_) => bail!(
            "cgroup write to {display} timed out after {}ms",
            timeout.as_millis()
        ),
    }
}

/// Check whether an anyhow error chain contains a specific OS error.
fn is_os_error(err: &anyhow::Error, errno: i32) -> bool {
    for cause in err.chain() {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>()
            && io_err.raw_os_error() == Some(errno)
        {
            return true;
        }
    }
    false
}

/// ESRCH: task exited between listing and migration
/// (`cgroup_procs_write_start` -> `find_task_by_vpid` returns NULL).
fn is_esrch(err: &anyhow::Error) -> bool {
    is_os_error(err, libc::ESRCH)
}

/// EBUSY: either the cgroup v2 no-internal-process constraint
/// (`cgroup_migrate_vet_dst` when `subtree_control` is set) or a
/// transient rejection from a sched_ext BPF `cgroup_prep_move`
/// callback (`scx_cgroup_can_attach`).
fn is_ebusy(err: &anyhow::Error) -> bool {
    is_os_error(err, libc::EBUSY)
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
    ///
    /// For nested paths (e.g. `"cg_0/narrow"`), enables controllers on
    /// each intermediate cgroup's `subtree_control` so the leaf has
    /// controller files available. The kernel requires each parent to
    /// have the controller in `subtree_control` for its children to
    /// have the corresponding files (`cgroup_control()` returns
    /// `parent->subtree_control`).
    pub fn create_cgroup(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name);
        if !p.exists() {
            fs::create_dir_all(&p).with_context(|| format!("mkdir {}", p.display()))?;
        }
        self.enable_subtree_cpuset(name);
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

    /// Enable `+cpuset` on `cgroup.subtree_control` for each ancestor
    /// of the leaf in a nested cgroup path. For `"cg_0/narrow"`, writes
    /// `+cpuset` to `{parent}/cgroup.subtree_control` and
    /// `{parent}/cg_0/cgroup.subtree_control`. No-op for
    /// single-component paths.
    fn enable_subtree_cpuset(&self, name: &str) {
        let components: Vec<&str> = name.split('/').collect();
        if components.len() < 2 {
            return;
        }
        let mut cur = self.parent.clone();
        for c in &components[..components.len() - 1] {
            let sc = cur.join("cgroup.subtree_control");
            if sc.exists()
                && let Err(e) = write_with_timeout(&sc, "+cpuset", CGROUP_WRITE_TIMEOUT)
            {
                tracing::warn!(path = %sc.display(), err = %e, "failed to enable cpuset");
            }
            cur = cur.join(c);
        }
        // Write at the last intermediate (direct parent of the leaf).
        let sc = cur.join("cgroup.subtree_control");
        if sc.exists()
            && let Err(e) = write_with_timeout(&sc, "+cpuset", CGROUP_WRITE_TIMEOUT)
        {
            tracing::warn!(path = %sc.display(), err = %e, "failed to enable cpuset");
        }
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

    /// Move multiple tasks into a child cgroup by PID.
    ///
    /// Tolerates ESRCH (task exited between listing and migration).
    /// Retries EBUSY up to 3 times with 100ms backoff for transient
    /// rejections from sched_ext BPF `cgroup_prep_move` callbacks
    /// (`scx_cgroup_can_attach`). Propagates EBUSY after retries
    /// exhausted. Propagates all other errors immediately.
    pub fn move_tasks(&self, name: &str, tids: &[u32]) -> Result<()> {
        for &t in tids {
            if let Err(e) = self.move_task_with_retry(name, t) {
                if is_esrch(&e) {
                    tracing::warn!(tid = t, cgroup = name, "task vanished during migration");
                    continue;
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Move a single task with bounded EBUSY retry.
    fn move_task_with_retry(&self, name: &str, tid: u32) -> Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = Duration::from_millis(100);

        for attempt in 0..MAX_RETRIES {
            match self.move_task(name, tid) {
                Ok(()) => return Ok(()),
                Err(e) if is_ebusy(&e) && attempt + 1 < MAX_RETRIES => {
                    tracing::debug!(
                        tid,
                        cgroup = name,
                        attempt = attempt + 1,
                        "EBUSY on cgroup.procs write, retrying"
                    );
                    std::thread::sleep(RETRY_DELAY);
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    /// Clear `subtree_control` on a child cgroup by writing an empty
    /// string. Disables all controllers for the cgroup's children.
    ///
    /// Required before moving tasks into a cgroup that has
    /// `subtree_control` set: the kernel's no-internal-process
    /// constraint (`cgroup_migrate_vet_dst`) returns EBUSY when
    /// tasks are written to `cgroup.procs` of a cgroup with
    /// controllers in `subtree_control`.
    pub fn clear_subtree_control(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name).join("cgroup.subtree_control");
        if !p.exists() {
            return Ok(());
        }
        // Read current controllers and disable each one.
        let content = fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let content = content.trim();
        if content.is_empty() {
            return Ok(());
        }
        // Each controller name needs a "-" prefix to disable.
        let disable: Vec<String> = content
            .split_whitespace()
            .map(|c| format!("-{c}"))
            .collect();
        let disable_str = disable.join(" ");
        write_with_timeout(&p, &disable_str, CGROUP_WRITE_TIMEOUT)
            .with_context(|| format!("clear subtree_control on {name}"))
    }

    /// Move all tasks from a child cgroup to the cgroup root.
    ///
    /// Drains to `/sys/fs/cgroup/cgroup.procs` instead of the parent
    /// because the parent has `subtree_control` set (enabling cpuset
    /// for children), and the kernel's no-internal-process constraint
    /// rejects writes to `cgroup.procs` when `subtree_control` is
    /// active. The root cgroup is exempt from this constraint.
    pub fn drain_tasks(&self, name: &str) -> Result<()> {
        let src = self.parent.join(name).join("cgroup.procs");
        if !src.exists() {
            return Ok(());
        }
        drain_pids_to_root(&src, name);
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

/// Drain all tasks from `procs_path` to the cgroup filesystem root.
///
/// The root cgroup is exempt from the no-internal-process constraint,
/// so writes to `/sys/fs/cgroup/cgroup.procs` succeed even when
/// intermediate cgroups have `subtree_control` set.
/// ESRCH (task exited) is silently tolerated; other errors are logged.
fn drain_pids_to_root(procs_path: &Path, context: &str) {
    let dst = Path::new("/sys/fs/cgroup/cgroup.procs");
    if let Ok(content) = fs::read_to_string(procs_path) {
        for line in content.lines() {
            if let Ok(pid) = line.trim().parse::<u32>()
                && let Err(e) = write_with_timeout(dst, &pid.to_string(), CGROUP_WRITE_TIMEOUT)
                && !is_esrch(&e)
            {
                tracing::warn!(pid, cgroup = context, err = %e, "failed to drain task");
            }
        }
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
    drain_pids_to_root(&path.join("cgroup.procs"), &path.display().to_string());
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
        let dir = std::env::temp_dir().join(format!("ktstr-cg-test-{}", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("ktstr-cg-idem-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("cg_0").unwrap();
        cg.create_cgroup("cg_0").unwrap(); // should not error
        assert!(dir.join("cg_0").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_all_on_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-test-path");
        assert!(cg.cleanup_all().is_ok());
    }

    #[test]
    fn remove_cgroup_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-test-path");
        assert!(cg.remove_cgroup("no_such_cgroup").is_ok());
    }

    #[test]
    fn cleanup_removes_child_dirs() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-clean-{}", std::process::id()));
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
        let cg = CgroupManager::new("/nonexistent/ktstr-drain-test");
        assert!(cg.drain_tasks("missing_cgroup").is_ok());
    }

    #[test]
    fn setup_non_cgroup_path() {
        // setup() on a non-cgroup path should still create the dir
        let dir = std::env::temp_dir().join(format!("ktstr-setup-{}", std::process::id()));
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.setup(true).unwrap();
        assert!(dir.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_timeout_success() {
        let dir = std::env::temp_dir().join(format!("ktstr-wt-{}", std::process::id()));
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
        let cg = CgroupManager::new("/nonexistent/ktstr-move-test");
        assert!(cg.move_task("no_cgroup", 1).is_err());
    }

    #[test]
    fn set_cpuset_empty() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-cpuset-{}", std::process::id()));
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
        // move_tasks propagates non-ESRCH errors immediately
        let cg = CgroupManager::new("/nonexistent/ktstr-partial");
        let err = cg.move_tasks("cg", &[1, 2, 3]).unwrap_err();
        // The error comes from the first tid (write to nonexistent path)
        let msg = format!("{err:#}");
        assert!(msg.contains("cgroup.procs"), "unexpected error: {msg}");
    }

    #[test]
    fn drain_tasks_empty_cgroup() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-drain-{}", std::process::id()));
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
    fn is_esrch_detects_esrch_in_chain() {
        let io_err = std::io::Error::from_raw_os_error(libc::ESRCH);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(is_esrch(&anyhow_err));
    }

    #[test]
    fn is_esrch_rejects_enoent() {
        let io_err = std::io::Error::from_raw_os_error(libc::ENOENT);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(!is_esrch(&anyhow_err));
    }

    #[test]
    fn is_ebusy_detects_ebusy_in_chain() {
        let io_err = std::io::Error::from_raw_os_error(libc::EBUSY);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(is_ebusy(&anyhow_err));
    }

    #[test]
    fn is_ebusy_rejects_esrch() {
        let io_err = std::io::Error::from_raw_os_error(libc::ESRCH);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(!is_ebusy(&anyhow_err));
    }

    #[test]
    fn clear_subtree_control_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-clear-sc");
        // No subtree_control file → no-op success.
        assert!(cg.clear_subtree_control("cg_0").is_ok());
    }

    #[test]
    fn clear_subtree_control_empty() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-sc-{}", std::process::id()));
        let dir_a = dir.join("cg_a");
        fs::create_dir_all(&dir_a).unwrap();
        fs::write(dir_a.join("cgroup.subtree_control"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // Empty subtree_control → no-op success.
        assert!(cg.clear_subtree_control("cg_a").is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_timeout_blocks_on_fifo() {
        use std::ffi::CString;
        let dir = std::env::temp_dir().join(format!("ktstr-cg-fifo-{}", std::process::id()));
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
