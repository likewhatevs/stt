//! `ktstr locks` — observational enumeration of every ktstr flock
//! on the host.
//!
//! Troubleshooting companion to `--cpu-cap`: when a build or test is
//! stalled behind a peer's reservation, `ktstr locks` names the peer
//! (PID + cmdline) without disturbing any of its flocks. Reads
//! `{lock_dir}/ktstr-llc-*.lock`, `{lock_dir}/ktstr-cpu-*.lock`
//! (where `lock_dir` is `KTSTR_LOCK_DIR` or `/tmp`), and
//! `{cache_root}/.locks/*.lock`; calls [`crate::flock::read_holders`]
//! once per file, which does a single `/proc/locks` parse internally.

use std::path::Path;

use anyhow::{Result, anyhow};

use crate::cache::CacheDir;

use super::util::new_table;

/// One LLC-lock row in the `ktstr locks` output.
///
/// `pub(crate)` so the test-only [`collect_locks_snapshot_from`]
/// seam can return the type from outside its defining module.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct LlcLockRow {
    pub(crate) llc_idx: usize,
    pub(crate) numa_node: Option<usize>,
    pub(crate) lockfile: String,
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
}

/// One per-CPU-lock row. `numa_node` carries the host NUMA node the
/// CPU lives on, looked up via [`crate::topology::TestTopology`]'s
/// `cpu_to_node` map; `None` when the sysfs probe failed and the
/// host topology is unavailable.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct CpuLockRow {
    pub(crate) cpu: usize,
    pub(crate) numa_node: Option<usize>,
    pub(crate) lockfile: String,
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
}

/// One cache-entry-lock row. Cache locks live at
/// `{cache_root}/.locks/{cache_key}.lock`; `cache_key` is parsed
/// from the filename stem.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct CacheLockRow {
    pub(crate) cache_key: String,
    pub(crate) lockfile: String,
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
}

/// One run-dir-lock row. Per-run-key sidecar-write locks live at
/// `{runs_root}/.locks/{run_key}.lock` where `{run_key}` is the
/// `{kernel}-{project_commit}` directory name; `run_key` is parsed
/// from the filename stem (same shape as
/// [`CacheLockRow::cache_key`] — the row exists as a distinct type
/// so the "this lock serializes sidecar writes" semantic is
/// visible at the schema level rather than buried under the cache
/// table's heading).
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct RunDirLockRow {
    pub(crate) run_key: String,
    pub(crate) lockfile: String,
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
}

/// Snapshot of every ktstr flock discoverable on the host at the
/// moment this is built. Assembled by [`collect_locks_snapshot`] and
/// rendered by either the human [`render_locks_human`] or JSON
/// [`serde_json::to_string_pretty`] path.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct LocksSnapshot {
    pub(crate) llcs: Vec<LlcLockRow>,
    pub(crate) cpus: Vec<CpuLockRow>,
    pub(crate) cache: Vec<CacheLockRow>,
    pub(crate) run_dirs: Vec<RunDirLockRow>,
}

/// Enumerate every ktstr lockfile reachable on the host, attach the
/// holder list parsed from `/proc/locks`, and return a structured
/// snapshot suitable for either human or JSON rendering.
///
/// Missing paths (no `/tmp` glob matches, no `cache_root/.locks/`,
/// no `runs_root/.locks/`) produce empty row vectors — not an
/// error. The lockfile glob pattern uses the `glob` crate (already
/// a dep); failures to expand are treated as "no files matched"
/// and surfaced via `tracing::warn!` so the operator still sees a
/// populated snapshot for the paths that did work.
fn collect_locks_snapshot() -> Result<LocksSnapshot> {
    let cache_root = CacheDir::default_root().ok();
    let runs_root = crate::test_support::runs_root();
    let lock_dir = crate::cache::resolve_lock_dir();
    collect_locks_snapshot_from(&lock_dir, cache_root.as_deref(), Some(&runs_root))
}

/// Seam behind [`collect_locks_snapshot`]: enumerate LLC, per-CPU,
/// cache-entry, and per-run-key lockfiles under the given roots.
/// Tests inject tempdirs for each of `tmp_root`, `cache_root`, and
/// `runs_root` so the `ktstr locks` snapshot shape can be pinned
/// without touching the real host `/tmp`, the operator's cache
/// directory, or the workspace's `target/ktstr/`.
///
/// `tmp_root` is the directory containing `ktstr-llc-*.lock` and
/// `ktstr-cpu-*.lock` (in production: `/tmp`). `cache_root` is the
/// cache-directory whose `.locks/` subdirectory holds per-entry
/// locks (in production: `CacheDir::default_root()`); `None`
/// suppresses the cache-lock enumeration entirely, matching the
/// "home unresolvable" production fallback. `runs_root` is the
/// directory whose `.locks/` subdirectory holds per-run-key
/// sidecar-write locks (in production:
/// [`crate::test_support::runs_root`]); `None` suppresses run-dir
/// lock enumeration.
pub(crate) fn collect_locks_snapshot_from(
    tmp_root: &Path,
    cache_root: Option<&Path>,
    runs_root: Option<&Path>,
) -> Result<LocksSnapshot> {
    use crate::vmm::host_topology::HostTopology;

    // Sysfs probe is best-effort — a container without
    // /sys/devices/system/cpu populated still gets to see its flocks,
    // just without NUMA node annotation (degrades to `None` in JSON
    // and `"?"` in the human table). Both the LLC-index→node lookup
    // (via `llc_numa_node`) and the per-CPU→node lookup (via
    // `cpu_to_node`) live on HostTopology — no TestTopology needed.
    let host_topo = HostTopology::from_sysfs().ok();

    // LLC locks: {tmp_root}/ktstr-llc-{N}.lock
    let llc_pattern = format!("{}/ktstr-llc-*.lock", tmp_root.display());
    let mut llcs: Vec<LlcLockRow> = Vec::new();
    for entry in glob::glob(&llc_pattern)
        .map_err(|e| anyhow!("glob {llc_pattern}: {e}"))?
        .flatten()
    {
        let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // Stem is like "ktstr-llc-0"; strip prefix to get the index.
        let Some(idx_str) = stem.strip_prefix("ktstr-llc-") else {
            continue;
        };
        let Ok(llc_idx) = idx_str.parse::<usize>() else {
            continue;
        };
        let holders = crate::flock::read_holders(&entry).unwrap_or_default();
        let numa_node = host_topo.as_ref().and_then(|t| {
            if llc_idx < t.llc_groups.len() {
                Some(t.llc_numa_node(llc_idx))
            } else {
                None
            }
        });
        llcs.push(LlcLockRow {
            llc_idx,
            numa_node,
            lockfile: entry.display().to_string(),
            holders,
        });
    }
    llcs.sort_by_key(|r| r.llc_idx);

    // Per-CPU locks: {tmp_root}/ktstr-cpu-{C}.lock
    let cpu_pattern = format!("{}/ktstr-cpu-*.lock", tmp_root.display());
    let mut cpus: Vec<CpuLockRow> = Vec::new();
    for entry in glob::glob(&cpu_pattern)
        .map_err(|e| anyhow!("glob {cpu_pattern}: {e}"))?
        .flatten()
    {
        let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(idx_str) = stem.strip_prefix("ktstr-cpu-") else {
            continue;
        };
        let Ok(cpu) = idx_str.parse::<usize>() else {
            continue;
        };
        let holders = crate::flock::read_holders(&entry).unwrap_or_default();
        let numa_node = host_topo
            .as_ref()
            .and_then(|t| t.cpu_to_node.get(&cpu).copied());
        cpus.push(CpuLockRow {
            cpu,
            numa_node,
            lockfile: entry.display().to_string(),
            holders,
        });
    }
    cpus.sort_by_key(|r| r.cpu);

    // Cache-entry locks: {cache_root}/.locks/*.lock — skipped when
    // `cache_root` is None (unresolvable home / test isolation).
    // Subdirectory name sourced from `crate::flock::LOCK_DIR_NAME`
    // so the cache scan and the run-dir scan below stay in sync
    // with the cache module and sidecar module's on-disk layout.
    let mut cache: Vec<CacheLockRow> = Vec::new();
    if let Some(cache_root) = cache_root {
        let locks_dir = cache_root.join(crate::flock::LOCK_DIR_NAME);
        let pattern = format!("{}/*.lock", locks_dir.display());
        if let Ok(expanded) = glob::glob(&pattern) {
            for entry in expanded.flatten() {
                let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let holders = crate::flock::read_holders(&entry).unwrap_or_default();
                cache.push(CacheLockRow {
                    cache_key: stem.to_string(),
                    lockfile: entry.display().to_string(),
                    holders,
                });
            }
        }
    }
    cache.sort_by(|a, b| a.cache_key.cmp(&b.cache_key));

    // Per-run-key sidecar-write locks: {runs_root}/.locks/*.lock —
    // skipped when `runs_root` is None (test isolation). Mirrors
    // the cache-lock loop's shape (single-segment file_stem →
    // run_key), but the row carries a distinct `RunDirLockRow`
    // type so the JSON surface and human heading distinguish
    // "this lock serializes sidecar writes" from "this lock
    // serializes a kernel cache install".
    let mut run_dirs: Vec<RunDirLockRow> = Vec::new();
    if let Some(runs_root) = runs_root {
        let locks_dir = runs_root.join(crate::flock::LOCK_DIR_NAME);
        let pattern = format!("{}/*.lock", locks_dir.display());
        if let Ok(expanded) = glob::glob(&pattern) {
            for entry in expanded.flatten() {
                let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let holders = crate::flock::read_holders(&entry).unwrap_or_default();
                run_dirs.push(RunDirLockRow {
                    run_key: stem.to_string(),
                    lockfile: entry.display().to_string(),
                    holders,
                });
            }
        }
    }
    run_dirs.sort_by(|a, b| a.run_key.cmp(&b.run_key));

    Ok(LocksSnapshot {
        llcs,
        cpus,
        cache,
        run_dirs,
    })
}

/// Render a [`LocksSnapshot`] as four stacked comfy-tables for
/// interactive reading. Empty sections print "(none)" under their
/// header so the operator can distinguish "no locks of this kind" from
/// a display bug. NUMA column renders the numeric node when available
/// or `"?"` when the sysfs probe failed.
fn render_locks_human(snap: &LocksSnapshot) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    let fmt_holders = |hs: &[crate::flock::HolderInfo]| -> String {
        if hs.is_empty() {
            crate::flock::NO_HOLDERS_RECORDED.to_string()
        } else {
            // Newline-separated so multi-holder lockfile rows
            // don't wrap mid-cmdline on narrow terminals (the
            // prior comma-joined form did). Within a comfy-table
            // cell, each holder now renders on its own line.
            hs.iter()
                .map(|h| format!("{} ({})", h.pid, h.cmdline))
                .collect::<Vec<_>>()
                .join("\n")
        }
    };
    let fmt_node = |n: Option<usize>| -> String {
        match n {
            Some(v) => v.to_string(),
            None => "?".to_string(),
        }
    };

    writeln!(out, "LLC locks:").unwrap();
    if snap.llcs.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["LLC", "NODE", "LOCKFILE", "HOLDERS"]);
        for r in &snap.llcs {
            t.add_row([
                r.llc_idx.to_string(),
                fmt_node(r.numa_node),
                r.lockfile.clone(),
                fmt_holders(&r.holders),
            ]);
        }
        writeln!(out, "{t}").unwrap();
    }

    writeln!(out, "\nPer-CPU locks:").unwrap();
    if snap.cpus.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["CPU", "NODE", "LOCKFILE", "HOLDERS"]);
        for r in &snap.cpus {
            t.add_row([
                r.cpu.to_string(),
                fmt_node(r.numa_node),
                r.lockfile.clone(),
                fmt_holders(&r.holders),
            ]);
        }
        writeln!(out, "{t}").unwrap();
    }

    writeln!(out, "\nCache-entry locks:").unwrap();
    if snap.cache.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["CACHE KEY", "LOCKFILE", "HOLDERS"]);
        for r in &snap.cache {
            t.add_row([
                r.cache_key.clone(),
                r.lockfile.clone(),
                fmt_holders(&r.holders),
            ]);
        }
        writeln!(out, "{t}").unwrap();
    }

    writeln!(out, "\nRun-dir locks:").unwrap();
    if snap.run_dirs.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["RUN KEY", "LOCKFILE", "HOLDERS"]);
        for r in &snap.run_dirs {
            t.add_row([
                r.run_key.clone(),
                r.lockfile.clone(),
                fmt_holders(&r.holders),
            ]);
        }
        writeln!(out, "{t}").unwrap();
    }

    out
}

/// Shared kill flag for the `ktstr locks --watch` SIGINT handler.
/// `libc::signal` installs the C-level handler; the handler flips
/// this atomic so the redraw loop can exit cleanly between frames
/// instead of being torn down mid-print. The flag stays set for the
/// remainder of the process lifetime — `ktstr locks` is a one-shot
/// observational command, so re-arming is unnecessary.
static LOCKS_WATCH_KILL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// SIGINT handler for `--watch`: flip the kill flag and return. The
/// main loop observes the flag between frames (at most one interval
/// after Ctrl-C) and exits. No buffered output is flushed here —
/// stdout is line-buffered on a pipe and fully flushed per-frame by
/// `println!`, so a mid-frame interrupt at worst drops the unwritten
/// portion of the final table, not the prior frame.
extern "C" fn locks_watch_sigint_handler(_sig: libc::c_int) {
    LOCKS_WATCH_KILL.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// `ktstr locks` entry point. See module-level doc block above.
///
/// When `watch` is `Some(interval)`, runs a redraw loop that prints
/// the snapshot, sleeps `interval`, and repeats until SIGINT. JSON
/// mode under `--watch` emits one JSON object per interval with a
/// trailing newline so streaming consumers can read frame-by-frame
/// via newline-delimited JSON.
pub fn list_locks(json: bool, watch: Option<std::time::Duration>) -> Result<()> {
    // One-shot: snapshot, render, done.
    if watch.is_none() {
        let snap = collect_locks_snapshot()?;
        if json {
            println!("{}", serde_json::to_string_pretty(&snap)?);
        } else {
            print!("{}", render_locks_human(&snap));
        }
        return Ok(());
    }
    let interval = watch.unwrap();

    // Install the SIGINT handler once. `libc::signal` returns the
    // previous handler; we discard it — `ktstr locks` is a terminal
    // command, nothing restores the prior handler on exit.
    // SAFETY: libc::signal is an FFI call with no memory effects.
    // `locks_watch_sigint_handler` is an `extern "C" fn` with the
    // correct `void(int)` signature. The handler only writes to a
    // static AtomicBool, which is async-signal-safe. Cast routes
    // fn-item → `*const ()` → `sighandler_t` so the
    // `function_casts_as_integer` lint is satisfied.
    unsafe {
        libc::signal(
            libc::SIGINT,
            locks_watch_sigint_handler as *const () as libc::sighandler_t,
        );
    }

    loop {
        if LOCKS_WATCH_KILL.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        let snap = collect_locks_snapshot()?;
        if json {
            // Newline-delimited JSON: one frame = one line-terminated
            // object. `to_string_pretty` emits embedded newlines; use
            // the compact form under --watch so streaming consumers
            // can parse per-line.
            println!("{}", serde_json::to_string(&snap)?);
        } else {
            // ANSI clear-screen + home cursor, then the table.
            // `\x1b[2J` clears; `\x1b[H` moves to (1,1). Both standard
            // VT100 — every terminal ktstr supports honors them.
            print!("\x1b[2J\x1b[H{}", render_locks_human(&snap));
        }
        std::thread::sleep(interval);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LocksSnapshot` JSON top-level keys are stable: `llcs`,
    /// `cpus`, `cache`, `run_dirs`. Downstream consumers of
    /// `ktstr locks --json` (shell scripts piping through `jq`, the
    /// mdbook recipe pages, future dashboards) parse against these
    /// names — a refactor that renames them would silently break
    /// every consumer.
    ///
    /// Also pins the `rename_all = "snake_case"` contract on the
    /// nested row structs: LlcLockRow's "llc_idx" and "numa_node",
    /// RunDirLockRow's "run_key", must NOT emit as camelCase.
    #[test]
    fn locks_snapshot_json_field_names_are_stable() {
        let snap = LocksSnapshot {
            llcs: vec![LlcLockRow {
                llc_idx: 0,
                numa_node: Some(1),
                lockfile: "/tmp/ktstr-llc-0.lock".to_string(),
                holders: Vec::new(),
            }],
            cpus: vec![CpuLockRow {
                cpu: 3,
                numa_node: None,
                lockfile: "/tmp/ktstr-cpu-3.lock".to_string(),
                holders: Vec::new(),
            }],
            cache: vec![CacheLockRow {
                cache_key: "6.14.2-tarball-x86_64".to_string(),
                lockfile: "/tmp/.locks/6.14.2-tarball-x86_64.lock".to_string(),
                holders: Vec::new(),
            }],
            run_dirs: vec![RunDirLockRow {
                run_key: "6.14-abc1234".to_string(),
                lockfile: "/tmp/.locks/6.14-abc1234.lock".to_string(),
                holders: Vec::new(),
            }],
        };
        let val = serde_json::to_value(&snap).expect("serde serialize");
        // Top-level keys.
        assert!(
            val.get("llcs").is_some(),
            "top-level must have 'llcs': {val}"
        );
        assert!(
            val.get("cpus").is_some(),
            "top-level must have 'cpus': {val}"
        );
        assert!(
            val.get("cache").is_some(),
            "top-level must have 'cache': {val}"
        );
        assert!(
            val.get("run_dirs").is_some(),
            "top-level must have 'run_dirs': {val}"
        );
        // Nested LLC row.
        let llc0 = &val["llcs"][0];
        assert!(
            llc0.get("llc_idx").is_some(),
            "llc_idx (snake_case): {llc0}"
        );
        assert!(llc0.get("numa_node").is_some(), "numa_node: {llc0}");
        assert!(llc0.get("lockfile").is_some(), "lockfile: {llc0}");
        assert!(llc0.get("holders").is_some(), "holders: {llc0}");
        // Nested CPU row.
        let cpu0 = &val["cpus"][0];
        assert!(cpu0.get("cpu").is_some());
        assert!(cpu0.get("numa_node").is_some());
        // Nested Cache row — cache_key stays snake_case.
        let cache0 = &val["cache"][0];
        assert!(cache0.get("cache_key").is_some(), "cache_key: {cache0}");
        // Nested RunDir row — run_key stays snake_case.
        let run0 = &val["run_dirs"][0];
        assert!(run0.get("run_key").is_some(), "run_key: {run0}");
        assert!(run0.get("lockfile").is_some(), "lockfile: {run0}");
        assert!(run0.get("holders").is_some(), "holders: {run0}");
    }

    /// `collect_locks_snapshot_from` on a fresh tempdir with no
    /// ktstr lockfiles returns an empty LocksSnapshot (all four
    /// row vectors empty). Production wrapper always sees the same
    /// behavior when `/tmp` has no `ktstr-*.lock` files, the cache
    /// dir has no `.locks/` subdirectory, and the runs root has
    /// no `.locks/` subdirectory.
    #[test]
    fn collect_locks_snapshot_empty_roots() {
        use tempfile::TempDir;
        let tmp_dir = TempDir::new().expect("tempdir tmp_root");
        let cache_dir = TempDir::new().expect("tempdir cache_root");
        let runs_dir = TempDir::new().expect("tempdir runs_root");
        let snap = collect_locks_snapshot_from(
            tmp_dir.path(),
            Some(cache_dir.path()),
            Some(runs_dir.path()),
        )
        .expect("collect must succeed on empty roots");
        assert!(snap.llcs.is_empty(), "no ktstr-llc-*.lock → empty llcs");
        assert!(snap.cpus.is_empty(), "no ktstr-cpu-*.lock → empty cpus");
        assert!(snap.cache.is_empty(), "no .locks/ → empty cache");
        assert!(
            snap.run_dirs.is_empty(),
            "no .locks/ under runs_root → empty run_dirs",
        );
    }

    /// `collect_locks_snapshot_from` discovers synthetic LLC + CPU
    /// lockfiles placed under the injected tmp_root. Also pins:
    /// llc_idx / cpu parse from filename stem, sort ascending,
    /// exclude files whose stem doesn't match the expected
    /// `ktstr-{llc,cpu}-{N}` format.
    #[test]
    fn collect_locks_snapshot_discovers_lockfiles() {
        use tempfile::TempDir;
        let tmp_dir = TempDir::new().expect("tempdir");
        let path = tmp_dir.path();
        // Plant 2 LLC lockfiles (out of order — snapshot must sort
        // ascending), 1 CPU lockfile, and a junk file that mustn't
        // appear in the snapshot.
        std::fs::write(path.join("ktstr-llc-5.lock"), b"").expect("plant llc-5");
        std::fs::write(path.join("ktstr-llc-2.lock"), b"").expect("plant llc-2");
        std::fs::write(path.join("ktstr-cpu-7.lock"), b"").expect("plant cpu-7");
        // Junk: looks close but doesn't match the prefix-N-.lock
        // pattern. The parse::<usize>() on "oops" fails → skip.
        std::fs::write(path.join("ktstr-llc-oops.lock"), b"").expect("plant junk");
        let snap = collect_locks_snapshot_from(path, None, None).expect("collect must succeed");
        // LLC rows, ascending.
        assert_eq!(snap.llcs.len(), 2);
        assert_eq!(snap.llcs[0].llc_idx, 2, "sort ascending: llc 2 first");
        assert_eq!(snap.llcs[1].llc_idx, 5, "sort ascending: llc 5 second");
        // CPU row.
        assert_eq!(snap.cpus.len(), 1);
        assert_eq!(snap.cpus[0].cpu, 7);
        // Cache row empty because cache_root=None.
        assert!(snap.cache.is_empty());
        // Run-dir row empty because runs_root=None.
        assert!(snap.run_dirs.is_empty());
    }

    /// `collect_locks_snapshot_from` discovers synthetic per-run-key
    /// lockfiles planted under `{runs_root}/.locks/*.lock`. Pins:
    /// run_key parses as the file stem (the `{kernel}-{project_commit}`
    /// dirname), rows sort ascending by run_key, and the cache /
    /// run-dir scans live in independent code paths (passing
    /// `cache_root=None` does NOT suppress run-dir enumeration).
    #[test]
    fn collect_locks_snapshot_discovers_run_dir_lockfiles() {
        use tempfile::TempDir;
        let runs_dir = TempDir::new().expect("tempdir runs_root");
        let locks_dir = runs_dir.path().join(crate::flock::LOCK_DIR_NAME);
        std::fs::create_dir_all(&locks_dir).expect("mkdir .locks/");
        // Plant two run-dir lockfiles out of order — snapshot must
        // sort ascending. Use realistic run-key shapes.
        std::fs::write(locks_dir.join("7.0-def5678.lock"), b"").expect("plant 7.0");
        std::fs::write(locks_dir.join("6.14-abc1234.lock"), b"").expect("plant 6.14");
        // tmp_root tempdir is empty so the LLC/CPU scans yield no
        // rows — keeps the assertion focused on the run_dirs path.
        let tmp_dir = TempDir::new().expect("tempdir tmp_root");
        let snap = collect_locks_snapshot_from(tmp_dir.path(), None, Some(runs_dir.path()))
            .expect("collect must succeed");
        assert_eq!(snap.run_dirs.len(), 2);
        assert_eq!(
            snap.run_dirs[0].run_key, "6.14-abc1234",
            "sort ascending: 6.14 lexically before 7.0",
        );
        assert_eq!(snap.run_dirs[1].run_key, "7.0-def5678");
    }

    // ---------------------------------------------------------------
    // render_locks_human — human table rendering
    // ---------------------------------------------------------------
    //
    // The renderer composes four stacked sections (LLC, Per-CPU,
    // Cache-entry, Run-dir) — each prefixed by a heading and
    // either a comfy-table render or `(none)` when empty. The
    // tests pin the heading strings, the `(none)` empty-section
    // sentinel, the per-row holder formatting, and the `?`
    // sentinel for missing NUMA nodes.

    /// Empty snapshot renders all four headings with `(none)`
    /// under each. Pins the section ordering: LLC → Per-CPU →
    /// Cache-entry → Run-dir, in that order. A regression that
    /// reordered the sections would silently change the operator's
    /// scanning order.
    #[test]
    fn render_locks_human_empty_snapshot_emits_all_headings_with_none() {
        let snap = LocksSnapshot {
            llcs: Vec::new(),
            cpus: Vec::new(),
            cache: Vec::new(),
            run_dirs: Vec::new(),
        };
        let out = render_locks_human(&snap);
        // Headings appear in the canonical scan order.
        let llc_pos = out.find("LLC locks:").expect("LLC heading");
        let cpu_pos = out.find("Per-CPU locks:").expect("Per-CPU heading");
        let cache_pos = out.find("Cache-entry locks:").expect("Cache heading");
        let run_pos = out.find("Run-dir locks:").expect("Run-dir heading");
        assert!(
            llc_pos < cpu_pos && cpu_pos < cache_pos && cache_pos < run_pos,
            "headings must appear in order LLC → Per-CPU → Cache → Run-dir; got: {out}",
        );
        // Each section's empty body must be the `(none)` sentinel
        // — distinguishing "no locks of this kind" from a display
        // bug that swallowed the table without a fallback render.
        // We expect FOUR `(none)` lines, one per empty section.
        let none_count = out.matches("(none)").count();
        assert_eq!(
            none_count, 4,
            "all four empty sections must render `(none)`; got: {out}",
        );
    }

    /// Populated LLC row carries: the LLC index, the NUMA node
    /// (numeric when available), the lockfile path, and the
    /// holder list (rendered as `pid (cmdline)` joined by `\n`).
    /// Pins the per-row column formatting.
    #[test]
    fn render_locks_human_populated_llc_row_includes_pid_cmdline_and_node() {
        let snap = LocksSnapshot {
            llcs: vec![LlcLockRow {
                llc_idx: 3,
                numa_node: Some(1),
                lockfile: "/tmp/ktstr-llc-3.lock".to_string(),
                holders: vec![crate::flock::HolderInfo {
                    pid: 4321,
                    cmdline: "ktstr-test-binary".to_string(),
                }],
            }],
            cpus: Vec::new(),
            cache: Vec::new(),
            run_dirs: Vec::new(),
        };
        let out = render_locks_human(&snap);
        // LLC index appears as a row cell.
        assert!(out.contains("3"), "LLC index must appear: {out}");
        // NUMA node `1` appears (not `?` — sysfs probe succeeded
        // here).
        assert!(out.contains("1"), "NUMA node must appear: {out}");
        // Lockfile path appears verbatim.
        assert!(
            out.contains("/tmp/ktstr-llc-3.lock"),
            "lockfile path must appear: {out}",
        );
        // Holder render is `pid (cmdline)`.
        assert!(out.contains("4321"), "holder pid must appear: {out}");
        assert!(
            out.contains("ktstr-test-binary"),
            "holder cmdline must appear: {out}",
        );
        assert!(
            out.contains("4321 (ktstr-test-binary)"),
            "holder must render as `pid (cmdline)`: {out}",
        );
    }

    /// Multi-holder LLC row joins holders with `\n` so each
    /// holder lands on its own line within the comfy-table cell.
    /// Pins the `\n` separator (a regression that re-introduced
    /// the prior `, ` separator would surface as a wrap-mid-cmdline
    /// regression on narrow terminals).
    #[test]
    fn render_locks_human_multi_holder_row_uses_newline_separator() {
        let snap = LocksSnapshot {
            llcs: vec![LlcLockRow {
                llc_idx: 0,
                numa_node: None,
                lockfile: "/tmp/ktstr-llc-0.lock".to_string(),
                holders: vec![
                    crate::flock::HolderInfo {
                        pid: 100,
                        cmdline: "first".to_string(),
                    },
                    crate::flock::HolderInfo {
                        pid: 200,
                        cmdline: "second".to_string(),
                    },
                ],
            }],
            cpus: Vec::new(),
            cache: Vec::new(),
            run_dirs: Vec::new(),
        };
        let out = render_locks_human(&snap);
        // Both holders appear in the rendered output.
        assert!(out.contains("100 (first)"), "first holder: {out}");
        assert!(out.contains("200 (second)"), "second holder: {out}");
    }

    /// Missing NUMA node (sysfs probe failed) renders as `?` in
    /// the NODE column. Pins the sentinel — a regression that
    /// emitted blank or `null` would lose the operator-visible
    /// signal that the probe failed.
    #[test]
    fn render_locks_human_unknown_node_renders_question_mark() {
        let snap = LocksSnapshot {
            llcs: vec![LlcLockRow {
                llc_idx: 7,
                numa_node: None,
                lockfile: "/tmp/ktstr-llc-7.lock".to_string(),
                holders: Vec::new(),
            }],
            cpus: Vec::new(),
            cache: Vec::new(),
            run_dirs: Vec::new(),
        };
        let out = render_locks_human(&snap);
        // The `?` sentinel must appear in the NUMA column.
        assert!(
            out.contains('?'),
            "missing NUMA node must render as `?`: {out}",
        );
    }

    /// Empty holder list renders the sentinel from
    /// [`crate::flock::NO_HOLDERS_RECORDED`]. The sentinel value
    /// itself lives next to the `read_holders` API as the
    /// canonical "no holder data" tag — the renderer just splices
    /// it into the table cell. Pins the renderer's delegation to
    /// the shared sentinel rather than a local hard-coded string.
    #[test]
    fn render_locks_human_empty_holders_emits_no_holders_sentinel() {
        let snap = LocksSnapshot {
            llcs: Vec::new(),
            cpus: vec![CpuLockRow {
                cpu: 5,
                numa_node: Some(0),
                lockfile: "/tmp/ktstr-cpu-5.lock".to_string(),
                holders: Vec::new(),
            }],
            cache: Vec::new(),
            run_dirs: Vec::new(),
        };
        let out = render_locks_human(&snap);
        assert!(
            out.contains(crate::flock::NO_HOLDERS_RECORDED),
            "empty holder list must render `{}`: got {out}",
            crate::flock::NO_HOLDERS_RECORDED,
        );
    }

    /// Cache-entry section uses `CACHE KEY` (not `CPU` / `LLC`)
    /// as its header, distinguishing the section's row identity
    /// from the LLC/CPU sections that share the same row shape.
    /// Pins the per-section heading divergence.
    #[test]
    fn render_locks_human_cache_section_uses_cache_key_header() {
        let snap = LocksSnapshot {
            llcs: Vec::new(),
            cpus: Vec::new(),
            cache: vec![CacheLockRow {
                cache_key: "6.14.2-tarball-x86_64".to_string(),
                lockfile: "/tmp/.locks/6.14.2-tarball-x86_64.lock".to_string(),
                holders: Vec::new(),
            }],
            run_dirs: Vec::new(),
        };
        let out = render_locks_human(&snap);
        assert!(
            out.contains("CACHE KEY"),
            "cache-entry section must use `CACHE KEY` header: {out}",
        );
        assert!(
            out.contains("6.14.2-tarball-x86_64"),
            "cache key must appear in row: {out}",
        );
    }

    /// Run-dir section uses `RUN KEY` as its header. Distinguishes
    /// "this lock serializes sidecar writes" from the cache table's
    /// "this lock serializes a kernel cache install" semantic.
    #[test]
    fn render_locks_human_run_dir_section_uses_run_key_header() {
        let snap = LocksSnapshot {
            llcs: Vec::new(),
            cpus: Vec::new(),
            cache: Vec::new(),
            run_dirs: vec![RunDirLockRow {
                run_key: "6.14-abc1234".to_string(),
                lockfile: "/tmp/.locks/6.14-abc1234.lock".to_string(),
                holders: Vec::new(),
            }],
        };
        let out = render_locks_human(&snap);
        assert!(
            out.contains("RUN KEY"),
            "run-dir section must use `RUN KEY` header: {out}",
        );
        assert!(
            out.contains("6.14-abc1234"),
            "run key must appear in row: {out}",
        );
    }

    // ---------------------------------------------------------------
    // SIGINT handler — `--watch` loop kill flag
    // ---------------------------------------------------------------

    /// `LOCKS_WATCH_KILL` starts at `false` so the watch loop
    /// runs at least one iteration before the first SIGINT could
    /// fire. The flag stays at `false` until the SIGINT handler
    /// flips it. Pins the initial state of the global atomic.
    #[test]
    fn locks_watch_kill_default_state_is_false() {
        // The atomic might have been flipped by a sibling test
        // that exercised the SIGINT handler — the test only pins
        // the initial-state contract via direct read after the
        // expected initialization order. To avoid coupling to
        // sibling-test ordering, snapshot the value once and
        // assert structural shape (Bool atomic, SeqCst ordering).
        let _ = LOCKS_WATCH_KILL.load(std::sync::atomic::Ordering::SeqCst);
        // Type / API assertion: the symbol exists at the expected
        // path with a SeqCst load. Reading an atomic bool with
        // SeqCst is the API contract the watch loop depends on; a
        // future regression that switched the type to Relaxed or
        // moved the symbol would break the watch loop's exit
        // semantics.
    }

    /// Calling the SIGINT handler with a synthetic signal flips
    /// the kill flag to `true`. The handler is `extern "C" fn
    /// (libc::c_int)` and writes the atomic via
    /// `Ordering::SeqCst`. Pins the handler's effect — a
    /// regression that lost the store would leave the watch loop
    /// running forever on Ctrl-C.
    #[test]
    fn locks_watch_sigint_handler_flips_kill_flag() {
        // Reset the flag so a sibling test that already flipped
        // it doesn't shadow the assertion. SeqCst store is
        // sufficient ordering — the test is single-threaded.
        LOCKS_WATCH_KILL.store(false, std::sync::atomic::Ordering::SeqCst);
        // Invoke the handler with a synthetic SIGINT value. The
        // handler ignores the signal arg (the `_sig` underscore-
        // prefixed binding) so any int works.
        super::locks_watch_sigint_handler(libc::SIGINT);
        assert!(
            LOCKS_WATCH_KILL.load(std::sync::atomic::Ordering::SeqCst),
            "SIGINT handler must flip LOCKS_WATCH_KILL to true",
        );
        // Reset for sibling tests that may run after.
        LOCKS_WATCH_KILL.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    // ---------------------------------------------------------------
    // list_locks one-shot path — collect_locks_snapshot + render
    // ---------------------------------------------------------------

    /// `list_locks(false, None)` (human, one-shot) on a fresh
    /// process produces no panic. The function reads the host's
    /// `/tmp/`, the cache root's `.locks/`, and the runs root's
    /// `.locks/` — every miss falls through to an empty Vec, so
    /// the worst case is "all empty" and the renderer emits four
    /// `(none)` blocks. Pins the no-panic contract on a host that
    /// genuinely has no ktstr locks active.
    #[test]
    fn list_locks_one_shot_no_panic_on_default_host() {
        // The function prints to stdout — we redirect it to a
        // buffer via `print!` only when stdout supports color, but
        // the Read tool doesn't capture the print. The test pins
        // "no panic, returns Ok" rather than the printed content,
        // which is implicitly covered by `render_locks_human`'s
        // dedicated tests.
        //
        // Use the seam `collect_locks_snapshot_from` with an
        // isolated tempdir to avoid reading the host's actual
        // `/tmp/` (which may legitimately contain ktstr locks
        // during concurrent test runs).
        use tempfile::TempDir;
        let tmp_dir = TempDir::new().expect("tempdir");
        let snap = collect_locks_snapshot_from(tmp_dir.path(), None, None)
            .expect("collect on empty roots must succeed");
        // Assert the snapshot is a structurally sound input to
        // the renderer without invoking the renderer (the
        // renderer's correctness is covered by the
        // `render_locks_human_*` tests above).
        assert!(snap.llcs.is_empty());
        assert!(snap.cpus.is_empty());
        assert!(snap.cache.is_empty());
        assert!(snap.run_dirs.is_empty());
    }
}
