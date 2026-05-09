use super::*;

#[test]
fn parse_rq_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_kernel_offsets(&path);
    assert_ne!(
        offsets.rq_nr_running, offsets.rq_clock,
        "rq_nr_running and rq_clock offsets must be distinct"
    );
    assert!(offsets.rq_clock > 0);
    assert!(offsets.rq_scx > 0);
    assert!(offsets.dsq_nr > 0);
}

#[test]
fn parse_event_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_kernel_offsets(&path);
    // Event offsets are optional — only assert if present.
    if let Some(ev) = &offsets.event_offsets {
        // All event counter fields are s64, so offsets must differ.
        let mut all = vec![
            ev.ev_select_cpu_fallback,
            ev.ev_dispatch_local_dsq_offline,
            ev.ev_dispatch_keep_last,
            ev.ev_enq_skip_exiting,
            ev.ev_enq_skip_migration_disabled,
        ];
        for off in [
            ev.ev_reenq_immed,
            ev.ev_reenq_local_repeat,
            ev.ev_refill_slice_dfl,
            ev.ev_bypass_duration,
            ev.ev_bypass_dispatch,
            ev.ev_bypass_activate,
            ev.ev_insert_not_owned,
            ev.ev_sub_bypass_dispatch,
        ]
        .into_iter()
        .flatten()
        {
            all.push(off);
        }
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "event counter offsets must be distinct");
            }
        }
    }
}

#[test]
fn parse_schedstat_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_kernel_offsets(&path);
    // Schedstat offsets are optional — only assert if present.
    if let Some(ss) = &offsets.schedstat_offsets {
        // rq_sched_info must be at a nonzero offset (it's not the first
        // field of struct rq).
        assert!(ss.rq_sched_info > 0);
        // pcount is the first field in struct sched_info, so its offset
        // can be 0. run_delay follows pcount, so it must be > 0.
        assert!(
            ss.sched_info_run_delay > 0,
            "run_delay must follow pcount in struct sched_info"
        );
        assert_ne!(
            ss.sched_info_pcount, ss.sched_info_run_delay,
            "pcount and run_delay offsets must be distinct"
        );
        // All rq-level fields must be at distinct nonzero offsets.
        let rq_fields = [
            ss.rq_yld_count,
            ss.rq_sched_count,
            ss.rq_sched_goidle,
            ss.rq_ttwu_count,
            ss.rq_ttwu_local,
        ];
        for &off in &rq_fields {
            assert!(off > 0, "schedstat rq field offset must be nonzero");
        }
        for i in 0..rq_fields.len() {
            for j in (i + 1)..rq_fields.len() {
                assert_ne!(
                    rq_fields[i], rq_fields[j],
                    "schedstat rq field offsets must be distinct"
                );
            }
        }
    }
}

#[test]
fn parse_sched_domain_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_kernel_offsets(&path);
    // Sched domain offsets are optional — only assert if present.
    if let Some(sd) = &offsets.sched_domain_offsets {
        // sd fields must be at distinct nonzero offsets (none is the
        // first field of struct sched_domain — parent is).
        assert!(sd.rq_sd > 0, "rq.sd must be at nonzero offset");
        // parent can be offset 0 (first field). level and name must
        // differ from parent.
        assert_ne!(
            sd.sd_level, sd.sd_parent,
            "level and parent offsets must be distinct"
        );
        assert_ne!(
            sd.sd_name, sd.sd_parent,
            "name and parent offsets must be distinct"
        );
        // Runtime fields that are always present must be at nonzero offsets.
        let always_present = [
            sd.sd_balance_interval,
            sd.sd_nr_balance_failed,
            sd.sd_max_newidle_lb_cost,
        ];
        for &off in &always_present {
            assert!(off > 0, "sched_domain runtime field offset must be nonzero");
        }
        // newidle_call/newidle_success/newidle_ratio are optional
        // (added in 6.19; absent on older kernels). When present,
        // they must be at nonzero offsets.
        for off in [
            sd.sd_newidle_call,
            sd.sd_newidle_success,
            sd.sd_newidle_ratio,
        ]
        .into_iter()
        .flatten()
        {
            assert!(
                off > 0,
                "optional newidle field offset must be nonzero when present"
            );
        }
        // Stats offsets are optional (CONFIG_SCHEDSTATS).
        if let Some(so) = &sd.stats_offsets {
            let array_fields = [
                so.sd_lb_count,
                so.sd_lb_failed,
                so.sd_lb_balanced,
                so.sd_lb_imbalance_load,
                so.sd_lb_imbalance_util,
                so.sd_lb_imbalance_task,
                so.sd_lb_imbalance_misfit,
                so.sd_lb_gained,
                so.sd_lb_hot_gained,
                so.sd_lb_nobusyg,
                so.sd_lb_nobusyq,
            ];
            for i in 0..array_fields.len() {
                for j in (i + 1)..array_fields.len() {
                    assert_ne!(
                        array_fields[i], array_fields[j],
                        "sched_domain array field offsets must be distinct"
                    );
                }
            }
            let scalar_fields = [
                so.sd_alb_count,
                so.sd_alb_failed,
                so.sd_alb_pushed,
                so.sd_ttwu_wake_remote,
                so.sd_ttwu_move_affine,
                so.sd_ttwu_move_balance,
            ];
            for &off in &scalar_fields {
                assert!(off > 0, "sched_domain scalar field offset must be nonzero");
            }
            for i in 0..scalar_fields.len() {
                for j in (i + 1)..scalar_fields.len() {
                    assert_ne!(
                        scalar_fields[i], scalar_fields[j],
                        "sched_domain scalar field offsets must be distinct"
                    );
                }
            }
        }
    }
}

#[test]
fn parse_bpf_map_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_bpf_map_offsets(&path);
    // All offsets should be nonzero in a real kernel BTF.
    assert!(offsets.map_name > 0);
    assert!(offsets.map_type > 0);
    assert!(offsets.value_size > 0);
    assert!(offsets.array_value > 0);
    // BTF-related offsets should be resolved.
    // btf_data can be 0 (first field in struct btf), so just verify
    // that parsing succeeded without error. btf_data_size cannot be
    // the first field (data comes before it), so it must be nonzero.
    assert!(offsets.map_btf > 0);
    assert!(offsets.map_btf_value_type_id > 0);
    assert!(offsets.btf_data_size > offsets.btf_data);
}

#[test]
fn parse_bpf_prog_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_bpf_prog_offsets(&path);
    assert!(offsets.prog_aux > 0);
    assert!(offsets.aux_verified_insns > 0);
    assert!(offsets.aux_name > 0);
}

/// Validate that optional BTF offsets (watchdog, event) are
/// internally consistent.
///
/// `watchdog_offsets` requires the post-refactor `scx_sched` layout
/// (with `watchdog_timeout` field). `event_offsets` can resolve via
/// either path (6.18+ `pcpu` or 6.16-6.17 `event_stats_cpu`).
/// `watchdog_offsets` being present implies `event_offsets` is also
/// present, but not vice versa.
///
/// Assertions that overlap with parse_rq_offsets_from_vmlinux and
/// parse_event_offsets_from_vmlinux are intentionally omitted.
#[test]
fn btf_optional_offsets_consistent() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = match KernelOffsets::from_vmlinux(&path) {
        Ok(o) => o,
        Err(e) => skip!("vmlinux BTF resolution failed: {e}"),
    };

    assert_ne!(
        offsets.rq_nr_running, offsets.rq_scx,
        "rq_nr_running and rq_scx offsets must be distinct"
    );

    if let Some(ref ev) = offsets.event_offsets {
        assert!(ev.percpu_ptr_off > 0);
    }

    if let Some(ref wd) = offsets.watchdog_offsets {
        assert!(
            wd.scx_sched_watchdog_timeout_off > 0,
            "watchdog_timeout offset must be nonzero within scx_sched"
        );
        assert!(
            offsets.event_offsets.is_some(),
            "watchdog_offsets present implies event_offsets must also resolve"
        );
    }
}

#[test]
fn from_vmlinux_nonexistent() {
    let path = std::path::Path::new("/nonexistent/vmlinux");
    assert!(KernelOffsets::from_vmlinux(path).is_err());
}

#[test]
fn from_vmlinux_empty_file() {
    let dir = std::env::temp_dir().join(format!("ktstr-btf-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("vmlinux");
    std::fs::write(&f, b"").unwrap();
    assert!(KernelOffsets::from_vmlinux(&f).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

// -- BTF sidecar cache --
//
// These tests exercise the `<path>.btf` sidecar pipeline:
//   * pure helpers (path derivation, magic check, freshness)
//     directly;
//   * end-to-end `load_btf_from_path` behavior against a real-ELF
//     fixture when one is available on the host;
//   * the cache-root membership guard that suppresses sidecar
//     reads/writes for vmlinux paths outside the cache, including
//     symlink-resolution semantics and relative-path handling.

#[test]
fn btf_sidecar_path_appends_dot_btf() {
    let p = std::path::Path::new("/cache/vmlinux");
    assert_eq!(
        btf_sidecar_path(p),
        std::path::PathBuf::from("/cache/vmlinux.btf"),
    );
}

#[test]
fn btf_sidecar_path_preserves_existing_extension() {
    // Append-suffix semantics, NOT `with_extension` which would
    // replace `.elf` with `.btf`.
    let p = std::path::Path::new("/cache/vmlinux.elf");
    assert_eq!(
        btf_sidecar_path(p),
        std::path::PathBuf::from("/cache/vmlinux.elf.btf"),
    );
}

#[test]
fn is_raw_btf_accepts_little_endian_magic() {
    // Little-endian BTF begins with bytes 0x9F 0xEB in file
    // order. `is_raw_btf` accepts only little-endian BTF: the
    // host architectures ktstr supports are LE, so a big-endian
    // BTF blob is an unsupported configuration even though
    // btf-rs itself could parse it (see the sibling
    // `is_raw_btf_rejects_wrong_magic_and_short_input` where
    // the BE magic is explicitly rejected).
    assert!(is_raw_btf(&[0x9F, 0xEB, 0x01, 0x00]));
}

#[test]
fn is_raw_btf_rejects_wrong_magic_and_short_input() {
    // ELF magic — bytes-wise distinct from BTF magic.
    assert!(!is_raw_btf(&[0x7F, b'E', b'L', b'F']));
    // Big-endian BTF magic: file-order bytes 0xEB 0x9F. btf-rs
    // itself would parse such a blob (branches on the magic at
    // cbtf::btf_header::from_reader), but ktstr supports only
    // LE hosts, so `is_raw_btf` deliberately rejects BE and
    // lets the caller surface "not recognized as raw BTF" via
    // the ELF-parse fallback.
    assert!(!is_raw_btf(&[0xEB, 0x9F, 0x01, 0x00]));
    // Too short to carry the 2-byte magic.
    assert!(!is_raw_btf(&[0x9F]));
    assert!(!is_raw_btf(&[]));
}

#[test]
fn sidecar_fresh_false_when_either_file_missing() {
    let dir =
        std::env::temp_dir().join(format!("ktstr-btf-sidecar-missing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let vmlinux = dir.join("vmlinux");
    let sidecar = dir.join("vmlinux.btf");
    std::fs::write(&vmlinux, b"vmlinux-bytes").unwrap();
    // sidecar missing → not fresh
    assert!(!sidecar_fresh(&sidecar, &vmlinux));
    std::fs::write(&sidecar, b"cached-btf").unwrap();
    // both present → fresh (sidecar written after vmlinux)
    assert!(sidecar_fresh(&sidecar, &vmlinux));
    // vmlinux missing → not fresh (safe default)
    std::fs::remove_file(&vmlinux).unwrap();
    assert!(!sidecar_fresh(&sidecar, &vmlinux));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Vmlinux staged inside a private cache root for sidecar tests.
///
/// Field declaration order pins drop order — Rust drops struct
/// fields top-to-bottom. `_cache_env` (the `KTSTR_CACHE_DIR`
/// `EnvVarGuard`) is declared first so it drops first, restoring
/// the env BEFORE `_root` drops and removes the temporary
/// directory. Without that ordering, `KTSTR_CACHE_DIR` would
/// transiently point at a deleted directory while the next test
/// runs — a dangling-env-ref hazard.
///
/// `entry_dir` and `vmlinux` are simple `PathBuf`s with no
/// drop side effects, so their position only documents intent.
struct CacheStagedVmlinux {
    _cache_env: crate::test_support::test_helpers::EnvVarGuard,
    entry_dir: std::path::PathBuf,
    vmlinux: std::path::PathBuf,
    _root: tempfile::TempDir,
}

/// Stage a vmlinux copy at `<cache_root>/<entry>/vmlinux` so the
/// sidecar guard treats writes as in-cache, and point
/// `KTSTR_CACHE_DIR` at the cache root for the returned value's
/// lifetime. See [`CacheStagedVmlinux`] for drop semantics.
fn stage_in_cache(src: &std::path::Path) -> CacheStagedVmlinux {
    let root = tempfile::TempDir::new().expect("cache-root tempdir");
    let entry_dir = root.path().join("kentry");
    std::fs::create_dir_all(&entry_dir).expect("create cache entry dir");
    let vmlinux = entry_dir.join("vmlinux");
    std::fs::copy(src, &vmlinux).expect("copy vmlinux into cache-staged dir");
    let _cache_env =
        crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", root.path());
    CacheStagedVmlinux {
        _cache_env,
        entry_dir,
        vmlinux,
        _root: root,
    }
}

/// End-to-end: first load extracts BTF from ELF vmlinux and
/// writes the sidecar; second load reads the sidecar bytes
/// directly and parses them. Exercises both branches of
/// `load_btf_from_path` against a real vmlinux.
///
/// Skipped when no test vmlinux is available or when
/// `find_test_vmlinux` resolves to raw BTF (sysfs), which
/// exercises a different branch that never writes a sidecar.
#[test]
fn load_btf_writes_sidecar_then_hits_cache_on_second_load() {
    use std::time::Duration;

    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        // Raw BTF input never writes a sidecar — wrong branch.
        return;
    }

    // Stage the vmlinux inside a private KTSTR_CACHE_DIR so the
    // sidecar membership guard permits the write. lock_env held
    // for the test's lifetime — KTSTR_CACHE_DIR is process-wide.
    let _env = crate::test_support::test_helpers::lock_env();
    let staged = stage_in_cache(&path);
    let vmlinux = staged.vmlinux.as_path();
    let sidecar = btf_sidecar_path(vmlinux);
    // Ensure vmlinux mtime is strictly less than whatever the
    // sidecar write will stamp — avoids a same-second tie that
    // could false-pass the freshness check on low-resolution
    // filesystems.
    std::thread::sleep(Duration::from_millis(10));

    // Pre-state: no sidecar exists.
    assert!(
        !sidecar.exists(),
        "precondition: sidecar should not exist before first load",
    );

    // First load: extract + write sidecar.
    let btf1 = load_btf_from_path(vmlinux).expect("first load must succeed");
    // Consume btf1 so the optimizer cannot elide the parse.
    let _ = format!("{:?}", btf1.resolve_types_by_name("task_struct").is_ok());
    assert!(
        sidecar.exists(),
        "first load must write sidecar at {}",
        sidecar.display(),
    );
    let sidecar_bytes = std::fs::read(&sidecar).unwrap();
    assert!(
        is_raw_btf(&sidecar_bytes),
        "sidecar contents must carry the raw BTF 0x9FEB magic",
    );

    // Sanity: sidecar mtime is at/after vmlinux mtime so the
    // freshness check on the second load picks it up.
    assert!(
        sidecar_fresh(&sidecar, vmlinux),
        "sidecar mtime must be ≥ vmlinux mtime after first load",
    );

    // Second load: should hit the cache. We verify by deleting
    // the ELF so that any fallback to ELF parsing would fail —
    // but wait, the function reads the ELF path first for its
    // bytes; deletion would break even the sidecar branch
    // (since the function reads `path` unconditionally at the
    // top). Instead, pin the behavior by checking that a second
    // load still succeeds AND the sidecar mtime is unchanged
    // (a second write would bump it).
    let sidecar_mtime_before = std::fs::metadata(&sidecar).unwrap().modified().unwrap();
    // Sleep a bit so a spurious sidecar rewrite would be
    // detectable via an mtime bump.
    std::thread::sleep(Duration::from_millis(50));
    let btf2 = load_btf_from_path(vmlinux).expect("second load must succeed");
    let _ = format!("{:?}", btf2.resolve_types_by_name("task_struct").is_ok());
    let sidecar_mtime_after = std::fs::metadata(&sidecar).unwrap().modified().unwrap();
    assert_eq!(
        sidecar_mtime_before, sidecar_mtime_after,
        "second load must hit sidecar cache — mtime bump proves a \
         redundant rewrite",
    );
}

/// Simulating a stale sidecar by making its mtime older than
/// vmlinux's: the next load must ignore the cached bytes and
/// re-extract from ELF, then overwrite the sidecar. Exercises
/// the `mtime(sidecar) < mtime(vmlinux)` staleness guard.
#[test]
fn load_btf_rejects_stale_sidecar() {
    use std::time::{Duration, SystemTime};

    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    let staged = stage_in_cache(&path);
    let vmlinux = staged.vmlinux.as_path();
    let sidecar = btf_sidecar_path(vmlinux);

    // Plant a sidecar that predates vmlinux by writing garbage
    // and stamping its mtime into the past. `set_times` is the
    // portable way to force a past mtime.
    std::fs::write(&sidecar, b"stale-sidecar-bytes").unwrap();
    let past = SystemTime::now() - Duration::from_secs(3600);
    let f = std::fs::File::options().write(true).open(&sidecar).unwrap();
    f.set_modified(past).unwrap();
    drop(f);

    // Precondition: sidecar is older than vmlinux.
    assert!(
        !sidecar_fresh(&sidecar, vmlinux),
        "precondition: planted sidecar must be stale",
    );

    let btf = load_btf_from_path(vmlinux)
        .expect("load must succeed via ELF fallback despite stale sidecar");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    // Post-condition: sidecar has been overwritten with fresh
    // BTF bytes (must now start with the BTF magic, not the
    // garbage we planted).
    let sidecar_bytes = std::fs::read(&sidecar).unwrap();
    assert!(
        is_raw_btf(&sidecar_bytes),
        "load must overwrite stale sidecar with fresh BTF bytes",
    );
    assert!(
        sidecar_fresh(&sidecar, vmlinux),
        "sidecar must be fresh again after re-extraction",
    );
}

/// Sidecar with correct mtime but garbage contents (no 0x9FEB
/// magic): the load must recover by falling through to ELF
/// extraction and overwriting the corrupt sidecar. Exercises
/// the "fresh but lacks magic" branch of the match inside
/// `load_btf_from_path`.
#[test]
fn load_btf_recovers_from_corrupt_sidecar() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    let staged = stage_in_cache(&path);
    let vmlinux = staged.vmlinux.as_path();
    let sidecar = btf_sidecar_path(vmlinux);
    // Plant a sidecar that is newer than vmlinux but whose
    // contents do not carry the BTF magic.
    std::fs::write(&sidecar, b"not-btf-bytes").unwrap();
    assert!(
        sidecar_fresh(&sidecar, vmlinux),
        "precondition: planted sidecar must be mtime-fresh",
    );

    let btf =
        load_btf_from_path(vmlinux).expect("load must recover when sidecar is fresh-but-corrupt");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    // Corrupt sidecar should have been overwritten.
    let sidecar_bytes = std::fs::read(&sidecar).unwrap();
    assert!(
        is_raw_btf(&sidecar_bytes),
        "corrupt sidecar must be overwritten on next load",
    );
}

/// When the sidecar would be written to a read-only directory,
/// the load must still succeed — sidecar writes are
/// best-effort and never surface as errors. Exercises the
/// tracing::warn fallback in `write_btf_sidecar`'s error path
/// while the path IS inside the cache root, so the
/// membership-guard skip cannot be the reason no sidecar
/// appears.
#[test]
#[cfg(unix)]
fn load_btf_survives_readonly_sidecar_dir() {
    use std::os::unix::fs::PermissionsExt;

    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }
    // Root skips DAC permission checks entirely, so a
    // read-only directory still lets root write inside. The
    // test cannot distinguish "sidecar write skipped due to
    // best-effort" from "sidecar write succeeded because we
    // are root" — skip under euid 0 to avoid a false-pass on
    // CI runners that sandbox as root.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    let staged = stage_in_cache(&path);
    let vmlinux = staged.vmlinux.as_path();
    let entry_dir = staged.entry_dir.as_path();
    // Mark entry dir read-only after the vmlinux is in place so
    // the sidecar's tempfile+rename within `write_btf_sidecar`
    // fails on tempfile creation.
    std::fs::set_permissions(entry_dir, std::fs::Permissions::from_mode(0o555)).unwrap();

    // Load must succeed despite the sidecar write failing.
    let btf =
        load_btf_from_path(vmlinux).expect("load must succeed even when sidecar dir is read-only");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    // Sidecar must not exist — write should have failed at the
    // best-effort layer, not at the membership guard.
    let sidecar = btf_sidecar_path(vmlinux);
    assert!(
        !sidecar.exists(),
        "sidecar must not exist after write to read-only dir",
    );

    // Restore permissions so the tempdir cleanup (TempDir drop)
    // can recurse into the entry dir.
    let _ = std::fs::set_permissions(entry_dir, std::fs::Permissions::from_mode(0o755));
}

/// Raw-BTF inputs (files that already carry 0x9FEB magic) must
/// never have a sidecar written alongside them — the input file
/// IS the BTF blob, and a sidecar would be a byte-for-byte
/// copy of itself. Exercises the raw-BTF early-return branch.
#[test]
fn load_btf_skips_sidecar_for_raw_btf_input() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if !path.starts_with("/sys/") {
        // Generate a raw-BTF file from the ELF so this test
        // exercises the raw-BTF path even when
        // find_test_vmlinux returns an ELF.
        let dir =
            std::env::temp_dir().join(format!("ktstr-btf-sidecar-raw-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src_data = std::fs::read(&path).unwrap();
        let elf = match goblin::elf::Elf::parse(&src_data) {
            Ok(e) => e,
            Err(_) => {
                // Raw BTF already — skip the ELF extraction.
                let raw = dir.join("vmlinux.btf-raw");
                std::fs::copy(&path, &raw).unwrap();
                let _ = load_btf_from_path(&raw).expect("raw-BTF load must succeed");
                let sidecar = btf_sidecar_path(&raw);
                assert!(
                    !sidecar.exists(),
                    "raw-BTF input must not produce a sidecar",
                );
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        };
        let btf_shdr = elf
            .section_headers
            .iter()
            .find(|sh| elf.shdr_strtab.get_at(sh.sh_name) == Some(".BTF"));
        let shdr = match btf_shdr {
            Some(s) => s,
            None => {
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        };
        let offset = shdr.sh_offset as usize;
        let size = shdr.sh_size as usize;
        let raw_bytes = &src_data[offset..offset + size];
        let raw = dir.join("vmlinux.btf-raw");
        std::fs::write(&raw, raw_bytes).unwrap();
        let _ = load_btf_from_path(&raw).expect("raw-BTF load must succeed");
        // The sidecar would be at `<raw>.btf` — must NOT exist.
        let sidecar = btf_sidecar_path(&raw);
        assert!(
            !sidecar.exists(),
            "raw-BTF input must not produce a sidecar at {}",
            sidecar.display(),
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
    // /sys/kernel/btf/vmlinux path: raw BTF on a read-only
    // filesystem. Any sidecar write would fail anyway, and the
    // sidecar path itself ("/sys/kernel/btf/vmlinux.btf") is
    // not writable — we cannot assert much here beyond "load
    // must succeed," which the pre-existing tests already
    // cover.
}

/// A vmlinux that lives outside the configured cache root must
/// never have a sidecar written next to it. Models the
/// kernel-source-tree pollution shape that motivated the guard:
/// `KTSTR_CACHE_DIR` points at one tempdir, the vmlinux lives
/// in a sibling tempdir (the "source tree"), and the load must
/// produce parsed BTF without touching the source-tree
/// directory.
#[test]
fn sidecar_skipped_when_path_outside_cache_root() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    // KTSTR_CACHE_DIR points at one tempdir.
    let cache_root = tempfile::TempDir::new().expect("cache root tempdir");
    let _cache_env =
        crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
    // vmlinux lives in a sibling tempdir — outside the cache
    // root, simulating a kernel source tree.
    let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
    let vmlinux = source_tree.path().join("vmlinux");
    std::fs::copy(&path, &vmlinux).expect("copy vmlinux into source-tree dir");

    let btf =
        load_btf_from_path(&vmlinux).expect("load must succeed even when sidecar is suppressed");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    let sidecar = btf_sidecar_path(&vmlinux);
    assert!(
        !sidecar.exists(),
        "sidecar must not be written when vmlinux path is outside cache root, got {}",
        sidecar.display(),
    );
}

/// A vmlinux that lives inside the configured cache root must
/// have its sidecar written. Sibling assertion to
/// `sidecar_skipped_when_path_outside_cache_root`: the guard
/// must not be a blanket suppression, only an out-of-cache
/// suppression.
#[test]
fn sidecar_written_when_path_inside_cache_root() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    let staged = stage_in_cache(&path);
    let vmlinux = staged.vmlinux.as_path();

    let sidecar = btf_sidecar_path(vmlinux);
    assert!(
        !sidecar.exists(),
        "precondition: sidecar must not exist before the load — \
         a leftover from a prior test would falsely pass the post-load \
         existence check",
    );

    let btf = load_btf_from_path(vmlinux).expect("load must succeed inside cache root");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    assert!(
        sidecar.exists(),
        "sidecar must be written when vmlinux path is inside cache root, expected at {}",
        sidecar.display(),
    );
    let bytes = std::fs::read(&sidecar).unwrap();
    assert!(
        is_raw_btf(&bytes),
        "sidecar must contain raw BTF (0x9FEB magic) when written inside cache root",
    );
}

/// Cache root that cannot be resolved (every cascade variable
/// removed) must produce `path_inside_cache_root == false` and
/// suppress the sidecar. The load itself must still succeed —
/// "no cache root" is not a failure mode for BTF resolution,
/// just for sidecar caching.
#[test]
fn sidecar_skipped_when_cache_root_unresolvable() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    // Strip every variable in the resolution cascade so
    // resolve_cache_root_with_suffix has nothing to walk.
    let _no_ktstr = crate::test_support::test_helpers::EnvVarGuard::remove("KTSTR_CACHE_DIR");
    let _no_xdg = crate::test_support::test_helpers::EnvVarGuard::remove("XDG_CACHE_HOME");
    let _no_home = crate::test_support::test_helpers::EnvVarGuard::remove("HOME");

    let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
    let vmlinux = source_tree.path().join("vmlinux");
    std::fs::copy(&path, &vmlinux).expect("copy vmlinux");

    let btf =
        load_btf_from_path(&vmlinux).expect("load must succeed when cache root is unresolvable");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    let sidecar = btf_sidecar_path(&vmlinux);
    assert!(
        !sidecar.exists(),
        "sidecar must not be written when cache root is unresolvable, got {}",
        sidecar.display(),
    );
}

/// Symlink E2E: real vmlinux LIVES in the cache, a symlink to
/// it lives in a source tree. Loading via the symlink path
/// must canonicalize through to the cache and write the
/// sidecar NEXT TO THE REAL FILE — not next to the symlink.
/// The sidecar derivation MUST track the same canonical path
/// as the membership check.
#[test]
#[cfg(unix)]
fn load_btf_symlink_into_cache_writes_sidecar_in_cache_only() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    let staged = stage_in_cache(&path);
    let real_vmlinux = staged.vmlinux.as_path();
    let real_sidecar = btf_sidecar_path(real_vmlinux);
    assert!(
        !real_sidecar.exists(),
        "precondition: real sidecar must not exist before the load",
    );

    // Symlink in a sibling tempdir (the "source tree") pointing
    // at the real cached vmlinux.
    let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
    let symlink_path = source_tree.path().join("vmlinux");
    std::os::unix::fs::symlink(real_vmlinux, &symlink_path)
        .expect("create symlink to real vmlinux");
    let lexical_sidecar = btf_sidecar_path(&symlink_path);

    // Load via the symlink path. Post-fix, canonicalize at the
    // top of load_btf_from_path resolves the symlink so the
    // sidecar writes to <cache>/kentry/vmlinux.btf next to the
    // real file. Pre-fix this flow missed the cache entirely:
    // lexical parent (<source-tree>) canonicalizes outside the
    // cache, the membership gate returns false, and the sidecar
    // is suppressed for what is, after symlink resolution, a
    // genuine cache entry.
    let btf = load_btf_from_path(&symlink_path)
        .expect("load via symlink must succeed and resolve the target");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    assert!(
        real_sidecar.exists(),
        "sidecar must land at the canonical path inside cache, expected {}",
        real_sidecar.display(),
    );
    assert!(
        !lexical_sidecar.exists(),
        "sidecar must NOT land next to the symlink in the source tree, \
         got pollution at {}",
        lexical_sidecar.display(),
    );
}

/// Symlink E2E inverse: real vmlinux lives in a source tree,
/// symlink to it lives in the cache. Loading via the symlink
/// path must canonicalize through to the source-tree real file
/// — and the membership check on the canonical path returns
/// false, so NO sidecar is written anywhere. The cache
/// directory must remain free of sidecar files for symlinks
/// pointing OUT.
#[test]
#[cfg(unix)]
fn load_btf_symlink_out_of_cache_writes_no_sidecar() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    // Cache root with no real vmlinux inside it.
    let cache_root = tempfile::TempDir::new().expect("cache-root tempdir");
    let _cache_env =
        crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
    // Real vmlinux in source tree (outside cache).
    let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
    let real_vmlinux = source_tree.path().join("vmlinux");
    std::fs::copy(&path, &real_vmlinux).expect("copy vmlinux into source tree");
    // Symlink in cache pointing at the real source-tree vmlinux.
    let symlink_in_cache = cache_root.path().join("vmlinux");
    std::os::unix::fs::symlink(&real_vmlinux, &symlink_in_cache)
        .expect("create symlink to source-tree vmlinux");

    let btf = load_btf_from_path(&symlink_in_cache).expect("load via symlink must succeed");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    let real_sidecar = btf_sidecar_path(&real_vmlinux);
    let lexical_sidecar = btf_sidecar_path(&symlink_in_cache);
    assert!(
        !real_sidecar.exists(),
        "sidecar must not land in source tree (outside cache), got {}",
        real_sidecar.display(),
    );
    assert!(
        !lexical_sidecar.exists(),
        "sidecar must not land at the symlink path in cache either — \
         canonicalize-at-top resolves to the source-tree real file, \
         which is outside the cache",
    );
}

/// Relative path: pass a path that does not start with `/`,
/// confirm no sidecar lands at either the lexical relative
/// path's location or the absolute target.
///
/// Production callers reach `load_btf_from_path` through several
/// paths: `crate::vmm::find_vmlinux` (absolute paths derived from
/// the kernel-cache entry or distro debug locations) and the `None`
/// fallback in `crate::probe::btf::parse_btf_functions` /
/// `crate::probe::btf::resolve_field_specs` (the absolute literal
/// `/sys/kernel/btf/vmlinux`). All emit absolute paths. A
/// relative-path invocation is unusual, and its semantics
/// depend on the test process's CWD: if CWD is unrelated to
/// the relative path's parent (the typical case during a test
/// run), the initial `fs::read` fails and the function returns
/// Err before reaching any sidecar branch. This pins:
///
///   * the function does not panic on a relative input;
///   * no sidecar is written at the lexical relative-path
///     location, so a CWD-relative pollution shape cannot leak
///     past the membership gate even when canonicalize would
///     otherwise have reached the cache;
///   * no sidecar is written at the absolute target either —
///     the read step never resolves to it.
#[test]
fn load_btf_relative_path_suppresses_sidecar() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    // Point KTSTR_CACHE_DIR somewhere isolated. The cache root
    // is irrelevant for the assertion — we just want the load
    // path to NOT be inside whatever it points at.
    let cache_root = tempfile::TempDir::new().expect("cache-root tempdir");
    let _cache_env =
        crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
    // Stage a real vmlinux in a tempdir, then build a relative
    // path referring to it. The relative path is constructed
    // by stripping the leading `/` from the absolute path; from
    // the test process's CWD (the cargo workspace root), this
    // relative path will not resolve to a file, so the load's
    // initial `fs::read` step fails and the function returns
    // Err. The point of the test is the post-condition: NO
    // sidecar appears anywhere as a side effect.
    let outside = tempfile::TempDir::new().expect("outside tempdir");
    let abs_vmlinux = outside.path().join("vmlinux");
    std::fs::copy(&path, &abs_vmlinux).expect("copy vmlinux into outside dir");
    let rel_str = abs_vmlinux
        .to_str()
        .expect("test vmlinux path must be UTF-8")
        .strip_prefix('/')
        .expect("absolute path expected to start with /");
    let rel = std::path::Path::new(rel_str);
    assert!(
        !rel.is_absolute(),
        "precondition: constructed path must be relative, got {}",
        rel.display(),
    );

    let _ = load_btf_from_path(rel);
    let abs_sidecar = btf_sidecar_path(&abs_vmlinux);
    let rel_sidecar = btf_sidecar_path(rel);
    assert!(
        !abs_sidecar.exists(),
        "sidecar must not appear at the absolute target, got {}",
        abs_sidecar.display(),
    );
    assert!(
        !rel_sidecar.exists(),
        "sidecar must not appear at the relative path's lexical \
         location, got {}",
        rel_sidecar.display(),
    );
}

/// Empty `KTSTR_CACHE_DIR=""` falls through the cascade per
/// `resolve_cache_root_with_suffix`. With the rest of the
/// cascade pointed at an isolated tempdir, the membership
/// check succeeds for paths inside the resolved root. Models
/// the operator who clears KTSTR_CACHE_DIR expecting
/// XDG/HOME to take over.
#[test]
fn load_btf_empty_ktstr_cache_dir_falls_through() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    let xdg = tempfile::TempDir::new().expect("xdg tempdir");
    let _g_ktstr = crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", "");
    let _g_xdg = crate::test_support::test_helpers::EnvVarGuard::set("XDG_CACHE_HOME", xdg.path());
    // Resolved root: <xdg>/ktstr/kernels.
    let resolved_root = xdg.path().join("ktstr").join("kernels");
    let entry = resolved_root.join("kentry");
    std::fs::create_dir_all(&entry).expect("create cache entry under XDG fallback");
    let vmlinux = entry.join("vmlinux");
    std::fs::copy(&path, &vmlinux).expect("copy vmlinux into XDG-derived cache");
    let sidecar = btf_sidecar_path(&vmlinux);
    assert!(
        !sidecar.exists(),
        "precondition: sidecar must not pre-exist",
    );

    let btf =
        load_btf_from_path(&vmlinux).expect("load must succeed inside XDG-derived cache root");
    let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

    assert!(
        sidecar.exists(),
        "sidecar must be written even when cascade resolves via XDG_CACHE_HOME \
         (KTSTR_CACHE_DIR=\"\")",
    );
}

/// Mid-process `KTSTR_CACHE_DIR` change: a load that wrote a
/// sidecar under cache_a must produce no sidecar under cache_b
/// for the same vmlinux on the next call after the env points
/// at cache_b. Pins that membership resolution does not stick
/// to a memoized first-call answer.
#[test]
fn load_btf_fresh_resolution_per_call() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    if path.starts_with("/sys/") {
        return;
    }

    let _env = crate::test_support::test_helpers::lock_env();
    // Two cache roots; vmlinux always sits inside cache_a.
    let cache_a = tempfile::TempDir::new().expect("cache_a tempdir");
    let cache_b = tempfile::TempDir::new().expect("cache_b tempdir");
    let entry_a = cache_a.path().join("kentry");
    std::fs::create_dir_all(&entry_a).expect("create cache_a entry");
    let vmlinux = entry_a.join("vmlinux");
    std::fs::copy(&path, &vmlinux).expect("copy vmlinux into cache_a");
    let sidecar = btf_sidecar_path(&vmlinux);

    // First call: KTSTR_CACHE_DIR points at cache_a → in-cache,
    // sidecar written.
    {
        let _g =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", cache_a.path());
        assert!(
            !sidecar.exists(),
            "precondition: sidecar must not pre-exist"
        );
        let btf = load_btf_from_path(&vmlinux).expect("first load must succeed");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());
        assert!(
            sidecar.exists(),
            "first load (KTSTR_CACHE_DIR=cache_a) must write sidecar",
        );
        // Wipe sidecar so the second call's outcome is unambiguous.
        std::fs::remove_file(&sidecar).expect("remove sidecar between calls");
    }

    // Second call: KTSTR_CACHE_DIR moved to cache_b. The vmlinux
    // is still under cache_a, so it is now outside the active
    // cache → no sidecar should be written. A memoized cache
    // root resolution would surface here as a stale `true` and
    // the sidecar would reappear.
    {
        let _g =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", cache_b.path());
        let btf = load_btf_from_path(&vmlinux).expect("second load must succeed");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());
        assert!(
            !sidecar.exists(),
            "second load (KTSTR_CACHE_DIR=cache_b) must NOT write sidecar — \
             the vmlinux is now outside the active cache root",
        );
    }
}

// ---- probe.bpf.o atomic-op verification -----------------------
//
// The probe BPF program publishes the error-exit latch via
// `__sync_val_compare_and_swap(&ktstr_err_exit_detected, 0u, 1u)`.
// Cross-core ordering on weakly-ordered architectures (aarch64)
// depends on the BPF backend lowering this to a real
// `BPF_STX | BPF_ATOMIC | BPF_W` instruction with `BPF_CMPXCHG`
// in the imm field — NOT a plain store.
//
// A toolchain regression that silently degraded the cmpxchg to a
// non-atomic store would leave the latch's publication
// unsynchronized, causing the freeze coordinator on a different
// core to miss the transition under TSO-violating reorder. This
// test pins the BPF bytecode against that regression by parsing
// the compiled `probe.o` and asserting at least one atomic op is
// present in the `tp_btf/sched_ext_exit` program section.
//
// BPF instruction encoding (uapi/linux/bpf.h):
//   - opcode byte: bits[2:0] = class (BPF_STX = 0x03),
//     bits[4:3] = size (BPF_W = 0x00, BPF_DW = 0x18),
//     bits[7:5] = mode (BPF_ATOMIC = 0xc0).
//     STX | ATOMIC | W = 0xc3.
//   - imm field (4 bytes, little-endian): atomic op type.
//     BPF_CMPXCHG = 0xf1 (= 0xf0 | BPF_FETCH).
#[test]
fn probe_bpf_object_emits_atomic_for_err_exit_latch() {
    // probe.o is produced by build.rs at OUT_DIR/probe.o. A
    // missing or unparseable file is a HARD FAIL, not a skip —
    // the whole point of the test is catching silent
    // regressions in the BPF backend lowering, and a silent
    // skip when the artifact is gone defeats that. build.rs
    // produces probe.o on every cargo build of the lib, so a
    // missing artifact here means the build pipeline is
    // broken and the test should surface that loudly.
    //
    // Limitation: this test counts ANY BPF_CMPXCHG instruction
    // in the `tp_btf/sched_ext_exit` section, not specifically
    // the cmpxchg targeting `ktstr_err_exit_detected`. Today
    // the section contains exactly one
    // `__sync_val_compare_and_swap` call (against the latch),
    // so any cmpxchg present must be the latch's; if a future
    // change adds a second atomic in the same handler, the
    // assert still passes but stops being a tight check on
    // the latch specifically. A future refinement could parse
    // the ELF relocation entries to constrain by symbol name
    // (look for a relocation referencing
    // `ktstr_err_exit_detected` adjacent to the cmpxchg
    // instruction).
    let probe_obj_path = std::path::PathBuf::from(env!("OUT_DIR")).join("probe.o");
    let bytes = std::fs::read(&probe_obj_path).unwrap_or_else(|e| {
        panic!(
            "probe.o missing or unreadable at {}: {e}. \
             build.rs failed to produce the BPF skeleton — fix the \
             build pipeline before re-running this test.",
            probe_obj_path.display()
        )
    });
    let elf = goblin::elf::Elf::parse(&bytes).unwrap_or_else(|e| {
        panic!(
            "probe.o at {} is not a valid ELF: {e}. \
             The BPF skeleton emitter changed format or the file is \
             corrupted — re-run the build to regenerate.",
            probe_obj_path.display()
        )
    });
    // Locate the program section. libbpf names sections after
    // the SEC() macro argument; tp_btf programs land in
    // `tp_btf/sched_ext_exit`. Match exact name; a future
    // restructure that splits the program into a different
    // section would surface as a clear test failure with the
    // expected name.
    const TARGET_SECTION: &str = "tp_btf/sched_ext_exit";
    let mut found_section = false;
    let mut atomic_count: usize = 0;
    for sh in &elf.section_headers {
        let Some(name) = elf.shdr_strtab.get_at(sh.sh_name) else {
            continue;
        };
        if name != TARGET_SECTION {
            continue;
        }
        found_section = true;
        // BPF programs are SHT_PROGBITS sections of `n` 8-byte
        // instructions. Read the section bytes via offset/size.
        let off = sh.sh_offset as usize;
        let sz = sh.sh_size as usize;
        assert!(
            sz.is_multiple_of(8),
            "BPF section size {sz} must be a multiple of 8 (instruction width)"
        );
        let prog = &bytes[off..off + sz];
        // BPF_STX | BPF_ATOMIC | BPF_W = 0xc3
        // BPF_STX | BPF_ATOMIC | BPF_DW = 0xc3 | 0x18 = 0xdb
        // The latch is u32, so we expect 0xc3 specifically — but
        // accept either width to keep the test robust against
        // a future widening to u64.
        const STX_ATOMIC_W: u8 = 0xc3;
        const STX_ATOMIC_DW: u8 = 0xdb;
        // BPF_CMPXCHG = 0xf0 | BPF_FETCH(0x01) = 0xf1
        const BPF_CMPXCHG_IMM: i32 = 0xf1;
        for chunk in prog.chunks_exact(8) {
            let opcode = chunk[0];
            if opcode == STX_ATOMIC_W || opcode == STX_ATOMIC_DW {
                let imm = i32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                if imm == BPF_CMPXCHG_IMM {
                    atomic_count += 1;
                }
            }
        }
    }
    assert!(
        found_section,
        "probe.o is missing the expected `{TARGET_SECTION}` section — \
         SEC() macro changed?"
    );
    assert!(
        atomic_count >= 1,
        "probe.o `{TARGET_SECTION}` section has no BPF_STX|BPF_ATOMIC|cmpxchg \
         instruction — `__sync_val_compare_and_swap` was silently \
         lowered to a non-atomic store. Cross-core ordering on aarch64 \
         would be broken by this regression."
    );
}

/// `ScxWalkerOffsets::missing_groups` enumerates exactly the
/// sub-groups that failed to resolve, naming each with the kernel
/// struct name the freeze coordinator surfaces in the failure
/// dump's `scx_walker_unavailable` field. Operators parsing the
/// failure-dump JSON look for these exact names — drift in the
/// `missing.push("...")` literals here breaks the human-visible
/// diagnostic string AND the structural pattern matching in any
/// downstream tooling that buckets walker availability by group.
///
/// Pin every sub-group's missing-name string by constructing
/// `ScxWalkerOffsets` instances with each `Option` field set to
/// `None` in isolation, asserting `missing_groups()` returns
/// exactly that one name. A regression that renamed any push
/// literal trips here. Also pins the empty-vec contract for the
/// fully-populated case (no groups missing) and the multi-missing
/// case (every group missing — the diagnostic must list all 10).
#[test]
fn scx_walker_missing_groups_pins_every_group_name() {
    // Construct a ScxWalkerOffsets with every sub-group resolved
    // to `Some(...)` placeholder — used as the baseline; tests
    // override one field at a time to None to exercise each push
    // arm of `missing_groups()`.
    fn full() -> ScxWalkerOffsets {
        ScxWalkerOffsets {
            rq: Some(RqStructOffsets { scx: 0, curr: 0 }),
            scx_rq: Some(ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 0,
                nr_running: 0,
                flags: 0,
                cpu_released: 0,
                ops_qseq: 0,
                kick_sync: Some(0),
                nr_immed: Some(0),
                clock: Some(0),
            }),
            task: Some(TaskStructCoreOffsets {
                comm: 0,
                pid: 0,
                scx: 0,
            }),
            see: Some(SchedExtEntityOffsets {
                runnable_node: 0,
                runnable_at: 0,
                weight: 0,
                slice: 0,
                dsq_vtime: 0,
                dsq: 0,
                dsq_list: 0,
                flags: 0,
                dsq_flags: 0,
                sticky_cpu: 0,
                holding_cpu: 0,
                tasks_node: 0,
            }),
            dsq_lnode: Some(ScxDsqListNodeOffsets { node: 0, flags: 0 }),
            dsq: Some(ScxDispatchQOffsets {
                list: 0,
                nr: 0,
                seq: 0,
                id: 0,
                hash_node: 0,
            }),
            sched: Some(ScxSchedOffsets {
                dsq_hash: 0,
                pnode: Some(0),
                pcpu: Some(0),
                aborting: Some(0),
                bypass_depth: Some(0),
                exit_kind: 0,
            }),
            sched_pnode: Some(ScxSchedPnodeOffsets {
                global_dsq: Some(0),
            }),
            sched_pcpu: Some(ScxSchedPcpuOffsets {
                bypass_dsq: Some(0),
            }),
            rht: Some(RhashtableOffsets {
                tbl: 0,
                nelems: 0,
                bucket_table_size: 0,
                bucket_table_buckets: 0,
                rhash_head_next: 0,
            }),
        }
    }

    // Fully-populated: no sub-group missing. Empty vec is the
    // "every walker pass has data" sentinel — the freeze
    // coordinator only writes a partial-degradation diagnostic
    // when this list is non-empty.
    let all = full();
    assert!(
        all.missing_groups().is_empty(),
        "fully-populated offsets must report no missing groups; got {:?}",
        all.missing_groups(),
    );

    // Pin each group's missing-name string in isolation: drop
    // exactly one field, expect exactly one entry whose string
    // matches the canonical name. The pairs below enumerate the
    // 10 sub-groups; a regression adding/removing/renaming a
    // push arm trips here.
    #[allow(clippy::type_complexity)]
    let cases: &[(fn(&mut ScxWalkerOffsets), &'static str)] = &[
        (
            (|o: &mut ScxWalkerOffsets| o.rq = None) as fn(&mut ScxWalkerOffsets),
            "rq",
        ),
        (|o: &mut ScxWalkerOffsets| o.scx_rq = None, "scx_rq"),
        (|o: &mut ScxWalkerOffsets| o.task = None, "task_struct"),
        (|o: &mut ScxWalkerOffsets| o.see = None, "sched_ext_entity"),
        (
            |o: &mut ScxWalkerOffsets| o.dsq_lnode = None,
            "scx_dsq_list_node",
        ),
        (|o: &mut ScxWalkerOffsets| o.dsq = None, "scx_dispatch_q"),
        (|o: &mut ScxWalkerOffsets| o.sched = None, "scx_sched"),
        (
            |o: &mut ScxWalkerOffsets| o.sched_pnode = None,
            "scx_sched_pnode",
        ),
        (
            |o: &mut ScxWalkerOffsets| o.sched_pcpu = None,
            "scx_sched_pcpu",
        ),
        (
            |o: &mut ScxWalkerOffsets| o.rht = None,
            "rhashtable/bucket_table/rhash_head",
        ),
    ];
    for (drop_fn, expected_name) in cases {
        let mut o = full();
        drop_fn(&mut o);
        let missing = o.missing_groups();
        assert_eq!(
            missing.len(),
            1,
            "exactly one group should be missing; expected {expected_name:?}, got {missing:?}",
        );
        assert_eq!(
            missing[0], *expected_name,
            "missing-group name string drifted: expected {expected_name:?}, got {:?}",
            missing[0],
        );
    }

    // Every group missing: the order of the names must match
    // the order of the `if self.<field>.is_none()` arms in
    // `missing_groups()` so a downstream consumer reading the
    // failure dump sees a stable, predictable sequence (rq,
    // scx_rq, task_struct, sched_ext_entity, scx_dsq_list_node,
    // scx_dispatch_q, scx_sched, scx_sched_pnode, scx_sched_pcpu,
    // rhashtable/bucket_table/rhash_head). A regression that
    // shuffled the arms would silently break that ordering.
    let empty = ScxWalkerOffsets {
        rq: None,
        scx_rq: None,
        task: None,
        see: None,
        dsq_lnode: None,
        dsq: None,
        sched: None,
        sched_pnode: None,
        sched_pcpu: None,
        rht: None,
    };
    let missing = empty.missing_groups();
    assert_eq!(
        missing,
        vec![
            "rq",
            "scx_rq",
            "task_struct",
            "sched_ext_entity",
            "scx_dsq_list_node",
            "scx_dispatch_q",
            "scx_sched",
            "scx_sched_pnode",
            "scx_sched_pcpu",
            "rhashtable/bucket_table/rhash_head",
        ],
        "all-missing order must match the if-chain order in `missing_groups()`",
    );
}

// -- vmlinux BTF resolution for new offset types ------------------
//
// `BpfMapOffsets::from_vmlinux` populates `task_storage_offsets`,
// `struct_ops_offsets`, `ringbuf_offsets`, and `stackmap_offsets` as
// optional sub-structs (each `Result<T>::ok()`). Verify against a
// real vmlinux that the offsets resolve and that fields within each
// resolved struct are at distinct nonzero offsets where required.
//
// Each test skips silently when `find_test_vmlinux` returns None,
// matching the pattern of the existing vmlinux-backed tests above.

/// `bpf_struct_ops_map` and `bpf_struct_ops_value` resolve to the two
/// offsets needed for the value-bytes read path: `kvalue` (the
/// embedded value within the map struct) and `value_data` (the
/// data[] flex array start within the value).
#[test]
fn parse_struct_ops_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_bpf_map_offsets(&path);
    let Some(so) = &offsets.struct_ops_offsets else {
        // struct_ops support is optional — kernel built without
        // CONFIG_BPF_JIT or with stripped BTF can elide these
        // types. Skip silently rather than fail.
        return;
    };
    // kvalue must be > 0: bpf_struct_ops_map starts with bpf_map at
    // offset 0, so its kvalue field cannot be at 0.
    assert!(
        so.kvalue > 0,
        "kvalue must follow bpf_map prefix in bpf_struct_ops_map"
    );
    // value_data must be ≥ 8: bpf_struct_ops_value's common header
    // (refcnt + state) is at least 8 bytes, and `data` follows.
    // The cacheline-aligned flex-array placement may push it
    // further (typical: 64). Pin the lower bound at the
    // common-header size.
    assert!(
        so.value_data >= 8,
        "value_data must follow bpf_struct_ops_common_value (refcnt + state)"
    );
}

/// `bpf_local_storage_map`, `bpf_local_storage_map_bucket`,
/// `bpf_local_storage_elem`, `bpf_local_storage_data`,
/// `bpf_local_storage`, and `hlist_node` all resolve, with field
/// offsets honoring the offset-0 invariants the resolver enforces.
#[test]
fn parse_task_storage_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_bpf_map_offsets(&path);
    let Some(ts) = &offsets.task_storage_offsets else {
        // local-storage subsystem types missing from BTF; kernels
        // without CONFIG_BPF_SYSCALL or stripped BTF land here.
        return;
    };
    // Resolver assertion: hlist_node.next is at offset 0. If a
    // future kernel reorders hlist_node, resolve_task_storage_offsets
    // returns Err and this test runs the None path above.
    assert_eq!(
        ts.hlist_node_next, 0,
        "hlist_node.next must be at offset 0 (resolver invariant)"
    );
    // smap_buckets and smap_bucket_log must be at distinct nonzero
    // offsets. bpf_local_storage_map starts with bpf_map at offset 0,
    // so neither of these embedded fields can land at 0.
    assert!(
        ts.smap_buckets > 0,
        "smap_buckets must follow bpf_map prefix"
    );
    assert!(
        ts.smap_bucket_log > 0,
        "smap_bucket_log must follow bpf_map prefix"
    );
    assert_ne!(
        ts.smap_buckets, ts.smap_bucket_log,
        "buckets pointer and bucket_log must be distinct fields"
    );
    // bucket_size > 0 (the bucket struct contains at least an
    // hlist_head with a `first` pointer).
    assert!(ts.bucket_size > 0, "bpf_local_storage_map_bucket size > 0");
    // hlist_head_first is the first member of struct hlist_head, so
    // it can be 0.
    // elem_local_storage and elem_sdata must be distinct nonzero
    // (elem starts with map_node at offset 0; both follow).
    assert!(
        ts.elem_local_storage > 0,
        "elem.local_storage follows map_node"
    );
    assert!(ts.elem_sdata > 0, "elem.sdata follows map_node");
    assert_ne!(
        ts.elem_local_storage, ts.elem_sdata,
        "elem.local_storage and elem.sdata must be distinct fields"
    );
}

/// `bpf_ringbuf_map` and `bpf_ringbuf` resolve to the five offsets
/// needed for ringbuf state capture. mask/consumer_pos/producer_pos/
/// pending_pos must all be at distinct offsets (they're co-located
/// in struct bpf_ringbuf).
#[test]
fn parse_ringbuf_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_bpf_map_offsets(&path);
    let Some(rb) = &offsets.ringbuf_offsets else {
        // ringbuf subsystem missing from BTF.
        return;
    };
    // rbm_rb: bpf_ringbuf_map starts with bpf_map at offset 0; rb is
    // the next field, so > 0.
    assert!(
        rb.rbm_rb > 0,
        "bpf_ringbuf_map.rb must follow embedded bpf_map"
    );
    // The four position fields live on bpf_ringbuf, which starts
    // with mask. Beyond that, every field must be at a distinct
    // offset since they're back-to-back u64/unsigned long fields.
    let position_fields = [
        rb.rb_mask,
        rb.rb_consumer_pos,
        rb.rb_producer_pos,
        rb.rb_pending_pos,
    ];
    for i in 0..position_fields.len() {
        for j in (i + 1)..position_fields.len() {
            assert_ne!(
                position_fields[i], position_fields[j],
                "ringbuf position offsets must be distinct: \
                 mask={}, consumer_pos={}, producer_pos={}, pending_pos={}",
                rb.rb_mask, rb.rb_consumer_pos, rb.rb_producer_pos, rb.rb_pending_pos,
            );
        }
    }
    // Spacing sanity: consumer/producer/pending are placed on
    // separate cachelines per the kernel layout (each at >= 4096
    // bytes apart in modern kernels for false-sharing avoidance).
    // Don't pin the exact spacing — just ensure the layout is
    // monotonically nonzero past mask.
    assert!(
        rb.rb_consumer_pos > rb.rb_mask,
        "consumer_pos must follow mask in bpf_ringbuf"
    );
}

/// `bpf_stack_map` and `stack_map_bucket` resolve to the four
/// offsets needed for stack-trace bucket walking. n_buckets and
/// buckets are co-located in bpf_stack_map; nr and data are
/// co-located in stack_map_bucket.
#[test]
fn parse_stackmap_offsets_from_vmlinux() {
    let path = match crate::monitor::find_test_vmlinux() {
        Some(p) => p,
        None => return,
    };
    let offsets = crate::test_support::require_bpf_map_offsets(&path);
    let Some(sm) = &offsets.stackmap_offsets else {
        // STACK_TRACE / bpf_stack_map missing from BTF.
        return;
    };
    // bpf_stack_map starts with bpf_map at offset 0; n_buckets and
    // buckets follow.
    assert!(
        sm.smap_n_buckets > 0,
        "n_buckets must follow embedded bpf_map"
    );
    assert!(sm.smap_buckets > 0, "buckets must follow embedded bpf_map");
    assert_ne!(
        sm.smap_n_buckets, sm.smap_buckets,
        "n_buckets and buckets pointer must be distinct fields"
    );
    // stack_map_bucket: nr is a u32 typically near the head, data
    // is the trailing flex array. nr can be at 0 if the bucket has
    // no rcu_head/links preceding it; data must follow.
    assert!(
        sm.smb_data > sm.smb_nr || sm.smb_nr > 0,
        "stack_map_bucket layout: nr and data must be distinguishable \
         (data follows nr OR nr is past offset 0)"
    );
    assert_ne!(
        sm.smb_nr, sm.smb_data,
        "stack_map_bucket nr and data must be distinct fields"
    );
}

/// `cached_vmlinux_btf` returns the same `Arc<Btf>` on every call
/// once the cache is populated. `Arc::ptr_eq` is the load-bearing
/// assertion — content equality would also hold from a re-parse, but
/// only the cache hit produces a shared allocation.
///
/// Skipped when `/sys/kernel/btf/vmlinux` is unreadable (the function
/// returns `None` on the first call). On test hosts where the file
/// exists but is unreadable for the test user, the cache will not
/// populate and the assertion would fail spuriously; the early
/// `is_none` short-circuit lets such an environment skip the test
/// rather than misreport a cache bug.
#[test]
fn cached_vmlinux_btf_hits_on_second_call() {
    let first = match super::cached_vmlinux_btf() {
        Some(b) => b,
        None => {
            // `/sys/kernel/btf/vmlinux` unreadable on this host —
            // every probe-pipeline call site that consumes the cache
            // also handles `None` by falling back to the no-BTF
            // path, so there is nothing to assert here.
            return;
        }
    };
    let second = super::cached_vmlinux_btf().expect(
        "second call must succeed when first did — the cache slot is populated and \
         no error path is taken on cache hit",
    );
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "cached_vmlinux_btf must return the same Arc on every call once populated; \
         got fresh allocations, indicating the cache hit path did not fire",
    );
}
