use super::*;

// -- parse_config_hz / vcpu_preemption_threshold_ns tests --

#[test]
fn parse_config_hz_standard() {
    let config = "# comment\nCONFIG_HZ_1000=y\nCONFIG_HZ=1000\n";
    assert_eq!(parse_config_hz(config), Some(1000));
}

#[test]
fn parse_config_hz_250() {
    let config = "CONFIG_HZ=250\n";
    assert_eq!(parse_config_hz(config), Some(250));
}

#[test]
fn parse_config_hz_100() {
    let config = "CONFIG_HZ=100\n";
    assert_eq!(parse_config_hz(config), Some(100));
}

#[test]
fn parse_config_hz_missing() {
    let config = "CONFIG_PREEMPT=y\nCONFIG_HZ_1000=y\n";
    assert_eq!(parse_config_hz(config), None);
}

#[test]
fn parse_config_hz_garbage_value() {
    let config = "CONFIG_HZ=abc\n";
    assert_eq!(parse_config_hz(config), None);
}

#[test]
fn parse_config_hz_whitespace() {
    let config = "  CONFIG_HZ=1000  \n";
    assert_eq!(parse_config_hz(config), Some(1000));
}

#[test]
fn parse_config_hz_commented_out() {
    let config = "# CONFIG_HZ=1000\nCONFIG_HZ_1000=y\n";
    assert_eq!(parse_config_hz(config), None);
}

#[test]
fn vcpu_threshold_reasonable_range() {
    // With no kernel path, falls back to host config or DEFAULT_HZ=250.
    // Threshold should be between 10ms (HZ=1000) and 100ms (HZ=100).
    let t = vcpu_preemption_threshold_ns(None);
    assert!(
        (10_000_000..=100_000_000).contains(&t),
        "threshold {t} ns outside expected range 10ms-100ms"
    );
}

#[test]
fn vcpu_threshold_default_hz_fallback() {
    // Nonexistent kernel path -> falls back to host config or default.
    let t = vcpu_preemption_threshold_ns(Some(std::path::Path::new("/nonexistent/bzImage")));
    assert!(
        (10_000_000..=100_000_000).contains(&t),
        "fallback threshold {t} ns outside expected range"
    );
}

/// Regression for the "host config leaks into guest HZ" bug:
/// when `kernel_path` is `Some`, `guest_kernel_hz` must not fall
/// back to `/boot/config-$(uname -r)`. A cached/built guest
/// kernel's HZ is independent of the host's HZ, so silently
/// picking up host HZ would yield wrong tick-dependent thresholds
/// on any mismatch.
///
/// This test points `kernel_path` at a nonexistent file. The
/// IKCONFIG and `.config` lookups both fail, and the function
/// must return exactly [`DEFAULT_HZ`] — NOT whatever the host's
/// `/boot/config` happens to contain.
#[test]
fn guest_kernel_hz_gated_on_kernel_path() {
    let bogus = std::path::Path::new("/nonexistent/ktstr-kernel/bzImage");
    let hz = guest_kernel_hz(Some(bogus));
    assert_eq!(
        hz, DEFAULT_HZ,
        "kernel_path=Some with no IKCONFIG/.config must fall back \
         to DEFAULT_HZ, not host /boot/config; got {hz}"
    );
}

/// Complement: with `kernel_path=None` (virtme-style run), the
/// host config IS authoritative and may legitimately override
/// `DEFAULT_HZ`. Check the returned value is a plausible HZ
/// value — i.e., the code path still works when we explicitly
/// want host fallback.
#[test]
fn guest_kernel_hz_none_consults_host_config() {
    let hz = guest_kernel_hz(None);
    // Accept any known Linux HZ value (DEFAULT_HZ=250 is in this set).
    assert!(
        matches!(hz, 100 | 250 | 300 | 1000),
        "guest_kernel_hz(None) = {hz} outside plausible HZ set"
    );
}

// -- IKCONFIG extraction tests --

/// Build a synthetic blob: padding + IKCFG_ST marker + gzip(config_text) + IKCFG_ED marker.
fn make_ikconfig_blob(config_text: &str) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    let mut blob = vec![0u8; 64]; // padding
    blob.extend_from_slice(IKCONFIG_MAGIC);
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(config_text.as_bytes()).unwrap();
    blob.extend(encoder.finish().unwrap());
    blob.extend_from_slice(b"IKCFG_ED");
    blob
}

#[test]
fn ikconfig_extracts_hz_1000() {
    let blob = make_ikconfig_blob("CONFIG_HZ=1000\nCONFIG_PREEMPT=y\n");
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-1000");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, &blob).unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), Some(1000));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ikconfig_extracts_hz_250() {
    let blob = make_ikconfig_blob("CONFIG_HZ=250\n");
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-250");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, &blob).unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), Some(250));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ikconfig_no_marker_returns_none() {
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-none");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, b"no marker here").unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ikconfig_missing_config_hz_returns_none() {
    let blob = make_ikconfig_blob("CONFIG_PREEMPT=y\n");
    let dir = std::env::temp_dir().join("ktstr-ikconfig-test-nohz");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vmlinux");
    std::fs::write(&path, &blob).unwrap();
    assert_eq!(read_hz_from_ikconfig(&path), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn empty_samples_default_summary() {
    let summary = MonitorSummary::from_samples(&[]);
    assert_eq!(summary.total_samples, 0);
    assert_eq!(summary.max_imbalance_ratio, 0.0);
    assert_eq!(summary.max_local_dsq_depth, 0);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn single_sample_imbalanced_cpus() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                local_dsq_depth: 3,
                rq_clock: 1000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 4,
                local_dsq_depth: 1,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    assert!((summary.max_imbalance_ratio - 4.0).abs() < f64::EPSILON);
    assert_eq!(summary.max_local_dsq_depth, 3);
    assert!(!summary.stuck_detected);
    // avg fields: single sample with cpus [nr_running=1, nr_running=4]
    assert!((summary.avg_imbalance_ratio - 4.0).abs() < f64::EPSILON);
    assert!((summary.avg_nr_running - 2.5).abs() < f64::EPSILON);
    assert!((summary.avg_local_dsq_depth - 2.0).abs() < f64::EPSILON);
}

#[test]
fn stuck_detected_when_clock_stuck() {
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 6000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000, // stuck
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    assert!(summary.stuck_detected);
}

#[test]
fn balanced_cpus_ratio_one() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 50,
        cpus: vec![
            CpuSnapshot {
                nr_running: 3,
                rq_clock: 100,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 3,
                rq_clock: 200,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert!(!summary.stuck_detected);
    assert!((summary.avg_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert!((summary.avg_nr_running - 3.0).abs() < f64::EPSILON);
    assert!((summary.avg_local_dsq_depth - 0.0).abs() < f64::EPSILON);
}

#[test]
fn single_cpu_no_division_by_zero() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![CpuSnapshot {
            nr_running: 5,
            local_dsq_depth: 2,
            rq_clock: 1000,
            ..Default::default()
        }],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    // Single CPU: min == max, ratio = 1.0
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert_eq!(summary.max_local_dsq_depth, 2);
    assert!(!summary.stuck_detected);
}

#[test]
fn all_zero_snapshots() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    // nr_running=0 for all CPUs: max/max(min,1) = 0/1 = 0.0, but
    // initial max_imbalance_ratio is 1.0 and 0.0 < 1.0, so stays 1.0.
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert_eq!(summary.max_local_dsq_depth, 0);
    // rq_clock=0 is excluded from stall detection
    assert!(!summary.stuck_detected);
    // avg: valid sample with 2 all-zero CPUs
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn empty_cpus_in_sample() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    // Empty cpus slice is skipped via `continue`
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    // avg: sample skipped (empty cpus), no valid readings
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn min_nr_zero_division_guard() {
    // All CPUs have nr_running=0. The code uses min_nr.max(1) as
    // divisor, so ratio = 0/1 = 0.0, which is < initial 1.0.
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 100,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 200,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    // Should not panic from division by zero.
    // max_imbalance_ratio stays at initial 1.0 since 0/1=0 < 1.0.
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
}

#[test]
fn min_nr_zero_max_nr_nonzero() {
    // min_nr=0, max_nr=5: ratio = 5/max(0,1) = 5.0
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 100,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 5,
                rq_clock: 200,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert!((summary.max_imbalance_ratio - 5.0).abs() < f64::EPSILON);
}

#[test]
fn advancing_clocks_no_stuck() {
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1500,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2500,
                ..Default::default()
            },
        ],
    };
    let s3 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 300,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 3000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2, s3]);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.total_samples, 3);
}

#[test]
fn different_length_cpu_vecs() {
    // First sample has 2 CPUs, second has 3. Stall detection uses
    // min(prev.len, curr.len) = 2, so only CPUs 0-1 are compared.
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1500,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2500,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 3000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.total_samples, 2);
    // max_local_dsq_depth comes from all CPUs in all samples.
    assert_eq!(summary.max_local_dsq_depth, 0);
}

// -- MonitorThresholds tests --

fn balanced_sample(elapsed_ms: u64, clock_base: u64) -> MonitorSample {
    MonitorSample {
        prog_stats: None,
        elapsed_ms,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base,
                local_dsq_depth: 3,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base + 100,
                local_dsq_depth: 2,
                ..Default::default()
            },
        ],
    }
}

/// Wire-format pin: `enforce` defaults to `false` (report-only
/// mode). Tests that want hard pass/fail must explicitly set
/// `enforce: true` in their `MonitorThresholds` literal OR use the
/// `Assert` builder's `.with_monitor_defaults()` helper (which sets
/// `enforce_monitor_thresholds = true`).
///
/// Without this canary, a future commit that flips the default back
/// to `enforce: true` (or any other value) would silently re-enable
/// enforcement for all tests using `..Default::default()` or
/// `MonitorThresholds::DEFAULT` — masking real bugs that the
/// report-only mode is designed to surface as warnings instead of
/// failures.
#[test]
fn enforce_defaults_to_false() {
    let t = MonitorThresholds::default();
    assert!(
        !t.enforce,
        "enforce must default to false (report-only mode)"
    );
    let d = MonitorThresholds::DEFAULT;
    assert!(!d.enforce, "DEFAULT.enforce must match default()");
}

#[test]
fn thresholds_default_values() {
    // Regression guard for `MonitorThresholds::DEFAULT`. Every
    // field is asserted: changing a default silently shifts what
    // "passes by default" across every test that inherits
    // defaults via `Assert::default_checks()` + per-scheduler
    // merge. If a default moves, the rationale belongs in the
    // doc comment on `DEFAULT` first; the test failure then
    // prompts the rationale update.
    let t = MonitorThresholds::default();
    assert!(
        (t.max_imbalance_ratio - 4.0).abs() < f64::EPSILON,
        "default max_imbalance_ratio drifted: {}",
        t.max_imbalance_ratio,
    );
    assert_eq!(
        t.max_local_dsq_depth, 50,
        "default max_local_dsq_depth drifted",
    );
    assert!(t.fail_on_stall, "default fail_on_stall drifted");
    assert_eq!(t.sustained_samples, 5, "default sustained_samples drifted");
    assert!(
        (t.max_fallback_rate - 200.0).abs() < f64::EPSILON,
        "default max_fallback_rate drifted: {}",
        t.max_fallback_rate,
    );
    assert!(
        (t.max_keep_last_rate - 100.0).abs() < f64::EPSILON,
        "default max_keep_last_rate drifted: {}",
        t.max_keep_last_rate,
    );
}

#[test]
fn thresholds_default_matches_const() {
    // `Default::default()` and `DEFAULT` must agree — the impl
    // forwards, but the forward is a single expression that a
    // drive-by refactor could break.
    let a = MonitorThresholds::default();
    let b = MonitorThresholds::DEFAULT;
    assert!((a.max_imbalance_ratio - b.max_imbalance_ratio).abs() < f64::EPSILON);
    assert_eq!(a.max_local_dsq_depth, b.max_local_dsq_depth);
    assert_eq!(a.fail_on_stall, b.fail_on_stall);
    assert_eq!(a.sustained_samples, b.sustained_samples);
    assert!((a.max_fallback_rate - b.max_fallback_rate).abs() < f64::EPSILON);
    assert!((a.max_keep_last_rate - b.max_keep_last_rate).abs() < f64::EPSILON);
}

#[test]
fn thresholds_empty_report_passes() {
    let t = MonitorThresholds::default();
    let report = MonitorReport {
        samples: vec![],
        summary: MonitorSummary::default(),
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed);
    assert!(v.details.is_empty());
}

#[test]
fn thresholds_balanced_samples_pass() {
    let t = MonitorThresholds::default();
    let samples: Vec<_> = (0..10)
        .map(|i| balanced_sample(i * 100, 1000 + i * 500))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "balanced samples should pass: {:?}", v.details);
}

#[test]
fn thresholds_imbalance_below_sustained_passes() {
    let t = MonitorThresholds {
        sustained_samples: 5,
        max_imbalance_ratio: 4.0,
        ..Default::default()
    };
    // 4 consecutive imbalanced samples (below sustained_samples=5).
    let mut samples = Vec::new();
    for i in 0..4 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    // Then a balanced one to break the streak.
    samples.push(balanced_sample(400, 3000));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "4 imbalanced < sustained_samples=5: {:?}",
        v.details
    );
}

#[test]
fn thresholds_imbalance_at_sustained_fails() {
    let t = MonitorThresholds {
        sustained_samples: 5,
        max_imbalance_ratio: 4.0,
        enforce: true,
        ..Default::default()
    };
    // 5 consecutive imbalanced samples (ratio=10, threshold=4).
    let mut samples = Vec::new();
    for i in 0..5u64 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed);
    assert!(v.details.iter().any(|d| d.contains("imbalance")));
}

#[test]
fn thresholds_dsq_depth_sustained_fails() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_local_dsq_depth: 10,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let mut samples = Vec::new();
    for i in 0..3u64 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 20,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 5,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed);
    assert!(v.details.iter().any(|d| d.contains("DSQ depth")));
}

#[test]
fn thresholds_dsq_depth_below_sustained_passes() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_local_dsq_depth: 10,
        fail_on_stall: false,
        ..Default::default()
    };
    // Only 2 consecutive DSQ violations, then a clean sample.
    let mut samples = Vec::new();
    for i in 0..2u64 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 20,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 5,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    samples.push(balanced_sample(200, 2000));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "2 DSQ violations < sustained=3: {:?}", v.details);
}

#[test]
fn thresholds_stuck_detected_fails() {
    // Stuck checks use the sustained_samples window. With sustained_samples=1,
    // a single stuck pair triggers failure. `enforce: true` opts out of
    // the report-only default so the violation flips `passed` to false.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                }, // stuck
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed);
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn thresholds_stuck_disabled_passes() {
    let t = MonitorThresholds {
        fail_on_stall: false,
        sustained_samples: 100,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                ..Default::default()
            }],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                }, // stuck but stall check disabled
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "stall disabled should pass: {:?}", v.details);
}

#[test]
fn thresholds_imbalance_interrupted_by_balanced_resets() {
    // 3 imbalanced, 1 balanced, 3 imbalanced — never reaches sustained=5.
    let t = MonitorThresholds {
        sustained_samples: 5,
        max_imbalance_ratio: 4.0,
        fail_on_stall: false,
        ..Default::default()
    };
    let mut samples = Vec::new();
    for i in 0..3u64 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    samples.push(balanced_sample(300, 2500));
    for i in 4..7u64 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 3000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 3100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "interrupted imbalance should pass: {:?}",
        v.details
    );
}

#[test]
fn thresholds_multiple_violations() {
    // Both imbalance and stall in the same report. Both need to
    // reach sustained_samples to trigger. 3 samples = 2 consecutive
    // stall pairs for cpu0 (clock stuck at 1000), 2 consecutive
    // imbalance violations (ratio=5.0 > 2.0).
    let t = MonitorThresholds {
        sustained_samples: 2,
        max_imbalance_ratio: 2.0,
        fail_on_stall: true,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                }, // stall + imbalance
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 300,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                }, // stall continues
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 4000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed);
    assert!(v.details.iter().any(|d| d.contains("imbalance")));
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn thresholds_empty_cpus_samples_pass() {
    let t = MonitorThresholds::default();
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed);
}

#[test]
fn thresholds_uninitialized_memory_passes() {
    // Simulates what happens when monitor reads guest memory before
    // kernel initialization: all rq_clocks identical, DSQ depths garbage.
    let t = MonitorThresholds::default();
    let garbage_clock = 10314579376562252011u64;
    let samples: Vec<_> = (0..10)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: garbage_clock,
                    local_dsq_depth: 1550435906,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: garbage_clock,
                    local_dsq_depth: 1550435906,
                    ..Default::default()
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "uninitialized guest memory should be skipped: {:?}",
        v.details
    );
}

#[test]
fn thresholds_all_same_clocks_passes() {
    // All clocks identical across all CPUs and samples = uninitialized.
    let t = MonitorThresholds {
        fail_on_stall: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "all-same clocks should be treated as uninitialized: {:?}",
        v.details
    );
}

#[test]
fn thresholds_dsq_over_plausibility_ceiling_passes() {
    let t = MonitorThresholds::default();
    let samples = vec![MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1000,
                local_dsq_depth: 50000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                local_dsq_depth: 5,
                ..Default::default()
            },
        ],
    }];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "implausible DSQ depth should skip evaluation: {:?}",
        v.details
    );
}

#[test]
fn thresholds_single_cpu_single_sample_valid() {
    // A single reading cannot be compared, so all_clocks_same with
    // total_readings=1 should still be treated as valid.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 5000,
            ..Default::default()
        }],
    }];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "single reading should be valid: {:?}", v.details);
}

// -- Event counter rate threshold tests --

/// Build a sample with event counters. Each CPU gets the same counter
/// values so the total across CPUs = ncpus * per_cpu_value.
fn sample_with_events(
    elapsed_ms: u64,
    clock_base: u64,
    fallback: i64,
    keep_last: i64,
) -> MonitorSample {
    MonitorSample {
        prog_stats: None,
        elapsed_ms,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base,
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: fallback,
                    dispatch_keep_last: keep_last,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base + 100,
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: fallback,
                    dispatch_keep_last: keep_last,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
    }
}

#[test]
fn thresholds_fallback_rate_sustained_fails() {
    // sustained_samples=3, max_fallback_rate=10.0.
    // 100ms intervals, 2 CPUs. Each CPU increments fallback by 10
    // per sample -> delta = 20 total per interval / 0.1s = 200/s > 10.
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_fallback_rate: 10.0,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..4)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed);
    assert!(v.details.iter().any(|d| d.contains("fallback rate")));
}

#[test]
fn thresholds_fallback_rate_below_sustained_passes() {
    // 2 violating intervals then a clean one — below sustained=3.
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_fallback_rate: 10.0,
        fail_on_stall: false,
        ..Default::default()
    };
    let mut samples: Vec<_> = (0..3)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
        .collect();
    // 4th sample: same fallback as 3rd -> rate = 0.
    samples.push(sample_with_events(300, 2500, 20, 0));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
}

#[test]
fn thresholds_keep_last_rate_sustained_fails() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_keep_last_rate: 10.0,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..4)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed);
    assert!(v.details.iter().any(|d| d.contains("keep_last rate")));
}

#[test]
fn thresholds_keep_last_rate_below_sustained_passes() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_keep_last_rate: 10.0,
        fail_on_stall: false,
        ..Default::default()
    };
    let mut samples: Vec<_> = (0..3)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
        .collect();
    // Reset: same keep_last as previous -> rate = 0.
    samples.push(sample_with_events(300, 2500, 0, 20));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
}

#[test]
fn thresholds_event_rate_interrupted_resets() {
    // 2 violating intervals, 1 clean, 2 violating — never reaches sustained=3.
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_fallback_rate: 10.0,
        fail_on_stall: false,
        ..Default::default()
    };
    let mut samples = Vec::new();
    // 3 samples = 2 intervals of high fallback rate.
    for i in 0..3u64 {
        samples.push(sample_with_events(
            i * 100,
            1000 + i * 500,
            i as i64 * 10,
            0,
        ));
    }
    // Clean interval: same fallback -> rate = 0.
    samples.push(sample_with_events(300, 2500, 20, 0));
    // 3 more samples = 2 intervals of high fallback rate (not 3).
    // The fallback delta for the first interval covers sample 3->4,
    // which is (30-20)/0.1 = 100/s (violating), then 4->5 is also
    // violating. That's 2 intervals, below sustained=3.
    for i in 0..2u64 {
        samples.push(sample_with_events(
            400 + i * 100,
            3000 + i * 500,
            30 + (i + 1) as i64 * 10,
            0,
        ));
    }
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "interrupted rate violations should pass: {:?}",
        v.details
    );
}

#[test]
fn thresholds_no_event_counters_skips_rate_check() {
    // Samples without event counters should not trigger rate violations.
    let t = MonitorThresholds {
        sustained_samples: 1,
        max_fallback_rate: 0.0, // any rate would fail
        max_keep_last_rate: 0.0,
        fail_on_stall: false,
        ..Default::default()
    };
    let samples: Vec<_> = (0..5)
        .map(|i| balanced_sample(i * 100, 1000 + i * 500))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "no event counters should skip rate check: {:?}",
        v.details
    );
}

#[test]
fn thresholds_default_event_rate_values() {
    let t = MonitorThresholds::default();
    assert!((t.max_fallback_rate - 200.0).abs() < f64::EPSILON);
    assert!((t.max_keep_last_rate - 100.0).abs() < f64::EPSILON);
}

#[test]
fn summary_keep_last_rate_computed() {
    // 2 CPUs, each with keep_last incrementing by 5 per sample.
    // 3 samples over 200ms -> total delta = 2*10 = 20, rate = 20/0.2 = 100.
    let samples = vec![
        sample_with_events(0, 1000, 0, 0),
        sample_with_events(100, 1500, 0, 5),
        sample_with_events(200, 2000, 0, 10),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    assert!((deltas.keep_last_rate - 100.0).abs() < f64::EPSILON);
}

// -- compute_event_deltas edge cases --

#[test]
fn event_deltas_none_without_counters() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.event_deltas.is_none());
}

#[test]
fn event_deltas_single_sample() {
    // Only one sample with events -> first == last, duration=0, rates=0.
    let samples = vec![sample_with_events(100, 1000, 50, 25)];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    assert_eq!(deltas.fallback_rate, 0.0);
    assert_eq!(deltas.keep_last_rate, 0.0);
}

#[test]
fn event_deltas_max_fallback_burst() {
    // 3 samples: burst between samples 1 and 2.
    let samples = vec![
        sample_with_events(0, 1000, 0, 0),
        sample_with_events(100, 1500, 5, 0),
        sample_with_events(200, 2000, 100, 0),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    // Per-CPU: burst is (100-5)*2 = 190 across 2 CPUs.
    assert!(deltas.max_fallback_burst > 0);
}

#[test]
fn event_deltas_counter_reset_clamps_to_zero() {
    // A scheduler restart between samples resets the per-CPU
    // counters to smaller (or zero) values. The raw delta
    // `last - first` is then negative — which would flow through
    // as a negative fallback_rate / negative total. Clamp to zero
    // so the downstream rate is sane.
    //
    // Sample 0 at t=0ms has high counters (pre-restart).
    // Sample 1 at t=1000ms has low counters (post-restart).
    let samples = vec![
        sample_with_events(0, 1000, 1000, 500),
        sample_with_events(1000, 2000, 5, 2),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    assert!(
        deltas.total_fallback >= 0,
        "reset must not produce negative total_fallback, got {}",
        deltas.total_fallback
    );
    assert!(
        deltas.fallback_rate >= 0.0,
        "reset must not produce negative fallback_rate, got {}",
        deltas.fallback_rate
    );
    assert!(
        deltas.total_dispatch_keep_last >= 0,
        "reset must not produce negative keep_last total, got {}",
        deltas.total_dispatch_keep_last
    );
    assert!(
        deltas.keep_last_rate >= 0.0,
        "reset must not produce negative keep_last_rate, got {}",
        deltas.keep_last_rate
    );
}

#[test]
fn event_deltas_all_counters_computed() {
    let make = |elapsed_ms, fb, kl, dsq_off, exit, migdis| MonitorSample {
        prog_stats: None,
        elapsed_ms,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: elapsed_ms * 10,
            event_counters: Some(ScxEventCounters {
                select_cpu_fallback: fb,
                dispatch_local_dsq_offline: dsq_off,
                dispatch_keep_last: kl,
                enq_skip_exiting: exit,
                enq_skip_migration_disabled: migdis,
                ..Default::default()
            }),
            ..Default::default()
        }],
    };
    let samples = vec![
        make(100, 10, 20, 30, 40, 50),
        make(200, 110, 120, 130, 140, 150),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.event_deltas.unwrap();
    assert_eq!(d.total_fallback, 100);
    assert_eq!(d.total_dispatch_keep_last, 100);
    assert_eq!(d.total_dispatch_offline, 100);
    assert_eq!(d.total_enq_skip_exiting, 100);
    assert_eq!(d.total_enq_skip_migration_disabled, 100);
}

// -- data_looks_valid tests --

#[test]
fn data_looks_valid_empty() {
    assert!(MonitorThresholds::data_looks_valid(&[]));
}

#[test]
fn data_looks_valid_normal() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 2000)];
    assert!(MonitorThresholds::data_looks_valid(&samples));
}

#[test]
fn data_looks_valid_all_same_clocks() {
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
            ],
        },
    ];
    assert!(!MonitorThresholds::data_looks_valid(&samples));
}

#[test]
fn data_looks_valid_dsq_over_ceiling() {
    let samples = vec![MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: 50000,
            rq_clock: 1000,
            ..Default::default()
        }],
    }];
    assert!(!MonitorThresholds::data_looks_valid(&samples));
}

// -- MonitorSample::imbalance_ratio tests --

#[test]
fn imbalance_ratio_empty_cpus() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![],
    };
    assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_single_cpu() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot {
            nr_running: 5,
            ..Default::default()
        }],
    };
    assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_balanced() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                nr_running: 3,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 3,
                ..Default::default()
            },
        ],
    };
    assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_imbalanced() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 8,
                ..Default::default()
            },
        ],
    };
    assert!((s.imbalance_ratio() - 4.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_zero_min() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 5,
                ..Default::default()
            },
        ],
    };
    // min=0, max(0,1)=1, ratio=5/1=5.0
    assert!((s.imbalance_ratio() - 5.0).abs() < f64::EPSILON);
}

// -- MonitorSample::sum_event_field tests --

#[test]
fn sum_event_field_none_when_no_counters() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
    };
    assert!(s.sum_event_field(|e| e.select_cpu_fallback).is_none());
}

#[test]
fn sum_event_field_sums_across_cpus() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: 10,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot {
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: 20,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
    };
    assert_eq!(s.sum_event_field(|e| e.select_cpu_fallback), Some(30));
}

#[test]
fn sum_event_field_mixed_some_none() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                event_counters: Some(ScxEventCounters {
                    dispatch_keep_last: 7,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot::default(),
        ],
    };
    assert_eq!(s.sum_event_field(|e| e.dispatch_keep_last), Some(7));
}

// -- sample_looks_valid tests --

#[test]
fn sample_looks_valid_normal() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: 5,
            ..Default::default()
        }],
    };
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_at_ceiling() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: DSQ_PLAUSIBILITY_CEILING,
            ..Default::default()
        }],
    };
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_over_ceiling() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: DSQ_PLAUSIBILITY_CEILING + 1,
            ..Default::default()
        }],
    };
    assert!(!sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_empty_cpus() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![],
    };
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_zero_initialized() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
    };
    // All fields zero, local_dsq_depth=0 <= DSQ_PLAUSIBILITY_CEILING.
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_multiple_cpus_one_over() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                local_dsq_depth: 5,
                ..Default::default()
            },
            CpuSnapshot {
                local_dsq_depth: DSQ_PLAUSIBILITY_CEILING + 1,
                ..Default::default()
            },
        ],
    };
    // One CPU over ceiling invalidates the entire sample.
    assert!(!sample_looks_valid(&s));
}

// -- MonitorSummary field value assertions --

#[test]
fn from_samples_fields_sane_values() {
    let samples: Vec<_> = (0..5u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: (i as u32 + 1),
                    scx_nr_running: i as u32,
                    local_dsq_depth: (i as u32) % 3,
                    rq_clock: 1000 + i * 500,
                    scx_flags: 0,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: i as i64 * 2,
                        dispatch_keep_last: i as i64,
                        ..Default::default()
                    }),
                    schedstat: None,
                    vcpu_cpu_time_ns: None,
                    vcpu_perf: None,
                    sched_domains: None,
                },
                CpuSnapshot {
                    nr_running: (i as u32 + 2),
                    scx_nr_running: i as u32 + 1,
                    local_dsq_depth: 0,
                    rq_clock: 1100 + i * 600,
                    scx_flags: 0,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: i as i64 * 3,
                        dispatch_keep_last: i as i64 * 2,
                        ..Default::default()
                    }),
                    schedstat: None,
                    vcpu_cpu_time_ns: None,
                    vcpu_perf: None,
                    sched_domains: None,
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    // total_samples matches input count
    assert_eq!(summary.total_samples, 5);
    // max_imbalance_ratio: all samples have nr_running differing by 1,
    // worst case is sample 0: nr_running=[1,2] -> ratio=2.0
    assert!(
        summary.max_imbalance_ratio >= 1.0,
        "ratio must be >= 1.0: {}",
        summary.max_imbalance_ratio
    );
    assert!(
        summary.max_imbalance_ratio <= 10.0,
        "ratio must be reasonable: {}",
        summary.max_imbalance_ratio
    );
    // max_local_dsq_depth: worst is (4 % 3) = 1 on cpu0 at i=4, or (3 % 3)=0 at i=3, (2%3)=2 at i=2
    assert!(
        summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
        "dsq depth must be below plausibility ceiling: {}",
        summary.max_local_dsq_depth
    );
    assert!(
        summary.max_local_dsq_depth <= 10,
        "dsq depth must be small in this controlled test: {}",
        summary.max_local_dsq_depth
    );
    // stuck_detected: rq_clock advances each sample, so no stuck
    assert!(
        !summary.stuck_detected,
        "no stuck expected with advancing rq_clock"
    );
    // event_deltas: should be computed
    let deltas = summary
        .event_deltas
        .as_ref()
        .expect("event deltas must be present");
    assert!(
        deltas.total_fallback >= 0,
        "fallback count must be non-negative"
    );
    assert!(
        deltas.total_dispatch_keep_last >= 0,
        "keep_last count must be non-negative"
    );
    assert!(
        deltas.fallback_rate >= 0.0,
        "fallback rate must be non-negative"
    );
    assert!(
        deltas.keep_last_rate >= 0.0,
        "keep_last rate must be non-negative"
    );
    // avg fields: must be positive with non-zero nr_running input
    assert!(
        summary.avg_imbalance_ratio >= 1.0,
        "avg imbalance must be >= 1.0: {}",
        summary.avg_imbalance_ratio,
    );
    assert!(
        summary.avg_nr_running > 0.0,
        "avg nr_running must be positive: {}",
        summary.avg_nr_running,
    );
    assert!(
        summary.avg_local_dsq_depth >= 0.0,
        "avg dsq_depth must be non-negative: {}",
        summary.avg_local_dsq_depth,
    );
}

#[test]
fn from_samples_empty_all_defaults() {
    // Check that every field of MonitorSummary defaults correctly for empty input,
    // including event_deltas which empty_samples_default_summary does not check.
    let summary = MonitorSummary::from_samples(&[]);
    assert_eq!(summary.total_samples, 0);
    assert_eq!(summary.max_imbalance_ratio, 0.0);
    assert_eq!(summary.max_local_dsq_depth, 0);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
    assert!(
        summary.event_deltas.is_none(),
        "empty input must not produce event deltas"
    );
}

// ---------------------------------------------------------------
// Negative tests: check monitor diagnostics catch controlled failures
// ---------------------------------------------------------------

#[test]
fn neg_tight_imbalance_threshold_catches_mild_imbalance() {
    let t = MonitorThresholds {
        max_imbalance_ratio: 1.0,
        sustained_samples: 2,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 3,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        summary.max_imbalance_ratio >= 1.5,
        "summary must capture ratio"
    );
    assert!(!summary.stuck_detected, "no stall in this scenario");
    assert_eq!(summary.total_samples, 3);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "imbalance=1.5 must fail threshold=1.0");
    // Format: "imbalance ratio 1.5 exceeded threshold 1.0 for 2 consecutive samples (ending at sample 2)"
    let detail = v.details.iter().find(|d| d.contains("imbalance")).unwrap();
    assert!(detail.contains("ratio"), "must include 'ratio': {detail}");
    assert!(
        detail.contains("exceeded threshold"),
        "must include threshold: {detail}"
    );
    assert!(
        detail.contains("1.0"),
        "must show threshold value: {detail}"
    );
    assert!(
        detail.contains("consecutive samples"),
        "must show sustained count: {detail}"
    );
    assert!(
        detail.contains("ending at sample"),
        "must show sample index: {detail}"
    );
    assert!(
        v.summary.contains("FAILED"),
        "summary must say FAILED: {}",
        v.summary
    );
}

#[test]
fn neg_tight_dsq_threshold_catches_small_depth() {
    let t = MonitorThresholds {
        max_local_dsq_depth: 1,
        sustained_samples: 2,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    local_dsq_depth: 3,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    local_dsq_depth: 0,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.max_local_dsq_depth, 3,
        "summary must capture max depth"
    );
    assert!(
        summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
        "depth must be plausible"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "dsq_depth=3 must fail threshold=1");
    // Format: "local DSQ depth 3 on cpu0 exceeded threshold 1 for 2 consecutive samples (ending at sample 2)"
    let detail = v.details.iter().find(|d| d.contains("DSQ depth")).unwrap();
    assert!(detail.contains("3"), "must show depth value: {detail}");
    assert!(detail.contains("cpu0"), "must show CPU number: {detail}");
    assert!(
        detail.contains("threshold 1"),
        "must show threshold: {detail}"
    );
    assert!(
        detail.contains("consecutive samples"),
        "must show count: {detail}"
    );
}

#[test]
fn neg_stuck_detection_catches_frozen_rq_clock() {
    // Stuck checks use sustained_samples window. sustained_samples=1 means
    // a single stuck pair triggers failure.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        summary.stuck_detected,
        "summary.stuck_detected must be true"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "frozen rq_clock must be detected");
    let detail = v
        .details
        .iter()
        .find(|d| d.contains("rq_clock stall"))
        .unwrap();
    assert!(detail.contains("cpu0"), "must name frozen CPU: {detail}");
    assert!(
        detail.contains("consecutive samples"),
        "must show sustained count: {detail}"
    );
    assert!(
        detail.contains("clock=5000"),
        "must include frozen clock value: {detail}"
    );
}

#[test]
fn neg_combined_imbalance_and_stuck_both_reported() {
    let t = MonitorThresholds {
        max_imbalance_ratio: 2.0,
        sustained_samples: 1,
        fail_on_stall: true,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.stuck_detected);
    assert!(summary.max_imbalance_ratio >= 10.0);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed);
    let imb = v.details.iter().find(|d| d.contains("imbalance")).unwrap();
    assert!(
        imb.contains("exceeded threshold 2.0"),
        "imbalance format: {imb}"
    );
    let stall = v
        .details
        .iter()
        .find(|d| d.contains("rq_clock stall"))
        .unwrap();
    assert!(stall.contains("cpu0"), "stall format: {stall}");
    assert!(
        v.details.len() >= 2,
        "both violations must be reported, got {}",
        v.details.len()
    );
    assert!(v.summary.contains("FAILED"), "summary: {}", v.summary);
}

#[test]
fn stuck_idle_cpu_exempt() {
    // nr_running==0 on both samples: idle CPU, NOHZ tick stopped.
    // rq_clock not advancing is expected, not a stall.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000, // stuck but idle
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        !summary.stuck_detected,
        "idle CPU should not trigger stall in summary"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "idle CPU should not trigger stall: {:?}",
        v.details
    );
}

#[test]
fn stuck_idle_to_busy_not_exempt() {
    // nr_running transitions from 0 to 1 — the CPU woke up but
    // rq_clock didn't advance. This IS a stall (the CPU is now
    // busy but the scheduler tick hasn't fired).
    // Second CPU has a different clock value so data_looks_valid passes.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck, but now busy
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        summary.stuck_detected,
        "busy CPU with frozen clock is a stall"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        !v.passed,
        "busy CPU with frozen clock must fail: {:?}",
        v.details
    );
}

#[test]
fn stuck_sustained_window_filters_transient() {
    // With sustained_samples=3, a 2-sample stall doesn't trigger.
    // Second CPU has a different clock value so data_looks_valid passes.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 3,
        ..Default::default()
    };
    let mut samples = Vec::new();
    // 3 samples: 2 consecutive stall pairs for cpu0, then clock advances.
    for i in 0..3u64 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck for all 3
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000 + i * 500, // advancing
                    ..Default::default()
                },
            ],
        });
    }
    // Break the streak: clock advances in 4th sample.
    samples.push(MonitorSample {
        prog_stats: None,
        elapsed_ms: 300,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 6000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7500,
                ..Default::default()
            },
        ],
    });
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    // 2 consecutive stall pairs < sustained_samples=3
    assert!(v.passed, "2 stall pairs < sustained=3: {:?}", v.details);
}

#[test]
fn stuck_sustained_window_catches_real_stuck() {
    // With sustained_samples=3, 3+ consecutive stall pairs trigger.
    // Second CPU has a different clock value so data_looks_valid passes.
    // `enforce: true` opts the verdict out of report-only mode (the
    // default), so the recorded stall violation flips `passed` to
    // false.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 3,
        enforce: true,
        ..Default::default()
    };
    // 4 samples = 3 consecutive stall pairs for cpu0. cpu1 advances.
    let samples: Vec<_> = (0..4u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000 + i * 500, // advancing
                    ..Default::default()
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "3 consecutive stall pairs must fail");
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn from_samples_idle_cpu_no_stuck() {
    // from_samples should not flag stall when both samples have
    // nr_running==0 on the stuck CPU.
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 5000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 6000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 5000, // stuck but idle
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    assert!(!summary.stuck_detected);
}

#[test]
fn stuck_below_sustained_passes() {
    // 1 stall pair with sustained_samples=5 should pass.
    // Second CPU has a different clock value so data_looks_valid passes.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 5,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
        // Clock recovers.
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 300,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 8000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "1 stall < sustained=5: {:?}", v.details);
}

#[test]
fn neg_fallback_rate_threshold_fires() {
    let t = MonitorThresholds {
        sustained_samples: 2,
        max_fallback_rate: 5.0,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        summary.event_deltas.is_some(),
        "event deltas must be computed"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "fallback rate must be caught");
    // Format: "fallback rate 200.0/s exceeded threshold 5.0/s for 2 consecutive intervals (ending at sample 2)"
    let detail = v
        .details
        .iter()
        .find(|d| d.contains("fallback rate"))
        .unwrap();
    assert!(detail.contains("/s"), "must include rate unit: {detail}");
    assert!(
        detail.contains("exceeded threshold"),
        "must state threshold: {detail}"
    );
    assert!(
        detail.contains("5.0/s"),
        "must show threshold value: {detail}"
    );
    assert!(
        detail.contains("consecutive intervals"),
        "must show sustained count: {detail}"
    );
}

#[test]
fn neg_keep_last_rate_threshold_fires() {
    let t = MonitorThresholds {
        sustained_samples: 2,
        max_keep_last_rate: 5.0,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.event_deltas.is_some());
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "keep_last rate must be caught");
    // Format: "keep_last rate .../s exceeded threshold 5.0/s for 2 consecutive intervals ..."
    let detail = v
        .details
        .iter()
        .find(|d| d.contains("keep_last rate"))
        .unwrap();
    assert!(detail.contains("/s"), "must include rate unit: {detail}");
    assert!(
        detail.contains("exceeded threshold"),
        "must state threshold: {detail}"
    );
    assert!(
        detail.contains("5.0/s"),
        "must show threshold value: {detail}"
    );
}

// -- vCPU CPU time gating tests --

#[test]
fn evaluate_suppresses_stuck_when_vcpu_preempted() {
    // vcpu_cpu_time_ns shows < threshold advancement -> vCPU was
    // preempted, stall should be suppressed. Use explicit threshold
    // (10ms) to avoid host CONFIG_HZ dependency.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,                        // stuck
                    vcpu_cpu_time_ns: Some(1_000_500_000), // 0.5ms < 10ms threshold
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    vcpu_cpu_time_ns: Some(1_010_000_000),
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples_with_threshold(&samples, 10_000_000);
    assert!(
        !summary.stuck_detected,
        "preempted vCPU should not flag stall in summary"
    );
    let report = MonitorReport {
        samples,
        summary,
        preemption_threshold_ns: 10_000_000,
        watchdog_observation: None,
        page_offset: 0,
    };
    let v = t.evaluate(&report);
    assert!(
        v.passed,
        "preempted vCPU should suppress stall: {:?}",
        v.details
    );
}

#[test]
fn evaluate_catches_stuck_when_vcpu_running() {
    // vcpu_cpu_time_ns shows advancement >= threshold -> vCPU was
    // running, stall is real. Use explicit threshold (10ms) to avoid
    // host CONFIG_HZ dependency (DEFAULT_HZ=250 gives 40ms threshold,
    // which would mask the 10ms advance).
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,                        // stuck
                    vcpu_cpu_time_ns: Some(1_010_000_000), // 10ms advance
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    vcpu_cpu_time_ns: Some(1_010_000_000),
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples_with_threshold(&samples, 10_000_000);
    assert!(
        summary.stuck_detected,
        "running vCPU with stuck clock is a stall"
    );
    let report = MonitorReport {
        samples,
        summary,
        preemption_threshold_ns: 10_000_000,
        watchdog_observation: None,
        page_offset: 0,
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "running vCPU stall must fail: {:?}", v.details);
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn evaluate_stuck_none_vcpu_time_falls_back_to_current_behavior() {
    // vcpu_cpu_time_ns is None -> assume vCPU was running (don't suppress).
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck, no vcpu_cpu_time_ns
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        summary.stuck_detected,
        "None vcpu time should not suppress stall"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        !v.passed,
        "None vcpu time should detect stall: {:?}",
        v.details
    );
}

#[test]
fn from_samples_suppresses_stuck_when_vcpu_preempted() {
    // from_samples_with_threshold should respect vcpu_cpu_time_ns
    // gating. Use explicit threshold to avoid host CONFIG_HZ dependency.
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                vcpu_cpu_time_ns: Some(1_000_000_000),
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 6000,
                vcpu_cpu_time_ns: Some(1_000_000_000),
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,                        // stuck
                vcpu_cpu_time_ns: Some(1_000_100_000), // 0.1ms < 10ms threshold
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7000,
                vcpu_cpu_time_ns: Some(1_010_000_000),
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples_with_threshold(&[s1, s2], 10_000_000);
    assert!(
        !summary.stuck_detected,
        "preempted vCPU should not flag stall"
    );
}

// -- SchedstatDeltas tests --

fn sample_with_schedstat(
    elapsed_ms: u64,
    clock_base: u64,
    run_delay: u64,
    pcount: u64,
    sched_count: u32,
    ttwu_count: u32,
) -> MonitorSample {
    MonitorSample {
        prog_stats: None,
        elapsed_ms,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base,
                schedstat: Some(RqSchedstat {
                    run_delay,
                    pcount,
                    sched_count,
                    ttwu_count,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base + 100,
                schedstat: Some(RqSchedstat {
                    run_delay,
                    pcount,
                    sched_count,
                    ttwu_count,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
    }
}

#[test]
fn schedstat_deltas_computed_from_samples() {
    // 2 CPUs, each starting at run_delay=1000, ending at 5000.
    // Total delta = 2 * (5000 - 1000) = 8000.
    let samples = vec![
        sample_with_schedstat(0, 1000, 1000, 10, 50, 30),
        sample_with_schedstat(1000, 2000, 5000, 20, 100, 60),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    assert_eq!(d.total_run_delay, 8000);
    assert_eq!(d.total_pcount, 20);
    assert_eq!(d.total_sched_count, 100);
    assert_eq!(d.total_ttwu_count, 60);
    // Rate: 8000 ns / 1.0 s = 8000.0 ns/s.
    assert!((d.run_delay_rate - 8000.0).abs() < f64::EPSILON);
    assert!((d.sched_count_rate - 100.0).abs() < f64::EPSILON);
}

#[test]
fn schedstat_deltas_none_without_schedstat() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.schedstat_deltas.is_none());
}

#[test]
fn schedstat_deltas_single_sample() {
    // Single sample -> first == last, duration=0, rates=0.
    let samples = vec![sample_with_schedstat(100, 1000, 5000, 10, 50, 30)];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    assert_eq!(d.run_delay_rate, 0.0);
    assert_eq!(d.sched_count_rate, 0.0);
    assert_eq!(d.total_run_delay, 0);
}

#[test]
fn schedstat_deltas_rates() {
    // 1 CPU, 500ms window. run_delay increases by 2000, sched_count by 40.
    // run_delay_rate = 2000 / 0.5 = 4000.0 ns/s.
    // sched_count_rate = 40 / 0.5 = 80.0 /s.
    let samples = vec![
        sample_with_schedstat(0, 1000, 1000, 5, 10, 20),
        sample_with_schedstat(500, 2000, 3000, 15, 50, 40),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    // 2 CPUs, each delta = 2000, total = 4000.
    assert_eq!(d.total_run_delay, 4000);
    // rate = 4000 / 0.5s = 8000.0
    assert!((d.run_delay_rate - 8000.0).abs() < f64::EPSILON);
    // 2 CPUs, each sched_count delta = 40, total = 80.
    assert_eq!(d.total_sched_count, 80);
    // rate = 80 / 0.5s = 160.0
    assert!((d.sched_count_rate - 160.0).abs() < f64::EPSILON);
}

#[test]
fn schedstat_deltas_all_fields() {
    let make = |elapsed_ms, rd, pc, yc, sc, sg, tc, tl| MonitorSample {
        prog_stats: None,
        elapsed_ms,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: elapsed_ms * 10,
            schedstat: Some(RqSchedstat {
                run_delay: rd,
                pcount: pc,
                yld_count: yc,
                sched_count: sc,
                sched_goidle: sg,
                ttwu_count: tc,
                ttwu_local: tl,
            }),
            ..Default::default()
        }],
    };
    let samples = vec![
        make(100, 100, 10, 1, 20, 5, 30, 15),
        make(200, 500, 25, 4, 50, 12, 70, 35),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    assert_eq!(d.total_run_delay, 400);
    assert_eq!(d.total_pcount, 15);
    assert_eq!(d.total_yld_count, 3);
    assert_eq!(d.total_sched_count, 30);
    assert_eq!(d.total_sched_goidle, 7);
    assert_eq!(d.total_ttwu_count, 40);
    assert_eq!(d.total_ttwu_local, 20);
}

// -- SustainedViolationTracker direct tests --

#[test]
fn sustained_tracker_no_violations() {
    let t = SustainedViolationTracker::default();
    assert!(!t.sustained(3));
    assert_eq!(t.worst_run, 0);
}

#[test]
fn sustained_tracker_single_violation_not_sustained() {
    let mut t = SustainedViolationTracker::default();
    t.record(true, 5.0, 0);
    assert!(!t.sustained(3));
    assert_eq!(t.worst_run, 1);
    assert_eq!(t.worst_at, 0);
    assert!((t.worst_value - 5.0).abs() < f64::EPSILON);
}

#[test]
fn sustained_tracker_meets_threshold() {
    let mut t = SustainedViolationTracker::default();
    t.record(true, 2.0, 0);
    t.record(true, 3.0, 1);
    t.record(true, 4.0, 2);
    assert!(t.sustained(3));
    assert_eq!(t.worst_run, 3);
    assert_eq!(t.worst_at, 2);
    assert!((t.worst_value - 4.0).abs() < f64::EPSILON);
}

#[test]
fn sustained_tracker_reset_on_non_violation() {
    let mut t = SustainedViolationTracker::default();
    t.record(true, 1.0, 0);
    t.record(true, 2.0, 1);
    t.record(false, 0.0, 2); // reset
    t.record(true, 3.0, 3);
    assert!(!t.sustained(3));
    assert_eq!(t.worst_run, 2); // longest consecutive run was 2
    assert_eq!(t.consecutive, 1); // current run is 1
}

#[test]
fn sustained_tracker_worst_run_preserved_after_reset() {
    let mut t = SustainedViolationTracker::default();
    for i in 0..5 {
        t.record(true, i as f64, i);
    }
    t.record(false, 0.0, 5);
    t.record(true, 99.0, 6);
    t.record(true, 100.0, 7);
    // Worst run is 5 from the first sequence.
    assert_eq!(t.worst_run, 5);
    assert!(t.sustained(5));
    assert!(!t.sustained(6));
}
