//! Host-side VM result evaluation for `#[ktstr_test]` runs.
//!
//! The core [`run_ktstr_test_inner`] orchestrates a single test run:
//! boot the guest VM with the scheduler and workload, collect profraw
//! + stimulus events from SHM, then hand off to [`evaluate_vm_result`]
//!   for pass/fail judgment and error-message construction.
//!
//! [`evaluate_vm_result`] is factored out of the VM-boot path so error
//! formatting can be unit-tested with synthetic `VmResult` values.
//!
//! Supporting items:
//! - [`resolve_scheduler`] / [`resolve_test_kernel`] locate the
//!   scheduler binary and kernel image from env + cache + filesystem.
//! - [`nextest_setup`] is the `setup-script` entry point that warms the
//!   SHM initramfs cache before nextest starts running tests.
//! - [`scheduler_label`] formats the `[sched=...]` bracket in error
//!   headers.
//! - [`format_monitor_section`] and [`trim_settle_samples`] handle the
//!   `--- monitor ---` block in failed-test output.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::assert::AssertResult;
use crate::timeline::StimulusEvent;
use crate::vmm;

use super::output::{
    classify_init_stage, extract_kernel_version, extract_panic_message, extract_sched_ext_dump,
    format_console_diagnostics, parse_assert_result, parse_assert_result_shm,
    sched_log_fingerprint,
};
use super::probe::attempt_auto_repro;
use super::profraw::{MSG_TYPE_PROFRAW, write_profraw};
use super::sidecar::{write_sidecar, write_skip_sidecar};
use super::topo::TopoOverride;
use super::{KtstrTestEntry, SchedulerSpec, Topology};
use crate::verifier::{SCHED_OUTPUT_START, parse_sched_output};

use super::runtime::{config_file_parts, verbose};

// ---------------------------------------------------------------------------
// Failure-message constants
// ---------------------------------------------------------------------------
//
// Shared between the production error-formatting paths in this module
// and the tests that pin those messages. Editing a production string
// here without updating the test (or vice versa) is caught at compile
// time instead of as a runtime test assertion drift.

/// Header body for a timed-out run with no parseable AssertResult.
/// Pinned by `eval_timeout_no_result` and `eval_timeout_with_sched_includes_diagnostics`.
pub(crate) const ERR_TIMED_OUT_NO_RESULT: &str = "timed out (no result in SHM or COM2)";

/// Header body for a run whose scenario passed but whose monitor
/// verdict failed. Pinned by `eval_monitor_fail_has_fingerprint` and
/// `eval_monitor_fail_includes_sched_log`.
pub(crate) const ERR_MONITOR_FAILED_AFTER_SCENARIO: &str = "passed scenario but monitor failed";

/// Reason body when a scheduler is running but no AssertResult was
/// received from the guest. Pinned by `eval_sched_dies_no_com2_output`
/// and `eval_sched_dies_with_sched_log`.
pub(crate) const ERR_NO_TEST_RESULT_FROM_GUEST: &str = "no test result received from guest \
     (no AssertResult arrived via SHM or COM2; check kernel log and \
     scheduler exit status)";

/// Reason body when EEVDF (no scheduler) produced no AssertResult.
/// Pinned by `eval_eevdf_no_com2_output` and `eval_payload_exits_no_check_result`.
pub(crate) const ERR_NO_TEST_FUNCTION_OUTPUT: &str =
    "test function produced no output (no test result found)";

/// Prefix for the `guest crashed: ...` reason body. Pinned by
/// `eval_crash_in_output_says_guest_crashed`, `eval_crash_eevdf_says_guest_crashed`,
/// and `eval_crash_message_from_shm`.
pub(crate) const ERR_GUEST_CRASHED_PREFIX: &str = "guest crashed:";

/// Write a skip sidecar for `entry` + `active_flags`, logging to
/// stderr on failure without propagating the error. Used at five
/// sites — the three in [`run_ktstr_test_inner`] (performance_mode
/// gate plus the two `ResourceContention` arms at VM build + VM
/// run) and the two in `super::dispatch` (performance_mode gates
/// at the plain-run and flag-profile entry points) — all of which
/// must record the skip for stats tooling but cannot meaningfully
/// handle a sidecar-write failure beyond logging it. The skip
/// itself is still valid; only post-run stats tooling loses
/// visibility.
pub(crate) fn record_skip_sidecar(entry: &KtstrTestEntry, active_flags: &[String]) {
    if let Err(e) = write_skip_sidecar(entry, active_flags) {
        eprintln!("ktstr_test: {e:#}");
    }
}

/// Run a single ktstr_test and return the VM's AssertResult.
pub(crate) fn run_ktstr_test_inner(
    entry: &KtstrTestEntry,
    topo: Option<&TopoOverride>,
    active_flags: &[String],
) -> Result<AssertResult> {
    entry.validate().context("KtstrTestEntry validation")?;
    if let Some(t) = topo {
        t.validate().context("TopoOverride validation")?;
    }
    if entry.performance_mode && std::env::var("KTSTR_NO_PERF_MODE").is_ok() {
        crate::report::test_skip(format_args!(
            "{}: test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
            entry.name,
        ));
        // Record the skip so stats tooling sees every skipped run,
        // not just the ones that made it to the VM-run site. A sidecar
        // write failure is logged but not propagated: the skip itself
        // is still valid — only post-run stats tooling loses visibility.
        record_skip_sidecar(entry, active_flags);
        return Ok(AssertResult::skip(
            "test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
        ));
    }
    ensure_kvm()?;
    let kernel = resolve_test_kernel()?;
    let scheduler = match entry.scheduler.scheduler_binary() {
        Some(b) => resolve_scheduler(b)?,
        None => None,
    };
    let ktstr_bin = crate::resolve_current_exe()?;

    let guest_args = vec![
        "run".to_string(),
        "--ktstr-test-fn".to_string(),
        entry.name.to_string(),
    ];

    let cmdline_extra = super::runtime::build_cmdline_extra(entry);

    let (vm_topology, memory_mb) = super::runtime::resolve_vm_topology(entry, topo);

    let no_perf_mode = std::env::var("KTSTR_NO_PERF_MODE").is_ok();
    let mut builder = super::runtime::build_vm_builder_base(
        entry,
        &kernel,
        &ktstr_bin,
        scheduler.as_deref(),
        vm_topology,
        memory_mb,
        &cmdline_extra,
        &guest_args,
        no_perf_mode,
    )
    .performance_mode(entry.performance_mode);

    // Merge order: default_checks -> scheduler.assert -> per-test assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(entry.scheduler.assert())
        .merge(&entry.assert);

    if let Some(SchedulerSpec::KernelBuiltin { enable, disable }) =
        entry.scheduler.scheduler_binary()
    {
        builder = builder.sched_enable_cmds(enable);
        builder = builder.sched_disable_cmds(disable);
    }
    if entry.scheduler.has_active_scheduling() {
        builder = builder.monitor_thresholds(merged_assert.monitor_thresholds());
    }

    let mut sched_args: Vec<String> = Vec::new();
    if let Some((archive_path, host_path, guest_path)) = config_file_parts(entry) {
        builder = builder.include_files(vec![(archive_path, host_path)]);
        sched_args.push("--config".to_string());
        sched_args.push(guest_path);
    }
    super::runtime::append_base_sched_args(entry, &mut sched_args);
    for flag_name in active_flags {
        if let Some(args) = entry.scheduler.flag_args(flag_name) {
            sched_args.extend(args.iter().map(|s| s.to_string()));
        }
    }
    if !sched_args.is_empty() {
        builder = builder.sched_args(&sched_args);
    }

    // Catch ResourceContention before .context() wraps it —
    // downcast_ref only checks the outermost error type, so
    // .context() would hide ResourceContention from the skip
    // logic in result_to_exit_code. Also record a skip sidecar at
    // the propagation point: a ResourceContention-skipped run is
    // otherwise invisible to stats tooling that enumerates
    // sidecars.
    let vm = match builder.build() {
        Ok(vm) => vm,
        Err(e)
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some() =>
        {
            record_skip_sidecar(entry, active_flags);
            return Err(e);
        }
        Err(e) => return Err(e.context("build ktstr_test VM")),
    };

    let result = match vm.run() {
        Ok(r) => r,
        Err(e)
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some() =>
        {
            record_skip_sidecar(entry, active_flags);
            return Err(e);
        }
        Err(e) => return Err(e.context("run ktstr_test VM")),
    };

    // Drop the VM to release CPU/LLC flock fds before auto-repro.
    drop(vm);

    // Log verifier stats count for visibility.
    if !result.verifier_stats.is_empty() {
        eprintln!(
            "ktstr_test: verifier_stats: {} struct_ops programs",
            result.verifier_stats.len(),
        );
    }

    // When running with a struct_ops scheduler, check that host-side
    // BPF program enumeration found programs with non-zero verified_insns.
    if entry.scheduler.has_active_scheduling() && result.success && result.verifier_stats.is_empty()
    {
        eprintln!("ktstr_test: WARNING: scheduler loaded but verifier_stats is empty");
    }

    // Extract profraw from SHM ring buffer and collect stimulus
    // events + per-payload metrics.
    let mut stimulus_events = Vec::new();
    let mut payload_metrics: Vec<crate::test_support::PayloadMetrics> = Vec::new();
    if let Some(ref shm) = result.shm_data {
        for entry in &shm.entries {
            if entry.msg_type == MSG_TYPE_PROFRAW
                && entry.crc_ok
                && !entry.payload.is_empty()
                && let Err(e) = write_profraw(&entry.payload)
            {
                eprintln!("ktstr_test: write guest profraw: {e}");
            }
            if entry.msg_type == crate::vmm::shm_ring::MSG_TYPE_STIMULUS
                && entry.crc_ok
                && let Some(ev) = crate::vmm::shm_ring::StimulusEvent::from_payload(&entry.payload)
            {
                stimulus_events.push(crate::timeline::StimulusEvent {
                    elapsed_ms: ev.elapsed_ms as u64,
                    label: format!("StepStart[{}]", ev.step_index),
                    op_kind: Some(format!("ops={}", ev.op_count)),
                    detail: Some(format!(
                        "{} cgroups, {} workers",
                        ev.cgroup_count, ev.worker_count,
                    )),
                    total_iterations: if ev.total_iterations > 0 {
                        Some(ev.total_iterations)
                    } else {
                        None
                    },
                });
            }
            if entry.msg_type == crate::vmm::shm_ring::MSG_TYPE_PAYLOAD_METRICS && entry.crc_ok {
                match serde_json::from_slice::<crate::test_support::PayloadMetrics>(&entry.payload)
                {
                    Ok(pm) => payload_metrics.push(pm),
                    Err(e) => eprintln!("ktstr_test: decode payload metrics from SHM: {e}"),
                }
            }
        }
    }

    // auto_repro is enabled when:
    // - entry.auto_repro is true (default)
    // - a scheduler is running (not EEVDF)
    // - the test does not expect failure (expect_err = false)
    let effective_auto_repro = entry.auto_repro && scheduler.is_some() && !entry.expect_err;
    let repro_fn = |output: &str| -> Option<String> {
        if !effective_auto_repro {
            return None;
        }
        let repro = attempt_auto_repro(
            entry,
            &kernel,
            scheduler.as_deref(),
            &ktstr_bin,
            output,
            &result.stderr,
            topo,
        );
        // When auto-repro was attempted but produced no data, return a
        // diagnostic so the user knows it was tried.
        Some(repro.unwrap_or_else(|| {
            "auto-repro: no probe data — the scheduler may have \
             exited before probes could capture events, or the \
             crash did not reproduce in the repro VM. Re-run with \
             RUST_LOG=debug for probe pipeline diagnostics. Check \
             the sched_ext dump and scheduler log sections above \
             for crash details."
                .to_string()
        }))
    };

    evaluate_vm_result(
        entry,
        &result,
        &merged_assert,
        &stimulus_events,
        &payload_metrics,
        &vm_topology,
        active_flags,
        &repro_fn,
    )
}

/// Evaluate a VM result and produce the appropriate error or Ok.
///
/// This is the core result-evaluation logic, extracted from
/// `run_ktstr_test_inner` so that error message formatting can be tested
/// without booting a VM. The `repro_fn` callback handles auto-repro
/// (which requires a second VM boot) when provided. `payload_metrics`
/// is the per-invocation accumulator drained from the guest SHM ring;
/// the sidecar writer receives it verbatim so stats tooling sees one
/// entry per `ctx.payload(X).run()` / `.spawn().wait()`.
#[allow(clippy::too_many_arguments)]
fn evaluate_vm_result(
    entry: &KtstrTestEntry,
    result: &vmm::VmResult,
    merged_assert: &crate::assert::Assert,
    stimulus_events: &[StimulusEvent],
    payload_metrics: &[crate::test_support::PayloadMetrics],
    topo: &Topology,
    active_flags: &[String],
    repro_fn: &dyn Fn(&str) -> Option<String>,
) -> Result<AssertResult> {
    // Build timeline from stimulus events + monitor samples.
    let timeline = result
        .monitor
        .as_ref()
        .map(|m| crate::timeline::Timeline::build(stimulus_events, &m.samples));

    let sched_label = match entry.scheduler.scheduler_binary() {
        Some(b) => scheduler_label(b),
        None => String::new(),
    };
    let output = &result.output;
    let dump_section = extract_sched_ext_dump(&result.stderr)
        .map(|d| format!("\n\n--- sched_ext dump ---\n{d}"))
        .unwrap_or_default();
    let sched_log_section = parse_sched_output(output)
        .map(|s| {
            let collapsed = crate::verifier::collapse_cycles(s);
            format!("\n\n--- scheduler log ---\n{collapsed}")
        })
        .unwrap_or_default();
    let fingerprint_line = sched_log_fingerprint(output)
        .map(|fp| {
            if crate::cli::stderr_color() {
                format!("\x1b[1;31m{fp}\x1b[0m\n")
            } else {
                format!("{fp}\n")
            }
        })
        .unwrap_or_default();

    let tl_ctx = crate::timeline::TimelineContext {
        kernel: extract_kernel_version(&result.stderr),
        topology: Some(format!("{topo} ({} cpus)", topo.total_cpus())),
        scheduler: Some(entry.scheduler.scheduler_name().to_string()),
        scenario: Some(entry.name.to_string()),
        duration_s: Some(result.duration.as_secs_f64()),
    };

    // Section builders shared by every error branch in this function.
    // Timeline skips phaseless runs; monitor only reports when an
    // active scheduler exposes rq data (EEVDF reads would be junk).
    let build_timeline_section = || -> String {
        timeline
            .as_ref()
            .filter(|t| !t.phases.is_empty())
            .map(|t| format!("\n\n{}", t.format_with_context(&tl_ctx)))
            .unwrap_or_default()
    };
    let build_monitor_section = || -> String {
        if entry.scheduler.has_active_scheduling()
            && let Some(ref monitor) = result.monitor
        {
            format_monitor_section(monitor, merged_assert)
        } else {
            String::new()
        }
    };

    if let Ok(check_result) =
        parse_assert_result_shm(result.shm_data.as_ref()).or_else(|_| parse_assert_result(output))
    {
        // Write sidecar before checking pass/fail so both outcomes are captured.
        // A sidecar write failure is logged but not propagated: the test
        // verdict itself is still valid — only post-run stats tooling
        // loses visibility.
        let args: Vec<String> = std::env::args().collect();
        let work_type =
            super::args::extract_work_type_arg(&args).unwrap_or_else(|| "CpuSpin".to_string());
        if let Err(e) = write_sidecar(
            entry,
            result,
            stimulus_events,
            &check_result,
            &work_type,
            active_flags,
            payload_metrics,
        ) {
            eprintln!("ktstr_test: {e:#}");
        }

        if !check_result.passed {
            let details = check_result
                .details
                .iter()
                .map(|d| d.message.as_str())
                .collect::<Vec<_>>()
                .join("\n  ");
            let repro = if entry.scheduler.has_active_scheduling() {
                repro_fn(output)
            } else {
                None
            };
            let repro_section = repro
                .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
                .unwrap_or_default();
            let timeline_section = build_timeline_section();
            let stats_section = if !check_result.stats.cgroups.is_empty() {
                let s = &check_result.stats;
                let mut lines = vec![format!(
                    "\n\n--- stats ---\n{} workers, {} cpus, {} migrations, worst_spread={:.1}%, worst_gap={}ms",
                    s.total_workers,
                    s.total_cpus,
                    s.total_migrations,
                    s.worst_spread,
                    s.worst_gap_ms,
                )];
                for (i, cg) in s.cgroups.iter().enumerate() {
                    lines.push(format!(
                        "  cg{}: workers={} cpus={} spread={:.1}% gap={}ms migrations={} iter={}",
                        i,
                        cg.num_workers,
                        cg.num_cpus,
                        cg.spread,
                        cg.max_gap_ms,
                        cg.total_migrations,
                        cg.total_iterations,
                    ));
                }
                lines.join("\n")
            } else {
                String::new()
            };
            let console_section = if check_result
                .details
                .iter()
                .any(|d| {
                    d.contains("scheduler process exited")
                        || d.contains("scheduler process no longer running")
                })
                || verbose()
            {
                let init_stage = classify_init_stage(output);
                format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
            } else {
                String::new()
            };
            let monitor_section = build_monitor_section();
            let msg = format!(
                "{}ktstr_test '{}'{} [topo={}] failed:\n  {}{}{}{}{}{}{}{}",
                fingerprint_line,
                entry.name,
                sched_label,
                topo,
                details,
                stats_section,
                console_section,
                timeline_section,
                sched_log_section,
                monitor_section,
                dump_section,
                repro_section,
            );
            anyhow::bail!("{msg}");
        }

        // Evaluate monitor data against thresholds when a scheduler is running.
        // Without a scheduler (EEVDF), monitor reads rq data that may be
        // uninitialized or irrelevant — skip evaluation in that case.
        //
        // Skip early monitor warmup samples: during boot, BPF verification,
        // and initramfs unpacking the scheduler tick may not fire for hundreds
        // of milliseconds. These transient stalls are real but not indicative
        // of scheduler bugs.
        if entry.scheduler.has_active_scheduling()
            && let Some(ref monitor) = result.monitor
        {
            let eval_report = trim_settle_samples(monitor);
            let thresholds = merged_assert.monitor_thresholds();
            let verdict = thresholds.evaluate(&eval_report);
            if !verdict.passed {
                let details = verdict.details.join("\n  ");
                let timeline_section = build_timeline_section();
                let monitor_section = format_monitor_section(monitor, merged_assert);
                let msg = format!(
                    "{}ktstr_test '{}'{} [topo={}] {ERR_MONITOR_FAILED_AFTER_SCENARIO}:\n  {}{}{}{}{}",
                    fingerprint_line,
                    entry.name,
                    sched_label,
                    topo,
                    details,
                    timeline_section,
                    monitor_section,
                    sched_log_section,
                    dump_section,
                );
                anyhow::bail!("{msg}");
            }
        }

        return Ok(check_result);
    }

    // No parseable result — no AssertResult found in SHM or COM2.
    // When a scheduler is running this typically means the scheduler died;
    // without a scheduler (EEVDF) it means the payload itself failed.
    // Attempt auto-repro if enabled and a scheduler was running.
    // Any scheduler failure that prevents producing a test result warrants
    // repro — BPF verifier failures, scx_bpf_error() exits, crashes, and
    // stalls all land here. Previous code required specific string patterns
    // ("SCHEDULER_DIED", "sched_ext:" + "disabled") which missed mid-test
    // deaths where the sched_exit_monitor writes to SHM but not COM2.
    let repro_section = if entry.scheduler.has_active_scheduling() {
        repro_fn(output)
            .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
            .unwrap_or_default()
    } else {
        String::new()
    };

    // Build a diagnostic section from COM1 kernel console output and exit code.
    // When COM2 has scheduler output markers, sched_log_section and dump_section
    // carry the diagnostics and the kernel console is noise (BIOS, ACPI boot).
    // When COM2 has NO scheduler output (crash before writing), the kernel console
    // is the ONLY source of crash info — include it unconditionally as a fallback.
    let has_sched_output = output.contains(SCHED_OUTPUT_START);
    let console_section = if !has_sched_output || verbose() {
        let init_stage = classify_init_stage(output);
        format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
    } else {
        String::new()
    };

    let timeline_section = build_timeline_section();

    // Build monitor section for error paths where neither SHM nor COM2 had a parseable result.
    let monitor_section = build_monitor_section();

    if result.timed_out {
        let msg = format!(
            "{}ktstr_test '{}'{} [topo={}] {ERR_TIMED_OUT_NO_RESULT}{}{}{}{}{}{}",
            fingerprint_line,
            entry.name,
            sched_label,
            topo,
            console_section,
            timeline_section,
            sched_log_section,
            dump_section,
            monitor_section,
            repro_section,
        );
        anyhow::bail!("{msg}");
    }

    let reason = if let Some(ref shm_crash) = result.crash_message {
        format!("{ERR_GUEST_CRASHED_PREFIX}\n{shm_crash}")
    } else if let Some(crash_msg) = extract_panic_message(output) {
        format!("{ERR_GUEST_CRASHED_PREFIX} {crash_msg}")
    } else if entry.scheduler.has_active_scheduling() {
        ERR_NO_TEST_RESULT_FROM_GUEST.to_string()
    } else {
        ERR_NO_TEST_FUNCTION_OUTPUT.to_string()
    };
    let msg = format!(
        "{}ktstr_test '{}'{} [topo={}] {}{}{}{}{}{}{}",
        fingerprint_line,
        entry.name,
        sched_label,
        topo,
        reason,
        console_section,
        timeline_section,
        sched_log_section,
        dump_section,
        monitor_section,
        repro_section,
    );
    anyhow::bail!("{msg}")
}

/// Format the `--- monitor ---` section for failure output.
///
/// Shows peak values, averaged metrics, event counter rates, schedstat
/// rates, and the monitor verdict. All values are from the post-warmup
/// evaluation window (boot-settle samples trimmed).
pub(crate) fn format_monitor_section(
    monitor: &crate::monitor::MonitorReport,
    merged_assert: &crate::assert::Assert,
) -> String {
    let eval_report = trim_settle_samples(monitor);
    let s = &eval_report.summary;
    let thresholds = merged_assert.monitor_thresholds();
    let verdict = thresholds.evaluate(&eval_report);
    let verdict_line = if verdict.passed {
        verdict.summary.clone()
    } else {
        format!("{}: {}", verdict.summary, verdict.details.join("; "))
    };

    let mut lines = vec![
        format!(
            "samples={} max_imbalance={:.2} max_dsq_depth={} stall={}",
            s.total_samples, s.max_imbalance_ratio, s.max_local_dsq_depth, s.stall_detected,
        ),
        format!(
            "avg: imbalance={:.2} nr_running/cpu={:.1} dsq/cpu={:.1}",
            s.avg_imbalance_ratio, s.avg_nr_running, s.avg_local_dsq_depth,
        ),
    ];

    if let Some(ref ev) = s.event_deltas {
        lines.push(format!(
            "events: fallback={} ({:.1}/s) keep_last={} ({:.1}/s) offline={}",
            ev.total_fallback,
            ev.fallback_rate,
            ev.total_dispatch_keep_last,
            ev.keep_last_rate,
            ev.total_dispatch_offline,
        ));
        let mut extra = Vec::new();
        if ev.total_reenq_immed != 0 {
            extra.push(format!("reenq_immed={}", ev.total_reenq_immed));
        }
        if ev.total_reenq_local_repeat != 0 {
            extra.push(format!(
                "reenq_local_repeat={}",
                ev.total_reenq_local_repeat
            ));
        }
        if ev.total_refill_slice_dfl != 0 {
            extra.push(format!("refill_slice_dfl={}", ev.total_refill_slice_dfl));
        }
        if ev.total_bypass_activate != 0 {
            extra.push(format!("bypass_activate={}", ev.total_bypass_activate));
        }
        if ev.total_bypass_dispatch != 0 {
            extra.push(format!("bypass_dispatch={}", ev.total_bypass_dispatch));
        }
        if ev.total_bypass_duration != 0 {
            extra.push(format!("bypass_duration={}ns", ev.total_bypass_duration));
        }
        if ev.total_insert_not_owned != 0 {
            extra.push(format!("insert_not_owned={}", ev.total_insert_not_owned));
        }
        if ev.total_sub_bypass_dispatch != 0 {
            extra.push(format!(
                "sub_bypass_dispatch={}",
                ev.total_sub_bypass_dispatch
            ));
        }
        if !extra.is_empty() {
            lines.push(format!("events+: {}", extra.join(" ")));
        }
    }

    if let Some(ref ss) = s.schedstat_deltas {
        lines.push(format!(
            "schedstat: csw={} ({:.0}/s) run_delay={:.0}ns/s ttwu={} goidle={}",
            ss.total_sched_count,
            ss.sched_count_rate,
            ss.run_delay_rate,
            ss.total_ttwu_count,
            ss.total_sched_goidle,
        ));
    }

    if let Some(ref progs) = s.prog_stats_deltas {
        for p in progs {
            if p.cnt > 0 {
                lines.push(format!(
                    "bpf: {} cnt={} {:.0}ns/call",
                    p.name, p.cnt, p.nsecs_per_call,
                ));
            }
        }
    }

    lines.push(format!("verdict: {verdict_line}"));

    format!("\n\n--- monitor ---\n{}", lines.join("\n"))
}

/// Number of monitor samples to skip at the start of evaluation.
///
/// During VM boot the kernel performs BPF verification, initramfs
/// unpacking, and scheduler loading. These memory-intensive operations
/// cause the scheduler tick to stall for hundreds of milliseconds.
/// The stalls are real but transient — evaluating them produces false
/// positives, especially in low-memory VMs.
///
/// 20 samples at ~100ms interval = ~2 seconds of warmup. This covers
/// the boot settling period after the scheduler attaches.
const MONITOR_WARMUP_SAMPLES: usize = 20;

/// Skip boot-settle samples from a MonitorReport for threshold evaluation.
///
/// Returns a report with the first `MONITOR_WARMUP_SAMPLES` removed so
/// that transient boot-time stalls don't trigger sustained-window
/// violations.
pub(crate) fn trim_settle_samples(
    report: &crate::monitor::MonitorReport,
) -> crate::monitor::MonitorReport {
    if report.samples.len() <= MONITOR_WARMUP_SAMPLES {
        return report.clone();
    }

    let trimmed = report.samples[MONITOR_WARMUP_SAMPLES..].to_vec();
    let summary = crate::monitor::MonitorSummary::from_samples_with_threshold(
        &trimmed,
        report.preemption_threshold_ns,
    );
    crate::monitor::MonitorReport {
        samples: trimmed,
        summary,
        preemption_threshold_ns: report.preemption_threshold_ns,
        watchdog_observation: report.watchdog_observation,
    }
}

/// Check that `/dev/kvm` is accessible for read+write.
///
/// Pre-flight check for VM-booting test runs: every ktstr test needs
/// a KVM fd, and failing fast here yields an actionable error
/// ("add your user to the kvm group") before the VM builder starts
/// allocating memory / fetching kernels.
fn ensure_kvm() -> Result<()> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
        .context(
            "/dev/kvm not accessible — KVM is required for ktstr_test. \
             Check that KVM is enabled and your user is in the kvm group.",
        )?;
    Ok(())
}

/// Setup function for nextest `setup-script` integration.
///
/// Validates KVM access, discovers a kernel, writes `KTSTR_TEST_KERNEL`
/// to `env_writer`, and warms the SHM initramfs cache for each binary.
pub fn nextest_setup(binaries: &[&Path], env_writer: &mut dyn Write) -> Result<()> {
    ensure_kvm()?;
    let kernel = resolve_test_kernel()?;
    writeln!(env_writer, "KTSTR_TEST_KERNEL={}", kernel.display())
        .context("write KTSTR_TEST_KERNEL to env")?;

    for bin in binaries {
        let key = vmm::BaseKey::new(bin, None, None, None, false)?;
        let _ = vmm::get_or_build_base(bin, &[], &[], false, &key, false)?;
    }

    // Eager-conditional prefetch: if any registered test declares
    // OutputFormat::LlmExtract, make sure the default model is in
    // the cache before tests start. Fetch failures surface as a
    // warning rather than a hard setup error so scheduler-only test
    // runs remain decoupled from the model cache's availability;
    // the LlmExtract invocation path fails loudly when the model is
    // genuinely missing at test time.
    match super::model::prefetch_if_required() {
        Ok(Some(path)) => {
            tracing::info!(path = %path.display(), "model cache ready");
        }
        Ok(None) => {}
        Err(e) => {
            tracing::warn!(
                error = ?e,
                "model prefetch failed; LlmExtract tests may fail",
            );
        }
    }

    Ok(())
}

/// Format a label for the scheduler spec, for use in test output.
///
/// Returns an empty string for `SchedulerSpec::Eevdf` so the failure
/// header reads `ktstr_test 'name' [topo=...]` with no sched
/// bracket — every other variant renders `" [sched=X]"` where `X`
/// comes from [`SchedulerSpec::display_name`].
fn scheduler_label(spec: &SchedulerSpec) -> String {
    if matches!(spec, SchedulerSpec::Eevdf) {
        String::new()
    } else {
        format!(" [sched={}]", spec.display_name())
    }
}

// ---------------------------------------------------------------------------
// Scheduler resolution
// ---------------------------------------------------------------------------

/// Resolve a scheduler binary from a `SchedulerSpec`.
///
/// Returns `Ok(None)` for `SchedulerSpec::Eevdf` (EEVDF).
/// For `Discover`, searches: `KTSTR_SCHEDULER` env, sibling of current_exe,
/// `target/debug/`, `target/release/`.
/// For `Path`, validates the file exists.
pub fn resolve_scheduler(spec: &SchedulerSpec) -> Result<Option<PathBuf>> {
    match spec {
        SchedulerSpec::Eevdf | SchedulerSpec::KernelBuiltin { .. } => Ok(None),
        SchedulerSpec::Path(p) => {
            let path = PathBuf::from(p);
            anyhow::ensure!(path.exists(), "scheduler not found: {p}");
            Ok(Some(path))
        }
        SchedulerSpec::Discover(name) => {
            // 1. KTSTR_SCHEDULER env var
            if let Ok(p) = std::env::var("KTSTR_SCHEDULER") {
                let path = PathBuf::from(&p);
                if path.exists() {
                    return Ok(Some(path));
                }
            }

            // 2. Sibling of current executable (or parent of deps/)
            if let Ok(exe) = crate::resolve_current_exe()
                && let Some(dir) = exe.parent()
            {
                let candidate = dir.join(name);
                if candidate.exists() {
                    return Ok(Some(candidate));
                }
                // Integration tests and nextest place test binaries in
                // target/{debug,release}/deps/. The scheduler binary is
                // one level up in target/{debug,release}/.
                if dir.file_name().is_some_and(|d| d == "deps")
                    && let Some(parent) = dir.parent()
                {
                    let candidate = parent.join(name);
                    if candidate.exists() {
                        return Ok(Some(candidate));
                    }
                }
            }

            // 3. target/debug/
            let candidate = PathBuf::from("target/debug").join(name);
            if candidate.exists() {
                return Ok(Some(candidate));
            }

            // 4. target/release/
            let candidate = PathBuf::from("target/release").join(name);
            if candidate.exists() {
                return Ok(Some(candidate));
            }

            // 5. Build the scheduler package on demand.
            match crate::build_and_find_binary(name) {
                Ok(path) => return Ok(Some(path)),
                Err(e) => eprintln!("ktstr: auto-build scheduler '{name}' failed: {e:#}"),
            }

            anyhow::bail!(
                "scheduler '{name}' not found. Set KTSTR_SCHEDULER or \
                 place it next to the test binary or in target/{{debug,release}}/"
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel resolution
// ---------------------------------------------------------------------------

/// Find a kernel image for running tests.
///
/// Checks `KTSTR_TEST_KERNEL` env var first (direct image path),
/// then delegates to [`crate::find_kernel()`] for cache and
/// filesystem discovery. Bails with actionable hints on failure.
pub fn resolve_test_kernel() -> Result<PathBuf> {
    // Check environment variable first.
    if let Ok(path) = std::env::var("KTSTR_TEST_KERNEL") {
        let p = PathBuf::from(&path);
        anyhow::ensure!(p.exists(), "KTSTR_TEST_KERNEL not found: {path}");
        return Ok(p);
    }

    // Standard locations.
    if let Some(p) = crate::find_kernel()? {
        return Ok(p);
    }

    anyhow::bail!(
        "no kernel found\n  \
         hint: run `cargo ktstr kernel build` to download and build the latest stable kernel\n  \
         hint: or set KTSTR_KERNEL=/path/to/linux\n  \
         hint: or set KTSTR_TEST_KERNEL=/path/to/{image_name}",
        image_name = if cfg!(target_arch = "aarch64") {
            "Image"
        } else {
            "bzImage"
        }
    )
}

#[cfg(test)]
mod tests {
    use super::super::output::{
        RESULT_END, RESULT_START, STAGE_INIT_NOT_STARTED, STAGE_INIT_STARTED_NO_PAYLOAD,
        STAGE_PAYLOAD_STARTED_NO_RESULT,
    };
    use super::super::test_helpers::{
        EVAL_TOPO, EnvVarGuard, eevdf_entry, lock_env, make_vm_result, no_repro, sched_entry,
    };
    use super::*;
    use crate::verifier::SCHED_OUTPUT_END;

    // -- resolve_test_kernel tests --

    #[test]
    fn resolve_test_kernel_with_env_var() {
        let _lock = lock_env();
        let exe = crate::resolve_current_exe().unwrap();
        let _env = EnvVarGuard::set("KTSTR_TEST_KERNEL", &exe);
        let result = resolve_test_kernel();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), exe);
    }

    #[test]
    fn resolve_test_kernel_with_nonexistent_env_path() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set("KTSTR_TEST_KERNEL", "/nonexistent/kernel/path");
        let result = resolve_test_kernel();
        assert!(result.is_err());
    }

    // -- KVM check --

    #[test]
    fn kvm_accessible_on_test_host() {
        // Checks that /dev/kvm is accessible with read+write permissions.
        ensure_kvm().expect("/dev/kvm not accessible");
    }

    // -- resolve_scheduler tests --

    #[test]
    fn resolve_scheduler_eevdf() {
        let result = resolve_scheduler(&SchedulerSpec::Eevdf).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn resolve_scheduler_path_exists() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = resolve_scheduler(&SchedulerSpec::Path(Box::leak(
            exe.to_str().unwrap().to_string().into_boxed_str(),
        )))
        .unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn resolve_scheduler_path_missing() {
        let result = resolve_scheduler(&SchedulerSpec::Path("/nonexistent/scheduler"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_scheduler_discover_missing() {
        let _lock = lock_env();
        let _env = EnvVarGuard::remove("KTSTR_SCHEDULER");
        let result = resolve_scheduler(&SchedulerSpec::Discover("__nonexistent_scheduler_xyz__"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_scheduler_discover_via_env() {
        let _lock = lock_env();
        let exe = crate::resolve_current_exe().unwrap();
        let _env = EnvVarGuard::set("KTSTR_SCHEDULER", &exe);
        let result = resolve_scheduler(&SchedulerSpec::Discover("anything"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().unwrap(), exe);
    }

    // -- scheduler_label tests --

    #[test]
    fn scheduler_label_eevdf_empty() {
        assert_eq!(scheduler_label(&SchedulerSpec::Eevdf), "");
    }

    #[test]
    fn scheduler_label_discover() {
        assert_eq!(
            scheduler_label(&SchedulerSpec::Discover("scx_mitosis")),
            " [sched=scx_mitosis]"
        );
    }

    #[test]
    fn scheduler_label_path() {
        assert_eq!(
            scheduler_label(&SchedulerSpec::Path("/usr/bin/sched")),
            " [sched=/usr/bin/sched]"
        );
    }

    // -- nextest_setup --

    #[test]
    fn nextest_setup_writes_kernel_env() {
        let _lock = lock_env();
        let exe = crate::resolve_current_exe().unwrap();
        let _env = EnvVarGuard::set("KTSTR_TEST_KERNEL", &exe);

        let mut buf = Vec::new();
        let result = nextest_setup(&[exe.as_path()], &mut buf);

        assert!(result.is_ok(), "nextest_setup failed: {result:?}");
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.starts_with("KTSTR_TEST_KERNEL="),
            "expected KTSTR_TEST_KERNEL=..., got: {output}"
        );
    }

    // -- evaluate_vm_result error path tests --

    #[test]
    fn eval_eevdf_no_com2_output() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_eevdf_no_out__");
        let result = make_vm_result("", "boot log line\nKernel panic", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_FUNCTION_OUTPUT),
            "EEVDF with no COM2 output should say {ERR_NO_TEST_FUNCTION_OUTPUT:?}, got: {msg}",
        );
        assert!(
            !msg.contains("no test result received from guest"),
            "EEVDF error should not use the scheduler-path wording, got: {msg}",
        );
        assert!(
            msg.contains("exit_code=1"),
            "should include exit code, got: {msg}"
        );
        assert!(
            msg.contains("Kernel panic"),
            "should include console output, got: {msg}"
        );
    }

    #[test]
    fn eval_sched_dies_no_com2_output() {
        let entry = sched_entry("__eval_sched_dies__");
        let result = make_vm_result("", "boot ok", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
            "scheduler present with no output should take the scheduler-path fallback, got: {msg}",
        );
        assert!(
            !msg.contains("test function produced no output"),
            "should not say 'test function produced no output' when scheduler is set, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_dies_with_sched_log() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let sched_log = format!(
            "noise\n{SCHED_OUTPUT_START}\ndo_enqueue_task+0x1a0\nbalance_one+0x50\n{SCHED_OUTPUT_END}\nmore",
        );
        let entry = sched_entry("__eval_sched_log__");
        let result = make_vm_result(&sched_log, "", -1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_RESULT_FROM_GUEST),
            "should take the scheduler-path fallback, got: {msg}",
        );
        assert!(
            msg.contains("--- scheduler log ---"),
            "should include scheduler log section, got: {msg}",
        );
        assert!(
            msg.contains("do_enqueue_task"),
            "should include scheduler log content, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_mid_test_death_triggers_repro() {
        // Scheduler dies mid-test: sched_exit_monitor dumps log to COM2
        // but does NOT write "SCHEDULER_DIED". Auto-repro should still
        // trigger because has_active_scheduling() is true and no
        // AssertResult was produced.
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nError: BPF program error\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_mid_death_repro__");
        let result = make_vm_result(&sched_log, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let repro_called = std::sync::atomic::AtomicBool::new(false);
        let repro_fn = |_output: &str| -> Option<String> {
            repro_called.store(true, std::sync::atomic::Ordering::Relaxed);
            Some("repro data".to_string())
        };
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &repro_fn,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            repro_called.load(std::sync::atomic::Ordering::Relaxed),
            "repro_fn should be called for mid-test scheduler death without SCHEDULER_DIED marker",
        );
        assert!(
            msg.contains("--- auto-repro ---"),
            "error should include auto-repro section, got: {msg}",
        );
        assert!(
            msg.contains("repro data"),
            "error should include repro output, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_repro_no_data_shows_diagnostic() {
        // When repro_fn returns the fallback diagnostic, the error
        // output should include it so the user knows auto-repro was
        // tried and why it produced nothing.
        let entry = sched_entry("__eval_repro_no_data__");
        let result = make_vm_result("", "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let repro_fn = |_output: &str| -> Option<String> {
            Some(
                "auto-repro: no probe data — scheduler may have exited before \
                 probes could attach. Check the sched_ext dump and scheduler \
                 log sections above for crash details."
                    .to_string(),
            )
        };
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &repro_fn,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--- auto-repro ---"),
            "should include auto-repro section, got: {msg}",
        );
        assert!(
            msg.contains("no probe data"),
            "should include diagnostic message, got: {msg}",
        );
        assert!(
            msg.contains("sched_ext dump"),
            "should direct user to dump section, got: {msg}",
        );
    }

    #[test]
    fn eval_timeout_no_result() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_timeout__");
        let result = make_vm_result("", "booting...\nstill booting...", 0, true);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_TIMED_OUT_NO_RESULT),
            "should contain full timed-out reason {ERR_TIMED_OUT_NO_RESULT:?}, got: {msg}",
        );
        assert!(
            msg.contains("booting"),
            "should include console output, got: {msg}",
        );
        assert!(
            msg.contains("[topo="),
            "error should include topology, got: {msg}",
        );
    }

    #[test]
    fn eval_payload_exits_no_check_result() {
        // Payload wrote something to COM2 but not a valid AssertResult.
        let entry = eevdf_entry("__eval_no_check__");
        let result = make_vm_result(
            "some output but no delimiters",
            "Linux version 6.14.0\nboot complete",
            0,
            false,
        );
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_NO_TEST_FUNCTION_OUTPUT),
            "non-parseable COM2 with EEVDF should say {ERR_NO_TEST_FUNCTION_OUTPUT:?}, got: {msg}",
        );
        assert!(
            !msg.contains("no test result received from guest"),
            "EEVDF should not use the scheduler-path wording, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_ext_dump_included() {
        let dump_line = "ktstr-0 [001] 0.5: sched_ext_dump: Debug dump line";
        let entry = sched_entry("__eval_dump__");
        let result = make_vm_result("", dump_line, -1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--- sched_ext dump ---"),
            "should include dump section, got: {msg}",
        );
        assert!(
            msg.contains("sched_ext_dump: Debug dump"),
            "should include dump content, got: {msg}",
        );
    }

    #[test]
    fn eval_check_result_passed_returns_ok() {
        let json = r#"{"passed":true,"skipped":false,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_pass__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        assert!(
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro,
            )
            .is_ok(),
            "passing AssertResult should return Ok",
        );
    }

    #[test]
    fn eval_check_result_failed_includes_details() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"stuck 3000ms"},{"kind":"Unfair","message":"spread 45%"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_fail_details__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("failed:"), "got: {msg}");
        assert!(msg.contains("stuck 3000ms"), "got: {msg}");
        assert!(msg.contains("spread 45%"), "got: {msg}");
    }

    #[test]
    fn eval_assert_failure_includes_sched_log() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"worker 0 stuck 5000ms"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nscheduler noise line\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fail_sched_log__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("worker 0 stuck 5000ms"), "got: {msg}");
        assert!(msg.contains("scheduler noise"), "got: {msg}");
        assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
    }

    #[test]
    fn eval_assert_failure_has_fingerprint() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"stuck 3000ms"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let error_line = "Error: apply_cell_config BPF program returned error -2";
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fingerprint__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains(error_line), "got: {msg}");
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(fp_pos < name_pos, "got: {msg}");
    }

    #[test]
    fn eval_timeout_has_fingerprint() {
        let error_line = "Error: scheduler panicked";
        let output = format!("{SCHED_OUTPUT_START}\n{error_line}\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_timeout_fp__");
        let result = make_vm_result(&output, "", 0, true);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(error_line),
            "timeout should contain fingerprint, got: {msg}",
        );
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(
            fp_pos < name_pos,
            "fingerprint should appear before ktstr_test line, got: {msg}",
        );
    }

    #[test]
    fn eval_no_result_has_fingerprint() {
        let error_line = "Error: fatal scheduler crash";
        let output =
            format!("{SCHED_OUTPUT_START}\nstartup log\n{error_line}\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_no_result_fp__");
        let result = make_vm_result(&output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(error_line),
            "no-result failure should contain fingerprint, got: {msg}",
        );
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(
            fp_pos < name_pos,
            "fingerprint should appear before ktstr_test line, got: {msg}",
        );
    }

    #[test]
    fn eval_no_sched_output_no_fingerprint() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"stuck"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_no_fp__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.starts_with("ktstr_test"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_has_fingerprint() {
        let pass_json = r#"{"passed":true,"skipped":false,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let error_line = "Error: imbalance detected internally";
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",);
        let output = format!("{RESULT_START}\n{pass_json}\n{RESULT_END}\n{sched_log}");
        let entry = sched_entry("__eval_monitor_fp__");
        let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
            .map(|i| {
                crate::monitor::MonitorSample::new(
                    (i * 100) as u64,
                    vec![
                        crate::monitor::CpuSnapshot {
                            nr_running: 10,
                            scx_nr_running: 10,
                            local_dsq_depth: 0,
                            rq_clock: 1000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                        crate::monitor::CpuSnapshot {
                            nr_running: 1,
                            scx_nr_running: 1,
                            local_dsq_depth: 0,
                            rq_clock: 2000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                    ],
                )
            })
            .collect();
        let summary =
            crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
        let result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output,
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: imbalance_samples,
                summary,
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let assertions = crate::assert::Assert::default_checks();
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(
            msg.contains(ERR_MONITOR_FAILED_AFTER_SCENARIO),
            "got: {msg}"
        );
        assert!(msg.contains(error_line), "got: {msg}");
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(fp_pos < name_pos, "got: {msg}");
    }

    #[test]
    fn eval_timeout_with_sched_includes_diagnostics() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = sched_entry("__eval_timeout_sched__");
        let result = make_vm_result("", "Linux version 6.14.0\nkernel panic here", -1, true);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_TIMED_OUT_NO_RESULT),
            "should contain {ERR_TIMED_OUT_NO_RESULT:?}, got: {msg}"
        );
        assert!(
            msg.contains("[sched=test_sched_bin]"),
            "should include scheduler label, got: {msg}"
        );
        assert!(
            msg.contains("--- diagnostics ---"),
            "should include diagnostics, got: {msg}"
        );
        assert!(
            msg.contains("kernel panic here"),
            "should include console tail, got: {msg}"
        );
    }

    // -- sentinel integration in evaluate_vm_result --

    #[test]
    fn eval_no_sentinels_shows_initramfs_failure() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_no_sentinel__");
        let result = make_vm_result("", "Kernel panic", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(STAGE_INIT_NOT_STARTED),
            "no sentinels should indicate kernel/mount failure, got: {msg}",
        );
    }

    #[test]
    fn eval_init_started_but_no_payload() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_init_only__");
        let result = make_vm_result("KTSTR_INIT_STARTED\n", "boot log", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(STAGE_INIT_STARTED_NO_PAYLOAD),
            "init sentinel only should indicate cgroup/scheduler setup failure, got: {msg}",
        );
    }

    #[test]
    fn eval_payload_started_no_result() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let entry = eevdf_entry("__eval_payload_start__");
        let output = "KTSTR_INIT_STARTED\nKTSTR_PAYLOAD_STARTING\ngarbage";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(STAGE_PAYLOAD_STARTED_NO_RESULT),
            "both sentinels should indicate payload ran but failed, got: {msg}",
        );
    }

    // -- guest panic detection tests --

    #[test]
    fn eval_crash_in_output_says_guest_crashed() {
        let entry = sched_entry("__eval_crash_detect__");
        let output = "KTSTR_INIT_STARTED\nPANIC: panicked at src/foo.rs:42: assertion failed";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(ERR_GUEST_CRASHED_PREFIX), "got: {msg}");
        assert!(msg.contains("assertion failed"), "got: {msg}");
    }

    #[test]
    fn eval_crash_eevdf_says_guest_crashed() {
        let entry = eevdf_entry("__eval_crash_eevdf__");
        let output = "PANIC: panicked at src/bar.rs:10: index out of bounds";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(ERR_GUEST_CRASHED_PREFIX), "got: {msg}");
        assert!(msg.contains("index out of bounds"), "got: {msg}");
    }

    #[test]
    fn eval_crash_message_from_shm() {
        let entry = sched_entry("__eval_crash_shm__");
        let shm_crash = "PANIC: panicked at src/test.rs:42: assertion failed\n   \
                          0: ktstr::vmm::rust_init::ktstr_guest_init\n";
        // COM2 also has a PANIC: line (serial fallback). SHM must take priority.
        let output = "PANIC: panicked at src/test.rs:42: assertion failed";
        let mut result = make_vm_result(output, "", 1, false);
        result.crash_message = Some(shm_crash.to_string());
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(ERR_GUEST_CRASHED_PREFIX),
            "should say {ERR_GUEST_CRASHED_PREFIX:?}, got: {msg}",
        );
        assert!(
            msg.contains("ktstr_guest_init"),
            "SHM backtrace content should be present, got: {msg}",
        );
        // SHM path uses "guest crashed:\n{shm_crash}" (multiline),
        // COM2 path uses "guest crashed: {msg}" (single line).
        // The backtrace frame proves SHM was used, not COM2.
        assert!(
            msg.contains("0: ktstr::vmm::rust_init::ktstr_guest_init"),
            "full backtrace from SHM should appear, got: {msg}",
        );
    }

    // -- diagnostic section tests --

    #[test]
    fn eval_sched_died_includes_console() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Monitor","message":"scheduler process exited unexpectedly after completing step 1 of 2 (0.5s into test)"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = sched_entry("__eval_sched_died_console__");
        let result = make_vm_result(&output, "kernel panic\nsched_ext: disabled", 1, false);
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("--- diagnostics ---"), "got: {msg}");
        assert!(msg.contains("kernel panic"), "got: {msg}");
    }

    #[test]
    fn eval_sched_died_includes_monitor() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Monitor","message":"scheduler process exited unexpectedly during workload (2.0s into test)"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = sched_entry("__eval_sched_died_monitor__");
        let result = crate::vmm::VmResult {
            success: false,
            exit_code: 1,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: output.to_string(),
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: vec![],
                summary: crate::monitor::MonitorSummary {
                    total_samples: 5,
                    max_imbalance_ratio: 3.0,
                    max_local_dsq_depth: 2,
                    stall_detected: false,
                    event_deltas: None,
                    schedstat_deltas: None,
                    prog_stats_deltas: None,
                    ..Default::default()
                },
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let assertions = crate::assert::Assert::NO_OVERRIDES;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("--- monitor ---"), "got: {msg}");
        assert!(msg.contains("max_imbalance"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_includes_sched_log() {
        let pass_json = r#"{"passed":true,"skipped":false,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0,"worst_migration_ratio":0.0,"p99_wake_latency_us":0.0,"median_wake_latency_us":0.0,"wake_latency_cv":0.0,"total_iterations":0,"mean_run_delay_us":0.0,"worst_run_delay_us":0.0,"worst_page_locality":0.0,"worst_cross_node_migration_ratio":0.0}}"#;
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nscheduler debug output here\n{SCHED_OUTPUT_END}",);
        let output = format!("{RESULT_START}\n{pass_json}\n{RESULT_END}\n{sched_log}");
        let entry = sched_entry("__eval_monitor_fail_sched__");
        // Imbalance ratio 10.0 exceeds default threshold of 4.0,
        // sustained for 5+ samples past the 20-sample warmup window.
        let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
            .map(|i| {
                crate::monitor::MonitorSample::new(
                    (i * 100) as u64,
                    vec![
                        crate::monitor::CpuSnapshot {
                            nr_running: 10,
                            scx_nr_running: 10,
                            local_dsq_depth: 0,
                            rq_clock: 1000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                        crate::monitor::CpuSnapshot {
                            nr_running: 1,
                            scx_nr_running: 1,
                            local_dsq_depth: 0,
                            rq_clock: 2000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                    ],
                )
            })
            .collect();
        let summary =
            crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
        let result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output,
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: imbalance_samples,
                summary,
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let assertions = crate::assert::Assert::default_checks();
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(
            msg.contains(ERR_MONITOR_FAILED_AFTER_SCENARIO),
            "got: {msg}"
        );
        assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
    }
}
