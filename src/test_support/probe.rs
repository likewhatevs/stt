//! Auto-repro and BPF probe pipeline for `#[ktstr_test]`.
//!
//! When a scheduler crash is observed in a ktstr_test VM, the framework
//! boots a second "repro" VM with BPF kprobes/fentries attached to the
//! functions that appeared in the crash stack. Probe output is
//! serialized on the guest (COM2) and deserialized + formatted on the
//! host where DWARF is available.
//!
//! Probe attachment runs in two phases:
//! - **Phase A** ([`start_probe_phase_a`]) attaches kprobes, fexits,
//!   and the tracepoint trigger to kernel functions before the scheduler
//!   starts. Needed because kprobes must be in place before the
//!   first call to each traced function.
//! - **Phase B** ([`maybe_dispatch_vm_test_with_phase_a`]) discovers
//!   BPF symbols from the running scheduler and attaches fentries to
//!   BPF callbacks. Runs after the scheduler has loaded.
//!
//! The single-phase path ([`maybe_dispatch_vm_test_with_args`]) is used
//! when the kernel doesn't support Phase A; all probes attach after the
//! scheduler is up.

use std::path::Path;
use std::time::Duration;

use crate::assert::AssertResult;

use super::args::{
    extract_probe_stack_arg, extract_test_fn_arg, extract_work_type_arg, resolve_cgroup_root,
};
use super::entry::find_test;
use super::output::{extract_sched_ext_dump, print_assert_result};
use super::profraw::try_flush_profraw;
use super::runtime::{config_file_parts, verbose};
use super::{KtstrTestEntry, TopoOverride};
use crate::verifier::{SCHED_OUTPUT_END, SCHED_OUTPUT_START, parse_sched_output};

/// Sentinel value for `--ktstr-probe-stack` when no crash stack functions
/// were extracted. Triggers the guest-side probe path so
/// `discover_bpf_symbols()` can dynamically find the scheduler's BPF
/// programs. `filter_traceable` drops it (not in kallsyms).
const DISCOVER_SENTINEL: &str = "__discover__";

/// Propagate `RUST_BACKTRACE` and `RUST_LOG` from the guest kernel
/// cmdline into the process environment.
///
/// # Safety invariant
///
/// Performs `std::env::set_var`, which is unsound on Linux unless the
/// process is provably single-threaded. glibc mutates the global
/// `__environ` array without locks, so a concurrent reader or another
/// `set_var` produces UB. The two callers
/// ([`ktstr_test_early_dispatch`](super::dispatch::ktstr_test_early_dispatch)
/// and the `ktstr_guest_init` boot path in `vmm::rust_init`) both
/// invoke this before any probe / workload / test thread is spawned.
pub(crate) fn propagate_rust_env_from_cmdline() {
    let Ok(cmdline) = std::fs::read_to_string("/proc/cmdline") else {
        return;
    };
    for (key, val) in parse_rust_env_from_cmdline(&cmdline) {
        // SAFETY: called from ktstr_guest_init before any probe /
        // workload thread is spawned; single-threaded mutation of
        // `__environ` is sound.
        unsafe { std::env::set_var(key, val) };
    }
}

/// Pure parser for the cmdline side of `propagate_rust_env_from_cmdline`.
/// Returns `(key, value)` pairs for every `RUST_BACKTRACE=...` or
/// `RUST_LOG=...` token found in whitespace-split `cmdline`, in the
/// order they appear. Split from the env-mutating wrapper so the
/// parse logic is testable without touching the process environment.
fn parse_rust_env_from_cmdline(cmdline: &str) -> Vec<(&'static str, &str)> {
    let mut out = Vec::new();
    for token in cmdline.split_whitespace() {
        if let Some(val) = token.strip_prefix("RUST_BACKTRACE=") {
            out.push(("RUST_BACKTRACE", val));
        } else if let Some(val) = token.strip_prefix("RUST_LOG=") {
            out.push(("RUST_LOG", val));
        }
    }
    out
}

/// Delimiters for probe output in guest COM2 (written by emit_probe_payload).
pub(crate) const PROBE_OUTPUT_START: &str = "===PROBE_OUTPUT_START===";
pub(crate) const PROBE_OUTPUT_END: &str = "===PROBE_OUTPUT_END===";

/// Format the last `n` lines of `text` under a `--- header ---` delimiter.
/// Returns `None` if `text` is empty.
fn format_tail(text: &str, n: usize, header: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let start = lines.len().saturating_sub(n);
    Some(format!("--- {header} ---\n{}", lines[start..].join("\n")))
}

/// Classify the repro VM outcome into a single human-readable status
/// line, used when the probe pipeline produced no events and the
/// caller needs to tell the user *why* the repro VM did not yield
/// probe data.
///
/// The ordering of branches matters. Each check eliminates a
/// distinct failure mode so the most specific match wins:
///
/// 1. `timed_out` — VM wall clock exceeded. No further signals are
///    meaningful; the run never reached a natural exit.
/// 2. `SCHEDULER_NOT_ATTACHED` in `output` — the scheduler process
///    stayed alive but never completed attachment (BPF verifier
///    reject, ops mismatch, sysfs absent). `rust_init` emits this
///    sentinel with a reason suffix then force-reboots, which is
///    distinct from a scheduler crash. Checked *before* the crash
///    branch because the emission path prevents a subsequent
///    SCHEDULER_DIED write in the same run.
/// 3. `crash_message.is_some()` or `SCHEDULER_DIED` in `output` —
///    the scheduler process crashed or was reported dead by the
///    guest's sched_exit monitor.
/// 4. Nonzero exit code — something exited abnormally, but the
///    guest did not emit a classification sentinel.
/// 5. Clean exit — scheduler ran to completion; the first VM's
///    crash did not reproduce.
fn classify_repro_vm_status(
    timed_out: bool,
    has_crash_message: bool,
    output: &str,
    exit_code: i32,
) -> String {
    if timed_out {
        return "repro VM: timed out".to_string();
    }
    if let Some(reason) = extract_not_attached_reason(output) {
        return format!("repro VM: scheduler did not attach ({reason}) (exit code {exit_code})",);
    }
    if has_crash_message || output.contains(super::SENTINEL_SCHEDULER_DIED) {
        // Describe qemu's exit disposition precisely: a crash sentinel
        // in `output` can coincide with qemu itself exiting 0 (guest
        // panic handler + orderly reboot), >0 (propagated non-zero),
        // -1 (VMM internal sentinel — the boot CPU's run loop seeds
        // `VmResult::exit_code = -1` and leaves it at -1 on error
        // paths that did not deliver the guest's final exit message;
        // watchdog-fire is caught earlier via `timed_out`, so a -1
        // reaching THIS branch indicates a code-unsetting error
        // path, not a signal-kill), or <-1 (signal-kill, rendered
        // via `ExitStatus::signal()` as a negative i32 on unix).
        // Labeling all four as "crashed (exit N)" conflates the
        // guest-scheduler failure with qemu's own exit, making the
        // qemu-clean and VMM-sentinel cases especially misleading.
        // The -1 clause is phrased in end-user terms: the internals
        // (boot-CPU run loop, scheduler-exit IPC message) belong in
        // the code comment, not in the output a test operator reads
        // at the console.
        let exit_clause = if exit_code == -1 {
            "VM host reported no final exit status (the scheduler did not \
             deliver an exit signal before the VM ended)"
                .to_string()
        } else if exit_code < 0 {
            format!("killed by signal ({exit_code})")
        } else if exit_code == 0 {
            "exited cleanly".to_string()
        } else {
            format!("exited with non-zero status ({exit_code})")
        };
        return format!("repro VM: scheduler crashed — {exit_clause}");
    }
    if exit_code != 0 {
        return format!("repro VM: exited abnormally (exit code {exit_code})");
    }
    "repro VM: scheduler ran normally (crash did not reproduce)".to_string()
}

/// Extract the reason suffix after `SCHEDULER_NOT_ATTACHED:` on the
/// first line of `output` that carries the sentinel. Returns
/// `Some("timeout")` for the line `SCHEDULER_NOT_ATTACHED: timeout`,
/// `Some("sched_ext sysfs absent")` for the sysfs-absent emission,
/// or `None` when no line carries the sentinel.
///
/// The sentinel emission in `vmm::rust_init::start_scheduler` writes
/// `"SCHEDULER_NOT_ATTACHED: <reason>"` as a single COM2 line. The
/// parser splits at the first `:` after the sentinel and trims the
/// remainder so trailing whitespace from `write_com2` does not leak
/// into the reason string.
///
/// The FIRST line with the sentinel wins unconditionally. If that
/// first occurrence is malformed — no colon, or an empty/
/// whitespace-only suffix — the result is `None` and no subsequent
/// line is consulted. A malformed first occurrence indicates an
/// emitter bug; falling through to a later, "better" line would
/// paper over that bug. The caller handles `None` by routing to the
/// generic crashed / abnormal-exit branches, which already surface
/// exit code and crash-message diagnostics.
fn extract_not_attached_reason(output: &str) -> Option<&str> {
    let line = output
        .lines()
        .find(|l| l.contains(super::SENTINEL_SCHEDULER_NOT_ATTACHED))?;
    let idx = line.find(super::SENTINEL_SCHEDULER_NOT_ATTACHED)?;
    let after = &line[idx + super::SENTINEL_SCHEDULER_NOT_ATTACHED.len()..];
    let reason = after.strip_prefix(':')?.trim();
    if reason.is_empty() {
        return None;
    }
    Some(reason)
}

/// Attempt auto-repro: extract stack functions from COM2 scheduler output
/// or COM1 kernel console (fallback), boot a second VM with BPF probes
/// attached, and return formatted probe data. When no stack functions are
/// available (e.g. BPF text error without backtrace), falls back to
/// dynamic BPF program discovery in the repro VM.
/// `console_output` is COM1 kernel console text, used when COM2 has no
/// extractable functions (e.g. scheduler died before writing output).
/// Returns `None` if repro cannot be attempted or yields no data.
pub(crate) fn attempt_auto_repro(
    entry: &KtstrTestEntry,
    kernel: &Path,
    scheduler: Option<&Path>,
    ktstr_bin: &Path,
    first_vm_output: &str,
    console_output: &str,
    topo: Option<&TopoOverride>,
) -> Option<String> {
    use crate::probe::stack::extract_stack_functions_all;

    // Extract scheduler log from COM2 output.
    eprintln!(
        "ktstr_test: auto-repro: COM2 length={} has_sched_start={} has_sched_end={}",
        first_vm_output.len(),
        first_vm_output.contains(SCHED_OUTPUT_START),
        first_vm_output.contains(SCHED_OUTPUT_END),
    );
    let sched_output = parse_sched_output(first_vm_output);

    // Extract function names from COM2 scheduler log first, then
    // fall back to COM1 kernel console (which has kernel backtraces
    // including sched_ext_dump output).
    let stack_funcs = if let Some(sched) = sched_output {
        let funcs = extract_stack_functions_all(sched);
        if funcs.is_empty() {
            eprintln!("ktstr_test: auto-repro: no functions from COM2, trying COM1");
            extract_stack_functions_all(console_output)
        } else {
            funcs
        }
    } else {
        eprintln!("ktstr_test: auto-repro: no scheduler output on COM2, trying COM1");
        extract_stack_functions_all(console_output)
    };
    let func_names: Vec<String> = stack_funcs.iter().map(|f| f.raw_name.clone()).collect();

    // When no stack functions were extracted (e.g. BPF text error with no
    // backtrace), still boot the repro VM. The guest-side discover_bpf_symbols()
    // dynamically finds the scheduler's BPF programs. Pass a sentinel value
    // so extract_probe_stack_arg returns Some and the guest probe path activates.
    let probe_arg = if func_names.is_empty() {
        eprintln!("ktstr_test: auto-repro: no stack functions, using BPF discovery in repro VM");
        format!("--ktstr-probe-stack={DISCOVER_SENTINEL}")
    } else {
        eprintln!(
            "ktstr_test: auto-repro: probing {} functions in second VM",
            func_names.len()
        );
        format!("--ktstr-probe-stack={}", func_names.join(","))
    };

    // Build guest args for the repro VM.
    let guest_args = vec![
        "run".to_string(),
        "--ktstr-test-fn".to_string(),
        entry.name.to_string(),
        probe_arg,
    ];

    let cmdline_extra = super::runtime::build_cmdline_extra(entry);

    let (vm_topology, memory_mb) = super::runtime::resolve_vm_topology(entry, topo);

    let no_perf_mode = std::env::var("KTSTR_NO_PERF_MODE").is_ok();
    let mut builder = super::runtime::build_vm_builder_base(
        entry,
        kernel,
        ktstr_bin,
        scheduler,
        vm_topology,
        memory_mb,
        &cmdline_extra,
        &guest_args,
        no_perf_mode,
    );

    {
        let mut args: Vec<String> = Vec::new();
        if let Some((archive_path, host_path, guest_path)) = config_file_parts(entry) {
            builder = builder.include_files(vec![(archive_path, host_path)]);
            args.push("--config".to_string());
            args.push(guest_path);
        }
        super::runtime::append_base_sched_args(entry, &mut args);
        if !args.is_empty() {
            builder = builder.sched_args(&args);
        }
    }

    let vm = match builder.build() {
        Ok(vm) => vm,
        Err(e) => {
            eprintln!("ktstr_test: auto-repro: failed to build VM: {e:#}");
            return None;
        }
    };

    let repro_result = match vm.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ktstr_test: auto-repro: VM run failed: {e:#}");
            return None;
        }
    };

    // Forward guest stderr (COM1) and COM2 probe lines when verbose.
    if verbose() {
        eprintln!(
            "ktstr_test: auto-repro: COM1 stderr length={} COM2 stdout length={}",
            repro_result.stderr.len(),
            repro_result.output.len(),
        );
        for line in repro_result.stderr.lines() {
            eprintln!("  repro-vm-com1: {line}");
        }
        let mut in_probe = false;
        for line in repro_result.output.lines() {
            if line.contains("ktstr_test: probe:") {
                in_probe = true;
            }
            if in_probe {
                eprintln!("  repro-vm-com2: {line}");
            }
        }
    }

    // Extract probe JSON from the repro VM and format on the host with
    // kernel_dir so blazesym can resolve source locations via vmlinux
    // DWARF. derive_kernel_dir handles both build-tree and cache-entry
    // layouts; for Local cache entries whose source tree is still on
    // disk, prefer_source_tree_for_dwarf re-routes blazesym to the
    // unstripped vmlinux in the source tree. Tarball/git cache entries
    // still can't recover file:line — stripped cache vmlinux is all
    // we have.
    let kernel_dir = crate::kernel_path::derive_kernel_dir(kernel)
        .map(|dir| crate::cache::prefer_source_tree_for_dwarf(&dir).unwrap_or(dir))
        .and_then(|p| p.to_str().map(String::from));
    let kernel_dir_str = kernel_dir.as_deref();
    let probe_section = extract_probe_output(&repro_result.output, kernel_dir_str);

    // Build diagnostic tails from the repro VM's output.
    const REPRO_TAIL_LINES: usize = 40;

    let sched_log_tail = parse_sched_output(&repro_result.output).and_then(|log| {
        let collapsed = crate::verifier::collapse_cycles(log);
        format_tail(&collapsed, REPRO_TAIL_LINES, "repro VM scheduler log")
    });

    let dump_tail = extract_sched_ext_dump(&repro_result.stderr)
        .and_then(|dump| format_tail(&dump, REPRO_TAIL_LINES, "repro VM sched_ext dump"));

    // Filter sched_ext_dump lines from dmesg tail to avoid duplicating
    // the dump section. Only non-dump kernel console lines are shown.
    let dmesg_filtered: String = repro_result
        .stderr
        .lines()
        .filter(|l| !l.contains("sched_ext_dump"))
        .collect::<Vec<_>>()
        .join("\n");
    let dmesg_tail = format_tail(&dmesg_filtered, REPRO_TAIL_LINES, "repro VM dmesg");

    let tails: Vec<String> = [sched_log_tail, dump_tail, dmesg_tail]
        .into_iter()
        .flatten()
        .collect();

    if probe_section.is_none() && tails.is_empty() {
        return None;
    }

    let has_probe = probe_section.is_some();
    let mut out = probe_section.unwrap_or_default();

    // Crash reproduction status when probe data is absent.
    if !has_probe {
        out.push_str(&classify_repro_vm_status(
            repro_result.timed_out,
            repro_result.crash_message.is_some(),
            &repro_result.output,
            repro_result.exit_code,
        ));
    }

    // Duration line before tails.
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(&format!(
        "repro VM duration: {:.1}s",
        repro_result.duration.as_secs_f64(),
    ));

    for tail in &tails {
        out.push_str("\n\n");
        out.push_str(tail);
    }
    Some(out)
}

/// Extract probe JSON from guest COM2, deserialize, and format on the
/// host where vmlinux (DWARF) is available for source locations.
pub(crate) fn extract_probe_output(output: &str, kernel_dir: Option<&str>) -> Option<String> {
    let json = crate::probe::output::extract_section(output, PROBE_OUTPUT_START, PROBE_OUTPUT_END);
    if json.is_empty() {
        return None;
    }
    let payload: ProbeBytes = match serde_json::from_str(&json) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ktstr_test: probe payload deserialize failed: {e}");
            return None;
        }
    };
    let mut out = String::new();

    // Append pipeline diagnostics if present.
    if let Some(ref diag) = payload.diagnostics {
        out.push_str(&format_probe_diagnostics(&diag.pipeline, &diag.skeleton));
    }

    if payload.events.is_empty() {
        if out.is_empty() {
            return None;
        }
        return Some(out);
    }
    out.push_str(&crate::probe::output::format_probe_events_with_bpf_locs(
        &payload.events,
        &payload.func_names,
        kernel_dir,
        &payload.bpf_source_locs,
        payload.nr_cpus,
        &payload.param_names,
        &payload.render_hints,
    ));
    Some(out)
}

/// Format probe pipeline diagnostics into a human-readable summary.
pub(crate) fn format_probe_diagnostics(
    pipeline: &PipelineDiagnostics,
    skeleton: &crate::probe::process::ProbeDiagnostics,
) -> String {
    let mut out = String::new();
    out.push_str("--- probe pipeline ---\n");

    // Stage 1: extraction
    out.push_str(&format!(
        "  extracted:   {} functions from crash backtrace\n",
        pipeline.stack_extracted,
    ));

    // Stage 2: filter
    let passed = pipeline.stack_extracted as usize - pipeline.filter_dropped.len();
    if pipeline.filter_dropped.is_empty() {
        out.push_str(&format!("  traceable:   {passed} passed filter\n"));
    } else {
        out.push_str(&format!(
            "  traceable:   {passed} passed, {} dropped: {}\n",
            pipeline.filter_dropped.len(),
            pipeline.filter_dropped.join(", "),
        ));
    }

    // Stage 3: BPF discovery
    out.push_str(&format!(
        "  bpf_discover: {} programs found\n",
        pipeline.bpf_discovered,
    ));

    // Stage 4: expansion
    out.push_str(&format!(
        "  after_expand: {} total probe targets\n",
        pipeline.total_after_expand,
    ));

    // Stage 5: kprobe attach
    if skeleton.kprobe_attach_failed.is_empty() {
        out.push_str(&format!(
            "  kprobes:     {} attached\n",
            skeleton.kprobe_attached,
        ));
    } else {
        out.push_str(&format!(
            "  kprobes:     {} attached, {} failed: {}\n",
            skeleton.kprobe_attached,
            skeleton.kprobe_attach_failed.len(),
            skeleton
                .kprobe_attach_failed
                .iter()
                .map(|(n, e)| format!("{n} ({e})"))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    if !skeleton.kprobe_resolve_failed.is_empty() {
        out.push_str(&format!(
            "  kprobe_miss: {} unresolved: {}\n",
            skeleton.kprobe_resolve_failed.len(),
            skeleton.kprobe_resolve_failed.join(", "),
        ));
    }

    // Stage 6: fentry attach
    if skeleton.fentry_candidates > 0 {
        if skeleton.fentry_attach_failed.is_empty() {
            out.push_str(&format!(
                "  fentry:      {} attached\n",
                skeleton.fentry_attached,
            ));
        } else {
            out.push_str(&format!(
                "  fentry:      {} attached, {} failed: {}\n",
                skeleton.fentry_attached,
                skeleton.fentry_attach_failed.len(),
                skeleton
                    .fentry_attach_failed
                    .iter()
                    .map(|(n, e)| format!("{n} ({e})"))
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
    }

    // Stage 7: trigger
    let trigger_type = if skeleton.trigger_type.is_empty() {
        "unknown"
    } else {
        &skeleton.trigger_type
    };
    if let Some(ref err) = skeleton.trigger_attach_error {
        out.push_str(&format!("  trigger:     attach failed ({err})\n"));
    } else {
        out.push_str(&format!(
            "  trigger:     {} ({})\n",
            if skeleton.trigger_fired {
                "fired"
            } else {
                "not fired"
            },
            trigger_type,
        ));
    }

    // Stage 8: capture
    out.push_str(&format!(
        "  probe_data:  {} keys, {} unmatched IPs\n",
        skeleton.probe_data_keys, skeleton.probe_data_unmatched_ips,
    ));

    // Stage 9: events + stitching
    out.push_str(&format!(
        "  events:      {} captured, {} after stitch\n",
        skeleton.events_before_stitch, skeleton.events_after_stitch,
    ));

    // Stage 10: BPF-side counters
    if skeleton.bpf_kprobe_fires > 0
        || skeleton.bpf_trigger_fires > 0
        || skeleton.bpf_meta_misses > 0
    {
        out.push_str(&format!(
            "  bpf_counts:  {} kprobe fires, {} trigger fires, {} meta misses\n",
            skeleton.bpf_kprobe_fires, skeleton.bpf_trigger_fires, skeleton.bpf_meta_misses,
        ));
        if !skeleton.bpf_miss_ips.is_empty() {
            let ips: Vec<String> = skeleton
                .bpf_miss_ips
                .iter()
                .map(|ip| format!("0x{ip:x}"))
                .collect();
            out.push_str(&format!("  miss_ips:    {}\n", ips.join(", ")));
        }
    }

    out
}

/// Guest-side dispatch: check for `--ktstr-test-fn=NAME` in args, run the
/// registered function, write the result to SHM and stdout (COM2),
/// and exit. Profraw data is flushed via `try_flush_profraw()`
/// inline on both the success and failure paths before
/// `std::process::exit()` is invoked.
///
/// Called from `ktstr_test_early_dispatch()` (ctor) before `main()`, or
/// from `ktstr_guest_init()` when running as PID 1.
///
/// When called from PID 1 context, args must be pre-loaded into the
/// process args (the caller reads `/args` from the initramfs).
/// Returns `Some(exit_code)` if dispatched, `None` if not an
/// ktstr_test invocation.
pub(crate) fn maybe_dispatch_vm_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    maybe_dispatch_vm_test_with_args(&args)
}

/// Guest-side scenario context prelude shared by every VM-dispatch
/// entry point in this module.
///
/// `maybe_dispatch_vm_test_with_args` and
/// `maybe_dispatch_vm_test_with_phase_a` both construct a
/// [`crate::scenario::Ctx`] around the same topology / cgroup /
/// sched_pid / assert-merge inputs — the only difference is whether
/// a probe thread is attached. Moving the inputs into a single
/// helper keeps the two dispatch paths in sync so a change to the
/// settle duration, assert merge chain, or sysfs fallback behaviour
/// lands in both without drift.
///
/// Returns `(topo, cgroups, sched_pid, merged_assert)`. The caller
/// owns the returned values for the lifetime of the `Ctx` it builds;
/// `Ctx` fields that borrow from them (`&topo`, `&cgroups`) stay
/// valid until the caller drops this tuple.
fn build_dispatch_ctx_parts(
    entry: &KtstrTestEntry,
    args: &[String],
) -> (
    crate::topology::TestTopology,
    crate::cgroup::CgroupManager,
    Option<libc::pid_t>,
    crate::assert::Assert,
) {
    // Sysfs is ground truth: CPUID, ACPI MADT, and MPTABLE all
    // express the VM's actual topology. Fall back to from_vm_topology
    // only when sysfs read fails.
    let topo = match crate::topology::TestTopology::from_system() {
        Ok(sys) => sys,
        Err(e) => {
            eprintln!("ktstr_test: topology from sysfs failed ({e}), using VM spec fallback");
            crate::topology::TestTopology::from_vm_topology(&entry.topology)
        }
    };
    let cgroup_root = resolve_cgroup_root(args);
    let cgroups = crate::cgroup::CgroupManager::new(&cgroup_root);
    if let Err(e) = cgroups.setup(false) {
        eprintln!("ktstr_test: cgroup setup failed: {e}");
    }
    let sched_pid = std::env::var("SCHED_PID")
        .ok()
        .and_then(|s| s.parse::<libc::pid_t>().ok())
        .filter(|&pid| pid != 0);
    // Three-layer merge: default_checks → scheduler.assert → entry.assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(entry.scheduler.assert())
        .merge(&entry.assert);
    (topo, cgroups, sched_pid, merged_assert)
}

/// Like `maybe_dispatch_vm_test` but with explicit args. Used by
/// `ktstr_guest_init()` which reads args from `/args` in the initramfs.
///
/// The caller (`ktstr_guest_init`) must have invoked
/// [`propagate_rust_env_from_cmdline`] before any probe / workload
/// thread was spawned. Doing the env propagation here would race
/// with the probe thread `start_probe_phase_a` spawned in the split
/// path, so it lives in the caller instead.
pub(crate) fn maybe_dispatch_vm_test_with_args(args: &[String]) -> Option<i32> {
    let name = extract_test_fn_arg(args)?;

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    // Parse --ktstr-probe-stack=func1,func2,... for auto-repro mode.
    let probe_stack = extract_probe_stack_arg(args);

    // Parse --ktstr-work-type=NAME for work type override.
    let work_type_override = extract_work_type_arg(args).and_then(|s| {
        crate::workload::WorkType::from_name(&s).or_else(|| {
            // `from_name` is exact-match on the PascalCase canonical
            // form. A user typo (`cpuspin`, `CPUSPIN`) lands here;
            // call `WorkType::suggest` for the canonical spelling
            // and surface it in the diagnostic so the user doesn't
            // have to guess the correct casing.
            match crate::workload::WorkType::suggest(&s) {
                Some(canonical) => eprintln!(
                    "ktstr_test: unknown work type '{s}'; did you mean \
                     '{canonical}'? Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
                None => eprintln!(
                    "ktstr_test: unknown work type '{s}'. Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
            }
            None
        })
    });

    // Set up BPF probes if --ktstr-probe-stack was provided.
    let pipeline = ProbePipeline::new();
    let probe_stop = pipeline.stop.clone();
    let probe_handle: Option<ProbeHandle> = probe_stack.as_ref().and_then(|stack_input| {
        use crate::probe::stack::load_probe_stack;

        eprintln!("ktstr_test: probe: loading probe stack from --ktstr-probe-stack");
        let mut pipe_diag = PipelineDiagnostics::default();
        let raw_functions = load_probe_stack(stack_input);
        pipe_diag.stack_extracted = raw_functions.len() as u32;
        let pre_filter: Vec<String> = raw_functions.iter().map(|f| f.raw_name.clone()).collect();
        let mut functions = crate::probe::stack::filter_traceable(raw_functions);
        // Record which functions were dropped by filter_traceable.
        for name in &pre_filter {
            if !functions.iter().any(|f| f.raw_name == *name) {
                pipe_diag.filter_dropped.push(name.clone());
            }
        }
        // Discover BPF scheduler functions from the running scheduler.
        // Stack-extracted BPF names have stale prog IDs from the first VM;
        // discover_bpf_symbols finds the current scheduler's programs.
        let stack_display_names: Vec<&str> = functions
            .iter()
            .filter(|f| f.is_bpf)
            .map(|f| f.display_name.as_str())
            .collect();
        let bpf_syms = crate::probe::btf::discover_bpf_symbols(&stack_display_names);
        pipe_diag.bpf_discovered = bpf_syms.len() as u32;
        if !bpf_syms.is_empty() {
            eprintln!(
                "ktstr_test: probe: {} BPF symbols discovered",
                bpf_syms.len()
            );
            functions.extend(bpf_syms);
        }
        // Expand BPF functions to kernel-side callers for bridge kprobes,
        // keeping BPF functions for fentry attachment.
        let functions = crate::probe::stack::expand_bpf_to_kernel_callers(functions);
        pipe_diag.total_after_expand = functions.len() as u32;
        if functions.is_empty() {
            eprintln!("ktstr_test: no traceable functions from --ktstr-probe-stack");
            return None;
        }

        eprintln!(
            "ktstr_test: probe: {} functions loaded, spawning probe thread",
            functions.len()
        );

        // Resolve BTF signatures for kernel functions so probe output
        // gets decoded field names instead of raw register values.
        let kernel_names: Vec<&str> = functions
            .iter()
            .filter(|f| !f.is_bpf)
            .map(|f| f.raw_name.as_str())
            .collect();
        let mut btf_funcs = crate::probe::btf::parse_btf_functions(&kernel_names, None);
        // Parse BPF function signatures from BPF program BTF.
        let bpf_btf_args: Vec<(&str, u32)> = functions
            .iter()
            .filter(|f| f.is_bpf)
            .filter_map(|f| Some((f.display_name.as_str(), f.bpf_prog_id?)))
            .collect();
        if !bpf_btf_args.is_empty() {
            btf_funcs.extend(crate::probe::btf::parse_bpf_btf_functions(&bpf_btf_args));
        }

        // Build func_names from the filtered list so indices match
        // the func_idx values assigned by run_probe_skeleton.
        let func_names: Vec<(u32, String)> = functions
            .iter()
            .enumerate()
            .map(|(i, f)| (i as u32, f.display_name.clone()))
            .collect();

        // Pre-open BPF program FDs while the scheduler is alive.
        // Holding these FDs keeps programs alive via kernel refcounting
        // even after the scheduler crashes.
        let bpf_fds = crate::probe::process::open_bpf_prog_fds(&functions);
        let pnames = crate::probe::output::build_param_names(&btf_funcs);
        let rhints = crate::probe::output::build_render_hints(&btf_funcs);
        let pnames_thread = pnames.clone();
        let rhints_thread = rhints.clone();
        let thread_pipeline = pipeline.clone();
        let funcs = functions.clone();
        let fn_names = func_names.clone();
        let pd = pipe_diag.clone();
        let handle = std::thread::spawn(move || {
            use crate::probe::process::run_probe_skeleton;
            let (events, diag, accumulated_fn_names) = run_probe_skeleton(
                &funcs,
                &btf_funcs,
                &thread_pipeline.stop,
                &bpf_fds,
                &thread_pipeline.probes_ready,
                None,
            );
            let emit_fn_names = if accumulated_fn_names.is_empty() {
                &fn_names
            } else {
                &accumulated_fn_names
            };
            // Serialize probe output after the trigger fires or stop
            // is signaled. Runs before the thread returns so output
            // reaches COM2 even if the main thread is blocked.
            emit_probe_payload(
                events.as_deref().unwrap_or(&[]),
                emit_fn_names,
                &pd,
                &diag,
                &pnames_thread,
                &rhints_thread,
            );
            thread_pipeline
                .output_done
                .store(true, std::sync::atomic::Ordering::Release);
            (events, diag, accumulated_fn_names)
        });

        // Wait for probes to attach before starting the test function.
        // Without this, the test may crash the scheduler before probes
        // are active, resulting in 0 captured events.
        pipeline.probes_ready.wait();

        Some(ProbeHandle {
            thread: handle,
            func_names,
            pipeline_diag: pipe_diag,
            output_done: pipeline.output_done.clone(),
            param_names: pnames,
            render_hints: rhints,
        })
    });

    let (topo, cgroups, sched_pid, merged_assert) = build_dispatch_ctx_parts(entry, args);
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo)
        .duration(entry.duration)
        .workers_per_cgroup(entry.workers_per_cgroup as usize)
        .sched_pid(sched_pid)
        .settle(Duration::from_millis(500))
        .work_type_override(work_type_override)
        .assert(merged_assert)
        .wait_for_map_write(!entry.bpf_map_write.is_empty())
        .build();

    let result = match (entry.func)(&ctx) {
        Ok(r) => r,
        Err(e) => {
            let r = AssertResult {
                passed: false,
                skipped: false,
                details: vec![format!("{e:#}").into()],
                stats: Default::default(),
            };
            publish_result_and_collect(&r, probe_stop, probe_handle);
            return Some(1);
        }
    };

    let exit_code = if result.passed { 0 } else { 1 };
    publish_result_and_collect(&result, probe_stop, probe_handle);
    Some(exit_code)
}

/// Result returned by the probe thread: collected events, skeleton
/// diagnostics, and accumulated function names from both phases.
type ProbeThreadResult = (
    Option<Vec<crate::probe::process::ProbeEvent>>,
    crate::probe::process::ProbeDiagnostics,
    Vec<(u32, String)>,
);

/// Probe-thread handle and associated state returned by the setup path.
///
/// Owns the join handle plus everything `collect_and_print_probe_data`
/// needs on the stop side: function-name registry for event rendering,
/// pipeline diagnostics captured before skeleton spawn, the
/// `output_done` flag the thread flips when it has already written
/// `PROBE_PAYLOAD_*` to COM2, and the param-name / render-hint maps
/// used to pretty-print parameters.
struct ProbeHandle {
    thread: std::thread::JoinHandle<ProbeThreadResult>,
    func_names: Vec<(u32, String)>,
    pipeline_diag: PipelineDiagnostics,
    output_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    param_names: std::collections::HashMap<String, Vec<(String, String)>>,
    render_hints: std::collections::HashMap<String, crate::probe::btf::RenderHint>,
}

/// Cross-thread probe pipeline signals.
///
/// Groups the three signals the probe setup path has to hand to its
/// worker thread: `stop` (main thread asks the probe thread to shut
/// down), `output_done` (probe thread tells the main thread it has
/// already emitted `PROBE_PAYLOAD_*`), and `probes_ready` (probe
/// thread signals the main thread that kprobes/kfentries have
/// attached). `stop` and `output_done` remain `AtomicBool` because
/// they are consulted inside the probe-thread's ring-buffer poll loop
/// where a blocking wait would stall diagnostics collection;
/// `probes_ready` uses [`crate::sync::Latch`] so the
/// dispatch path blocks on a condvar instead of sleep-polling.
/// [`Clone`] is the expected way to produce the thread-side view
/// before calling `std::thread::spawn` — each clone bumps refcounts
/// only.
#[derive(Clone, Default)]
pub(crate) struct ProbePipeline {
    pub stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub output_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub probes_ready: std::sync::Arc<crate::sync::Latch>,
}

impl ProbePipeline {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Pre-skeleton pipeline diagnostics captured during guest probe setup.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PipelineDiagnostics {
    /// Functions from --ktstr-probe-stack before filter.
    pub stack_extracted: u32,
    /// Functions dropped by filter_traceable.
    pub filter_dropped: Vec<String>,
    /// BPF symbols discovered from running scheduler.
    pub bpf_discovered: u32,
    /// Functions after expand_bpf_to_kernel_callers.
    pub total_after_expand: u32,
}

/// State from Phase A probe attachment (before scheduler starts).
///
/// Returned by `start_probe_phase_a`. Contains the probe thread handle,
/// the channel to send Phase B input (BPF fentry targets), and metadata
/// needed by the readout phase.
pub(crate) struct ProbePhaseAState {
    pub handle: std::thread::JoinHandle<ProbeThreadResult>,
    pub phase_b_tx: std::sync::mpsc::Sender<crate::probe::process::PhaseBInput>,
    /// Shared pipeline atomics (`stop`, `output_done`, `probes_ready`)
    /// — grouped so Phase B consumers thread a single value through
    /// the join + publish tail rather than tracking each `Arc` by hand.
    pub pipeline: ProbePipeline,
    pub kernel_func_names: Vec<(u32, String)>,
    /// Number of functions in Phase A. Phase B uses this as func_idx_offset
    /// to avoid index collisions in the shared BPF maps.
    pub kernel_func_count: u32,
    pub pipe_diag: PipelineDiagnostics,
    pub param_names: std::collections::HashMap<String, Vec<(String, String)>>,
    pub render_hints: std::collections::HashMap<String, crate::probe::btf::RenderHint>,
}

/// Start Phase A of the probe pipeline (before scheduler starts).
///
/// Parses `--ktstr-probe-stack` from args, loads kernel functions,
/// attaches kprobes + trigger + kernel fexit, and spawns the probe
/// thread with a Phase B channel. Returns `None` if no probe stack
/// arg is present or no traceable functions remain.
pub(crate) fn start_probe_phase_a(args: &[String]) -> Option<ProbePhaseAState> {
    use crate::probe::stack::{filter_traceable, load_probe_stack};

    let stack_input = extract_probe_stack_arg(args)?;

    eprintln!("ktstr_test: probe phase_a: loading kernel functions");
    let mut pipe_diag = PipelineDiagnostics::default();
    let raw_functions = load_probe_stack(&stack_input);
    pipe_diag.stack_extracted = raw_functions.len() as u32;
    let pre_filter: Vec<String> = raw_functions.iter().map(|f| f.raw_name.clone()).collect();
    let functions = filter_traceable(raw_functions);
    for name in &pre_filter {
        if !functions.iter().any(|f| f.raw_name == *name) {
            pipe_diag.filter_dropped.push(name.clone());
        }
    }

    // Phase A only processes kernel functions (non-BPF). BPF functions
    // are handled in Phase B after the scheduler starts.
    let kernel_functions: Vec<crate::probe::stack::StackFunction> =
        functions.into_iter().filter(|f| !f.is_bpf).collect();

    // Resolve BTF for kernel functions.
    let kernel_names: Vec<&str> = kernel_functions
        .iter()
        .map(|f| f.raw_name.as_str())
        .collect();
    let btf_funcs = crate::probe::btf::parse_btf_functions(&kernel_names, None);

    let func_names: Vec<(u32, String)> = kernel_functions
        .iter()
        .enumerate()
        .map(|(i, f)| (i as u32, f.display_name.clone()))
        .collect();

    pipe_diag.total_after_expand = kernel_functions.len() as u32;

    let bpf_fds = std::collections::HashMap::new(); // No BPF FDs in Phase A
    let param_names = crate::probe::output::build_param_names(&btf_funcs);
    let render_hints = crate::probe::output::build_render_hints(&btf_funcs);

    let pipeline = ProbePipeline::new();

    let (phase_b_tx, phase_b_rx) = std::sync::mpsc::channel();

    let thread_pipeline = pipeline.clone();
    let funcs = kernel_functions.clone();
    let btf = btf_funcs.clone();
    let fn_names = func_names.clone();
    let pd = pipe_diag.clone();
    let pnames = param_names.clone();
    let rhints = render_hints.clone();

    let handle = std::thread::spawn(move || {
        let (events, diag, accumulated_fn_names) = crate::probe::process::run_probe_skeleton(
            &funcs,
            &btf,
            &thread_pipeline.stop,
            &bpf_fds,
            &thread_pipeline.probes_ready,
            Some(phase_b_rx),
        );
        let emit_fn_names = if accumulated_fn_names.is_empty() {
            &fn_names
        } else {
            &accumulated_fn_names
        };
        emit_probe_payload(
            events.as_deref().unwrap_or(&[]),
            emit_fn_names,
            &pd,
            &diag,
            &pnames,
            &rhints,
        );
        thread_pipeline
            .output_done
            .store(true, std::sync::atomic::Ordering::Release);
        (events, diag, accumulated_fn_names)
    });

    // Wait for Phase A probes (kprobes + trigger + kernel fexit) to attach.
    pipeline.probes_ready.wait();

    eprintln!(
        "ktstr_test: probe phase_a: {} kernel functions attached, waiting for Phase B",
        kernel_functions.len(),
    );

    let kernel_func_count = kernel_functions.len() as u32;

    Some(ProbePhaseAState {
        handle,
        phase_b_tx,
        pipeline,
        kernel_func_names: func_names,
        kernel_func_count,
        pipe_diag,
        param_names,
        render_hints,
    })
}

/// Complete the probe pipeline with Phase B (after scheduler starts).
///
/// Discovers BPF symbols from the running scheduler, opens BPF prog
/// FDs, sends Phase B input to the probe thread, waits for Phase B
/// attachment, then runs the test function and collects probe output.
///
/// Returns `Some(exit_code)` if dispatched, `None` if not.
pub(crate) fn maybe_dispatch_vm_test_with_phase_a(
    args: &[String],
    pa: ProbePhaseAState,
) -> Option<i32> {
    use crate::probe::btf::discover_bpf_symbols;
    use crate::probe::stack::expand_bpf_to_kernel_callers;

    // Env propagation cannot happen here: `pa` holds a live probe
    // thread spawned by `start_probe_phase_a`, so mutating
    // `std::env::__environ` now would race with that thread. The
    // caller (`ktstr_guest_init`) invokes `propagate_rust_env_from_cmdline`
    // before Phase A spawns the thread.
    let name = extract_test_fn_arg(args)?;

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    let work_type_override = extract_work_type_arg(args).and_then(|s| {
        crate::workload::WorkType::from_name(&s).or_else(|| {
            // `from_name` is exact-match on the PascalCase canonical
            // form. A user typo (`cpuspin`, `CPUSPIN`) lands here;
            // call `WorkType::suggest` for the canonical spelling
            // and surface it in the diagnostic so the user doesn't
            // have to guess the correct casing.
            match crate::workload::WorkType::suggest(&s) {
                Some(canonical) => eprintln!(
                    "ktstr_test: unknown work type '{s}'; did you mean \
                     '{canonical}'? Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
                None => eprintln!(
                    "ktstr_test: unknown work type '{s}'. Valid types: {:?}",
                    crate::workload::WorkType::ALL_NAMES,
                ),
            }
            None
        })
    });

    // Destructure Phase A state up front so later branches (Phase B
    // send / drop, handle construction, stop propagation) operate on
    // owned locals. Keeping `pa` whole across `drop(pa.phase_b_tx)`
    // would partial-move the value and block the final `ProbeHandle`
    // build.
    let ProbePhaseAState {
        handle: pa_handle,
        phase_b_tx: pa_phase_b_tx,
        pipeline: pa_pipeline,
        kernel_func_names: pa_kernel_func_names,
        kernel_func_count: pa_kernel_func_count,
        pipe_diag: pa_pipe_diag,
        param_names: pa_param_names,
        render_hints: pa_render_hints,
    } = pa;

    // Phase B: discover BPF symbols from the running scheduler.
    eprintln!("ktstr_test: probe phase_b: discovering BPF symbols");
    let stack_display_names: Vec<&str> = Vec::new(); // Discovery uses empty hint list
    let bpf_syms = discover_bpf_symbols(&stack_display_names);
    eprintln!(
        "ktstr_test: probe phase_b: {} BPF symbols discovered",
        bpf_syms.len()
    );

    if !bpf_syms.is_empty() {
        // Expand BPF to kernel callers. Both BPF callbacks (for fentry)
        // and kernel callers (for additional kprobes) are included in
        // Phase B input.
        let phase_b_functions = expand_bpf_to_kernel_callers(bpf_syms);

        // Open BPF program FDs while the scheduler is alive.
        let bpf_fds = crate::probe::process::open_bpf_prog_fds(&phase_b_functions);

        // Parse BPF function signatures from BPF program BTF.
        let bpf_btf_args: Vec<(&str, u32)> = phase_b_functions
            .iter()
            .filter(|f| f.is_bpf)
            .filter_map(|f| Some((f.display_name.as_str(), f.bpf_prog_id?)))
            .collect();
        let mut phase_b_btf = if !bpf_btf_args.is_empty() {
            crate::probe::btf::parse_bpf_btf_functions(&bpf_btf_args)
        } else {
            Vec::new()
        };
        // Parse BTF for kernel callers added by expand_bpf_to_kernel_callers.
        let kernel_caller_names: Vec<&str> = phase_b_functions
            .iter()
            .filter(|f| !f.is_bpf)
            .map(|f| f.raw_name.as_str())
            .collect();
        if !kernel_caller_names.is_empty() {
            phase_b_btf.extend(crate::probe::btf::parse_btf_functions(
                &kernel_caller_names,
                None,
            ));
        }

        let phase_b_done = std::sync::Arc::new(crate::sync::Latch::new());
        let phase_b_done_clone = phase_b_done.clone();

        let phase_b_input = crate::probe::process::PhaseBInput {
            functions: phase_b_functions,
            bpf_prog_fds: bpf_fds,
            btf_funcs: phase_b_btf,
            done: phase_b_done_clone,
            func_idx_offset: pa_kernel_func_count,
        };

        if let Err(e) = pa_phase_b_tx.send(phase_b_input) {
            eprintln!("ktstr_test: probe phase_b: failed to send: {e}");
        } else {
            // Wait for Phase B attachment to complete.
            phase_b_done.wait();
            eprintln!("ktstr_test: probe phase_b: BPF fentry attached");
        }
    } else {
        eprintln!("ktstr_test: probe phase_b: no BPF symbols, skipping fentry");
        // Drop the sender so the probe thread's try_recv sees Disconnected.
        drop(pa_phase_b_tx);
    }

    let (topo, cgroups, sched_pid, merged_assert) = build_dispatch_ctx_parts(entry, args);
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo)
        .duration(entry.duration)
        .workers_per_cgroup(entry.workers_per_cgroup as usize)
        .sched_pid(sched_pid)
        .settle(std::time::Duration::from_millis(500))
        .work_type_override(work_type_override)
        .assert(merged_assert)
        .wait_for_map_write(!entry.bpf_map_write.is_empty())
        .build();

    // Build the ProbeHandle up front from the destructured Phase A
    // locals — cheap (mostly Arc clones and already-owned Vecs) and
    // lets both the Ok and Err tails funnel through
    // `publish_result_and_collect` without re-assembling the handle.
    let stop = pa_pipeline.stop.clone();
    let handle = ProbeHandle {
        thread: pa_handle,
        func_names: pa_kernel_func_names,
        pipeline_diag: pa_pipe_diag,
        output_done: pa_pipeline.output_done,
        param_names: pa_param_names,
        render_hints: pa_render_hints,
    };
    let result = match (entry.func)(&ctx) {
        Ok(r) => r,
        Err(e) => {
            let r = AssertResult {
                passed: false,
                skipped: false,
                details: vec![format!("{e:#}").into()],
                stats: Default::default(),
            };
            publish_result_and_collect(&r, stop, Some(handle));
            return Some(1);
        }
    };

    let exit_code = if result.passed { 0 } else { 1 };
    publish_result_and_collect(&result, stop, Some(handle));
    Some(exit_code)
}

/// Serialized probe data sent from guest to host via COM2.
/// The host deserializes and formats with kernel_dir for source locations.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbeBytes {
    pub events: Vec<crate::probe::process::ProbeEvent>,
    pub func_names: Vec<(u32, String)>,
    pub bpf_source_locs: std::collections::HashMap<String, String>,
    pub diagnostics: Option<ProbeBytesDiagnostics>,
    /// Guest VM CPU count for cpumask masking. Populated by
    /// `emit_probe_payload` which runs inside the guest where
    /// sysfs reports the correct value.
    pub nr_cpus: Option<u32>,
    /// BTF-resolved parameter labels per function: func_name ->
    /// vec of (param_name, type_label). Used by the formatter to
    /// print named args instead of arg0/arg1.
    pub param_names: std::collections::HashMap<String, Vec<(String, String)>>,
    /// BTF-derived render hints for auto-discovered fields.
    /// Maps field key (e.g. `"ctx:task_ctx.data__sz"`) to display format.
    pub render_hints: std::collections::HashMap<String, crate::probe::btf::RenderHint>,
}

/// Combined diagnostics for the probe payload.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbeBytesDiagnostics {
    pub pipeline: PipelineDiagnostics,
    pub skeleton: crate::probe::process::ProbeDiagnostics,
}

/// Serialize probe payload to stdout (COM2) between delimiters.
/// Resolves BPF source locations from loaded programs before serializing.
fn emit_probe_payload(
    events: &[crate::probe::process::ProbeEvent],
    func_names: &[(u32, String)],
    pipeline_diag: &PipelineDiagnostics,
    skeleton_diag: &crate::probe::process::ProbeDiagnostics,
    param_names: &std::collections::HashMap<String, Vec<(String, String)>>,
    render_hints: &std::collections::HashMap<String, crate::probe::btf::RenderHint>,
) {
    let source_loc_names: Vec<&str> = func_names.iter().map(|(_, name)| name.as_str()).collect();
    let bpf_syms = crate::probe::btf::discover_bpf_symbols(&source_loc_names);
    let bpf_prog_ids: Vec<u32> = func_names
        .iter()
        .filter_map(|(_, name)| {
            bpf_syms
                .iter()
                .find(|s| s.display_name == *name)
                .and_then(|s| s.bpf_prog_id)
        })
        .collect();
    let bpf_source_locs = crate::probe::btf::resolve_bpf_source_locs(&bpf_prog_ids);

    let payload = ProbeBytes {
        events: events.to_vec(),
        func_names: func_names.to_vec(),
        bpf_source_locs,
        diagnostics: Some(ProbeBytesDiagnostics {
            pipeline: pipeline_diag.clone(),
            skeleton: skeleton_diag.clone(),
        }),
        nr_cpus: crate::probe::output::get_nr_cpus(),
        param_names: param_names.clone(),
        render_hints: render_hints.clone(),
    };
    println!("{PROBE_OUTPUT_START}");
    if let Ok(json) = serde_json::to_string(&payload) {
        println!("{json}");
    }
    println!("{PROBE_OUTPUT_END}");
}

/// Flush profraw, publish the assert result to guest stdout, then stop
/// and join the probe thread. Called on both success and error paths
/// at the tail of every dispatch entry point so the host sees the
/// verdict even when the guest aborts early. Owns the probe-handle
/// value so callers don't reuse it after publication.
fn publish_result_and_collect(
    result: &AssertResult,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
) {
    try_flush_profraw();
    print_assert_result(result);
    collect_and_print_probe_data(stop, handle);
}

/// Stop probes, join the probe thread. The probe thread emits output
/// directly when the trigger fires; this function only needs to set
/// `stop` and join. If the probe thread already emitted output, this
/// is a no-op.
fn collect_and_print_probe_data(
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
) {
    let Some(ph) = handle else {
        return;
    };

    stop.store(true, std::sync::atomic::Ordering::Release);
    let (events, skeleton_diag, accumulated_fn_names) = match ph.thread.join() {
        Ok((Some(events), diag, fnames)) => (events, diag, fnames),
        Ok((None, diag, fnames)) => (Vec::new(), diag, fnames),
        Err(_) => (
            Vec::new(),
            crate::probe::process::ProbeDiagnostics::default(),
            Vec::new(),
        ),
    };

    // Prefer accumulated func_names (includes both Phase A and Phase B).
    let effective_fn_names = if accumulated_fn_names.is_empty() {
        &ph.func_names
    } else {
        &accumulated_fn_names
    };

    // The probe thread already emitted output on trigger/stop.
    // Only emit here if it somehow didn't (e.g. thread panicked
    // before reaching emit_probe_payload).
    if !ph.output_done.load(std::sync::atomic::Ordering::Acquire) {
        emit_probe_payload(
            &events,
            effective_fn_names,
            &ph.pipeline_diag,
            &skeleton_diag,
            &ph.param_names,
            &ph.render_hints,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_probe_output_valid_json() {
        use crate::probe::process::ProbeEvent;
        let payload = ProbeBytes {
            events: vec![ProbeEvent {
                func_idx: 0,
                task_ptr: 1,
                ts: 100,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 42)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            }],
            func_names: vec![(0, "schedule".to_string())],
            bpf_source_locs: Default::default(),
            diagnostics: None,
            nr_cpus: None,
            param_names: Default::default(),
            render_hints: Default::default(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let output = format!("noise\n{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}\nmore");
        let parsed = extract_probe_output(&output, None);
        assert!(parsed.is_some());
        let formatted = parsed.unwrap();
        assert!(
            formatted.contains("schedule"),
            "should contain func name: {formatted}"
        );
        assert!(
            formatted.contains("pid"),
            "should contain field name: {formatted}"
        );
    }

    #[test]
    fn extract_probe_output_missing() {
        assert!(extract_probe_output("no markers", None).is_none());
    }

    #[test]
    fn extract_probe_output_empty() {
        let output = format!("{PROBE_OUTPUT_START}\n\n{PROBE_OUTPUT_END}");
        assert!(extract_probe_output(&output, None).is_none());
    }

    #[test]
    fn extract_probe_output_invalid_json() {
        let output = format!("{PROBE_OUTPUT_START}\nnot valid json\n{PROBE_OUTPUT_END}");
        assert!(extract_probe_output(&output, None).is_none());
    }

    #[test]
    fn extract_probe_output_enriched_fields() {
        use crate::probe::process::ProbeEvent;
        let payload = ProbeBytes {
            events: vec![
                ProbeEvent {
                    func_idx: 0,
                    task_ptr: 1,
                    ts: 100,
                    args: [0xDEAD, 0, 0, 0, 0, 0],
                    fields: vec![
                        ("prev:task_struct.pid".to_string(), 42),
                        ("prev:task_struct.scx_flags".to_string(), 0x1c),
                    ],
                    kstack: vec![],
                    str_val: None,
                    ..Default::default()
                },
                ProbeEvent {
                    func_idx: 1,
                    task_ptr: 1,
                    ts: 200,
                    args: [0; 6],
                    fields: vec![("rq:rq.cpu".to_string(), 3)],
                    kstack: vec![],
                    str_val: None,
                    ..Default::default()
                },
            ],
            func_names: vec![
                (0, "schedule".to_string()),
                (1, "pick_task_scx".to_string()),
            ],
            bpf_source_locs: Default::default(),
            diagnostics: None,
            nr_cpus: None,
            param_names: Default::default(),
            render_hints: Default::default(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let output = format!("{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}");
        let formatted = extract_probe_output(&output, None).unwrap();

        // Decoded fields present (not raw args).
        assert!(formatted.contains("pid"), "pid field: {formatted}");
        assert!(formatted.contains("42"), "pid value: {formatted}");
        assert!(
            formatted.contains("scx_flags"),
            "scx_flags field: {formatted}"
        );
        assert!(formatted.contains("cpu"), "cpu field: {formatted}");
        assert!(formatted.contains("3"), "cpu value: {formatted}");

        // Type header grouping for struct params.
        assert!(
            formatted.contains("task_struct *prev"),
            "type header for task_struct: {formatted}"
        );
        assert!(
            formatted.contains("rq *rq"),
            "type header for rq: {formatted}"
        );

        // Raw args suppressed when fields present.
        assert!(
            !formatted.contains("arg0"),
            "raw args should not appear when fields exist: {formatted}"
        );

        // Function names present.
        assert!(formatted.contains("schedule"), "func schedule: {formatted}");
        assert!(
            formatted.contains("pick_task_scx"),
            "func pick_task_scx: {formatted}"
        );
    }

    // -- format_tail --

    #[test]
    fn format_tail_empty_text_returns_none() {
        assert_eq!(format_tail("", 5, "scheduler"), None);
    }

    #[test]
    fn format_tail_fewer_lines_than_n_returns_all() {
        let out = format_tail("one\ntwo\nthree", 10, "scheduler").unwrap();
        assert_eq!(out, "--- scheduler ---\none\ntwo\nthree");
    }

    #[test]
    fn format_tail_trims_to_last_n_lines() {
        let out = format_tail("1\n2\n3\n4\n5", 3, "log").unwrap();
        assert_eq!(out, "--- log ---\n3\n4\n5");
    }

    #[test]
    fn format_tail_zero_n_returns_empty_body_under_header() {
        // saturating_sub keeps `start == lines.len()`, so the joined
        // slice is empty — the header alone survives.
        let out = format_tail("a\nb", 0, "hdr").unwrap();
        assert_eq!(out, "--- hdr ---\n");
    }

    #[test]
    fn format_tail_preserves_trailing_blank_lines() {
        // `str::lines` strips a single trailing newline but keeps
        // interior blanks. The tail should include the blank line.
        let out = format_tail("a\n\nb", 3, "hdr").unwrap();
        assert_eq!(out, "--- hdr ---\na\n\nb");
    }

    // -- parse_rust_env_from_cmdline --

    #[test]
    fn parse_rust_env_empty_cmdline_is_empty() {
        assert!(parse_rust_env_from_cmdline("").is_empty());
    }

    #[test]
    fn parse_rust_env_no_matches() {
        assert!(parse_rust_env_from_cmdline("console=ttyS0 ro quiet").is_empty());
    }

    #[test]
    fn parse_rust_env_backtrace_only() {
        let parsed = parse_rust_env_from_cmdline("console=ttyS0 RUST_BACKTRACE=1 ro");
        assert_eq!(parsed, vec![("RUST_BACKTRACE", "1")]);
    }

    #[test]
    fn parse_rust_env_log_only() {
        let parsed = parse_rust_env_from_cmdline("RUST_LOG=debug other=x");
        assert_eq!(parsed, vec![("RUST_LOG", "debug")]);
    }

    #[test]
    fn parse_rust_env_both() {
        let parsed = parse_rust_env_from_cmdline("RUST_BACKTRACE=full RUST_LOG=trace other=y");
        assert_eq!(
            parsed,
            vec![("RUST_BACKTRACE", "full"), ("RUST_LOG", "trace")]
        );
    }

    #[test]
    fn parse_rust_env_preserves_token_order() {
        let parsed = parse_rust_env_from_cmdline("RUST_LOG=info RUST_BACKTRACE=1");
        assert_eq!(parsed, vec![("RUST_LOG", "info"), ("RUST_BACKTRACE", "1")]);
    }

    #[test]
    fn parse_rust_env_empty_value() {
        // `RUST_LOG=` with no value yields an empty-string value,
        // matching the split semantics of `strip_prefix`.
        let parsed = parse_rust_env_from_cmdline("RUST_LOG=");
        assert_eq!(parsed, vec![("RUST_LOG", "")]);
    }

    #[test]
    fn parse_rust_env_ignores_prefix_mismatch() {
        // Tokens that merely contain the key substring but do not
        // start with it are ignored (e.g. `xRUST_LOG=...`).
        assert!(parse_rust_env_from_cmdline("xRUST_LOG=x").is_empty());
    }

    // -- extract_not_attached_reason --

    #[test]
    fn extract_not_attached_reason_timeout() {
        let output = "noise\nSCHEDULER_NOT_ATTACHED: timeout\nmore";
        assert_eq!(extract_not_attached_reason(output), Some("timeout"));
    }

    #[test]
    fn extract_not_attached_reason_sysfs_absent() {
        // Multi-word reason must survive through to the caller so the
        // user can distinguish "timeout" from "sched_ext sysfs absent".
        let output = "SCHEDULER_NOT_ATTACHED: sched_ext sysfs absent";
        assert_eq!(
            extract_not_attached_reason(output),
            Some("sched_ext sysfs absent"),
        );
    }

    #[test]
    fn extract_not_attached_reason_trims_trailing_whitespace() {
        // `write_com2` may append whitespace; the reason comparison
        // and display path should not expose that to the user.
        let output = "SCHEDULER_NOT_ATTACHED:  timeout  \n";
        assert_eq!(extract_not_attached_reason(output), Some("timeout"));
    }

    #[test]
    fn extract_not_attached_reason_absent_returns_none() {
        assert_eq!(extract_not_attached_reason(""), None);
        assert_eq!(
            extract_not_attached_reason("SCHEDULER_DIED\nKTSTR_EXIT=1"),
            None,
        );
    }

    #[test]
    fn extract_not_attached_reason_without_colon_returns_none() {
        // The sentinel token alone, with no `: reason` suffix, carries
        // no diagnostic value. `None` lets the caller fall through to
        // the generic abnormal-exit branch instead of surfacing an
        // empty reason.
        let output = "SCHEDULER_NOT_ATTACHED\nKTSTR_EXIT=1";
        assert_eq!(extract_not_attached_reason(output), None);
    }

    #[test]
    fn extract_not_attached_reason_empty_suffix_returns_none() {
        // `SCHEDULER_NOT_ATTACHED:` with no reason text is functionally
        // equivalent to no sentinel — no classification signal to
        // surface.
        let output = "SCHEDULER_NOT_ATTACHED:\n";
        assert_eq!(extract_not_attached_reason(output), None);
        let output_ws = "SCHEDULER_NOT_ATTACHED:   \n";
        assert_eq!(extract_not_attached_reason(output_ws), None);
    }

    #[test]
    fn extract_not_attached_reason_first_match_wins() {
        // Two sentinel lines should not be possible in production
        // (rust_init emits exactly one before force_reboot), but if
        // the harness ever concatenates outputs, pinning "first match"
        // keeps the classification stable.
        let output =
            "SCHEDULER_NOT_ATTACHED: timeout\nSCHEDULER_NOT_ATTACHED: sched_ext sysfs absent";
        assert_eq!(extract_not_attached_reason(output), Some("timeout"));
    }

    // -- classify_repro_vm_status --

    #[test]
    fn classify_repro_vm_status_timeout_wins_over_other_signals() {
        // Even with a crash message present, the VM-level timeout is
        // the primary classification — a timed-out VM may have dumped
        // any signal on the way out.
        let status = classify_repro_vm_status(
            /*timed_out*/ true,
            /*has_crash_message*/ true,
            "SCHEDULER_NOT_ATTACHED: timeout\nSCHEDULER_DIED",
            137,
        );
        assert_eq!(status, "repro VM: timed out");
    }

    #[test]
    fn classify_repro_vm_status_not_attached_with_reason() {
        let status = classify_repro_vm_status(
            false,
            false,
            "noise\nSCHEDULER_NOT_ATTACHED: sched_ext sysfs absent\nKTSTR_EXIT=1",
            1,
        );
        assert_eq!(
            status,
            "repro VM: scheduler did not attach (sched_ext sysfs absent) (exit code 1)",
        );
    }

    #[test]
    fn classify_repro_vm_status_not_attached_takes_precedence_over_crashed() {
        // The rust_init emission path writes SCHEDULER_NOT_ATTACHED and
        // then force-reboots before any SCHEDULER_DIED write could
        // happen. If both sentinels ever appear in the same output,
        // NOT_ATTACHED is the more specific classification and should
        // win.
        let status = classify_repro_vm_status(
            false,
            true,
            "SCHEDULER_DIED\nSCHEDULER_NOT_ATTACHED: timeout",
            1,
        );
        assert_eq!(
            status,
            "repro VM: scheduler did not attach (timeout) (exit code 1)",
        );
    }

    #[test]
    fn classify_repro_vm_status_crashed_from_sentinel() {
        // Positive exit code on the crash-sentinel branch → qemu
        // propagated a non-zero exit alongside the guest crash
        // sentinel. Clause format: "exited with non-zero status (N)".
        let status = classify_repro_vm_status(false, false, "SCHEDULER_DIED\n", 139);
        assert_eq!(
            status,
            "repro VM: scheduler crashed — exited with non-zero status (139)",
        );
    }

    #[test]
    fn classify_repro_vm_status_crashed_from_crash_message() {
        // crash_message set without a SCHEDULER_DIED sentinel (e.g.
        // VM-level crash detection from COM1) still routes to the
        // crashed branch. Positive exit code → non-zero-status clause.
        let status = classify_repro_vm_status(false, true, "no sentinels here", 134);
        assert_eq!(
            status,
            "repro VM: scheduler crashed — exited with non-zero status (134)",
        );
    }

    /// exit_code == 0 with a crash sentinel is the "guest panic
    /// handler + orderly reboot" case — qemu shut down cleanly but
    /// the guest emitted SCHEDULER_DIED. The old format conflated
    /// this with a true qemu-level crash; the branched format makes
    /// it unambiguous.
    #[test]
    fn classify_repro_vm_status_crashed_from_sentinel_qemu_clean_exit() {
        let status = classify_repro_vm_status(false, false, "SCHEDULER_DIED\n", 0);
        assert_eq!(status, "repro VM: scheduler crashed — exited cleanly");
    }

    /// Negative exit_code on the crash branch exercises the
    /// `<0` arm. The sign convention is the VMM's, not
    /// `std::process::ExitStatus`: `VmResult::exit_code`
    /// (vmm::mod.rs) is seeded to `-1` in the BSP run loop and
    /// left negative on watchdog-fire / non-normal exits, so
    /// negatives that reach `classify_repro_vm_status` are VMM
    /// sentinels rather than OS-reported signal codes
    /// (`ExitStatus::code()` returns `None`, never a negative
    /// i32, on signal-kill). Clause format: "killed by signal (N)".
    #[test]
    fn classify_repro_vm_status_crashed_from_sentinel_killed_by_signal() {
        let status = classify_repro_vm_status(false, false, "SCHEDULER_DIED\n", -9);
        assert_eq!(
            status,
            "repro VM: scheduler crashed — killed by signal (-9)",
        );
    }

    /// `exit_code == -1` on the crash branch is the VMM sentinel —
    /// `VmResult::exit_code` is seeded to `-1` at the top of the
    /// boot-CPU run loop and left there when the scheduler did not
    /// deliver its final exit message. Watchdog-fire is caught
    /// earlier via the `timed_out` branch, so a `-1` here means the
    /// boot-CPU ran a code-unsetting error path — not a signal-kill.
    /// Distinct clause so users don't misread this as "signal 1"
    /// (SIGHUP). The asserted string is phrased in end-user terms —
    /// the internals are in the implementation comment so the
    /// console output stays operator-readable without cross-
    /// referencing VMM source. A regression that swapped the clause
    /// back to "VMM exit-code sentinel" or any `BSP` /
    /// `VmResult::exit_code` / `MSG_TYPE_EXIT` phrasing would fail
    /// here.
    #[test]
    fn classify_repro_vm_status_crashed_from_sentinel_vmm_exit_code_unset() {
        let status = classify_repro_vm_status(false, false, "SCHEDULER_DIED\n", -1);
        assert_eq!(
            status,
            "repro VM: scheduler crashed — VM host reported no final exit \
             status (the scheduler did not deliver an exit signal before \
             the VM ended)",
        );
        // Negative assertions: none of the internal vocabulary may
        // leak into the user-facing status — each term is a
        // usability bug and must stay out of the rendered string.
        assert!(
            !status.contains("BSP"),
            "user-facing status leaks BSP: {status}"
        );
        assert!(
            !status.contains("VmResult::exit_code"),
            "user-facing status leaks VmResult::exit_code: {status}",
        );
        assert!(
            !status.contains("MSG_TYPE_EXIT"),
            "user-facing status leaks MSG_TYPE_EXIT: {status}",
        );
    }

    #[test]
    fn classify_repro_vm_status_abnormal_exit() {
        let status = classify_repro_vm_status(false, false, "clean output", 2);
        assert_eq!(status, "repro VM: exited abnormally (exit code 2)");
    }

    #[test]
    fn classify_repro_vm_status_clean_run() {
        let status = classify_repro_vm_status(false, false, "clean output", 0);
        assert_eq!(
            status,
            "repro VM: scheduler ran normally (crash did not reproduce)",
        );
    }

    #[test]
    fn classify_repro_vm_status_malformed_not_attached_falls_through() {
        // A SCHEDULER_NOT_ATTACHED token with no colon-delimited
        // reason does not count as a classification signal. With no
        // crash signals and exit_code=1 the result should be the
        // abnormal-exit branch, not a NOT_ATTACHED branch with an
        // empty reason.
        let status =
            classify_repro_vm_status(false, false, "SCHEDULER_NOT_ATTACHED\nKTSTR_EXIT=1", 1);
        assert_eq!(status, "repro VM: exited abnormally (exit code 1)");
    }
}
