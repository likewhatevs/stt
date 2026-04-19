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
use crate::vmm;

use super::args::{
    extract_probe_stack_arg, extract_test_fn_arg, extract_work_type_arg, resolve_cgroup_root,
};
use super::entry::find_test;
use super::output::{
    SCHED_OUTPUT_END, SCHED_OUTPUT_START, extract_sched_ext_dump, parse_sched_output,
    print_assert_result,
};
use super::profraw::try_flush_profraw;
use super::runtime::{KTSTR_TEST_SHM_SIZE, config_file_parts, verbose};
use super::{KtstrTestEntry, TopoOverride};

/// Sentinel value for `--ktstr-probe-stack` when no crash stack functions
/// were extracted. Triggers the guest-side probe path so
/// `discover_bpf_symbols()` can dynamically find the scheduler's BPF
/// programs. `filter_traceable` drops it (not in kallsyms).
const DISCOVER_SENTINEL: &str = "__discover__";

/// Propagate `RUST_BACKTRACE` and `RUST_LOG` from the guest kernel
/// cmdline into the process environment.
///
/// Must be called while the process is single-threaded — i.e. before
/// any probe thread is spawned (`start_probe_phase_a`) and before any
/// test thread runs. `std::env::set_var` is not thread-safe on Linux
/// because glibc mutates the global `__environ` array without locks;
/// a concurrent reader (or another `set_var`) produces UB. Callers on
/// the VM-boot path must invoke this from `ktstr_guest_init` after
/// `/args` is read and before `start_probe_phase_a` spawns the probe
/// thread.
pub fn propagate_rust_env_from_cmdline() {
    let Ok(cmdline) = std::fs::read_to_string("/proc/cmdline") else {
        return;
    };
    let parts: Vec<&str> = cmdline.split_whitespace().collect();
    if let Some(val) = parts
        .iter()
        .find(|s| s.starts_with("RUST_BACKTRACE="))
        .and_then(|s| s.strip_prefix("RUST_BACKTRACE="))
    {
        // SAFETY: called from ktstr_guest_init before any probe /
        // workload thread is spawned; single-threaded mutation of
        // `__environ` is sound.
        unsafe { std::env::set_var("RUST_BACKTRACE", val) };
    }
    if let Some(val) = parts
        .iter()
        .find(|s| s.starts_with("RUST_LOG="))
        .and_then(|s| s.strip_prefix("RUST_LOG="))
    {
        // SAFETY: see above.
        unsafe { std::env::set_var("RUST_LOG", val) };
    }
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
    let mut builder = vmm::KtstrVm::builder()
        .kernel(kernel)
        .init_binary(ktstr_bin)
        .with_topology(vm_topology)
        .memory_deferred_min(memory_mb)
        .cmdline(&cmdline_extra)
        .shm_size(KTSTR_TEST_SHM_SIZE)
        .run_args(&guest_args)
        .timeout(Duration::from_secs(60))
        .no_perf_mode(no_perf_mode);

    if let Some(sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
    }

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

    // Forward bpf_map_write and watchdog_timeout so the repro VM
    // reproduces the same exit as the first VM with probes attached.
    for bpf_write in entry.bpf_map_write {
        builder =
            builder.bpf_map_write(bpf_write.map_name_suffix, bpf_write.offset, bpf_write.value);
    }
    builder = builder.watchdog_timeout(entry.watchdog_timeout);

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
        let status = if repro_result.timed_out {
            "repro VM: timed out".to_string()
        } else if repro_result.crash_message.is_some()
            || repro_result.output.contains(super::SENTINEL_SCHEDULER_DIED)
        {
            format!(
                "repro VM: scheduler crashed (exit code {})",
                repro_result.exit_code,
            )
        } else if repro_result.exit_code != 0 {
            format!(
                "repro VM: exited abnormally (exit code {})",
                repro_result.exit_code,
            )
        } else {
            "repro VM: scheduler ran normally (crash did not reproduce)".to_string()
        };
        out.push_str(&status);
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
    let payload: ProbePayload = match serde_json::from_str(&json) {
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
    libc::pid_t,
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
        .unwrap_or(0);
    // Three-layer merge: default_checks → scheduler.assert → entry.assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
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
            eprintln!("ktstr_test: unknown work type '{s}'");
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
        while !pipeline
            .probes_ready
            .load(std::sync::atomic::Ordering::Acquire)
        {
            std::thread::sleep(Duration::from_millis(10));
        }

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
    let ctx = crate::scenario::Ctx {
        cgroups: &cgroups,
        topo: &topo,
        duration: entry.duration,
        workers_per_cgroup: entry.workers_per_cgroup as usize,
        sched_pid,
        settle: Duration::from_millis(500),
        work_type_override,
        assert: merged_assert,
        wait_for_map_write: !entry.bpf_map_write.is_empty(),
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
            try_flush_profraw();
            print_assert_result(&r);
            collect_and_print_probe_data(probe_stop, probe_handle);
            return Some(1);
        }
    };

    let exit_code = if result.passed { 0 } else { 1 };
    try_flush_profraw();
    print_assert_result(&result);
    collect_and_print_probe_data(probe_stop, probe_handle);
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

/// Cross-thread probe pipeline atomics.
///
/// Groups the three `Arc<AtomicBool>` signals the probe setup path
/// has to hand to its worker thread: `stop` (main thread asks the
/// probe thread to shut down), `output_done` (probe thread tells
/// the main thread it has already emitted `PROBE_PAYLOAD_*`), and
/// `probes_ready` (probe thread signals the main thread that
/// kprobes/kfentries have attached). [`Clone`] is the expected way
/// to produce the thread-side view before calling
/// `std::thread::spawn` — each clone bumps refcounts only.
#[derive(Clone, Default)]
pub(crate) struct ProbePipeline {
    pub stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub output_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub probes_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
    pub stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub kernel_func_names: Vec<(u32, String)>,
    /// Number of functions in Phase A. Phase B uses this as func_idx_offset
    /// to avoid index collisions in the shared BPF maps.
    pub kernel_func_count: u32,
    pub pipe_diag: PipelineDiagnostics,
    pub output_done: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
    while !pipeline
        .probes_ready
        .load(std::sync::atomic::Ordering::Acquire)
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    eprintln!(
        "ktstr_test: probe phase_a: {} kernel functions attached, waiting for Phase B",
        kernel_functions.len(),
    );

    let kernel_func_count = kernel_functions.len() as u32;

    Some(ProbePhaseAState {
        handle,
        phase_b_tx,
        stop: pipeline.stop.clone(),
        kernel_func_names: func_names,
        kernel_func_count,
        pipe_diag,
        output_done: pipeline.output_done.clone(),
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
            eprintln!("ktstr_test: unknown work type '{s}'");
            None
        })
    });

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

        let phase_b_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let phase_b_done_clone = phase_b_done.clone();

        let phase_b_input = crate::probe::process::PhaseBInput {
            functions: phase_b_functions,
            bpf_prog_fds: bpf_fds,
            btf_funcs: phase_b_btf,
            done: phase_b_done_clone,
            func_idx_offset: pa.kernel_func_count,
        };

        if let Err(e) = pa.phase_b_tx.send(phase_b_input) {
            eprintln!("ktstr_test: probe phase_b: failed to send: {e}");
        } else {
            // Wait for Phase B attachment to complete.
            while !phase_b_done.load(std::sync::atomic::Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            eprintln!("ktstr_test: probe phase_b: BPF fentry attached");
        }
    } else {
        eprintln!("ktstr_test: probe phase_b: no BPF symbols, skipping fentry");
        // Drop the sender so the probe thread's try_recv sees Disconnected.
        drop(pa.phase_b_tx);
    }

    let (topo, cgroups, sched_pid, merged_assert) = build_dispatch_ctx_parts(entry, args);
    let ctx = crate::scenario::Ctx {
        cgroups: &cgroups,
        topo: &topo,
        duration: entry.duration,
        workers_per_cgroup: entry.workers_per_cgroup as usize,
        sched_pid,
        settle: std::time::Duration::from_millis(500),
        work_type_override,
        assert: merged_assert,
        wait_for_map_write: !entry.bpf_map_write.is_empty(),
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
            try_flush_profraw();
            print_assert_result(&r);
            let probe_handle = Some(ProbeHandle {
                thread: pa.handle,
                func_names: pa.kernel_func_names,
                pipeline_diag: pa.pipe_diag,
                output_done: pa.output_done,
                param_names: pa.param_names,
                render_hints: pa.render_hints,
            });
            collect_and_print_probe_data(pa.stop, probe_handle);
            return Some(1);
        }
    };

    let exit_code = if result.passed { 0 } else { 1 };
    try_flush_profraw();
    print_assert_result(&result);
    let probe_handle = Some(ProbeHandle {
        thread: pa.handle,
        func_names: pa.kernel_func_names,
        pipeline_diag: pa.pipe_diag,
        output_done: pa.output_done,
        param_names: pa.param_names,
        render_hints: pa.render_hints,
    });
    collect_and_print_probe_data(pa.stop, probe_handle);
    Some(exit_code)
}

/// Serialized probe data sent from guest to host via COM2.
/// The host deserializes and formats with kernel_dir for source locations.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbePayload {
    pub events: Vec<crate::probe::process::ProbeEvent>,
    pub func_names: Vec<(u32, String)>,
    pub bpf_source_locs: std::collections::HashMap<String, String>,
    pub diagnostics: Option<ProbePayloadDiagnostics>,
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
pub(crate) struct ProbePayloadDiagnostics {
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

    let payload = ProbePayload {
        events: events.to_vec(),
        func_names: func_names.to_vec(),
        bpf_source_locs,
        diagnostics: Some(ProbePayloadDiagnostics {
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
        let payload = ProbePayload {
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
        let payload = ProbePayload {
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
}
