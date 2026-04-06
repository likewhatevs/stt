/// Return true if a function should not be probed.
///
/// Skips the trigger function (`scx_disable_workfn`), generic scheduler
/// entry points (`schedule`, `__schedule`), syscall handlers, and
/// low-level infrastructure (`_raw_spin_*`, `asm_*`, `entry_*`,
/// `sysvec_*`) that generate massive probe_data maps with no useful
/// scheduler decision data.
pub fn should_skip_probe(name: &str) -> bool {
    matches!(
        name,
        "scx_disable_workfn"
            | "schedule"
            | "__schedule"
            | "do_syscall_64"
            | "__do_sys_sched_yield"
            | "do_sched_yield"
            | "preempt_schedule_common"
            | "preempt_schedule_irq"
    ) || name.starts_with("_raw_spin_")
        || name.starts_with("asm_")
        || name.starts_with("entry_")
        || name.starts_with("__sysvec_")
        || name.starts_with("sysvec_")
}

/// Maps sched_ext BPF op name fragments to (kernel_caller, task_arg_idx).
/// When a BPF function name contains the op fragment, its kernel-side
/// caller is probed instead. The task_struct pointer is at arg{task_arg_idx}.
const BPF_OP_CALLERS: &[(&str, &str, u32)] = &[
    ("select_cpu", "do_enqueue_task", 1),
    ("enqueue", "do_enqueue_task", 1),
    ("dispatch", "balance_one", 0),
    ("running", "set_next_task_scx", 1),
    ("stopping", "put_prev_task_scx", 1),
    ("tick", "task_tick_scx", 1),
    ("set_cpumask", "set_cpus_allowed_scx", 1),
    ("init_task", "scx_enable_task", 1),
    ("enable", "scx_enable_task", 1),
];

/// Expand BPF functions by adding their kernel-side callers.
///
/// BPF functions are kept (for fentry attachment) and their kernel
/// callers are added (for bridge kprobes that capture the kernel-side
/// view). Uses [`BPF_OP_CALLERS`] to map sched_ext op name fragments
/// to kernel entry points (e.g. `enqueue` -> `do_enqueue_task`).
/// Deduplicates by raw_name.
pub fn expand_bpf_to_kernel_callers(functions: Vec<StackFunction>) -> Vec<StackFunction> {
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for f in functions {
        if !f.is_bpf {
            if seen.insert(f.raw_name.clone()) {
                result.push(f);
            }
            continue;
        }
        // Keep the BPF function for fentry attachment.
        if seen.insert(f.raw_name.clone()) {
            result.push(f.clone());
        }
        // Add the kernel caller for bridge kprobe.
        let caller = BPF_OP_CALLERS
            .iter()
            .find(|(op, _, _)| f.display_name.contains(op));
        if let Some((_, caller_name, _)) = caller
            && seen.insert(caller_name.to_string())
        {
            result.push(StackFunction {
                raw_name: caller_name.to_string(),
                display_name: caller_name.to_string(),
                is_bpf: false,
                bpf_prog_id: None,
            });
        }
    }
    result
}

/// Extract function names from a crash stack trace for the next run.
/// Deduplicates and skips generic functions.
#[allow(dead_code)]
pub fn extract_stack_functions(stack: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    stack
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim().trim_start_matches("  ");
            let func = trimmed.split('+').next()?;
            let func = func.trim();
            if func.is_empty()
                || func.contains(' ')
                || func.starts_with('[')
                || func.starts_with('#')
                || func.starts_with('=')
                || func.starts_with('-')
                || func.ends_with(':')
                || func.starts_with("bpf_prog_")
                || !func
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
                || should_skip_probe(func)
            {
                return None;
            }
            if seen.insert(func.to_string()) {
                Some(func.to_string())
            } else {
                None
            }
        })
        .collect()
}

// ---- Auto-probe: crash-stack-driven probing ----

/// A function to probe, from a crash stack or BPF program discovery.
///
/// `raw_name` is the symbol as it appears in kallsyms (e.g.
/// `bpf_prog_9_mitosis_enqueue` for BPF). `display_name` is the
/// short name used in output (e.g. `mitosis_enqueue`).
#[derive(Debug, Clone)]
pub struct StackFunction {
    pub raw_name: String,
    pub display_name: String,
    pub is_bpf: bool,
    #[allow(dead_code)]
    pub bpf_prog_id: Option<u32>,
}

/// Public API for auto-repro: extract function names as strings.
pub fn extract_stack_functions_all_pub(stack: &str) -> Vec<String> {
    extract_stack_functions_all(stack)
        .into_iter()
        .map(|f| f.raw_name)
        .collect()
}

/// Extract function names from a crash stack, including BPF programs.
pub fn extract_stack_functions_all(stack: &str) -> Vec<StackFunction> {
    let mut seen = std::collections::HashSet::new();
    stack
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim().trim_start_matches("  ");
            // Handle "func+0xOFFSET/0xSIZE" and "bpf_prog_HASH_name+0x..."
            let func = trimmed.split('+').next()?.trim();
            if func.is_empty()
                || func.contains(' ')
                || func.starts_with('[')
                || func.starts_with('#')
                || func.starts_with('=')
                || func.starts_with('-')
                || func.ends_with(':')
                || !func
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
                || should_skip_probe(func)
            {
                return None;
            }
            if !seen.insert(func.to_string()) {
                return None;
            }
            let is_bpf = func.starts_with("bpf_prog_");
            let display_name = if is_bpf {
                bpf_short_name(func).unwrap_or(func).to_string()
            } else {
                func.to_string()
            };
            Some(StackFunction {
                raw_name: func.to_string(),
                display_name,
                is_bpf,
                bpf_prog_id: None,
            })
        })
        .collect()
}

/// Extract the short function name from a BPF program symbol.
/// "bpf_prog_abc123_mitosis_enqueue" -> "mitosis_enqueue"
pub fn bpf_short_name(raw: &str) -> Option<&str> {
    let rest = raw.strip_prefix("bpf_prog_")?;
    let idx = rest.find('_')?;
    Some(&rest[idx + 1..])
}

/// Load --probe-stack input: file path, inline stack, or comma-separated names.
pub fn load_probe_stack(input: &str) -> Vec<StackFunction> {
    // File path?
    if std::path::Path::new(input).exists()
        && let Ok(contents) = std::fs::read_to_string(input)
    {
        return extract_stack_functions_all(&contents);
    }
    // Inline stack (has +0x or newlines)?
    if input.contains("+0x") || input.contains('\n') {
        return extract_stack_functions_all(input);
    }
    // Comma-separated function names
    input
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let s = s.trim();
            let is_bpf = s.starts_with("bpf_prog_");
            StackFunction {
                raw_name: s.to_string(),
                display_name: if is_bpf {
                    bpf_short_name(s).unwrap_or(s).to_string()
                } else {
                    s.to_string()
                },
                is_bpf,
                bpf_prog_id: None,
            }
        })
        .collect()
}

/// Ensure tracefs or debugfs is mounted so available_filter_functions
/// is readable. Only attempts mounts if the files don't already exist.
fn ensure_tracefs_mounted() {
    if std::path::Path::new("/sys/kernel/tracing/available_filter_functions").exists()
        || std::path::Path::new("/sys/kernel/debug/tracing/available_filter_functions").exists()
    {
        return;
    }
    // Try mounting tracefs first (lighter than debugfs).
    let _ = std::fs::create_dir_all("/sys/kernel/tracing");
    let _ = std::process::Command::new("mount")
        .args(["-t", "tracefs", "tracefs", "/sys/kernel/tracing"])
        .status();
    if std::path::Path::new("/sys/kernel/tracing/available_filter_functions").exists() {
        return;
    }
    // Fall back to debugfs which exposes tracefs under tracing/.
    let _ = std::process::Command::new("mount")
        .args(["-t", "debugfs", "debugfs", "/sys/kernel/debug"])
        .status();
}

/// Filter functions to only those traceable via kprobe.
///
/// Reads `available_filter_functions` from tracefs or debugfs. Falls
/// back to `/proc/kallsyms` as last resort (less accurate: includes
/// `notrace`/`noinstr` functions that kprobes will reject). Mounts
/// tracefs/debugfs if needed. BPF functions are matched by suffix
/// against kallsyms `bpf_prog_*` entries.
pub fn filter_traceable(functions: Vec<StackFunction>) -> Vec<StackFunction> {
    ensure_tracefs_mounted();

    let available = std::fs::read_to_string("/sys/kernel/tracing/available_filter_functions")
        .or_else(|_| {
            std::fs::read_to_string("/sys/kernel/debug/tracing/available_filter_functions")
        })
        .or_else(|_| std::fs::read_to_string("/proc/kallsyms"))
        .unwrap_or_default();

    let source = if std::path::Path::new("/sys/kernel/tracing/available_filter_functions").exists()
    {
        "tracefs"
    } else if std::path::Path::new("/sys/kernel/debug/tracing/available_filter_functions").exists()
    {
        "debugfs"
    } else if available.is_empty() {
        tracing::warn!("filter_traceable: no symbol source, skipping filter");
        return functions;
    } else {
        "kallsyms"
    };

    // Build a HashSet of available symbol names for O(1) lookup.
    let sym_set: std::collections::HashSet<&str> = available
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .collect();

    let before = functions.len();
    let filtered: Vec<StackFunction> = functions
        .into_iter()
        .filter(|f| {
            let found = if f.is_bpf {
                let short = bpf_short_name(&f.raw_name).unwrap_or("");
                let suffix = format!("_{short}");
                sym_set
                    .iter()
                    .any(|sym| sym.starts_with("bpf_prog_") && sym.ends_with(&suffix))
            } else {
                sym_set.contains(f.raw_name.as_str())
            };
            if !found {
                tracing::debug!(func = %f.raw_name, source, "filter_traceable: dropped");
            }
            found
        })
        .collect();

    tracing::debug!(pass = filtered.len(), before, source, "filter_traceable");
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- should_skip_probe --

    #[test]
    fn should_skip_probe_skips() {
        assert!(should_skip_probe("scx_disable_workfn"));
        assert!(should_skip_probe("_raw_spin_lock"));
        assert!(should_skip_probe("asm_exc_page_fault"));
        assert!(should_skip_probe("entry_SYSCALL_64"));
        assert!(should_skip_probe("__sysvec_apic_timer"));
        assert!(should_skip_probe("sysvec_apic_timer"));
    }

    #[test]
    fn should_skip_probe_keeps() {
        assert!(!should_skip_probe("scx_exit"));
        assert!(!should_skip_probe("do_enqueue_task"));
        assert!(!should_skip_probe("mitosis_enqueue"));
        assert!(!should_skip_probe("balance_one"));
    }

    // -- bpf_short_name --

    #[test]
    fn bpf_short_name_valid() {
        assert_eq!(
            bpf_short_name("bpf_prog_abc123_mitosis_enqueue"),
            Some("mitosis_enqueue")
        );
    }

    #[test]
    fn bpf_short_name_no_prefix() {
        assert_eq!(bpf_short_name("do_enqueue_task"), None);
    }

    #[test]
    fn bpf_short_name_no_underscore() {
        assert_eq!(bpf_short_name("bpf_prog_"), None);
    }

    // -- extract_stack_functions --

    #[test]
    fn extract_stack_functions_crash_stack() {
        let stack = "\
            do_enqueue_task+0x1a0/0x380\n\
            balance_one+0x50/0x100\n\
            _raw_spin_lock+0x10/0x20\n\
            do_enqueue_task+0x1a0/0x380\n\
            asm_exc_page_fault+0x30/0x40\n\
            set_next_task_scx+0x80/0x120\n";
        let fns = extract_stack_functions(stack);
        assert!(fns.contains(&"do_enqueue_task".to_string()));
        assert!(fns.contains(&"balance_one".to_string()));
        assert!(fns.contains(&"set_next_task_scx".to_string()));
        // Skipped
        assert!(!fns.iter().any(|f| f.contains("_raw_spin")));
        assert!(!fns.iter().any(|f| f.contains("asm_exc")));
        // Deduped
        assert_eq!(fns.iter().filter(|f| *f == "do_enqueue_task").count(), 1);
    }

    #[test]
    fn extract_stack_functions_empty() {
        assert!(extract_stack_functions("").is_empty());
    }

    #[test]
    fn extract_stack_functions_noise() {
        let stack = "=== CRASH ===\n#0 some_frame\n[ 123.456] boot msg\nbpf_prog_abc_foo+0x10\n";
        let fns = extract_stack_functions(stack);
        assert!(!fns.iter().any(|f| f.starts_with("===")));
        assert!(!fns.iter().any(|f| f.starts_with("#")));
        assert!(!fns.iter().any(|f| f.starts_with("[")));
        assert!(!fns.iter().any(|f| f.starts_with("bpf_prog_")));
    }

    // -- extract_stack_functions_all_pub --

    #[test]
    fn extract_stack_functions_all_pub_includes_bpf() {
        let stack = "do_enqueue_task+0x100/0x200\nbpf_prog_abc_mitosis_enqueue+0x50/0x80\n";
        let fns = extract_stack_functions_all_pub(stack);
        assert!(fns.contains(&"do_enqueue_task".to_string()));
        assert!(fns.contains(&"bpf_prog_abc_mitosis_enqueue".to_string()));
    }

    // -- load_probe_stack (comma-separated path, no file I/O) --

    #[test]
    fn load_probe_stack_comma_separated() {
        let fns = load_probe_stack("do_enqueue_task,balance_one,set_next_task_scx");
        assert_eq!(fns.len(), 3);
        assert_eq!(fns[0].raw_name, "do_enqueue_task");
        assert_eq!(fns[1].raw_name, "balance_one");
        assert!(!fns[0].is_bpf);
    }

    #[test]
    fn load_probe_stack_inline_stack() {
        let input = "do_enqueue_task+0x1a0/0x380\nbalance_one+0x50/0x100";
        let fns = load_probe_stack(input);
        assert_eq!(fns.len(), 2);
    }

    #[test]
    fn load_probe_stack_bpf_names() {
        let fns = load_probe_stack("bpf_prog_abc_mitosis_enqueue,do_enqueue_task");
        assert_eq!(fns.len(), 2);
        assert!(fns[0].is_bpf);
        assert_eq!(fns[0].display_name, "mitosis_enqueue");
        assert!(!fns[1].is_bpf);
    }

    // -- extract_stack_functions_all edge cases --

    #[test]
    fn extract_stack_functions_all_deduplicates() {
        let stack = "do_exit+0x10/0x20\ndo_exit+0x10/0x20\n";
        let fns = extract_stack_functions_all(stack);
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].raw_name, "do_exit");
    }

    #[test]
    fn extract_stack_functions_all_bpf_display_name() {
        let stack = "bpf_prog_abc_mitosis_enqueue+0x50/0x80\n";
        let fns = extract_stack_functions_all(stack);
        assert_eq!(fns.len(), 1);
        assert!(fns[0].is_bpf);
        assert_eq!(fns[0].display_name, "mitosis_enqueue");
        assert_eq!(fns[0].raw_name, "bpf_prog_abc_mitosis_enqueue");
    }

    #[test]
    fn extract_stack_functions_all_skips_entries_with_spaces() {
        let stack = "  some function name+0x10\nvalid_func+0x20/0x30\n";
        let fns = extract_stack_functions_all(stack);
        // "some function name" has spaces and should be skipped
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].raw_name, "valid_func");
    }

    #[test]
    fn extract_stack_functions_all_skips_bracket_entries() {
        let stack = "[<ffffffff81000000>] do_exit+0x10\n[unknown]+0x20\n";
        let fns = extract_stack_functions_all(stack);
        // Entries starting with '[' should be skipped
        for f in &fns {
            assert!(!f.raw_name.starts_with('['));
        }
    }

    #[test]
    fn extract_stack_functions_all_skips_colon_suffix() {
        let stack = "Call Trace:\ndo_exit+0x10/0x20\n";
        let fns = extract_stack_functions_all(stack);
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].raw_name, "do_exit");
    }

    // -- load_probe_stack edge cases --

    #[test]
    fn load_probe_stack_empty_string() {
        let fns = load_probe_stack("");
        assert!(fns.is_empty());
    }

    #[test]
    fn load_probe_stack_whitespace_only() {
        let fns = load_probe_stack("  ,  ,  ");
        assert!(fns.is_empty());
    }

    // -- bpf_short_name edge cases --

    #[test]
    fn bpf_short_name_only_hash() {
        // "bpf_prog_abcdef" -> hash is "abcdef", no second underscore
        assert_eq!(bpf_short_name("bpf_prog_abcdef"), None);
    }

    #[test]
    fn bpf_short_name_multiple_underscores() {
        assert_eq!(
            bpf_short_name("bpf_prog_abc_my_complex_func"),
            Some("my_complex_func")
        );
    }

    // -- expand_bpf_to_kernel_callers --

    #[test]
    fn expand_bpf_kernel_only_passthrough() {
        let funcs = vec![StackFunction {
            raw_name: "do_exit".into(),
            display_name: "do_exit".into(),
            is_bpf: false,
            bpf_prog_id: None,
        }];
        let result = expand_bpf_to_kernel_callers(funcs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].raw_name, "do_exit");
    }

    #[test]
    fn expand_bpf_enqueue_keeps_bpf_and_adds_caller() {
        let funcs = vec![StackFunction {
            raw_name: "bpf_prog_9_mitosis_enqueue".into(),
            display_name: "mitosis_enqueue".into(),
            is_bpf: true,
            bpf_prog_id: Some(9),
        }];
        let result = expand_bpf_to_kernel_callers(funcs);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].raw_name, "bpf_prog_9_mitosis_enqueue");
        assert!(result[0].is_bpf);
        assert_eq!(result[1].raw_name, "do_enqueue_task");
        assert!(!result[1].is_bpf);
    }

    #[test]
    fn expand_bpf_deduplicates_callers() {
        // Both enqueue and select_cpu map to do_enqueue_task.
        let funcs = vec![
            StackFunction {
                raw_name: "bpf_prog_9_mitosis_enqueue".into(),
                display_name: "mitosis_enqueue".into(),
                is_bpf: true,
                bpf_prog_id: Some(9),
            },
            StackFunction {
                raw_name: "bpf_prog_9_mitosis_select_cpu".into(),
                display_name: "mitosis_select_cpu".into(),
                is_bpf: true,
                bpf_prog_id: Some(9),
            },
        ];
        let result = expand_bpf_to_kernel_callers(funcs);
        // 2 BPF functions + 1 deduplicated kernel caller
        assert_eq!(result.len(), 3);
        // Order: bpf1, caller (from bpf1), bpf2 (caller deduped for bpf2).
        assert!(result[0].is_bpf);
        assert_eq!(result[1].raw_name, "do_enqueue_task");
        assert!(!result[1].is_bpf);
        assert!(result[2].is_bpf);
    }

    #[test]
    fn expand_bpf_mixed_kernel_and_bpf() {
        let funcs = vec![
            StackFunction {
                raw_name: "pick_task_scx".into(),
                display_name: "pick_task_scx".into(),
                is_bpf: false,
                bpf_prog_id: None,
            },
            StackFunction {
                raw_name: "bpf_prog_9_mitosis_dispatch".into(),
                display_name: "mitosis_dispatch".into(),
                is_bpf: true,
                bpf_prog_id: Some(9),
            },
        ];
        let result = expand_bpf_to_kernel_callers(funcs);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].raw_name, "pick_task_scx");
        assert_eq!(result[1].raw_name, "bpf_prog_9_mitosis_dispatch");
        assert!(result[1].is_bpf);
        assert_eq!(result[2].raw_name, "balance_one");
    }

    // -- should_skip_probe additional --

    #[test]
    fn should_skip_probe_schedule_variants() {
        assert!(should_skip_probe("schedule"));
        assert!(should_skip_probe("__schedule"));
        assert!(should_skip_probe("preempt_schedule_common"));
        assert!(should_skip_probe("preempt_schedule_irq"));
    }

    #[test]
    fn should_skip_probe_syscall_variants() {
        assert!(should_skip_probe("do_syscall_64"));
        assert!(should_skip_probe("__do_sys_sched_yield"));
        assert!(should_skip_probe("do_sched_yield"));
    }

    #[test]
    fn should_skip_probe_prefix_patterns() {
        assert!(should_skip_probe("_raw_spin_lock_irqsave"));
        assert!(should_skip_probe("asm_sysvec_call_function"));
        assert!(should_skip_probe("entry_SYSCALL_64_after_hwframe"));
        assert!(should_skip_probe("__sysvec_reschedule_ipi"));
        assert!(should_skip_probe("sysvec_reschedule_ipi"));
    }

    #[test]
    fn should_skip_probe_keeps_sched_ext_funcs() {
        assert!(!should_skip_probe("scx_enable_task"));
        assert!(!should_skip_probe("scx_dispatch_enqueue"));
        assert!(!should_skip_probe("task_tick_scx"));
        assert!(!should_skip_probe("set_next_task_scx"));
        assert!(!should_skip_probe("put_prev_task_scx"));
    }

    // -- BPF_OP_CALLERS table --

    #[test]
    fn bpf_op_callers_all_ops_have_kernel_callers() {
        for (op, caller, _) in BPF_OP_CALLERS {
            assert!(!op.is_empty(), "empty op in BPF_OP_CALLERS");
            assert!(!caller.is_empty(), "empty caller for op {op}");
        }
    }

    #[test]
    fn bpf_op_callers_no_duplicate_ops() {
        let ops: Vec<&str> = BPF_OP_CALLERS.iter().map(|(op, _, _)| *op).collect();
        let unique: std::collections::HashSet<&&str> = ops.iter().collect();
        assert_eq!(ops.len(), unique.len(), "duplicate ops in BPF_OP_CALLERS");
    }

    #[test]
    fn bpf_op_callers_covers_key_ops() {
        let ops: Vec<&str> = BPF_OP_CALLERS.iter().map(|(op, _, _)| *op).collect();
        assert!(ops.contains(&"enqueue"), "missing enqueue");
        assert!(ops.contains(&"dispatch"), "missing dispatch");
        assert!(ops.contains(&"select_cpu"), "missing select_cpu");
        assert!(ops.contains(&"running"), "missing running");
        assert!(ops.contains(&"stopping"), "missing stopping");
        assert!(ops.contains(&"tick"), "missing tick");
    }

    // -- expand_bpf_to_kernel_callers additional --

    #[test]
    fn expand_bpf_all_ops_resolve() {
        // Every op in BPF_OP_CALLERS should expand when the display_name
        // contains the op fragment.
        for (op, expected_caller, _) in BPF_OP_CALLERS {
            let funcs = vec![StackFunction {
                raw_name: format!("bpf_prog_99_test_{op}"),
                display_name: format!("test_{op}"),
                is_bpf: true,
                bpf_prog_id: Some(99),
            }];
            let result = expand_bpf_to_kernel_callers(funcs);
            let has_caller = result.iter().any(|f| f.raw_name == *expected_caller);
            assert!(
                has_caller,
                "op '{op}' should expand to caller '{expected_caller}', got: {:?}",
                result.iter().map(|f| &f.raw_name).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn expand_bpf_unknown_op_keeps_bpf_no_caller() {
        let funcs = vec![StackFunction {
            raw_name: "bpf_prog_9_unknown_op".into(),
            display_name: "unknown_op".into(),
            is_bpf: true,
            bpf_prog_id: Some(9),
        }];
        let result = expand_bpf_to_kernel_callers(funcs);
        // BPF function kept even without a known caller
        assert_eq!(result.len(), 1);
        assert!(result[0].is_bpf);
    }
}
