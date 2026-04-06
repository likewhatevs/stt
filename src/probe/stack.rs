/// Skip the trigger function and low-level infrastructure that generates
/// massive maps with no scheduler decision data.
pub fn should_skip_probe(name: &str) -> bool {
    name == "scx_exit"
        || name.starts_with("_raw_spin_")
        || name.starts_with("asm_")
        || name.starts_with("entry_")
        || name.starts_with("__sysvec_")
        || name.starts_with("sysvec_")
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

/// Filter functions to only those traceable via kprobe.
/// Falls back to /proc/kallsyms if tracefs unavailable.
pub fn filter_traceable(functions: Vec<StackFunction>) -> Vec<StackFunction> {
    let available = std::fs::read_to_string("/sys/kernel/tracing/available_filter_functions")
        .or_else(|_| std::fs::read_to_string("/proc/kallsyms"))
        .unwrap_or_default();

    if available.is_empty() {
        tracing::warn!("filter_traceable: no symbol source, skipping filter");
        return functions;
    }

    let before = functions.len();
    let filtered: Vec<StackFunction> = functions
        .into_iter()
        .filter(|f| {
            if f.is_bpf {
                let short = bpf_short_name(&f.raw_name).unwrap_or("");
                available.lines().any(|l| {
                    let sym = l.split_whitespace().next().unwrap_or("");
                    sym.starts_with("bpf_prog_") && sym.ends_with(&format!("_{short}"))
                })
            } else {
                available
                    .lines()
                    .any(|l| l.split_whitespace().next() == Some(f.raw_name.as_str()))
            }
        })
        .collect();

    tracing::debug!(before, after = filtered.len(), "filter_traceable");
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- should_skip_probe --

    #[test]
    fn should_skip_probe_skips() {
        assert!(should_skip_probe("scx_exit"));
        assert!(should_skip_probe("_raw_spin_lock"));
        assert!(should_skip_probe("asm_exc_page_fault"));
        assert!(should_skip_probe("entry_SYSCALL_64"));
        assert!(should_skip_probe("__sysvec_apic_timer"));
        assert!(should_skip_probe("sysvec_apic_timer"));
    }

    #[test]
    fn should_skip_probe_keeps() {
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
}
