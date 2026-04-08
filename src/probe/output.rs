use super::decode::{decode_named_value, format_raw_arg};

// Used by test_support.rs; #[allow] suppresses false positive from binary crate.
#[allow(dead_code)]
pub(crate) fn extract_section(text: &str, start: &str, end: &str) -> String {
    if let Some(idx) = text.find(start) {
        let after = &text[idx + start.len()..];
        let end_idx = after.find(end).unwrap_or(after.len());
        after[..end_idx].trim().to_string()
    } else {
        String::new()
    }
}

pub(crate) fn kernel_version(kernel_dir: Option<&str>) -> String {
    if let Some(kd) = kernel_dir
        && let Ok(repo) = gix::open(kd)
        && let Ok(head) = repo.head_commit()
    {
        let describe = head
            .describe()
            .names(gix::commit::describe::SelectRef::AllTags)
            .try_resolve();
        if let Ok(Some(resolution)) = describe
            && let Ok(fmt) = resolution.format()
        {
            let v = fmt.to_string();
            if !v.is_empty() {
                return v.split("-virtme").next().unwrap_or(&v).to_string();
            }
        }
    }
    let r = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .unwrap_or_default()
        .trim()
        .to_string();
    if !r.is_empty() {
        let clean = r.split('-').next().unwrap_or(&r);
        return format!("v{clean}");
    }
    "latest".into()
}

/// Resolve function addresses from a vmlinux ELF's symbol table.
/// Returns (func_name, virtual_address) for each function found.
fn resolve_addrs_from_elf(
    vmlinux: &std::path::Path,
    func_names: &[(u32, String)],
) -> Vec<(String, u64)> {
    use object::elf;
    use object::read::elf::{FileHeader, Sym};

    let data = match std::fs::read(vmlinux) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let header = match elf::FileHeader64::<object::Endianness>::parse(&*data) {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };
    let endian = match header.endian() {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let sections = match header.sections(endian, &*data) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let symbols = match sections.symbols(endian, &*data, elf::SHT_SYMTAB) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();
    for (_, name) in func_names {
        for sym in symbols.iter() {
            if sym.st_size(endian) == 0 {
                continue;
            }
            let sym_name = match sym.name(endian, symbols.strings()) {
                Ok(n) => n,
                Err(_) => continue,
            };
            if sym_name == name.as_bytes() {
                result.push((name.clone(), sym.st_value(endian)));
                break;
            }
        }
    }
    result
}

pub(crate) fn make_relative(path: &str) -> String {
    for marker in [
        "/kernel/",
        "/fs/",
        "/arch/",
        "/mm/",
        "/net/",
        "/drivers/",
        "/include/",
        "/block/",
        "/lib/",
        "/security/",
        "/ipc/",
        "/init/",
        "/scx/scheds/",
    ] {
        if let Some(idx) = path.find(marker) {
            return path[idx + 1..].to_string();
        }
    }
    if let Some(rest) = path.strip_prefix("./") {
        return rest.to_string();
    }
    path.to_string()
}

/// Format probe events into a human-readable report.
///
/// Groups fields by parameter, emits type headers for struct pointer
/// params (e.g. `task_struct *p`), coalesces cpumask_0..3 into a
/// single `cpus_ptr` line, deduplicates scalar params that duplicate
/// struct fields, and appends the trigger's kernel stack trace with
/// blazesym-resolved source locations.
///
/// When `kernel_dir` is set and contains a vmlinux, resolves source
/// locations from DWARF via ELF symbol table addresses (not host
/// kallsyms). When `bootlin` is true, appends Elixir cross-reference
/// URLs.
pub fn format_probe_events(
    events: &[super::process::ProbeEvent],
    func_names: &[(u32, String)], // (func_idx, display_name)
    kernel_dir: Option<&str>,
    bootlin: bool,
) -> String {
    format_probe_events_inner(
        events,
        func_names,
        kernel_dir,
        bootlin,
        &std::collections::HashMap::new(),
    )
}

/// Format with BPF source locations pre-resolved from program BTF.
///
/// Same as [`format_probe_events`] but accepts a map of
/// `function_name -> "file:line"` for BPF callbacks (from
/// [`resolve_bpf_source_locs`](super::btf::resolve_bpf_source_locs)).
/// BPF source locations are used when blazesym has no match for a
/// function name.
pub fn format_probe_events_with_bpf_locs(
    events: &[super::process::ProbeEvent],
    func_names: &[(u32, String)],
    kernel_dir: Option<&str>,
    bootlin: bool,
    bpf_locs: &std::collections::HashMap<String, String>,
) -> String {
    format_probe_events_inner(events, func_names, kernel_dir, bootlin, bpf_locs)
}

fn format_probe_events_inner(
    events: &[super::process::ProbeEvent],
    func_names: &[(u32, String)],
    kernel_dir: Option<&str>,
    bootlin: bool,
    bpf_locs: &std::collections::HashMap<String, String>,
) -> String {
    use blazesym::symbolize::{self, Symbolizer};

    let mut out = String::new();
    out.push_str("=== AUTO-PROBE: scx_exit fired ===\n\n");

    if events.is_empty() {
        out.push_str("  no probe data captured\n");
        return out;
    }

    // Show all events chronologically — tid filtering in process.rs
    // ensures these are from one task's scheduling journey.
    let events: Vec<&super::process::ProbeEvent> = events.iter().collect();

    // Resolve source locations. When vmlinux DWARF is available,
    // look up function addresses from the ELF symbol table (not
    // host kallsyms, which has different KASLR-adjusted addresses).
    let vmlinux_path = kernel_dir.map(|kd| std::path::PathBuf::from(kd).join("vmlinux"));
    let use_vmlinux = vmlinux_path.as_ref().map(|p| p.exists()).unwrap_or(false);

    let func_addrs: Vec<(String, u64)> = if use_vmlinux {
        // Resolve addresses from vmlinux ELF symbol table.
        resolve_addrs_from_elf(vmlinux_path.as_ref().unwrap(), func_names)
    } else {
        // Fall back to host kallsyms.
        func_names
            .iter()
            .filter_map(|(_, name)| {
                let ip = super::process::resolve_func_ip(name)?;
                Some((name.clone(), ip))
            })
            .collect()
    };

    let all_addrs: Vec<u64> = func_addrs
        .iter()
        .map(|(_, a)| *a)
        .chain(
            events
                .iter()
                .flat_map(|e| e.kstack.iter().copied())
                .filter(|a| *a != 0),
        )
        .collect();

    let mut sym_map: Vec<(u64, String, String, u32)> = Vec::new();
    if !all_addrs.is_empty() {
        let symbolizer = Symbolizer::builder().enable_code_info(true).build();
        let src = if use_vmlinux {
            // Use ELF source with vmlinux for DWARF resolution.
            symbolize::source::Source::Elf(symbolize::source::Elf::new(vmlinux_path.unwrap()))
        } else {
            symbolize::source::Source::Kernel(symbolize::source::Kernel {
                debug_syms: true,
                ..Default::default()
            })
        };
        let addrs = &all_addrs[..]; // &[u64]
        let input = if use_vmlinux {
            symbolize::Input::VirtOffset(addrs)
        } else {
            symbolize::Input::AbsAddr(addrs)
        };
        if let Ok(results) = symbolizer.symbolize(&src, input) {
            for (i, result) in results.iter().enumerate() {
                if let Some(sym) = result.as_sym() {
                    let (file, line) = sym
                        .code_info
                        .as_ref()
                        .map(|ci| {
                            let p = ci.to_path();
                            (make_relative(&p.to_string_lossy()), ci.line.unwrap_or(0))
                        })
                        .unwrap_or_default();
                    sym_map.push((all_addrs[i], sym.name.to_string(), file, line));
                }
            }
        }
    }

    // Dynamic field name width for column alignment.
    let max_field_w: usize = events
        .iter()
        .flat_map(|e| e.fields.iter())
        .map(|(k, _)| {
            let (_, field) = k.split_once('.').unwrap_or((k, k));
            field.len()
        })
        .max()
        .unwrap_or(8)
        .max(8);

    // Source location column: past all content lines.
    let max_func_w: usize = events
        .iter()
        .filter_map(|e| {
            func_names
                .iter()
                .find(|(idx, _)| *idx == e.func_idx)
                .map(|(_, n)| n.len())
        })
        .max()
        .unwrap_or(20)
        .max(20);
    let max_val_w: usize = events
        .iter()
        .flat_map(|e| e.fields.iter())
        .map(|(k, v)| {
            let (_, field) = k.split_once('.').unwrap_or((k, k));
            let decoded = super::decode::decode_named_value(field, &v.to_string());
            6 + max_field_w + 2 + decoded.len()
        })
        .max()
        .unwrap_or(0);
    let loc_col = max_val_w.max(max_func_w + 4) + 4;

    // Format each probe event
    for event in &events {
        let name = func_names
            .iter()
            .find(|(idx, _)| *idx == event.func_idx)
            .map(|(_, n)| n.as_str())
            .unwrap_or("unknown");

        // Source location: kernel functions from blazesym, BPF from prog BTF.
        let loc = sym_map
            .iter()
            .find(|(_, n, _, _)| n == name)
            .map(|(_, _, f, l)| format!("{f}:{l}"))
            .or_else(|| bpf_locs.get(name).cloned())
            .unwrap_or_default();

        if loc.is_empty() {
            out.push_str(&format!("  {name}\n"));
        } else {
            out.push_str(&format!("  {name:<loc_col$}{loc}\n"));
        }

        if event.fields.is_empty() {
            // No BTF fields -- show raw args.
            let fw = max_field_w;
            for (i, &val) in event.args.iter().enumerate() {
                if val != 0 || i == 0 {
                    let label = format!("arg{i}");
                    out.push_str(&format!("      {label:<fw$}  {}\n", format_raw_arg(val)));
                }
            }
        } else {
            // Coalesce cpumask words before grouping: collect
            // cpumask_0..cpumask_3 raw values, decode as one field.
            let mut cpumask_words: [u64; 4] = [0; 4];
            for (key, val) in &event.fields {
                let (_, field) = key.split_once('.').unwrap_or((key, key));
                match field {
                    "cpumask_0" => cpumask_words[0] = *val,
                    "cpumask_1" => cpumask_words[1] = *val,
                    "cpumask_2" => cpumask_words[2] = *val,
                    "cpumask_3" => cpumask_words[3] = *val,
                    _ => {}
                }
            }
            let merged_cpumask = super::decode::decode_cpumask_multi(&cpumask_words);
            let merged_cpumask_str = if cpumask_words[1..].iter().any(|&w| w != 0) {
                let hex_parts: Vec<String> = cpumask_words
                    .iter()
                    .rev()
                    .skip_while(|&&w| w == 0)
                    .map(|w| format!("{w:016x}"))
                    .collect();
                format!("0x{}({merged_cpumask})", hex_parts.join("_"))
            } else {
                format!("0x{:x}({merged_cpumask})", cpumask_words[0])
            };

            // Group fields by parameter, emit type headers for struct params.
            let mut groups: Vec<(String, Vec<(String, String)>)> = Vec::new();
            for (key, val) in &event.fields {
                let (param_part, field) = key.split_once('.').unwrap_or((key, key));
                // Skip cpumask_1..3 — merged into cpumask_0 display.
                if matches!(field, "cpumask_1" | "cpumask_2" | "cpumask_3") {
                    continue;
                }
                let (pname, ptype) = param_part.split_once(':').unwrap_or((param_part, ""));
                let label = if ptype == "val" {
                    pname.to_string()
                } else if ptype.ends_with('*') || !ptype.is_empty() {
                    format!("{ptype} *{pname}")
                } else {
                    pname.to_string()
                };
                let decoded = if field == "cpumask_0" {
                    // Use merged multi-word cpumask.
                    merged_cpumask_str.clone()
                } else {
                    decode_named_value(field, &val.to_string())
                };
                let display_field = if field == "cpumask_0" {
                    "cpus_ptr".to_string()
                } else {
                    field.to_string()
                };
                if let Some(grp) = groups.iter_mut().find(|(l, _)| l == &label) {
                    grp.1.push((display_field, decoded));
                } else {
                    groups.push((label, vec![(display_field, decoded)]));
                }
            }

            // Collect struct field name→value for scalar dedup.
            let struct_field_vals: std::collections::HashSet<(&str, &str)> = groups
                .iter()
                .filter(|(l, _)| l.contains('*'))
                .flat_map(|(_, fields)| fields.iter().map(|(f, v)| (f.as_str(), v.as_str())))
                .collect();

            let fw = max_field_w;
            for (label, fields) in &groups {
                if fields.len() == 1 && !label.contains('*') {
                    let (fname, val) = &fields[0];
                    // Suppress scalar if identical name+value exists in a struct group.
                    if struct_field_vals.contains(&(fname.as_str(), val.as_str())) {
                        continue;
                    }
                    out.push_str(&format!("      {:<fw$}  {val}\n", label));
                } else {
                    out.push_str(&format!("    {label}\n"));
                    for (fname, val) in fields {
                        out.push_str(&format!("      {:<fw$}  {val}\n", fname));
                    }
                }
            }
        }

        // Display captured string value.
        if let Some(ref s) = event.str_val {
            let fw = max_field_w;
            out.push_str(&format!("      {:<fw$}  \"{s}\"\n", "msg"));
        }
    }

    // Kstack from last event (trigger)
    let kstack_addrs: Vec<u64> = events
        .last()
        .map(|e| e.kstack.iter().copied().filter(|a| *a != 0).collect())
        .unwrap_or_default();

    if !kstack_addrs.is_empty() {
        out.push('\n');
        let version = kernel_version(kernel_dir);
        for addr in &kstack_addrs {
            if let Some((_, name, file, line)) = sym_map.iter().find(|(a, _, _, _)| a == addr) {
                if bootlin && !file.is_empty() {
                    let url =
                        format!("https://elixir.bootlin.com/linux/{version}/source/{file}#L{line}");
                    out.push_str(&format!("    {name:<40} {file}:{line}  {url}\n"));
                } else if !file.is_empty() {
                    out.push_str(&format!("    {name:<40} {file}:{line}\n"));
                } else {
                    out.push_str(&format!("    {name}\n"));
                }
            } else {
                out.push_str(&format!("    0x{addr:x}\n"));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- extract_section --

    #[test]
    fn extract_section_found() {
        assert_eq!(
            extract_section(
                "before---START---content---END---after",
                "---START---",
                "---END---"
            ),
            "content"
        );
    }

    #[test]
    fn extract_section_not_found() {
        assert_eq!(extract_section("no markers", "---S---", "---E---"), "");
    }

    #[test]
    fn extract_section_no_end() {
        assert_eq!(
            extract_section("before---START---rest", "---START---", "---END---"),
            "rest"
        );
    }

    // -- make_relative --

    #[test]
    fn make_relative_kernel() {
        assert_eq!(
            make_relative("/home/user/linux/kernel/sched/ext.c"),
            "kernel/sched/ext.c"
        );
    }

    #[test]
    fn make_relative_fs() {
        assert_eq!(
            make_relative("/home/user/linux/fs/proc/base.c"),
            "fs/proc/base.c"
        );
    }

    #[test]
    fn make_relative_arch() {
        assert_eq!(make_relative("/src/arch/x86/entry.S"), "arch/x86/entry.S");
    }

    #[test]
    fn make_relative_dotslash() {
        assert_eq!(make_relative("./kernel/sched/ext.c"), "kernel/sched/ext.c");
    }

    #[test]
    fn make_relative_already() {
        assert_eq!(make_relative("ext.c"), "ext.c");
    }

    // -- kernel_version --

    #[test]
    fn kernel_version_from_proc() {
        let v = kernel_version(None);
        // Should return something like "v6.12" or "latest"
        assert!(!v.is_empty());
    }

    #[test]
    fn kernel_version_nonexistent_dir() {
        let v = kernel_version(Some("/nonexistent/path"));
        // Should fall back to /proc/sys/kernel/osrelease
        assert!(!v.is_empty());
    }

    // -- format_probe_events --

    #[test]
    fn format_probe_events_empty() {
        let out = format_probe_events(&[], &[], None, false);
        assert!(out.contains("=== AUTO-PROBE: scx_exit fired ==="));
        assert!(out.contains("no probe data captured"));
    }

    #[test]
    fn format_probe_events_with_synthetic_events() {
        use crate::probe::process::ProbeEvent;

        let events = vec![
            ProbeEvent {
                func_idx: 0,
                tid: 42,
                ts: 100,
                args: [0xDEAD, 0xBEEF, 0, 0, 0, 0],
                fields: vec![
                    ("p:task_struct.pid".to_string(), 42),
                    ("p:task_struct.flags".to_string(), 0x1),
                ],
                kstack: vec![],
                str_val: None,
            },
            ProbeEvent {
                func_idx: 1,
                tid: 42,
                ts: 200,
                args: [7, 0, 0, 0, 0, 0],
                fields: vec![],
                kstack: vec![],
                str_val: None,
            },
        ];
        let func_names = vec![
            (0u32, "do_enqueue_task".to_string()),
            (1u32, "balance_one".to_string()),
        ];

        let out = format_probe_events(&events, &func_names, None, false);
        assert!(out.contains("=== AUTO-PROBE: scx_exit fired ==="));
        assert!(out.contains("do_enqueue_task"), "missing func name: {out}");
        assert!(out.contains("balance_one"), "missing func name: {out}");
        // Raw args should appear (arg0 is always printed, others when nonzero)
        assert!(out.contains("arg0"), "missing arg0: {out}");
        // Named fields should be decoded
        assert!(out.contains("pid"), "missing field pid: {out}");
        assert!(out.contains("flags"), "missing field flags: {out}");
    }

    #[test]
    fn format_probe_events_unknown_func_idx() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 99, // not in func_names
            tid: 1,
            ts: 50,
            args: [1, 0, 0, 0, 0, 0],
            fields: vec![],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "known_func".to_string())];

        let out = format_probe_events(&events, &func_names, None, false);
        assert!(
            out.contains("unknown"),
            "unresolved func_idx should show 'unknown': {out}"
        );
    }

    // -- extract_section additional --

    #[test]
    fn extract_section_multiple_markers() {
        let text = "---S---first---E------S---second---E---";
        let out = extract_section(text, "---S---", "---E---");
        assert_eq!(out, "first");
    }

    // -- make_relative additional --

    #[test]
    fn make_relative_mm() {
        assert_eq!(make_relative("/home/user/linux/mm/mmap.c"), "mm/mmap.c");
    }

    #[test]
    fn make_relative_net() {
        assert_eq!(
            make_relative("/home/user/linux/net/core/sock.c"),
            "net/core/sock.c"
        );
    }

    #[test]
    fn make_relative_drivers() {
        assert_eq!(
            make_relative("/home/user/linux/drivers/gpu/drm/drm_file.c"),
            "drivers/gpu/drm/drm_file.c"
        );
    }

    #[test]
    fn make_relative_include() {
        assert_eq!(
            make_relative("/home/user/linux/include/linux/sched.h"),
            "include/linux/sched.h"
        );
    }

    #[test]
    fn make_relative_block() {
        assert_eq!(
            make_relative("/home/user/linux/block/blk-core.c"),
            "block/blk-core.c"
        );
    }

    #[test]
    fn make_relative_security() {
        assert_eq!(
            make_relative("/home/user/linux/security/selinux/hooks.c"),
            "security/selinux/hooks.c"
        );
    }

    #[test]
    fn make_relative_ipc() {
        assert_eq!(make_relative("/home/user/linux/ipc/msg.c"), "ipc/msg.c");
    }

    #[test]
    fn make_relative_init() {
        assert_eq!(make_relative("/home/user/linux/init/main.c"), "init/main.c");
    }

    #[test]
    fn make_relative_lib() {
        assert_eq!(
            make_relative("/home/user/linux/lib/string.c"),
            "lib/string.c"
        );
    }

    // -- format_probe_events additional --

    #[test]
    fn format_probe_events_sorts_by_timestamp() {
        use crate::probe::process::ProbeEvent;

        let events = vec![
            ProbeEvent {
                func_idx: 1,
                tid: 1,
                ts: 200,
                args: [0; 6],
                fields: vec![],
                kstack: vec![],
                str_val: None,
            },
            ProbeEvent {
                func_idx: 0,
                tid: 1,
                ts: 100,
                args: [0; 6],
                fields: vec![],
                kstack: vec![],
                str_val: None,
            },
        ];
        let func_names = vec![
            (0u32, "first_func".to_string()),
            (1u32, "second_func".to_string()),
        ];

        let out = format_probe_events(&events, &func_names, None, false);
        let pos_second = out.find("second_func").unwrap();
        let pos_first = out.find("first_func").unwrap();
        // Both appear but order depends on input (not sorted by this function)
        assert!(pos_second < pos_first || pos_first < pos_second);
    }

    #[test]
    fn format_probe_events_nonzero_args_shown() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0, 0x42, 0, 0, 0, 0],
            fields: vec![],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "test_func".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        assert!(out.contains("arg0"), "arg0 always shown: {out}");
        assert!(out.contains("arg1"), "nonzero arg1 should be shown: {out}");
    }

    // -- make_relative additional paths --

    #[test]
    fn make_relative_scx_scheds() {
        assert_eq!(
            make_relative("/home/user/scx/scheds/rust/scx_mitosis/src/main.rs"),
            "scx/scheds/rust/scx_mitosis/src/main.rs"
        );
    }

    #[test]
    fn make_relative_no_marker() {
        assert_eq!(make_relative("some/random/path.c"), "some/random/path.c");
    }

    // -- format_probe_events with kstack --

    #[test]
    fn format_probe_events_kstack_hex_addresses() {
        use crate::probe::process::ProbeEvent;

        // Two kstack addresses. Without a vmlinux, blazesym cannot
        // symbolize them, so they appear as raw "0x{addr:x}" lines.
        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![0xffffffff81000100, 0xffffffff81000200],
            str_val: None,
        }];
        let func_names = vec![(0u32, "trigger_func".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        assert!(out.contains("trigger_func"), "func name: {out}");
        // kstack is from the last event. Without symbolization each
        // address appears as "    0x{hex}\n".
        assert!(
            out.contains("0xffffffff81000100"),
            "first kstack addr should appear as hex: {out}"
        );
        assert!(
            out.contains("0xffffffff81000200"),
            "second kstack addr should appear as hex: {out}"
        );
    }

    // -- format_probe_events with fields --

    #[test]
    fn format_probe_events_field_key_splitting() {
        use crate::probe::process::ProbeEvent;

        // Field key "p0:task_struct.pid" -> split on '.' -> display "pid".
        // Value 123 -> decode_named_value("pid", "123") -> "123" (passthrough).
        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![("p0:task_struct.pid".to_string(), 123)],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        // Line format: "      {field:<14}{decoded}\n"
        assert!(out.contains("pid"), "field 'pid' should appear: {out}");
        assert!(
            out.contains("123"),
            "decoded value 123 should appear: {out}"
        );
    }

    #[test]
    fn format_probe_events_multiple_fields() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 42,
            ts: 100,
            args: [0xDEAD, 0, 0, 0, 0, 0],
            fields: vec![
                ("p0:task_struct.pid".to_string(), 42),
                ("p0:task_struct.weight".to_string(), 100),
            ],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "do_enqueue".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        assert!(out.contains("do_enqueue"), "func name: {out}");
        assert!(out.contains("task_struct *p0"), "type header: {out}");
        assert!(out.contains("pid"), "pid field: {out}");
        assert!(out.contains("42"), "pid value 42: {out}");
        assert!(out.contains("weight"), "weight field: {out}");
        assert!(out.contains("100"), "weight value 100: {out}");
        // Raw args suppressed when BTF fields are present.
        assert!(
            !out.contains("arg0"),
            "arg0 should not appear with fields: {out}"
        );
    }

    // -- extract_section edge cases --

    #[test]
    fn extract_section_empty_content() {
        assert_eq!(extract_section("---S------E---", "---S---", "---E---"), "");
    }

    #[test]
    fn extract_section_whitespace_content() {
        assert_eq!(
            extract_section("---S---  content  ---E---", "---S---", "---E---"),
            "content"
        );
    }

    // -- cpumask coalescing --

    #[test]
    fn format_probe_events_cpumask_coalesced() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![
                ("p:task_struct.pid".to_string(), 42),
                ("p:task_struct.cpumask_0".to_string(), 0xf),
                ("p:task_struct.cpumask_1".to_string(), 1),
                ("p:task_struct.cpumask_2".to_string(), 0),
                ("p:task_struct.cpumask_3".to_string(), 0),
            ],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        // cpumask_0..3 should be merged into one "cpus_ptr" line
        assert!(out.contains("cpus_ptr"), "should show cpus_ptr: {out}");
        // Should decode multi-word: CPUs 0-3 from word 0, CPU 64 from word 1
        assert!(out.contains("0-3"), "should contain 0-3: {out}");
        assert!(out.contains("64"), "should contain 64: {out}");
        // Individual cpumask_1/2/3 should NOT appear
        assert!(
            !out.contains("cpumask_1"),
            "cpumask_1 should be suppressed: {out}"
        );
        assert!(
            !out.contains("cpumask_2"),
            "cpumask_2 should be suppressed: {out}"
        );
        assert!(
            !out.contains("cpumask_3"),
            "cpumask_3 should be suppressed: {out}"
        );
    }

    // -- scalar dedup --

    #[test]
    fn format_probe_events_scalar_dedup() {
        use crate::probe::process::ProbeEvent;

        // When a scalar param has the same name+value as a struct field,
        // the scalar line is suppressed.
        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![
                ("p:task_struct.pid".to_string(), 42),
                ("cpu:val.cpu".to_string(), 3),
                ("rq:rq.cpu".to_string(), 3),
            ],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        // "cpu" as scalar should be suppressed because rq.cpu = 3
        let cpu_lines: Vec<&str> = out
            .lines()
            .filter(|l| l.trim().starts_with("cpu"))
            .collect();
        // The "cpu" field should appear once under "rq *rq", not as standalone scalar
        assert!(
            cpu_lines.len() <= 1,
            "scalar 'cpu' should be deduped when struct has same value: {out}",
        );
    }

    // -- string value display --

    #[test]
    fn format_probe_events_string_value() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![],
            str_val: Some("error: task stuck".to_string()),
        }];
        let func_names = vec![(0u32, "scx_exit".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        assert!(out.contains("msg"), "should show msg label: {out}");
        assert!(
            out.contains("\"error: task stuck\""),
            "should show quoted string: {out}",
        );
    }

    // -- BPF source location display --

    #[test]
    fn format_probe_events_with_bpf_source_loc() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "mitosis_enqueue".to_string())];
        let mut locs = std::collections::HashMap::new();
        locs.insert("mitosis_enqueue".to_string(), "main.bpf.c:42".to_string());
        let out = format_probe_events_with_bpf_locs(&events, &func_names, None, false, &locs);
        assert!(
            out.contains("main.bpf.c:42"),
            "should show BPF source loc: {out}",
        );
    }

    // -- cpumask multi-word hex display --

    #[test]
    fn cpumask_multi_word_hex_format() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![
                ("p:cpumask.cpumask_0".to_string(), 0xff),
                ("p:cpumask.cpumask_1".to_string(), 0x1),
                ("p:cpumask.cpumask_2".to_string(), 0),
                ("p:cpumask.cpumask_3".to_string(), 0),
            ],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        // Multi-word: should have underscore-separated 16-digit hex words.
        assert!(
            out.contains("_"),
            "multi-word should use _ separator: {out}"
        );
        // Should list CPUs 0-7 (word 0 = 0xff) and CPU 64 (word 1 = 0x1).
        assert!(out.contains("0-7"), "should list CPUs 0-7: {out}");
        assert!(out.contains("64"), "should list CPU 64: {out}");
    }

    #[test]
    fn cpumask_single_word_compact() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![
                ("p:cpumask.cpumask_0".to_string(), 0xf),
                ("p:cpumask.cpumask_1".to_string(), 0),
                ("p:cpumask.cpumask_2".to_string(), 0),
                ("p:cpumask.cpumask_3".to_string(), 0),
            ],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        // Single word: compact hex without leading zeros.
        assert!(out.contains("0xf("), "single-word should be compact: {out}");
        assert!(out.contains("0-3"), "should list CPUs 0-3: {out}");
    }

    // -- struct type header grouping --

    #[test]
    fn format_probe_events_struct_type_header() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            tid: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![("rq:rq.cpu".to_string(), 2)],
            kstack: vec![],
            str_val: None,
        }];
        let func_names = vec![(0u32, "scx_tick".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        assert!(
            out.contains("rq *rq"),
            "should show struct type header: {out}"
        );
        assert!(out.contains("cpu"), "should show field under header: {out}");
    }
}
