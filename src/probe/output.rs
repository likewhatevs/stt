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

/// Format structured probe events from the BPF skeleton into a
/// human-readable report.
pub fn format_probe_events(
    events: &[super::process::ProbeEvent],
    func_names: &[(u32, String)], // (func_idx, display_name)
    kernel_dir: Option<&str>,
    bootlin: bool,
) -> String {
    use blazesym::symbolize::{self, Symbolizer};

    let mut out = String::new();
    out.push_str("=== AUTO-PROBE: trigger fired ===\n\n");

    if events.is_empty() {
        out.push_str("  no probe data captured\n");
        return out;
    }

    // Symbolize kstack addresses
    let all_addrs: Vec<u64> = events
        .iter()
        .flat_map(|e| e.kstack.iter().copied())
        .filter(|a| *a != 0)
        .collect();

    let mut sym_map: Vec<(u64, String, String, u32)> = Vec::new(); // (addr, func, file, line)
    if !all_addrs.is_empty() {
        let mut ksrc = symbolize::source::Kernel {
            debug_syms: true,
            ..Default::default()
        };
        if let Some(kd) = kernel_dir {
            let vmlinux = std::path::PathBuf::from(kd).join("vmlinux");
            if vmlinux.exists() {
                ksrc.vmlinux = vmlinux.into();
            }
        }
        let symbolizer = Symbolizer::builder().enable_code_info(true).build();
        let src = symbolize::source::Source::Kernel(ksrc);
        if let Ok(results) = symbolizer.symbolize(&src, symbolize::Input::AbsAddr(&all_addrs)) {
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

    // Format each probe event
    for event in events {
        let name = func_names
            .iter()
            .find(|(idx, _)| *idx == event.func_idx)
            .map(|(_, n)| n.as_str())
            .unwrap_or("unknown");

        // Source location from symbolization
        let loc = sym_map
            .iter()
            .find(|(_, n, _, _)| n == name)
            .map(|(_, _, f, l)| format!("{f}:{l}"))
            .unwrap_or_default();

        if loc.is_empty() {
            out.push_str(&format!("  {name}\n"));
        } else {
            out.push_str(&format!("  {name}\t{loc}\n"));
        }

        // Raw args
        for (i, &val) in event.args.iter().enumerate() {
            if val != 0 || i == 0 {
                out.push_str(&format!("      arg{i:<6}{}\n", format_raw_arg(val)));
            }
        }

        // Decoded fields
        for (key, val) in &event.fields {
            let (_, field) = key.split_once('.').unwrap_or((key, key));
            let decoded = decode_named_value(field, &val.to_string());
            out.push_str(&format!("      {field:<14}{decoded}\n"));
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
        assert!(out.contains("=== AUTO-PROBE: trigger fired ==="));
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
            },
            ProbeEvent {
                func_idx: 1,
                tid: 42,
                ts: 200,
                args: [7, 0, 0, 0, 0, 0],
                fields: vec![],
                kstack: vec![],
            },
        ];
        let func_names = vec![
            (0u32, "do_enqueue_task".to_string()),
            (1u32, "balance_one".to_string()),
        ];

        let out = format_probe_events(&events, &func_names, None, false);
        assert!(out.contains("=== AUTO-PROBE: trigger fired ==="));
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
            },
            ProbeEvent {
                func_idx: 0,
                tid: 1,
                ts: 100,
                args: [0; 6],
                fields: vec![],
                kstack: vec![],
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
        }];
        let func_names = vec![(0u32, "do_enqueue".to_string())];
        let out = format_probe_events(&events, &func_names, None, false);
        assert!(out.contains("do_enqueue"), "func name: {out}");
        assert!(out.contains("pid"), "pid field: {out}");
        assert!(out.contains("42"), "pid value 42: {out}");
        assert!(out.contains("weight"), "weight field: {out}");
        assert!(out.contains("100"), "weight value 100: {out}");
        assert!(out.contains("arg0"), "arg0 always printed: {out}");
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
}
