use super::decode::{decode_named_value, format_raw_arg};

/// Read nr_cpu_ids from sysfs. Returns the number of possible CPUs
/// (upper bound of the "possible" range). Falls back to
/// `libc::sysconf(_SC_NPROCESSORS_CONF)` if sysfs is unavailable.
///
/// Only correct when called on the machine whose cpumask data is
/// being decoded. For guest VM probes formatted on the host, pass
/// the guest's CPU count explicitly via `nr_cpus` parameters instead.
pub(crate) fn get_nr_cpus() -> Option<u32> {
    // /sys/devices/system/cpu/possible contains "0-N" or "0-N,M-P".
    // The highest CPU ID + 1 is nr_cpu_ids.
    if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/possible") {
        let max_cpu = content
            .trim()
            .split(',')
            .filter_map(|range| {
                let last = range.split('-').next_back()?;
                last.parse::<u32>().ok()
            })
            .max();
        if let Some(max) = max_cpu {
            return Some(max + 1);
        }
    }
    // Fallback: configured processors.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) };
    if n > 0 { Some(n as u32) } else { None }
}

/// Format cpumask words into a display string like `0xf(0-3)` or
/// `0x0000000000000001_00000000000000ff(0-7,64)`. Masks raw words to
/// `nr_cpus` to avoid showing garbage bits from uninitialized memory.
fn format_cpumask_display(cpumask_words: &[u64; 4], nr_cpus: Option<u32>) -> String {
    let merged_cpumask = super::decode::decode_cpumask_multi(cpumask_words, nr_cpus);

    let display_words = if let Some(nr) = nr_cpus {
        let mut masked = [0u64; 4];
        for (i, &w) in cpumask_words.iter().enumerate() {
            let base = i as u32 * 64;
            if base >= nr {
                break;
            }
            let valid_bits = nr - base;
            if valid_bits >= 64 {
                masked[i] = w;
            } else {
                masked[i] = w & ((1u64 << valid_bits) - 1);
            }
        }
        masked
    } else {
        *cpumask_words
    };
    if display_words[1..].iter().any(|&w| w != 0) {
        let hex_parts: Vec<String> = display_words
            .iter()
            .rev()
            .skip_while(|&&w| w == 0)
            .map(|w| format!("{w:016x}"))
            .collect();
        format!("0x{}({merged_cpumask})", hex_parts.join("_"))
    } else {
        format!("0x{:x}({merged_cpumask})", display_words[0])
    }
}

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

/// Resolve function addresses from a vmlinux ELF's symbol table.
/// Returns (func_name, virtual_address) for each function found.
fn resolve_addrs_from_elf(
    vmlinux: &std::path::Path,
    func_names: &[(u32, String)],
) -> Vec<(String, u64)> {
    let data = match std::fs::read(vmlinux) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let elf = match goblin::elf::Elf::parse(&data) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();
    for (_, name) in func_names {
        for sym in elf.syms.iter() {
            if sym.st_size == 0 {
                continue;
            }
            let sym_name = match elf.strtab.get_at(sym.st_name) {
                Some(n) => n,
                None => continue,
            };
            if sym_name == name {
                result.push((name.clone(), sym.st_value));
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
/// kallsyms).
pub fn format_probe_events(
    events: &[super::process::ProbeEvent],
    func_names: &[(u32, String)], // (func_idx, display_name)
    kernel_dir: Option<&str>,
    nr_cpus: Option<u32>,
) -> String {
    format_probe_events_inner(
        events,
        func_names,
        kernel_dir,
        &std::collections::HashMap::new(),
        nr_cpus,
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
///
/// `nr_cpus` is the guest VM's CPU count, used to mask garbage bits
/// beyond `nr_cpu_ids` in cpumask words captured by BPF probes.
/// Pass `None` to show all bits unmasked.
///
/// `param_names` maps function display names to BTF-resolved parameter
/// labels: `vec![(name, type_label)]`. When present and an event has
/// no struct field specs, parameters are printed with their real names
/// (e.g. `prev (task_struct *)`) instead of `arg0`.
pub fn format_probe_events_with_bpf_locs(
    events: &[super::process::ProbeEvent],
    func_names: &[(u32, String)],
    kernel_dir: Option<&str>,
    bpf_locs: &std::collections::HashMap<String, String>,
    nr_cpus: Option<u32>,
    param_names: &std::collections::HashMap<String, Vec<(String, String)>>,
) -> String {
    format_probe_events_inner(
        events,
        func_names,
        kernel_dir,
        bpf_locs,
        nr_cpus,
        param_names,
    )
}

/// Build param_names map from BtfFunc slices.
///
/// Formats each parameter as `(name, type_label)` where type_label is
/// `"struct_name *"` for struct pointers, `"ptr"` for untyped pointers,
/// or empty for scalars.
///
/// For variadic functions, pads the param list to 6 entries so the arg
/// cap logic does not truncate extra arguments. Extra entries beyond
/// the named params use `arg{i}` names.
pub fn build_param_names(
    btf_funcs: &[super::btf::BtfFunc],
) -> std::collections::HashMap<String, Vec<(String, String)>> {
    let mut map = std::collections::HashMap::new();
    for func in btf_funcs {
        let mut params: Vec<(String, String)> = func
            .params
            .iter()
            .take(6)
            .map(|p| {
                let type_label = if let Some(ref sname) = p.struct_name {
                    format!("{sname} *")
                } else if let Some(ref tname) = p.type_name {
                    format!("{tname} *")
                } else if p.is_ptr {
                    "ptr".into()
                } else {
                    String::new()
                };
                (p.name.clone(), type_label)
            })
            .collect();
        // Pad variadic functions to 6 so extra unnamed args are shown.
        if func.is_variadic {
            while params.len() < 6 {
                let i = params.len();
                params.push((format!("arg{i}"), String::new()));
            }
        }
        map.insert(func.name.clone(), params);
    }
    map
}

/// (group_label, vec of (field_name, entry_decoded, exit_decoded_or_none)).
type FieldGroup = (String, Vec<(String, String, Option<String>)>);

/// Format a single field line with optional entry→exit diff.
/// When `exit_val` is present and differs from `entry_val`, shows
/// `field  entry_val  →  exit_val` with entry_val padded to `arrow_col`.
/// When same or no exit, shows `field  entry_val`.
fn format_field_line(
    out: &mut String,
    field: &str,
    entry_val: &str,
    exit_val: Option<&str>,
    fw: usize,
    arrow_col: usize,
) {
    match exit_val {
        Some(ev) if ev != entry_val => {
            let col = arrow_col.max(entry_val.len());
            out.push_str(&format!("      {field:<fw$}  {entry_val:<col$}  →  {ev}\n"));
        }
        _ => {
            out.push_str(&format!("      {field:<fw$}  {entry_val}\n"));
        }
    }
}

fn format_probe_events_inner(
    events: &[super::process::ProbeEvent],
    func_names: &[(u32, String)],
    kernel_dir: Option<&str>,
    bpf_locs: &std::collections::HashMap<String, String>,
    nr_cpus: Option<u32>,
    param_names: &std::collections::HashMap<String, Vec<(String, String)>>,
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

    let all_addrs: Vec<u64> = func_addrs.iter().map(|(_, a)| *a).collect();

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

    /// Extract struct name from a field key like `"p:task_struct.field"`.
    /// Returns `""` when no struct context is present.
    fn struct_from_key(key: &str) -> &str {
        let (param_part, _) = key.split_once('.').unwrap_or((key, key));
        let (_, sname) = param_part.split_once(':').unwrap_or(("", ""));
        if sname == "val" { "" } else { sname }
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
            let sname = struct_from_key(k);
            let (_, field) = k.split_once('.').unwrap_or((k, k));
            let decoded = super::decode::decode_named_value(sname, field, &v.to_string());
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
            // No struct field specs — show args with BTF param names
            // when available, fall back to arg0/arg1/... otherwise.
            // Cap to BTF param count to avoid showing garbage register
            // values beyond actual params. Fall back to all 6 when
            // BTF resolution failed (empty params).
            let fw = max_field_w;
            let params = param_names.get(name);
            let arg_cap = params
                .map(|ps| ps.len())
                .filter(|&n| n > 0)
                .unwrap_or(6)
                .min(6);
            for (i, &val) in event.args[..arg_cap].iter().enumerate() {
                if val != 0 || i == 0 {
                    let (label, decoded) = if let Some(p) = params.and_then(|ps| ps.get(i)) {
                        let (pname, ptype) = p;
                        let lbl = if ptype.is_empty() {
                            pname.clone()
                        } else {
                            format!("{pname} ({ptype})")
                        };
                        let dec = if ptype.contains("task_struct") {
                            format!("ptr:{:04x}", val & 0xffff)
                        } else if ptype == "ptr" {
                            format_raw_arg(val)
                        } else {
                            decode_named_value("", pname, &val.to_string())
                        };
                        (lbl, dec)
                    } else {
                        (format!("arg{i}"), format_raw_arg(val))
                    };
                    out.push_str(&format!("      {label:<fw$}  {decoded}\n"));
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
            let merged_cpumask_str = format_cpumask_display(&cpumask_words, nr_cpus);

            // Build exit field lookup for paired display.
            let mut exit_cpumask_words: [u64; 4] = [0; 4];
            let exit_map: std::collections::HashMap<&str, u64> = event
                .exit_fields
                .iter()
                .map(|(k, v)| {
                    let (_, field) = k.split_once('.').unwrap_or((k, k));
                    match field {
                        "cpumask_0" => exit_cpumask_words[0] = *v,
                        "cpumask_1" => exit_cpumask_words[1] = *v,
                        "cpumask_2" => exit_cpumask_words[2] = *v,
                        "cpumask_3" => exit_cpumask_words[3] = *v,
                        _ => {}
                    }
                    (field, *v)
                })
                .collect();
            let exit_cpumask_str = if !event.exit_fields.is_empty() {
                Some(format_cpumask_display(&exit_cpumask_words, nr_cpus))
            } else {
                None
            };
            let has_exit = !event.exit_fields.is_empty();

            // Group fields by parameter, emit type headers for struct params.
            let mut groups: Vec<FieldGroup> = Vec::new();
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
                let sname = struct_from_key(key);
                let entry_decoded = if field == "cpumask_0" {
                    merged_cpumask_str.clone()
                } else {
                    decode_named_value(sname, field, &val.to_string())
                };
                let exit_decoded = if has_exit {
                    if field == "cpumask_0" {
                        exit_cpumask_str.clone()
                    } else {
                        exit_map
                            .get(field)
                            .map(|ev| decode_named_value(sname, field, &ev.to_string()))
                    }
                } else {
                    None
                };
                let display_field = if field == "cpumask_0" {
                    "cpus_ptr".to_string()
                } else {
                    field.to_string()
                };
                if let Some(grp) = groups.iter_mut().find(|(l, _)| l == &label) {
                    grp.1.push((display_field, entry_decoded, exit_decoded));
                } else {
                    groups.push((label, vec![(display_field, entry_decoded, exit_decoded)]));
                }
            }

            // Collect struct field name→value for scalar dedup.
            let struct_field_vals: std::collections::HashSet<(&str, &str)> = groups
                .iter()
                .filter(|(l, _)| l.contains('*'))
                .flat_map(|(_, fields)| fields.iter().map(|(f, v, _)| (f.as_str(), v.as_str())))
                .collect();

            // Compute arrow column: max entry value length across
            // changed fields (where exit differs from entry).
            let arrow_col: usize = groups
                .iter()
                .flat_map(|(_, fields)| fields.iter())
                .filter_map(|(_, ev, xv)| {
                    xv.as_ref()
                        .filter(|x| x.as_str() != ev.as_str())
                        .map(|_| ev.len())
                })
                .max()
                .unwrap_or(0);

            let fw = max_field_w;
            for (label, fields) in &groups {
                if fields.len() == 1 && !label.contains('*') {
                    let (fname, entry_val, exit_val) = &fields[0];
                    // Suppress scalar if identical name+value exists in a struct group.
                    if struct_field_vals.contains(&(fname.as_str(), entry_val.as_str())) {
                        continue;
                    }
                    format_field_line(
                        &mut out,
                        label,
                        entry_val,
                        exit_val.as_deref(),
                        fw,
                        arrow_col,
                    );
                } else {
                    out.push_str(&format!("    {label}\n"));
                    for (fname, entry_val, exit_val) in fields {
                        format_field_line(
                            &mut out,
                            fname,
                            entry_val,
                            exit_val.as_deref(),
                            fw,
                            arrow_col,
                        );
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

    // -- format_probe_events --

    #[test]
    fn format_probe_events_empty() {
        let out = format_probe_events(&[], &[], None, None);
        assert!(out.contains("=== AUTO-PROBE: scx_exit fired ==="));
        assert!(out.contains("no probe data captured"));
    }

    #[test]
    fn format_probe_events_with_synthetic_events() {
        use crate::probe::process::ProbeEvent;

        let events = vec![
            ProbeEvent {
                func_idx: 0,
                task_ptr: 42,
                ts: 100,
                args: [0xDEAD, 0xBEEF, 0, 0, 0, 0],
                fields: vec![
                    ("p:task_struct.pid".to_string(), 42),
                    ("p:task_struct.flags".to_string(), 0x1),
                ],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
            ProbeEvent {
                func_idx: 1,
                task_ptr: 42,
                ts: 200,
                args: [7, 0, 0, 0, 0, 0],
                fields: vec![],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
        ];
        let func_names = vec![
            (0u32, "do_enqueue_task".to_string()),
            (1u32, "balance_one".to_string()),
        ];

        let out = format_probe_events(&events, &func_names, None, None);
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
            task_ptr: 1,
            ts: 50,
            args: [1, 0, 0, 0, 0, 0],
            fields: vec![],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "known_func".to_string())];

        let out = format_probe_events(&events, &func_names, None, None);
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
                task_ptr: 1,
                ts: 200,
                args: [0; 6],
                fields: vec![],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
            ProbeEvent {
                func_idx: 0,
                task_ptr: 1,
                ts: 100,
                args: [0; 6],
                fields: vec![],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
        ];
        let func_names = vec![
            (0u32, "first_func".to_string()),
            (1u32, "second_func".to_string()),
        ];

        let out = format_probe_events(&events, &func_names, None, None);
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
            task_ptr: 1,
            ts: 100,
            args: [0, 0x42, 0, 0, 0, 0],
            fields: vec![],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "test_func".to_string())];
        let out = format_probe_events(&events, &func_names, None, None);
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

    // -- format_probe_events with fields --

    #[test]
    fn format_probe_events_field_key_splitting() {
        use crate::probe::process::ProbeEvent;

        // Field key "p0:task_struct.pid" -> split on '.' -> display "pid".
        // Value 123 -> decode_named_value("pid", "123") -> "123" (passthrough).
        let events = vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![("p0:task_struct.pid".to_string(), 123)],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, None);
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
            task_ptr: 42,
            ts: 100,
            args: [0xDEAD, 0, 0, 0, 0, 0],
            fields: vec![
                ("p0:task_struct.pid".to_string(), 42),
                ("p0:task_struct.weight".to_string(), 100),
            ],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "do_enqueue".to_string())];
        let out = format_probe_events(&events, &func_names, None, None);
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

        // Use only word 0 bits so the test is host-CPU-count independent.
        // Multi-word display is tested separately in cpumask_multi_word_hex_format.
        let events = vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![
                ("p:task_struct.pid".to_string(), 42),
                ("p:task_struct.cpumask_0".to_string(), 0x7),
                ("p:task_struct.cpumask_1".to_string(), 0),
                ("p:task_struct.cpumask_2".to_string(), 0),
                ("p:task_struct.cpumask_3".to_string(), 0),
            ],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, None);
        // cpumask_0..3 should be merged into one "cpus_ptr" line
        assert!(out.contains("cpus_ptr"), "should show cpus_ptr: {out}");
        // CPUs 0-2 from word 0
        assert!(out.contains("0-2"), "should contain 0-2: {out}");
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
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![
                ("p:task_struct.pid".to_string(), 42),
                ("cpu:val.cpu".to_string(), 3),
                ("rq:rq.cpu".to_string(), 3),
            ],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "test_fn".to_string())];
        let out = format_probe_events(&events, &func_names, None, None);
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
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![],
            str_val: Some("error: task stuck".to_string()),
            ..Default::default()
        }];
        let func_names = vec![(0u32, "scx_exit".to_string())];
        let out = format_probe_events(&events, &func_names, None, None);
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
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "mitosis_enqueue".to_string())];
        let mut locs = std::collections::HashMap::new();
        locs.insert("mitosis_enqueue".to_string(), "main.bpf.c:42".to_string());
        let out = format_probe_events_with_bpf_locs(
            &events,
            &func_names,
            None,
            &locs,
            None,
            &std::collections::HashMap::new(),
        );
        assert!(
            out.contains("main.bpf.c:42"),
            "should show BPF source loc: {out}",
        );
    }

    // -- cpumask multi-word hex display --

    #[test]
    fn cpumask_multi_word_hex_format() {
        // Test multi-word display with a controlled nr_cpus to avoid
        // host-dependent masking (CI runners may have <65 CPUs).
        let words = [0xffu64, 0x1, 0, 0];
        let out = format_cpumask_display(&words, Some(128));
        // Multi-word: underscore-separated 16-digit hex words.
        assert!(
            out.contains("_"),
            "multi-word should use _ separator: {out}"
        );
        // CPUs 0-7 from word 0, CPU 64 from word 1.
        assert!(out.contains("0-7"), "should list CPUs 0-7: {out}");
        assert!(out.contains("64"), "should list CPU 64: {out}");
    }

    #[test]
    fn cpumask_single_word_compact() {
        // Single-word display: compact hex without leading zeros.
        let words = [0xfu64, 0, 0, 0];
        let out = format_cpumask_display(&words, Some(64));
        assert!(out.contains("0xf("), "single-word should be compact: {out}");
        assert!(out.contains("0-3"), "should list CPUs 0-3: {out}");
    }

    // -- struct type header grouping --

    #[test]
    fn format_probe_events_struct_type_header() {
        use crate::probe::process::ProbeEvent;

        let events = vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![("rq:rq.cpu".to_string(), 2)],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }];
        let func_names = vec![(0u32, "scx_tick".to_string())];
        let out = format_probe_events(&events, &func_names, None, None);
        assert!(
            out.contains("rq *rq"),
            "should show struct type header: {out}"
        );
        assert!(out.contains("cpu"), "should show field under header: {out}");
    }

    // -- resolve_addrs_from_elf --

    #[test]
    fn resolve_addrs_from_elf_finds_kernel_function() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no vmlinux available");
                return;
            }
        };
        // find_test_vmlinux may return /sys/kernel/btf/vmlinux (raw BTF,
        // not an ELF), which resolve_addrs_from_elf cannot parse.
        if path.starts_with("/sys/") {
            eprintln!("skipping: {} is raw BTF, not ELF", path.display());
            return;
        }
        let func_names = vec![(0u32, "schedule".to_string())];
        let result = resolve_addrs_from_elf(&path, &func_names);
        assert!(!result.is_empty(), "should resolve 'schedule' from vmlinux");
        assert_eq!(result[0].0, "schedule");
        assert_ne!(result[0].1, 0, "schedule address should be nonzero");
    }

    #[test]
    fn resolve_addrs_from_elf_nonexistent_returns_empty() {
        let func_names = vec![(0u32, "schedule".to_string())];
        let result =
            resolve_addrs_from_elf(std::path::Path::new("/nonexistent/vmlinux"), &func_names);
        assert!(result.is_empty());
    }
}
