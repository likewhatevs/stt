use super::stack::StackFunction;

#[derive(Debug, Clone)]
pub struct BtfParam {
    pub name: String,
    /// Known struct name (in STRUCT_FIELDS) for field key generation
    pub struct_name: Option<String>,
    pub is_ptr: bool,
}

#[derive(Debug, Clone)]
pub struct BtfFunc {
    pub name: String,
    pub params: Vec<BtfParam>,
}

/// Known struct types and their fields for probe output decoding.
/// Maps struct name -> list of (field_access, output_key) pairs.
pub const STRUCT_FIELDS: &[(&str, &[(&str, &str)])] = &[
    (
        "task_struct",
        &[
            ("->pid", "pid"),
            ("->cpus_ptr->bits[0]", "cpus_ptr"),
            ("->scx.ddsp_dsq_id", "dsq_id"),
            ("->scx.ddsp_enq_flags", "enq_flags"),
            ("->scx.slice", "slice"),
            ("->scx.dsq_vtime", "vtime"),
            ("->scx.weight", "weight"),
            ("->scx.sticky_cpu", "sticky_cpu"),
            ("->scx.flags", "scx_flags"),
        ],
    ),
    ("rq", &[("->cpu", "cpu")]),
    ("scx_dispatch_q", &[("->id", "dsq_id")]),
    ("scx_init_task_args", &[("->fork", "fork")]),
    (
        "scx_exit_info",
        &[("->kind", "exit_kind"), ("->reason", "reason")],
    ),
    ("scx_cgroup_init_args", &[("->weight", "weight")]),
];

/// Resolve the BTF source path for a given kernel source directory.
/// If `{kernel_dir}/vmlinux` exists (uncompressed ELF with embedded BTF),
/// return its path. Otherwise return None to fall back to sysfs.
pub fn resolve_btf_path(kernel_dir: Option<&str>) -> Option<std::path::PathBuf> {
    let dir = kernel_dir?;
    let vmlinux = std::path::Path::new(dir).join("vmlinux");
    if vmlinux.exists() {
        tracing::debug!(path = %vmlinux.display(), "btf: using vmlinux from kernel_dir");
        Some(vmlinux)
    } else {
        tracing::debug!(path = %vmlinux.display(), "btf: vmlinux not found, falling back to sysfs");
        None
    }
}

/// Parse BTF from vmlinux for a set of function names using btf-rs.
pub fn parse_btf_functions(func_names: &[&str], vmlinux_path: Option<&str>) -> Vec<BtfFunc> {
    use btf_rs::{Btf, BtfType, Type};

    let btf_path = vmlinux_path.unwrap_or("/sys/kernel/btf/vmlinux");
    let btf = match Btf::from_file(btf_path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(%e, path = btf_path, "btf: failed to parse");
            return Vec::new();
        }
    };

    // Resolve a parameter's type_id to its underlying struct name,
    // following PTR/CONST/VOLATILE/TYPEDEF chains.
    let resolve_struct_name = |type_id: u32| -> Option<String> {
        let mut t = match btf.resolve_type_by_id(type_id) {
            Ok(t) => t,
            Err(_) => return None,
        };
        for _ in 0..10 {
            match &t {
                Type::Ptr(_)
                | Type::Const(_)
                | Type::Volatile(_)
                | Type::Typedef(_)
                | Type::Restrict(_)
                | Type::TypeTag(_) => {
                    t = match btf.resolve_chained_type(t.as_btf_type().unwrap()) {
                        Ok(next) => next,
                        Err(_) => return None,
                    };
                }
                Type::Struct(s) => {
                    return btf.resolve_name(s).ok();
                }
                Type::Union(u) => {
                    return btf.resolve_name(u).ok();
                }
                _ => return None,
            }
        }
        None
    };

    let is_ptr =
        |type_id: u32| -> bool { matches!(btf.resolve_type_by_id(type_id), Ok(Type::Ptr(_))) };

    let mut results = Vec::new();

    for func_name in func_names {
        let types = match btf.resolve_types_by_name(func_name) {
            Ok(t) => t,
            Err(_) => continue,
        };

        for t in &types {
            if let Type::Func(func) = t {
                // Resolve the FuncProto
                let proto = match btf.resolve_chained_type(func) {
                    Ok(Type::FuncProto(fp)) => fp,
                    _ => continue,
                };

                let mut params = Vec::new();
                for param in &proto.parameters {
                    let name = btf.resolve_name(param).unwrap_or_default();
                    let tid = param.get_type_id().unwrap_or(0);
                    let struct_name = resolve_struct_name(tid)
                        .filter(|n| STRUCT_FIELDS.iter().any(|(s, _)| s == n));
                    params.push(BtfParam {
                        name,
                        struct_name,
                        is_ptr: is_ptr(tid),
                    });
                }

                results.push(BtfFunc {
                    name: func_name.to_string(),
                    params,
                });
                break; // take first match
            }
        }
    }

    tracing::debug!(n = results.len(), "btf: parsed function signatures");
    results
}

/// Discover BPF sched_ext program symbols via libbpf-rs.
/// Finds struct_ops programs (the scheduler), then enumerates their
/// functions via BTF for fentry probing.
pub fn discover_bpf_symbols() -> Vec<StackFunction> {
    use libbpf_rs::btf;
    use libbpf_rs::query::ProgInfoIter;

    // Find struct_ops program IDs via libbpf-rs prog iteration
    let sched_prog_ids: Vec<u32> = ProgInfoIter::default()
        .filter(|info| info.ty == libbpf_rs::ProgramType::StructOps)
        .map(|info| info.id)
        .collect();

    if sched_prog_ids.is_empty() {
        tracing::debug!("discover_bpf_symbols: no struct_ops programs found");
        return Vec::new();
    }

    // Enumerate functions from each prog's BTF
    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();

    for prog_id in &sched_prog_ids {
        let prog_btf = match btf::Btf::from_prog_id(*prog_id) {
            Ok(b) => b,
            Err(_) => continue,
        };

        // Walk all Func types in the prog BTF
        for type_id in 1.. {
            use libbpf_rs::btf::{BtfKind, BtfType};
            let t = match prog_btf.type_by_id::<BtfType<'_>>(type_id.into()) {
                Some(t) => t,
                None => break,
            };
            if t.kind() != BtfKind::Func {
                continue;
            }
            let func_name = match t.name().and_then(|n| n.to_str()) {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };
            if !seen.insert(func_name.clone()) {
                continue;
            }
            results.push(StackFunction {
                raw_name: format!("bpf_prog_{prog_id}_{func_name}"),
                display_name: func_name,
                is_bpf: true,
                bpf_prog_id: Some(*prog_id),
            });
        }
    }

    tracing::debug!(n = results.len(), "discover_bpf_symbols");
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- STRUCT_FIELDS constant --

    #[test]
    fn struct_fields_has_task_struct() {
        let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
        assert!(entry.is_some());
        let (_, fields) = entry.unwrap();
        assert!(fields.iter().any(|(_, k)| *k == "pid"));
        assert!(fields.iter().any(|(_, k)| *k == "dsq_id"));
        assert!(fields.iter().any(|(_, k)| *k == "cpus_ptr"));
    }

    #[test]
    fn struct_fields_has_rq() {
        let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "rq");
        assert!(entry.is_some());
        let (_, fields) = entry.unwrap();
        assert!(fields.iter().any(|(_, k)| *k == "cpu"));
    }

    // -- parse_btf_functions (requires /sys/kernel/btf/vmlinux) --

    #[test]
    fn parse_btf_known_kernel_func() {
        if !std::path::Path::new("/sys/kernel/btf/vmlinux").exists() {
            return;
        }
        let funcs = parse_btf_functions(&["do_exit"], None);
        if funcs.is_empty() {
            return;
        }
        assert_eq!(funcs[0].name, "do_exit");
        assert!(!funcs[0].params.is_empty());
    }

    #[test]
    fn parse_btf_unknown_func_returns_empty() {
        let funcs = parse_btf_functions(&["__totally_fake_function_name__"], None);
        assert!(funcs.is_empty());
    }

    // -- discover_bpf_symbols --

    #[test]
    fn discover_bpf_symbols_no_scheduler() {
        let _ = discover_bpf_symbols();
    }

    // -- resolve_btf_path --

    #[test]
    fn resolve_btf_path_none() {
        assert!(resolve_btf_path(None).is_none());
    }

    #[test]
    fn resolve_btf_path_nonexistent() {
        assert!(resolve_btf_path(Some("/nonexistent/path")).is_none());
    }

    #[test]
    fn resolve_btf_path_dir_without_vmlinux() {
        let dir = std::env::temp_dir();
        assert!(resolve_btf_path(Some(dir.to_str().unwrap())).is_none());
    }

    // -- STRUCT_FIELDS invariants --

    #[test]
    fn struct_fields_all_entries_have_fields() {
        for (name, fields) in STRUCT_FIELDS {
            assert!(!fields.is_empty(), "struct {name} has no fields");
        }
    }

    #[test]
    fn struct_fields_no_duplicate_structs() {
        let names: Vec<&str> = STRUCT_FIELDS.iter().map(|(n, _)| *n).collect();
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "duplicate struct names in STRUCT_FIELDS"
        );
    }

    #[test]
    fn struct_fields_no_duplicate_keys_per_struct() {
        for (name, fields) in STRUCT_FIELDS {
            let keys: Vec<&str> = fields.iter().map(|(_, k)| *k).collect();
            let unique: std::collections::HashSet<&&str> = keys.iter().collect();
            assert_eq!(
                keys.len(),
                unique.len(),
                "struct {name} has duplicate field keys"
            );
        }
    }

    #[test]
    fn struct_fields_has_scx_dispatch_q() {
        let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "scx_dispatch_q");
        assert!(entry.is_some());
    }

    #[test]
    fn struct_fields_has_scx_exit_info() {
        let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "scx_exit_info");
        assert!(entry.is_some());
    }

    #[test]
    fn struct_fields_has_scx_init_task_args() {
        let entry = STRUCT_FIELDS
            .iter()
            .find(|(s, _)| *s == "scx_init_task_args");
        assert!(entry.is_some());
    }

    #[test]
    fn struct_fields_has_scx_cgroup_init_args() {
        let entry = STRUCT_FIELDS
            .iter()
            .find(|(s, _)| *s == "scx_cgroup_init_args");
        assert!(entry.is_some());
    }

    // -- BtfParam/BtfFunc construction --

    #[test]
    fn btf_param_debug_display() {
        let p = BtfParam {
            name: "test".into(),
            struct_name: Some("task_struct".into()),
            is_ptr: true,
        };
        let dbg = format!("{:?}", p);
        assert!(dbg.contains("task_struct"));
        assert!(dbg.contains("test"));
    }

    // -- BtfFunc construction --

    #[test]
    fn btf_func_empty_params() {
        let f = BtfFunc {
            name: "empty".into(),
            params: vec![],
        };
        assert_eq!(f.name, "empty");
        assert!(f.params.is_empty());
    }

    #[test]
    fn btf_func_multiple_params() {
        let f = BtfFunc {
            name: "multi".into(),
            params: vec![
                BtfParam {
                    name: "a".into(),
                    struct_name: Some("task_struct".into()),
                    is_ptr: true,
                },
                BtfParam {
                    name: "b".into(),
                    struct_name: Some("rq".into()),
                    is_ptr: true,
                },
                BtfParam {
                    name: "c".into(),
                    struct_name: None,
                    is_ptr: false,
                },
            ],
        };
        assert_eq!(f.params.len(), 3);
        assert!(f.params[0].is_ptr);
        assert!(f.params[2].struct_name.is_none());
    }

    // -- parse_btf_functions with multiple names --

    #[test]
    fn parse_btf_multiple_unknown_funcs() {
        let funcs = parse_btf_functions(&["__fake_a__", "__fake_b__", "__fake_c__"], None);
        assert!(funcs.is_empty());
    }

    // -- resolve_btf_path with existing vmlinux --

    #[test]
    fn resolve_btf_path_real_kernel_dir() {
        // If /usr/lib/debug/boot/vmlinux exists, this would return Some.
        // On most systems it doesn't, so we just verify no panic.
        let _ = resolve_btf_path(Some("/usr/lib/debug/boot"));
    }

    // -- STRUCT_FIELDS field access patterns --

    #[test]
    fn struct_fields_access_patterns_start_with_arrow() {
        for (name, fields) in STRUCT_FIELDS {
            for (access, _key) in *fields {
                assert!(
                    access.starts_with("->"),
                    "struct {name} field access '{access}' should start with '->'"
                );
            }
        }
    }
}
