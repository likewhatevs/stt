use super::stack::StackFunction;

/// BTF-resolved parameter metadata for a probed function.
///
/// Each param maps to one register (fentry/kprobe). Struct pointer
/// params have their fields expanded via [`STRUCT_FIELDS`] or
/// auto-discovered from vmlinux or BPF program BTF.
#[derive(Debug, Clone, Default)]
pub struct BtfParam {
    pub name: String,
    /// Known struct name (in STRUCT_FIELDS) for field key generation
    pub struct_name: Option<String>,
    pub is_ptr: bool,
    /// True if this is a char * / const char * (string pointer).
    pub is_string_ptr: bool,
    /// Auto-discovered fields from vmlinux or BPF program BTF for
    /// struct pointer types not in STRUCT_FIELDS.
    /// Vec of (field_name, access_pattern).
    pub auto_fields: Vec<(String, String)>,
    /// Type name for auto-discovered structs (used in output headers).
    pub type_name: Option<String>,
}

/// BTF-resolved function signature.
///
/// Produced by [`parse_btf_functions`] (kernel functions via vmlinux BTF)
/// or [`parse_bpf_btf_functions`] (BPF callbacks via program BTF).
/// Used by [`run_probe_skeleton`](super::process::run_probe_skeleton) to
/// populate field specs and by [`build_field_keys`](super::process) to
/// generate output labels.
#[derive(Debug, Clone, Default)]
pub struct BtfFunc {
    pub name: String,
    pub params: Vec<BtfParam>,
    /// True if BTF FuncProto has a variadic sentinel parameter
    /// (name_off=0, type=0). Variadic functions should not have
    /// their displayed arg count capped.
    pub is_variadic: bool,
}

/// Known struct types and their fields for probe output decoding.
/// Maps struct name -> list of (field_access, output_key) pairs.
pub const STRUCT_FIELDS: &[(&str, &[(&str, &str)])] = &[
    (
        "task_struct",
        &[
            ("->pid", "pid"),
            ("->cpus_ptr->bits[0]", "cpumask_0"),
            ("->cpus_ptr->bits[1]", "cpumask_1"),
            ("->cpus_ptr->bits[2]", "cpumask_2"),
            ("->cpus_ptr->bits[3]", "cpumask_3"),
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

/// Runtime-resolved field dereference spec for the BPF skeleton.
/// Maps 1:1 to the C `struct field_spec` in intf.h.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub param_idx: u32,
    pub offset: u32,
    pub size: u32,
    pub field_idx: u32,
    /// Byte offset to intermediate pointer for chained dereferences.
    /// 0 = single-level read. Nonzero = read ptr at base+ptr_offset,
    /// then read size bytes at ptr+offset.
    pub ptr_offset: u32,
}

/// Resolve struct field offsets from BTF for a single function.
///
/// Handles both STRUCT_FIELDS entries (curated fields with known output
/// keys) and auto-discovered fields from [`discover_vmlinux_struct_fields`]
/// (stored in `BtfParam::auto_fields`). STRUCT_FIELDS entries consume
/// field slots first; auto-discovered fields fill remaining budget up to
/// MAX_FIELDS (16). Warns when auto-discovered fields are truncated.
///
/// Handles chained pointer dereferences (e.g. `->cpus_ptr->bits[0]`)
/// by reading through intermediate pointers.
pub fn resolve_field_specs(btf_func: &BtfFunc, vmlinux_path: Option<&str>) -> Vec<FieldSpec> {
    use btf_rs::Btf;

    let btf_path = vmlinux_path.unwrap_or("/sys/kernel/btf/vmlinux");
    let btf = match Btf::from_file(btf_path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(%e, path = btf_path, "resolve_field_specs: failed to parse BTF");
            return Vec::new();
        }
    };

    let mut specs = Vec::new();
    let mut field_idx: u32 = 0;

    let max_params = btf_func.params.len().min(6);
    for (param_idx, param) in btf_func.params[..max_params].iter().enumerate() {
        if let Some(ref struct_name) = param.struct_name {
            // Known struct in STRUCT_FIELDS — curated field list.
            let fields = match STRUCT_FIELDS
                .iter()
                .find(|(s, _)| *s == struct_name.as_str())
            {
                Some((_, f)) => *f,
                None => continue,
            };

            let struct_type = match resolve_struct_type(&btf, struct_name) {
                Some(s) => s,
                None => {
                    // Skip field slots to stay aligned with build_field_keys.
                    field_idx += fields.len() as u32;
                    continue;
                }
            };

            for (access, _key) in fields {
                let access = access.trim_start_matches("->");

                if access.contains("->") {
                    let (ptr_member, target) = access.split_once("->").unwrap();
                    let ptr_off_result = resolve_member_offset(&btf, &struct_type, ptr_member);
                    if let Some((ptr_off, _)) = ptr_off_result {
                        let pointed = resolve_pointed_struct(&btf, &struct_type, ptr_member);
                        if let Some(pointed_struct) = pointed
                            && let Some((target_off, target_sz)) =
                                resolve_member_offset(&btf, &pointed_struct, target)
                        {
                            specs.push(FieldSpec {
                                param_idx: param_idx as u32,
                                offset: target_off,
                                size: target_sz,
                                field_idx,
                                ptr_offset: ptr_off,
                            });
                        }
                    } else {
                        tracing::debug!(
                            member = ptr_member,
                            "chained deref: member offset not found",
                        );
                    }
                } else {
                    if let Some((offset, size)) = resolve_member_offset(&btf, &struct_type, access)
                    {
                        specs.push(FieldSpec {
                            param_idx: param_idx as u32,
                            offset,
                            size,
                            field_idx,
                            ptr_offset: 0,
                        });
                    }
                }

                field_idx += 1;
                if field_idx >= 16 {
                    break;
                }
            }
        } else if !param.auto_fields.is_empty() {
            // Auto-discovered vmlinux struct fields.
            let sname = match param.type_name.as_deref() {
                Some(n) => n,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };
            let struct_type = match resolve_struct_type(&btf, sname) {
                Some(s) => s,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };

            let remaining = (16 - field_idx) as usize;
            if param.auto_fields.len() > remaining {
                tracing::warn!(
                    func = %btf_func.name,
                    struct_name = sname,
                    total = param.auto_fields.len(),
                    budget = remaining,
                    "auto-discovered fields truncated to MAX_FIELDS budget",
                );
            }

            for (_fname, access) in &param.auto_fields {
                if field_idx >= 16 {
                    break;
                }
                let access = access.trim_start_matches("->");

                if access.contains("->") {
                    if let Some((ptr_member, target)) = access.split_once("->") {
                        let ptr_off_result = resolve_member_offset(&btf, &struct_type, ptr_member);
                        if let Some((ptr_off, _)) = ptr_off_result {
                            let pointed = resolve_pointed_struct(&btf, &struct_type, ptr_member);
                            if let Some(pointed_struct) = pointed
                                && let Some((target_off, target_sz)) =
                                    resolve_member_offset(&btf, &pointed_struct, target)
                            {
                                specs.push(FieldSpec {
                                    param_idx: param_idx as u32,
                                    offset: target_off,
                                    size: target_sz,
                                    field_idx,
                                    ptr_offset: ptr_off,
                                });
                            }
                        }
                    }
                } else if let Some((offset, size)) =
                    resolve_member_offset(&btf, &struct_type, access)
                {
                    specs.push(FieldSpec {
                        param_idx: param_idx as u32,
                        offset,
                        size,
                        field_idx,
                        ptr_offset: 0,
                    });
                }

                field_idx += 1;
            }
        } else if !param.is_ptr {
            // Scalar param takes one field slot (matched by build_field_keys).
            field_idx += 1;
        }
    }

    tracing::debug!(n = specs.len(), func = %btf_func.name, "resolve_field_specs");
    specs
}

/// Find a BTF struct type by name.
fn resolve_struct_type(btf: &btf_rs::Btf, name: &str) -> Option<btf_rs::Struct> {
    let types = btf.resolve_types_by_name(name).ok()?;
    for t in types {
        if let btf_rs::Type::Struct(s) = t {
            return Some(s);
        }
    }
    None
}

/// Resolve byte offset and read size for a possibly nested field access.
/// Access is dot-separated (e.g. "scx.ddsp_dsq_id").
fn resolve_member_offset(
    btf: &btf_rs::Btf,
    struct_type: &btf_rs::Struct,
    access: &str,
) -> Option<(u32, u32)> {
    let parts: Vec<&str> = access.split('.').collect();
    let mut current_struct = struct_type.clone();
    let mut total_offset: u32 = 0;

    for (i, part) in parts.iter().enumerate() {
        // Strip array index (e.g. "bits[0]" -> "bits", extract 0).
        let (member_name, array_idx) = if let Some(bracket) = part.find('[') {
            let name = &part[..bracket];
            let idx_str = &part[bracket + 1..part.len() - 1];
            let idx: u32 = idx_str.parse().unwrap_or(0);
            (name, Some(idx))
        } else {
            (*part, None)
        };

        // Find the member in the current struct.
        let member = current_struct.members.iter().find(|m| {
            btf.resolve_name(*m)
                .map(|n| n == member_name)
                .unwrap_or(false)
        })?;

        let bit_off = member.bit_offset();
        if bit_off % 8 != 0 {
            // Bitfield -- skip.
            return None;
        }
        total_offset += bit_off / 8;

        // Add array element offset if indexed.
        if let Some(idx) = array_idx {
            let elem_size = resolve_type_size(btf, member)?;
            total_offset += idx * elem_size;
        }

        let is_last = i == parts.len() - 1;
        if is_last {
            // Resolve the member's type to determine read size.
            let size = resolve_type_size(btf, member)?;
            return Some((total_offset, size));
        }

        // Not the last part: the member must be an embedded struct/union.
        // Follow the type chain to find the struct.
        let member_type = follow_to_struct_or_union(btf, member)?;
        current_struct = member_type;
    }

    None
}

/// Follow a member's type through qualifiers to find the underlying
/// type size. Returns 8 for pointers, the type's size for int/enum/struct.
fn resolve_type_size(btf: &btf_rs::Btf, member: &btf_rs::Member) -> Option<u32> {
    use btf_rs::{BtfType, Type};

    let tid = member.get_type_id().ok()?;
    let mut t = btf.resolve_type_by_id(tid).ok()?;

    for _ in 0..20 {
        match t {
            Type::Ptr(_) => return Some(8),
            Type::Int(ref i) => return Some(i.size() as u32),
            Type::Enum(ref e) => return Some(e.size() as u32),
            Type::Enum64(_) => return Some(8),
            Type::Struct(ref s) => return Some(s.size() as u32),
            Type::Union(ref u) => return Some(u.size() as u32),
            Type::Array(ref a) => {
                // For array access like bits[0], return element size.
                let elem_tid = a.get_type_id().ok()?;
                let elem = btf.resolve_type_by_id(elem_tid).ok()?;
                match elem {
                    Type::Int(ref i) => return Some(i.size() as u32),
                    Type::Ptr(_) => return Some(8),
                    _ => return Some(8),
                }
            }
            Type::Const(_)
            | Type::Volatile(_)
            | Type::Restrict(_)
            | Type::Typedef(_)
            | Type::TypeTag(_) => {
                t = btf.resolve_chained_type(t.as_btf_type()?).ok()?;
            }
            _ => return None,
        }
    }
    None
}

/// Follow a member's type through qualifiers to find an embedded
/// struct or union type.
fn follow_to_struct_or_union(btf: &btf_rs::Btf, member: &btf_rs::Member) -> Option<btf_rs::Struct> {
    use btf_rs::{BtfType, Type};

    let tid = member.get_type_id().ok()?;
    let mut t = btf.resolve_type_by_id(tid).ok()?;

    for _ in 0..20 {
        match t {
            Type::Struct(s) | Type::Union(s) => return Some(s),
            Type::Const(_)
            | Type::Volatile(_)
            | Type::Restrict(_)
            | Type::Typedef(_)
            | Type::TypeTag(_) => {
                t = btf.resolve_chained_type(t.as_btf_type()?).ok()?;
            }
            _ => return None,
        }
    }
    None
}

/// Resolve a pointer member's pointed-to struct type.
/// Given a struct and a member name that is a pointer, follow the
/// pointer type to find the struct/union it points to.
fn resolve_pointed_struct(
    btf: &btf_rs::Btf,
    struct_type: &btf_rs::Struct,
    member_name: &str,
) -> Option<btf_rs::Struct> {
    use btf_rs::BtfType;

    // Strip array index (e.g. "bits[0]" -> "bits").
    let member_name = member_name.split('[').next().unwrap_or(member_name);

    let member = struct_type.members.iter().find(|m| {
        btf.resolve_name(*m)
            .map(|n| n == member_name)
            .unwrap_or(false)
    })?;

    let tid = member.get_type_id().ok()?;
    crate::monitor::bpf_map::resolve_to_struct(btf, tid)
}

/// Parse BTF from vmlinux for kernel function signatures.
///
/// Resolves parameter types via btf-rs, following PTR/CONST/VOLATILE/TYPEDEF
/// chains to identify struct pointers. Records `struct_name` for types
/// listed in [`STRUCT_FIELDS`]; other struct pointers get auto-discovered
/// fields via [`discover_vmlinux_struct_fields`] and `type_name` set.
/// Detects `char *` parameters (`is_string_ptr`) by chasing the type chain
/// to an `Int` of size 1.
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
        for _ in 0..20 {
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

    // Detect char * / const char * — Ptr → (Const/Volatile →) Int(size=1).
    let is_str_ptr = |type_id: u32| -> bool {
        let mut t = match btf.resolve_type_by_id(type_id) {
            Ok(t) => t,
            Err(_) => return false,
        };
        for _ in 0..20 {
            match &t {
                Type::Ptr(_)
                | Type::Const(_)
                | Type::Volatile(_)
                | Type::Restrict(_)
                | Type::Typedef(_)
                | Type::TypeTag(_) => {
                    t = match btf.resolve_chained_type(t.as_btf_type().unwrap()) {
                        Ok(next) => next,
                        Err(_) => return false,
                    };
                }
                Type::Int(i) => return i.size() == 1,
                _ => return false,
            }
        }
        false
    };

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

                let variadic = proto
                    .parameters
                    .last()
                    .map(|p| p.is_variadic())
                    .unwrap_or(false);

                let mut params = Vec::new();
                for param in &proto.parameters {
                    // Skip the variadic sentinel (name_off=0, type=0).
                    if param.is_variadic() {
                        continue;
                    }
                    let name = btf.resolve_name(param).unwrap_or_default();
                    let tid = param.get_type_id().unwrap_or(0);
                    let all_struct_name = resolve_struct_name(tid);
                    let known_struct = all_struct_name
                        .as_ref()
                        .filter(|n| STRUCT_FIELDS.iter().any(|(s, _)| *s == n.as_str()))
                        .cloned();
                    let param_is_ptr = is_ptr(tid);

                    // Auto-discover fields for struct pointers not in STRUCT_FIELDS.
                    let (auto_fields, type_name) = if param_is_ptr && known_struct.is_none() {
                        if let Some(ref sname) = all_struct_name {
                            let fields = discover_vmlinux_struct_fields(&btf, tid);
                            (fields, Some(sname.clone()))
                        } else {
                            (Vec::new(), None)
                        }
                    } else {
                        (
                            Vec::new(),
                            all_struct_name.filter(|_| known_struct.is_none()),
                        )
                    };

                    params.push(BtfParam {
                        name,
                        struct_name: known_struct,
                        is_ptr: param_is_ptr,
                        is_string_ptr: is_str_ptr(tid),
                        auto_fields,
                        type_name,
                    });
                }

                results.push(BtfFunc {
                    name: func_name.to_string(),
                    params,
                    is_variadic: variadic,
                });
                break; // take first match
            }
        }
    }

    tracing::debug!(n = results.len(), "btf: parsed function signatures");
    results
}

/// Discover loaded sched_ext BPF programs via libbpf-rs `ProgInfoIter`.
///
/// Discovers programs in two passes:
/// 1. All `StructOps` programs (scheduler callbacks).
/// 2. Any other loaded BPF program whose name matches a display name
///    in `stack_names` (e.g. `SEC("syscall")` programs like
///    `apply_cell_config` that appear in crash backtraces).
///
/// For non-StructOps matching, `bpf_prog_info.name` is truncated to
/// 15 characters (`BPF_OBJ_NAME_LEN - 1`). When a stack name is
/// longer than 15 chars, the truncated `info.name` is used as a
/// candidate prefix match, then the full name is confirmed via the
/// program's BTF.
///
/// Returns a [`StackFunction`] per discovered program with `is_bpf = true`
/// and the program's ID in `bpf_prog_id`. The `raw_name` is
/// `bpf_prog_{id}_{name}` to match kallsyms format.
pub fn discover_bpf_symbols(stack_names: &[&str]) -> Vec<StackFunction> {
    use libbpf_rs::query::ProgInfoIter;

    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();

    for info in ProgInfoIter::default() {
        let info_name = match info.name.to_str() {
            Ok(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };
        if info.ty == libbpf_rs::ProgramType::StructOps {
            // bpf_prog_info.name is truncated to 15 chars. Resolve
            // the full name from program BTF when truncated so that
            // set_attach_target can find the function.
            let full_name = if info_name.len() >= 15 {
                resolve_bpf_prog_full_name(info.id).unwrap_or(info_name.clone())
            } else {
                info_name.clone()
            };
            if !seen.insert(full_name.clone()) {
                continue;
            }
            results.push(StackFunction {
                raw_name: format!("bpf_prog_{}_{full_name}", info.id),
                display_name: full_name,
                is_bpf: true,
                bpf_prog_id: Some(info.id),
            });
        } else if !stack_names.is_empty() {
            // bpf_prog_info.name is truncated to 15 chars. For short
            // names, exact match works. For long names, check if any
            // stack name starts with the truncated info.name, then
            // confirm the full name via BTF.
            let matched_name = if stack_names.contains(&info_name.as_str()) {
                Some(info_name.clone())
            } else {
                // Candidate: a stack name whose prefix matches the
                // truncated info.name (only relevant when info.name
                // is at the 15-char limit).
                let candidate = stack_names
                    .iter()
                    .find(|sn| sn.len() > info_name.len() && sn.starts_with(&info_name));
                if let Some(target) = candidate {
                    resolve_bpf_prog_full_name(info.id).filter(|full| full == *target)
                } else {
                    None
                }
            };
            if let Some(func_name) = matched_name {
                if !seen.insert(func_name.clone()) {
                    continue;
                }
                tracing::debug!(
                    name = %func_name, id = info.id, ty = ?info.ty,
                    "discover_bpf_symbols: matched non-struct_ops program from stack",
                );
                results.push(StackFunction {
                    raw_name: format!("bpf_prog_{}_{func_name}", info.id),
                    display_name: func_name,
                    is_bpf: true,
                    bpf_prog_id: Some(info.id),
                });
            }
        }
    }

    tracing::debug!(n = results.len(), "discover_bpf_symbols");
    results
}

/// Resolve the full function name for a BPF program from its BTF.
///
/// `bpf_prog_info.name` is truncated to 15 characters. The full name
/// is resolved from `func_info[0].type_id` in the program's BTF,
/// which the kernel guarantees is the entry point (`insn_off == 0`).
fn resolve_bpf_prog_full_name(prog_id: u32) -> Option<String> {
    use libbpf_rs::AsRawLibbpf;
    use libbpf_rs::libbpf_sys;

    let prog_btf = libbpf_rs::btf::Btf::from_prog_id(prog_id).ok()?;
    let btf_ptr = prog_btf.as_libbpf_object().as_ptr();

    let fd = unsafe { libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
    if fd < 0 {
        return None;
    }

    let mut info = libbpf_sys::bpf_prog_info::default();
    let mut info_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
    let ret = unsafe {
        libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info as *mut _ as *mut _, &mut info_len)
    };
    if ret != 0 || info.nr_func_info == 0 {
        unsafe { libc::close(fd) };
        return None;
    }

    let fi_rec = info.func_info_rec_size as usize;
    let mut fi_buf = vec![0u8; info.nr_func_info as usize * fi_rec];

    let mut info2 = libbpf_sys::bpf_prog_info {
        nr_func_info: info.nr_func_info,
        func_info_rec_size: info.func_info_rec_size,
        func_info: fi_buf.as_mut_ptr() as u64,
        ..Default::default()
    };
    let mut info2_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
    let ret = unsafe {
        libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info2 as *mut _ as *mut _, &mut info2_len)
    };
    unsafe { libc::close(fd) };
    if ret != 0 {
        return None;
    }

    // func_info[0] is the entry point (insn_off == 0).
    let fi = unsafe { &*(fi_buf.as_ptr() as *const libbpf_sys::bpf_func_info) };
    let t = unsafe { libbpf_sys::btf__type_by_id(btf_ptr, fi.type_id) };
    if t.is_null() {
        return None;
    }
    let name_ptr = unsafe { libbpf_sys::btf__name_by_offset(btf_ptr, (*t).name_off) };
    if name_ptr.is_null() {
        return None;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) }
        .to_str()
        .ok()?
        .to_string();
    if name.is_empty() { None } else { Some(name) }
}

/// Resolve source locations for BPF functions from program BTF line_info.
///
/// Queries each program's `bpf_prog_info` via `bpf_obj_get_info_by_fd`
/// to get func_info and line_info buffers, then cross-references them
/// to find the first line_info entry at or after each function's insn_off.
/// Returns a map from function name to `"basename:line"`.
pub fn resolve_bpf_source_locs(prog_ids: &[u32]) -> std::collections::HashMap<String, String> {
    use libbpf_rs::AsRawLibbpf;
    use libbpf_rs::libbpf_sys;

    let mut locs = std::collections::HashMap::new();

    for prog_id in prog_ids {
        let prog_btf = match libbpf_rs::btf::Btf::from_prog_id(*prog_id) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let btf_ptr = prog_btf.as_libbpf_object().as_ptr();

        let fd = unsafe { libbpf_sys::bpf_prog_get_fd_by_id(*prog_id) };
        if fd < 0 {
            continue;
        }

        // First query: get func_info/line_info counts.
        let mut info = libbpf_sys::bpf_prog_info::default();
        let mut info_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
        let ret = unsafe {
            libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info as *mut _ as *mut _, &mut info_len)
        };
        if ret != 0 || info.nr_func_info == 0 || info.nr_line_info == 0 {
            unsafe { libc::close(fd) };
            continue;
        }

        let nr_fi = info.nr_func_info as usize;
        let nr_li = info.nr_line_info as usize;
        let fi_rec = info.func_info_rec_size as usize;
        let li_rec = info.line_info_rec_size as usize;
        let mut fi_buf = vec![0u8; nr_fi * fi_rec];
        let mut li_buf = vec![0u8; nr_li * li_rec];

        // Second query: populate func_info and line_info buffers.
        let mut info2 = libbpf_sys::bpf_prog_info {
            nr_func_info: nr_fi as u32,
            func_info_rec_size: fi_rec as u32,
            func_info: fi_buf.as_mut_ptr() as u64,
            nr_line_info: nr_li as u32,
            line_info_rec_size: li_rec as u32,
            line_info: li_buf.as_mut_ptr() as u64,
            ..Default::default()
        };
        let mut info2_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
        let ret = unsafe {
            libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info2 as *mut _ as *mut _, &mut info2_len)
        };
        unsafe { libc::close(fd) };
        if ret != 0 {
            continue;
        }

        // Cross-reference func_info with line_info to resolve source
        // locations for each function.
        for i in 0..nr_fi {
            let fi =
                unsafe { &*(fi_buf.as_ptr().add(i * fi_rec) as *const libbpf_sys::bpf_func_info) };
            let t = unsafe { libbpf_sys::btf__type_by_id(btf_ptr, fi.type_id) };
            if t.is_null() {
                continue;
            }
            let name_ptr = unsafe { libbpf_sys::btf__name_by_offset(btf_ptr, (*t).name_off) };
            if name_ptr.is_null() {
                continue;
            }
            let fname = unsafe { std::ffi::CStr::from_ptr(name_ptr) }
                .to_str()
                .unwrap_or("")
                .to_string();
            if fname.is_empty() {
                continue;
            }

            // Find the first line_info entry at or after this function's
            // instruction offset.
            let mut best: Option<&libbpf_sys::bpf_line_info> = None;
            for j in 0..nr_li {
                let li = unsafe {
                    &*(li_buf.as_ptr().add(j * li_rec) as *const libbpf_sys::bpf_line_info)
                };
                if li.insn_off >= fi.insn_off && best.is_none_or(|b| li.insn_off < b.insn_off) {
                    best = Some(li);
                }
            }
            if let Some(li) = best {
                let file_ptr =
                    unsafe { libbpf_sys::btf__name_by_offset(btf_ptr, li.file_name_off) };
                if !file_ptr.is_null() {
                    let file = unsafe { std::ffi::CStr::from_ptr(file_ptr) }
                        .to_str()
                        .unwrap_or("");
                    if !file.is_empty() {
                        let basename = file.rsplit('/').next().unwrap_or(file);
                        let line = li.line_col >> 10;
                        locs.insert(fname, format!("{basename}:{line}"));
                    }
                }
            }
        }
    }

    tracing::debug!(n = locs.len(), "resolve_bpf_source_locs");
    locs
}

/// Parse BTF from loaded BPF programs for callback signatures.
///
/// For each `(display_name, prog_id)`, resolves the typed params by:
/// 1. Looking for `____name` (inner function with typed params) in program BTF.
/// 2. Falling back to [`resolve_ops_callback_proto`] from vmlinux `sched_ext_ops`.
/// 3. Last resort: wrapper function with `void *ctx` (no useful params).
///
/// For struct pointer params not in [`STRUCT_FIELDS`], auto-discovers
/// scalar and cpumask pointer fields from BPF program BTF via
/// `discover_bpf_struct_fields`.
pub fn parse_bpf_btf_functions(
    func_names: &[(&str, u32)], // (display_name, prog_id)
) -> Vec<BtfFunc> {
    use libbpf_rs::btf;

    let mut by_prog: std::collections::HashMap<u32, Vec<&str>> = std::collections::HashMap::new();
    for (name, pid) in func_names {
        by_prog.entry(*pid).or_default().push(name);
    }

    let vmlinux = match btf::Btf::from_vmlinux() {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(%e, "parse_bpf_btf: failed to load vmlinux BTF");
            return Vec::new();
        }
    };

    let resolve_struct_name = |b: &btf::Btf<'_>, type_id: btf::TypeId| -> Option<String> {
        let t = b.type_by_id::<btf::BtfType<'_>>(type_id)?;
        let inner = t.skip_mods_and_typedefs();
        let deref = if inner.kind() == btf::BtfKind::Ptr {
            inner.next_type()?.skip_mods_and_typedefs()
        } else {
            inner
        };
        if deref.kind() == btf::BtfKind::Struct || deref.kind() == btf::BtfKind::Union {
            Some(deref.name()?.to_str()?.to_string())
        } else {
            None
        }
    };

    let mut results = Vec::new();

    for (prog_id, names) in &by_prog {
        let prog_btf = match btf::Btf::from_prog_id(*prog_id) {
            Ok(b) => b,
            Err(e) => {
                tracing::debug!(prog_id, %e, "parse_bpf_btf: failed to load prog BTF");
                continue;
            }
        };

        for func_name in names {
            // Resolve the real typed params for this struct_ops callback.
            // Strategy:
            // 1. Try ____name (inner function with typed params) — may not be in BTF
            // 2. Try vmlinux sched_ext_ops member for the callback signature
            // 3. Fall back to wrapper name (void *ctx — no useful params)
            let inner_name = format!("____{func_name}");
            let (proto, skip_first_param) =
                if let Some(f) = prog_btf.type_by_name::<btf::types::Func<'_>>(&inner_name) {
                    let bt: btf::BtfType<'_> = *f;
                    if let Some(pt) = bt
                        .next_type()
                        .filter(|t| t.kind() == btf::BtfKind::FuncProto)
                    {
                        if let Ok(p) = TryInto::<btf::types::FuncProto<'_>>::try_into(pt) {
                            (Some(p), true)
                        } else {
                            (None, false)
                        }
                    } else {
                        (None, false)
                    }
                } else {
                    (None, false)
                };

            // Fallback: resolve from vmlinux sched_ext_ops struct member.
            let ops_proto = if proto.is_none() {
                let p = resolve_ops_callback_proto(&vmlinux, func_name);
                tracing::debug!(
                    func = func_name,
                    ops_found = p.is_some(),
                    "bpf_btf: ____name not in BTF, trying ops fallback",
                );
                p
            } else {
                None
            };

            let (use_proto_from_ops, proto) = if let Some(ref p) = proto {
                (false, p)
            } else if let Some(ref p) = ops_proto {
                (true, p)
            } else {
                // Last resort: use the wrapper (void *ctx).
                let f = match prog_btf.type_by_name::<btf::types::Func<'_>>(func_name) {
                    Some(f) => f,
                    None => continue,
                };
                let bt: btf::BtfType<'_> = *f;
                let _pt = match bt
                    .next_type()
                    .filter(|t| t.kind() == btf::BtfKind::FuncProto)
                {
                    Some(t) => t,
                    None => continue,
                };
                // Can't hold the proto across the match — just push an empty BtfFunc.
                results.push(BtfFunc {
                    name: func_name.to_string(),
                    params: vec![],
                    is_variadic: false,
                });
                continue;
            };

            let btf_for_params: &btf::Btf<'_> = if use_proto_from_ops {
                &vmlinux
            } else {
                &prog_btf
            };

            let mut params = Vec::new();
            let param_iter: Vec<_> = if skip_first_param && !use_proto_from_ops {
                proto.iter().skip(1).collect()
            } else {
                proto.iter().collect()
            };
            for (param_pos, param) in param_iter.into_iter().enumerate() {
                let mut name = param
                    .name
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                // Infer param name from type when vmlinux FuncProto
                // has empty names (function pointer BTF often lacks them).
                if name.is_empty() {
                    let sname = resolve_struct_name(btf_for_params, param.ty);
                    name = match sname.as_deref() {
                        Some("task_struct") => "p".into(),
                        Some("rq") => "rq".into(),
                        Some("scx_exit_info") => "ei".into(),
                        Some("scx_init_task_args") => "args".into(),
                        Some("scx_cgroup_init_args") => "args".into(),
                        Some("scx_dispatch_q") => "dsq".into(),
                        _ => {
                            // For scalars, infer from position in known callbacks.
                            infer_scalar_param_name(func_name, param_pos)
                        }
                    };
                }
                let all_struct_name = resolve_struct_name(btf_for_params, param.ty);
                let known_struct = all_struct_name
                    .as_ref()
                    .filter(|n| STRUCT_FIELDS.iter().any(|(s, _)| *s == n.as_str()))
                    .cloned();
                let is_ptr = btf_for_params
                    .type_by_id::<btf::BtfType<'_>>(param.ty)
                    .map(|t| t.skip_mods_and_typedefs().kind() == btf::BtfKind::Ptr)
                    .unwrap_or(false);

                // For unknown struct pointers: auto-discover fields
                let (auto_fields, type_name) = if is_ptr && known_struct.is_none() {
                    if let Some(ref sname) = all_struct_name {
                        // Check if this is a vmlinux type (skip auto-discovery
                        // for those — they'd need vmlinux BTF offsets).
                        let is_vmlinux: Option<btf::types::Struct<'_>> =
                            vmlinux.type_by_name(sname);
                        if is_vmlinux.is_some() {
                            (Vec::new(), Some(sname.clone()))
                        } else {
                            let fields = discover_bpf_struct_fields(&prog_btf, param.ty);
                            (fields, Some(sname.clone()))
                        }
                    } else {
                        (Vec::new(), None)
                    }
                } else {
                    (
                        Vec::new(),
                        all_struct_name.filter(|_| known_struct.is_none()),
                    )
                };

                tracing::debug!(
                    func = func_name, param = %name, struct_name = ?known_struct,
                    is_ptr, auto_fields = auto_fields.len(),
                    "bpf_btf: resolved param",
                );
                params.push(BtfParam {
                    name,
                    struct_name: known_struct,
                    is_ptr,
                    auto_fields,
                    type_name,
                    ..Default::default()
                });
            }

            results.push(BtfFunc {
                name: func_name.to_string(),
                params,
                is_variadic: false,
            });
        }
    }

    tracing::debug!(
        n = results.len(),
        "parse_bpf_btf: parsed BPF function signatures"
    );
    results
}

/// Infer scalar param names for sched_ext_ops callbacks.
/// Used when vmlinux FuncProto has empty names.
fn infer_scalar_param_name(func_name: &str, param_pos: usize) -> String {
    // Common sched_ext callback param names.
    const OPS_SCALARS: &[(&str, &[&str])] = &[
        ("dispatch", &["cpu"]),
        ("select_cpu", &["", "prev_cpu", "wake_flags"]),
        ("set_weight", &["", "weight"]),
        ("update_idle", &["cpu", "idle"]),
        ("cpu_acquire", &["cpu"]),
        ("cpu_release", &["cpu"]),
        ("cpu_online", &["cpu"]),
        ("cpu_offline", &["cpu"]),
    ];
    for (op, names) in OPS_SCALARS {
        if func_name.ends_with(op)
            && let Some(name) = names.get(param_pos)
            && !name.is_empty()
        {
            return name.to_string();
        }
    }
    format!("arg{param_pos}")
}

/// Resolve callback signature from vmlinux BTF's `sched_ext_ops` struct.
///
/// Maps scheduler function names (e.g. `ktstr_enqueue`) to ops members
/// (e.g. `enqueue`) by suffix matching, then follows the member's type
/// through Ptr to reach the FuncProto with typed parameters.
pub(super) fn resolve_ops_callback_proto<'a>(
    vmlinux: &'a libbpf_rs::btf::Btf<'a>,
    func_name: &str,
) -> Option<libbpf_rs::btf::types::FuncProto<'a>> {
    use libbpf_rs::btf::{BtfKind, BtfType};

    // Map function name to ops member name by finding the suffix
    // that matches a sched_ext_ops member.
    let ops: libbpf_rs::btf::types::Struct<'_> = vmlinux.type_by_name("sched_ext_ops")?;

    for member in ops.iter() {
        let member_name = member.name.and_then(|n| n.to_str()).unwrap_or("");
        if member_name.is_empty() || !func_name.ends_with(member_name) {
            continue;
        }
        // Follow the member type to find FuncProto (through Ptr if needed).
        let mut t = vmlinux.type_by_id::<BtfType<'_>>(member.ty)?;
        for _ in 0..20 {
            let inner = t.skip_mods_and_typedefs();
            match inner.kind() {
                BtfKind::Ptr => {
                    t = inner.next_type()?;
                }
                BtfKind::FuncProto => {
                    return inner.try_into().ok();
                }
                _ => break,
            }
        }
    }
    None
}

/// Resolve field specs for a BPF function's auto-discovered fields.
///
/// Uses BPF program BTF (not vmlinux) for offset resolution. Handles
/// both single-level field access (`->field`) and chained pointer
/// dereferences (`->ptr->field`). Skips params that have `struct_name`
/// set (those are handled by [`resolve_field_specs`] with vmlinux BTF).
pub fn resolve_bpf_field_specs(btf_func: &BtfFunc, prog_id: u32) -> Vec<FieldSpec> {
    use libbpf_rs::btf;

    let prog_btf = match btf::Btf::from_prog_id(prog_id) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };

    let mut specs = Vec::new();
    let mut field_idx: u32 = 0;

    let max_params = btf_func.params.len().min(6);
    for (param_idx, param) in btf_func.params[..max_params].iter().enumerate() {
        if param.struct_name.is_some() {
            // Known struct — handled by resolve_field_specs with vmlinux BTF.
            if let Some((_, fields)) = STRUCT_FIELDS
                .iter()
                .find(|(s, _)| Some(*s) == param.struct_name.as_deref())
            {
                field_idx += fields.len() as u32;
            }
            continue;
        }
        if !param.auto_fields.is_empty() {
            // Resolve offsets from BPF program BTF.
            let sname = match param.type_name.as_deref() {
                Some(n) => n,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };
            // Find the struct in BPF program BTF.
            let struct_type: Option<btf::types::Struct<'_>> = prog_btf.type_by_name(sname);
            let composite = match struct_type {
                Some(s) => s,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };

            for (_fname, access) in &param.auto_fields {
                let access = access.trim_start_matches("->");
                // Simple single-level field access.
                if !access.contains("->") {
                    let member_name = access.split('[').next().unwrap_or(access);
                    if let Some(offset) = resolve_bpf_member_offset(&composite, member_name) {
                        let size = resolve_bpf_member_size(&prog_btf, &composite, member_name)
                            .unwrap_or(8);
                        specs.push(FieldSpec {
                            param_idx: param_idx as u32,
                            offset,
                            size,
                            field_idx,
                            ptr_offset: 0,
                        });
                    }
                } else if let Some((ptr_member, target)) = access.split_once("->") {
                    // Chained pointer dereference (e.g. cpumask->bits[0]).
                    // Read pointer at ptr_member offset, then read target
                    // field through it.
                    let ptr_member = ptr_member.split('[').next().unwrap_or(ptr_member);
                    if let Some(ptr_off) = resolve_bpf_member_offset(&composite, ptr_member) {
                        // Resolve target struct from the pointer member's type.
                        // Target may be dot-separated for nested embedded structs
                        // (e.g. "cpumask.bits[0]" for bpf_cpumask).
                        let target_stripped = target.split('[').next().unwrap_or(target);
                        let target_parts: Vec<&str> = target_stripped.split('.').collect();
                        let mut target_off = 0u32;
                        let mut target_sz = 8u32;
                        'resolve_target: for member in composite.iter() {
                            let name = member.name.and_then(|n| n.to_str()).unwrap_or("");
                            if name != ptr_member {
                                continue;
                            }
                            // Follow pointer to find the target struct.
                            if let Some(pointed) =
                                prog_btf.type_by_id::<libbpf_rs::btf::BtfType<'_>>(member.ty)
                            {
                                let deref = pointed.skip_mods_and_typedefs();
                                if deref.kind() == libbpf_rs::btf::BtfKind::Ptr
                                    && let Some(inner) = deref.next_type()
                                {
                                    let mut current = inner.skip_mods_and_typedefs();
                                    let mut accumulated_off = 0u32;
                                    // Walk dot-separated path (e.g. cpumask.bits).
                                    for (i, part) in target_parts.iter().enumerate() {
                                        if let Ok(cur_struct) =
                                            TryInto::<libbpf_rs::btf::types::Struct<'_>>::try_into(
                                                current,
                                            )
                                        {
                                            if let Some(off) =
                                                resolve_bpf_member_offset(&cur_struct, part)
                                            {
                                                accumulated_off += off;
                                            }
                                            if i == target_parts.len() - 1 {
                                                // Last part — resolve size.
                                                if let Some(sz) = resolve_bpf_member_size(
                                                    &prog_btf,
                                                    &cur_struct,
                                                    part,
                                                ) {
                                                    target_sz = sz;
                                                }
                                            } else {
                                                // Intermediate part — find member type and descend.
                                                for m in cur_struct.iter() {
                                                    let mn = m
                                                        .name
                                                        .and_then(|n| n.to_str())
                                                        .unwrap_or("");
                                                    if mn == *part {
                                                        if let Some(mt) = prog_btf
                                                            .type_by_id::<libbpf_rs::btf::BtfType<
                                                            '_,
                                                        >>(
                                                            m.ty
                                                        ) {
                                                            current = mt.skip_mods_and_typedefs();
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    target_off = accumulated_off;
                                }
                            }
                            break 'resolve_target;
                        }
                        specs.push(FieldSpec {
                            param_idx: param_idx as u32,
                            offset: target_off,
                            size: target_sz,
                            field_idx,
                            ptr_offset: ptr_off,
                        });
                    }
                }
                field_idx += 1;
                if field_idx >= 16 {
                    break;
                }
            }
        } else if !param.is_ptr {
            field_idx += 1;
        }
    }

    tracing::debug!(
        n = specs.len(),
        func = %btf_func.name,
        "resolve_bpf_field_specs",
    );
    specs
}

/// Resolve byte offset of a member within a BPF program BTF struct.
fn resolve_bpf_member_offset(
    composite: &libbpf_rs::btf::types::Struct<'_>,
    member_name: &str,
) -> Option<u32> {
    use libbpf_rs::btf::types::MemberAttr;
    for member in composite.iter() {
        let name = member.name.and_then(|n| n.to_str()).unwrap_or("");
        if name == member_name {
            let bit_off = match member.attr {
                MemberAttr::Normal { offset } => offset,
                MemberAttr::BitField { offset, .. } => offset,
            };
            if bit_off % 8 != 0 {
                return None; // bitfield
            }
            return Some(bit_off / 8);
        }
    }
    None
}

/// Resolve byte size of a member within a BPF program BTF struct.
fn resolve_bpf_member_size(
    btf: &libbpf_rs::btf::Btf<'_>,
    composite: &libbpf_rs::btf::types::Struct<'_>,
    member_name: &str,
) -> Option<u32> {
    use libbpf_rs::btf::{BtfKind, BtfType};

    for member in composite.iter() {
        let name = member.name.and_then(|n| n.to_str()).unwrap_or("");
        if name != member_name {
            continue;
        }
        let t = btf.type_by_id::<BtfType<'_>>(member.ty)?;
        let inner = t.skip_mods_and_typedefs();
        return match inner.kind() {
            BtfKind::Int => {
                let int_ty: Result<libbpf_rs::btf::types::Int<'_>, _> = inner.try_into();
                Some(int_ty.map(|i| (i.bits / 8) as u32).unwrap_or(8))
            }
            BtfKind::Enum => Some(4),
            BtfKind::Enum64 => Some(8),
            BtfKind::Ptr => Some(8),
            _ => Some(8),
        };
    }
    None
}

/// Auto-discover struct fields from BPF program BTF for types not in
/// STRUCT_FIELDS. Walks members one level deep, emitting access patterns
/// for scalar, enum, and cpumask pointer fields.
fn discover_bpf_struct_fields(
    btf: &libbpf_rs::btf::Btf<'_>,
    type_id: libbpf_rs::btf::TypeId,
) -> Vec<(String, String)> {
    use libbpf_rs::btf::{BtfKind, BtfType};

    let t = match btf.type_by_id::<BtfType<'_>>(type_id) {
        Some(t) => t.skip_mods_and_typedefs(),
        None => return Vec::new(),
    };
    let inner = if t.kind() == BtfKind::Ptr {
        match t.next_type() {
            Some(t) => t.skip_mods_and_typedefs(),
            None => return Vec::new(),
        }
    } else {
        t
    };

    if inner.kind() != BtfKind::Struct && inner.kind() != BtfKind::Union {
        return Vec::new();
    }

    let composite: libbpf_rs::btf::types::Struct<'_> = match inner.try_into() {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let mut fields = Vec::new();
    for member in composite.iter() {
        let fname = match member.name.and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };

        let member_type = match btf.type_by_id::<BtfType<'_>>(member.ty) {
            Some(t) => t.skip_mods_and_typedefs(),
            None => continue,
        };

        match member_type.kind() {
            BtfKind::Int | BtfKind::Enum | BtfKind::Enum64 => {
                fields.push((fname.clone(), format!("->{fname}")));
            }
            BtfKind::Ptr => {
                let deref = member_type.next_type().map(|t| t.skip_mods_and_typedefs());
                let pointed_name = deref
                    .as_ref()
                    .and_then(|t| t.name())
                    .and_then(|n| n.to_str());
                match pointed_name {
                    Some("cpumask") => {
                        fields.push((fname.clone(), format!("->{fname}->bits[0]")));
                    }
                    Some("bpf_cpumask") => {
                        fields.push((fname.clone(), format!("->{fname}->cpumask.bits[0]")));
                    }
                    _ => {} // skip unknown pointers
                }
            }
            _ => {}
        }
    }
    fields
}

/// Auto-discover struct fields from vmlinux BTF for types not in
/// STRUCT_FIELDS. Walks members one level deep via btf_rs, emitting
/// access patterns for scalar, enum, and cpumask pointer fields.
fn discover_vmlinux_struct_fields(btf: &btf_rs::Btf, type_id: u32) -> Vec<(String, String)> {
    use btf_rs::{BtfType, Type};

    let t = match btf.resolve_type_by_id(type_id) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    // Follow PTR/CONST/VOLATILE/TYPEDEF to find the struct.
    let mut current = t;
    let struct_type = loop {
        match current {
            Type::Ptr(_)
            | Type::Const(_)
            | Type::Volatile(_)
            | Type::Typedef(_)
            | Type::Restrict(_)
            | Type::TypeTag(_) => {
                current = match btf.resolve_chained_type(current.as_btf_type().unwrap()) {
                    Ok(next) => next,
                    Err(_) => return Vec::new(),
                };
            }
            Type::Struct(s) | Type::Union(s) => break s,
            _ => return Vec::new(),
        }
    };

    let mut fields = Vec::new();
    for member in &struct_type.members {
        let fname = match btf.resolve_name(member) {
            Ok(n) if !n.is_empty() => n,
            _ => continue,
        };

        let member_tid = match member.get_type_id() {
            Ok(tid) => tid,
            Err(_) => continue,
        };

        // Follow qualifiers to the underlying type.
        let mut member_type = match btf.resolve_type_by_id(member_tid) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for _ in 0..20 {
            match &member_type {
                Type::Const(_)
                | Type::Volatile(_)
                | Type::Restrict(_)
                | Type::Typedef(_)
                | Type::TypeTag(_) => {
                    member_type = match btf.resolve_chained_type(member_type.as_btf_type().unwrap())
                    {
                        Ok(next) => next,
                        Err(_) => break,
                    };
                }
                _ => break,
            }
        }

        match &member_type {
            Type::Int(_) | Type::Enum(_) | Type::Enum64(_) => {
                fields.push((fname.clone(), format!("->{fname}")));
            }
            Type::Ptr(_) => {
                // Follow pointer to check if it points to cpumask.
                let pointed = btf.resolve_chained_type(member_type.as_btf_type().unwrap());
                if let Ok(pointed_type) = pointed {
                    // Chase qualifiers on the pointed-to type.
                    let mut inner = pointed_type;
                    for _ in 0..20 {
                        match &inner {
                            Type::Const(_)
                            | Type::Volatile(_)
                            | Type::Restrict(_)
                            | Type::Typedef(_)
                            | Type::TypeTag(_) => {
                                inner = match btf.resolve_chained_type(inner.as_btf_type().unwrap())
                                {
                                    Ok(next) => next,
                                    Err(_) => break,
                                };
                            }
                            _ => break,
                        }
                    }
                    let pointed_name = match &inner {
                        Type::Struct(s) | Type::Union(s) => btf.resolve_name(s).ok(),
                        _ => None,
                    };
                    match pointed_name.as_deref() {
                        Some("cpumask") | Some("cpumask_t") => {
                            fields.push((fname.clone(), format!("->{fname}->bits[0]")));
                        }
                        _ => {} // skip unknown pointers
                    }
                }
            }
            _ => {}
        }
    }
    fields
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
        assert!(fields.iter().any(|(_, k)| *k == "cpumask_0"));
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
        let _ = discover_bpf_symbols(&[]);
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
            ..Default::default()
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
            is_variadic: false,
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
                    ..Default::default()
                },
                BtfParam {
                    name: "b".into(),
                    struct_name: Some("rq".into()),
                    is_ptr: true,
                    ..Default::default()
                },
                BtfParam {
                    name: "c".into(),
                    struct_name: None,
                    is_ptr: false,
                    ..Default::default()
                },
            ],
            is_variadic: false,
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
    fn struct_fields_has_cpumask_words() {
        let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
        let (_, fields) = entry.unwrap();
        assert!(fields.iter().any(|(_, k)| *k == "cpumask_0"));
        assert!(fields.iter().any(|(_, k)| *k == "cpumask_1"));
        assert!(fields.iter().any(|(_, k)| *k == "cpumask_2"));
        assert!(fields.iter().any(|(_, k)| *k == "cpumask_3"));
    }

    #[test]
    fn struct_fields_cpumask_access_patterns() {
        let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
        let (_, fields) = entry.unwrap();
        assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[0]"));
        assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[1]"));
        assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[2]"));
        assert!(fields.iter().any(|(a, _)| *a == "->cpus_ptr->bits[3]"));
    }

    #[test]
    fn struct_fields_task_struct_field_count() {
        let entry = STRUCT_FIELDS.iter().find(|(s, _)| *s == "task_struct");
        let (_, fields) = entry.unwrap();
        // pid + 4 cpumask words + dsq_id + enq_flags + slice + vtime +
        // weight + sticky_cpu + scx_flags = 12
        assert_eq!(fields.len(), 12);
    }

    #[test]
    fn struct_fields_auto_discover_cpumask_pattern() {
        let ts_fields = STRUCT_FIELDS
            .iter()
            .find(|(s, _)| *s == "task_struct")
            .unwrap()
            .1;
        let cpumask_accesses: Vec<&&str> = ts_fields
            .iter()
            .filter(|(a, _)| a.contains("bits["))
            .map(|(a, _)| a)
            .collect();
        assert_eq!(cpumask_accesses.len(), 4);
        for (i, a) in cpumask_accesses.iter().enumerate() {
            assert!(
                a.contains(&format!("bits[{i}]")),
                "expected bits[{i}] in {a}",
            );
        }
    }

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
