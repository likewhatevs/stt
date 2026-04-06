use super::scx_defs::*;

pub(crate) fn decode_dsq_id(id: u64) -> String {
    if id == 0 {
        return "0".into();
    }
    match id >> DSQ_TYPE_SHIFT {
        DSQ_TYPE_LOCAL_ON => format!("LOCAL_ON|{}", id & 0xffffffff),
        DSQ_TYPE_BUILTIN => match (id & 0xffffffff) as u32 {
            DSQ_INVALID => "SCX_DSQ_INVALID".into(),
            DSQ_GLOBAL => "GLOBAL".into(),
            DSQ_LOCAL => "LOCAL".into(),
            DSQ_BYPASS => "BYPASS".into(),
            v => format!("BUILTIN({v})"),
        },
        _ => format!("DSQ(0x{id:x})"),
    }
}

/// Decode a single 64-bit cpumask word into a CPU range string.
/// Handles CPUs 0-63 (one u64 word). Multi-word cpumasks for >64 CPUs
/// require the caller to decode each word separately.
pub(crate) fn decode_cpumask(bits: u64) -> String {
    if bits == 0 {
        return "none".into();
    }
    let mut cpus = Vec::new();
    for i in 0..64u32 {
        if bits & (1u64 << i) != 0 {
            cpus.push(i);
        }
    }
    let mut ranges = Vec::new();
    let (mut s, mut e) = (cpus[0], cpus[0]);
    for &c in &cpus[1..] {
        if c == e + 1 {
            e = c;
        } else {
            ranges.push(if s == e {
                format!("{s}")
            } else {
                format!("{s}-{e}")
            });
            s = c;
            e = c;
        }
    }
    ranges.push(if s == e {
        format!("{s}")
    } else {
        format!("{s}-{e}")
    });
    ranges.join(",")
}

pub(crate) fn decode_enq_flags(flags: u64) -> String {
    decode_bitflags(flags, ENQ_FLAG_NAMES)
}

pub(crate) fn decode_exit_kind(kind: u64) -> String {
    for &(val, name) in EXIT_KIND_NAMES {
        if kind == val {
            return name.into();
        }
    }
    format!("UNKNOWN({kind})")
}

/// Decode a bitmask using a table of (bit_value, name) pairs.
fn decode_bitflags(flags: u64, table: &[(u64, &str)]) -> String {
    let mut parts = Vec::new();
    for &(bit, name) in table {
        if flags & bit != 0 {
            parts.push(name);
        }
    }
    if parts.is_empty() {
        "NONE".into()
    } else {
        parts.join("|")
    }
}

/// Infer type and format a raw u64 value for unknown function args.
/// Uses type:value format (colon, NOT equals) to distinguish from named args.
pub(crate) fn format_raw_arg(val: u64) -> String {
    if val == 0 {
        "int:0".into()
    } else if val == 0xffffffffffffffff || val == 0xffffffff {
        "int:-1".into()
    } else if val == 1 {
        "bool:true".into()
    } else if (2..=0xff).contains(&val) {
        format!("int:{val}")
    } else if val > 0xff00000000000000 {
        format!("ptr:{:04x}", val & 0xffff)
    } else if val <= 0xffff && val.count_ones() >= 2 && val.count_ones() <= 16 {
        format!("mask:0x{val:x}({})", decode_cpumask(val))
    } else if val <= 0xffff {
        format!("int:{val}")
    } else if val >> DSQ_TYPE_SHIFT >= DSQ_TYPE_BUILTIN {
        format!("dsq:{}", decode_dsq_id(val))
    } else {
        format!("hex:0x{val:x}")
    }
}

/// Decode a named key=value pair from a known function.
pub(crate) fn decode_named_value(key: &str, val: &str) -> String {
    let as_u64 = || -> u64 {
        if let Some(hex) = val.strip_prefix("0x") {
            u64::from_str_radix(hex, 16).unwrap_or(0)
        } else {
            val.parse().unwrap_or(0)
        }
    };

    match key {
        "dsq_id" | "dsq" => decode_dsq_id(as_u64()),
        "cpus_ptr" | "cpus" | "cpumask" => {
            let v = as_u64();
            format!("0x{v:x}({cpus})", cpus = decode_cpumask(v))
        }
        "enforce" => {
            if val == "1" || val == "true" {
                "true".into()
            } else {
                "false".into()
            }
        }
        "enq_flags" | "enq" | "enqflags" => decode_enq_flags(as_u64()),
        "exit_kind" => decode_exit_kind(as_u64()),
        "sticky_cpu" | "sticky" => {
            let v = as_u64();
            if v == 0xffffffff || v == 0xffffffffffffffff {
                "-1".into()
            } else {
                format!("{v}")
            }
        }
        "cpu" | "rq_cpu" | "dst_cpu" | "dest_cpu" => val.to_string(),
        "pid" => val.to_string(),
        "task" => val.to_string(),
        "slice" | "vtime" => {
            let v = as_u64();
            format!("{v}")
        }
        "weight" => val.to_string(),
        "flags" | "scx_flags" => {
            let v = as_u64();
            let mut parts = Vec::new();
            if v & TASK_QUEUED != 0 {
                parts.push("QUEUED");
            }
            if v & TASK_RESET_RUNNABLE_AT != 0 {
                parts.push("RESET_RUNNABLE_AT");
            }
            if v & TASK_DEQD_FOR_SLEEP != 0 {
                parts.push("DEQD_FOR_SLEEP");
            }
            let state = (v >> TASK_STATE_SHIFT) & TASK_STATE_MASK;
            match state {
                TASK_STATE_INIT => parts.push("INIT"),
                TASK_STATE_READY => parts.push("READY"),
                TASK_STATE_ENABLED => parts.push("ENABLED"),
                _ => {}
            }
            if parts.is_empty() {
                "NONE".into()
            } else {
                parts.join("|")
            }
        }
        _ => val.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- decode_dsq_id --

    #[test]
    fn decode_dsq_id_zero() {
        assert_eq!(decode_dsq_id(0), "0");
    }

    #[test]
    fn decode_dsq_id_global() {
        assert_eq!(decode_dsq_id((1u64 << 63) | 1), "GLOBAL");
    }

    #[test]
    fn decode_dsq_id_local() {
        assert_eq!(decode_dsq_id((1u64 << 63) | 2), "LOCAL");
    }

    #[test]
    fn decode_dsq_id_bypass() {
        assert_eq!(decode_dsq_id((1u64 << 63) | 3), "BYPASS");
    }

    #[test]
    fn decode_dsq_id_invalid() {
        assert_eq!(decode_dsq_id(1u64 << 63), "SCX_DSQ_INVALID");
    }

    #[test]
    fn decode_dsq_id_local_on() {
        let id = (1u64 << 63) | (1u64 << 62) | 7;
        assert_eq!(decode_dsq_id(id), "LOCAL_ON|7");
    }

    #[test]
    fn decode_dsq_id_user_dsq() {
        assert_eq!(decode_dsq_id(42), "DSQ(0x2a)");
    }

    // -- decode_cpumask --

    #[test]
    fn decode_cpumask_none() {
        assert_eq!(decode_cpumask(0), "none");
    }

    #[test]
    fn decode_cpumask_single() {
        assert_eq!(decode_cpumask(1), "0");
    }

    #[test]
    fn decode_cpumask_contiguous() {
        assert_eq!(decode_cpumask(0xf), "0-3");
    }

    #[test]
    fn decode_cpumask_gaps() {
        assert_eq!(decode_cpumask(0x33), "0-1,4-5");
    }

    #[test]
    fn decode_cpumask_scattered() {
        assert_eq!(decode_cpumask(0x15), "0,2,4");
    }

    // -- decode_enq_flags --

    #[test]
    fn decode_enq_flags_none() {
        assert_eq!(decode_enq_flags(0), "NONE");
    }

    #[test]
    fn decode_enq_flags_wakeup() {
        assert_eq!(decode_enq_flags(ENQ_WAKEUP), "WAKEUP");
    }

    #[test]
    fn decode_enq_flags_multi() {
        assert_eq!(decode_enq_flags(ENQ_WAKEUP | ENQ_HEAD), "WAKEUP|HEAD");
    }

    #[test]
    fn decode_enq_flags_preempt() {
        assert_eq!(decode_enq_flags(ENQ_PREEMPT), "PREEMPT");
    }

    #[test]
    fn decode_enq_flags_reenq() {
        assert_eq!(decode_enq_flags(ENQ_REENQ), "REENQ");
    }

    // -- decode_exit_kind --

    #[test]
    fn decode_exit_kind_all() {
        assert_eq!(decode_exit_kind(EXIT_NONE), "NONE");
        assert_eq!(decode_exit_kind(EXIT_DONE), "DONE");
        assert_eq!(decode_exit_kind(EXIT_UNREG), "UNREG");
        assert_eq!(decode_exit_kind(EXIT_UNREG_BPF), "UNREG_BPF");
        assert_eq!(decode_exit_kind(EXIT_UNREG_KERN), "UNREG_KERN");
        assert_eq!(decode_exit_kind(EXIT_SYSRQ), "SYSRQ");
        assert_eq!(decode_exit_kind(EXIT_ERROR), "ERROR");
        assert_eq!(decode_exit_kind(EXIT_ERROR_BPF), "ERROR_BPF");
        assert_eq!(decode_exit_kind(EXIT_ERROR_STALL), "ERROR_STALL");
        assert_eq!(decode_exit_kind(9999), "UNKNOWN(9999)");
    }

    // -- format_raw_arg --

    #[test]
    fn format_raw_arg_zero() {
        assert_eq!(format_raw_arg(0), "int:0");
    }

    #[test]
    fn format_raw_arg_one() {
        assert_eq!(format_raw_arg(1), "bool:true");
    }

    #[test]
    fn format_raw_arg_minus_one() {
        assert_eq!(format_raw_arg(0xffffffffffffffff), "int:-1");
    }

    #[test]
    fn format_raw_arg_minus_one_32() {
        assert_eq!(format_raw_arg(0xffffffff), "int:-1");
    }

    #[test]
    fn format_raw_arg_small_int() {
        assert_eq!(format_raw_arg(42), "int:42");
    }

    #[test]
    fn format_raw_arg_cpumask() {
        let v = 0x303u64;
        let out = format_raw_arg(v);
        assert!(out.starts_with("mask:"), "got: {out}");
        assert!(out.contains("0-1,8-9"));
    }

    #[test]
    fn format_raw_arg_kernel_ptr() {
        let out = format_raw_arg(0xffff888100123456);
        assert!(out.starts_with("ptr:"), "got: {out}");
    }

    #[test]
    fn format_raw_arg_dsq_id() {
        let v = (1u64 << 63) | 1;
        let out = format_raw_arg(v);
        assert!(out.contains("GLOBAL"), "got: {out}");
    }

    // -- decode_named_value --

    #[test]
    fn decode_named_value_dsq_id() {
        let v = (1u64 << 63) | 2;
        assert_eq!(decode_named_value("dsq_id", &v.to_string()), "LOCAL");
    }

    #[test]
    fn decode_named_value_cpus_ptr() {
        let out = decode_named_value("cpus_ptr", "15");
        assert!(out.contains("0-3"), "got: {out}");
    }

    #[test]
    fn decode_named_value_enq_flags() {
        assert_eq!(decode_named_value("enq_flags", "1"), "WAKEUP");
    }

    #[test]
    fn decode_named_value_exit_kind() {
        assert_eq!(decode_named_value("exit_kind", "1024"), "ERROR");
    }

    #[test]
    fn decode_named_value_sticky_minus_one() {
        assert_eq!(
            decode_named_value("sticky_cpu", &0xffffffffu64.to_string()),
            "-1"
        );
    }

    #[test]
    fn decode_named_value_pid_passthrough() {
        assert_eq!(decode_named_value("pid", "1234"), "1234");
    }

    #[test]
    fn decode_named_value_unknown_key() {
        assert_eq!(decode_named_value("foobar", "hello"), "hello");
    }

    #[test]
    fn decode_named_value_enforce_true() {
        assert_eq!(decode_named_value("enforce", "1"), "true");
    }

    #[test]
    fn decode_named_value_scx_flags() {
        let v = TASK_QUEUED | (TASK_STATE_ENABLED << TASK_STATE_SHIFT);
        assert_eq!(
            decode_named_value("scx_flags", &v.to_string()),
            "QUEUED|ENABLED"
        );
    }

    // -- scx_defs invariants --

    #[test]
    fn enq_flag_names_no_overlap() {
        for (i, &(a_val, a_name)) in ENQ_FLAG_NAMES.iter().enumerate() {
            for &(b_val, b_name) in &ENQ_FLAG_NAMES[i + 1..] {
                assert_eq!(
                    a_val & b_val,
                    0,
                    "flag overlap: {a_name} (0x{a_val:x}) & {b_name} (0x{b_val:x})",
                );
            }
        }
    }

    #[test]
    fn exit_kind_names_no_duplicate_values() {
        for (i, &(a_val, a_name)) in EXIT_KIND_NAMES.iter().enumerate() {
            for &(b_val, b_name) in &EXIT_KIND_NAMES[i + 1..] {
                assert_ne!(
                    a_val, b_val,
                    "duplicate exit kind value: {a_name} and {b_name} both = {a_val}",
                );
            }
        }
    }

    #[test]
    fn dsq_type_shift_and_builtin_values() {
        assert_eq!(DSQ_TYPE_SHIFT, 62);
        // DSQ_TYPE_BUILTIN is 2, so bits [63:62] = 10.
        assert_eq!(DSQ_TYPE_BUILTIN << DSQ_TYPE_SHIFT, 1u64 << 63);
        // DSQ_TYPE_LOCAL_ON is 3, so bits [63:62] = 11.
        assert_eq!(
            DSQ_TYPE_LOCAL_ON << DSQ_TYPE_SHIFT,
            (1u64 << 63) | (1u64 << 62)
        );
        // Builtin DSQ lower values are sequential from 0..=3.
        assert_eq!(DSQ_INVALID, 0);
        assert_eq!(DSQ_GLOBAL, 1);
        assert_eq!(DSQ_LOCAL, 2);
        assert_eq!(DSQ_BYPASS, 3);
    }

    #[test]
    fn task_state_constants_sequential() {
        assert_eq!(TASK_STATE_INIT, 1);
        assert_eq!(TASK_STATE_READY, 2);
        assert_eq!(TASK_STATE_ENABLED, 3);
        // Mask covers all three states.
        assert_eq!(TASK_STATE_MASK, 3);
    }

    // -- additional decode tests --

    #[test]
    fn decode_enq_flags_all_set() {
        let all = ENQ_WAKEUP
            | ENQ_HEAD
            | ENQ_PREEMPT
            | ENQ_REENQ
            | ENQ_LAST
            | ENQ_CLEAR_OPSS
            | ENQ_DSQ_PRIQ
            | ENQ_NESTED;
        let out = decode_enq_flags(all);
        for &(_, name) in ENQ_FLAG_NAMES {
            assert!(out.contains(name), "missing flag {name} in '{out}'");
        }
    }

    #[test]
    fn decode_exit_kind_between_known() {
        // Value 2 is between EXIT_DONE(1) and EXIT_UNREG(64).
        assert_eq!(decode_exit_kind(2), "UNKNOWN(2)");
        // Value 100 is between EXIT_SYSRQ(67) and EXIT_ERROR(1024).
        assert_eq!(decode_exit_kind(100), "UNKNOWN(100)");
    }

    #[test]
    fn format_raw_arg_boundary_0x100() {
        // 0x100 = 256: outside 2..=0xff range, inside <=0xffff range.
        // count_ones(256) = 1, which is <2, so it's not a mask. It's int.
        let out = format_raw_arg(0x100);
        assert_eq!(out, "int:256");
    }

    #[test]
    fn format_raw_arg_boundary_0x10000() {
        // 0x10000: outside <=0xffff range, below DSQ_TYPE_BUILTIN shifted.
        let out = format_raw_arg(0x10000);
        assert_eq!(out, "hex:0x10000");
    }

    #[test]
    fn format_raw_arg_boundary_0xff() {
        // 0xff = 255: inside 2..=0xff range.
        assert_eq!(format_raw_arg(0xff), "int:255");
    }

    #[test]
    fn decode_cpumask_bit_63() {
        // Only bit 63 set: should produce "63".
        assert_eq!(decode_cpumask(1u64 << 63), "63");
    }

    #[test]
    fn decode_cpumask_all_bits() {
        // All 64 bits set: should produce "0-63".
        assert_eq!(decode_cpumask(u64::MAX), "0-63");
    }

    #[test]
    fn decode_dsq_id_builtin_unknown() {
        // Builtin type (bits [63:62] = 10) with lower 32 bits = 99.
        // Not INVALID(0), GLOBAL(1), LOCAL(2), or BYPASS(3) -> BUILTIN(99).
        let id = (DSQ_TYPE_BUILTIN << DSQ_TYPE_SHIFT) | 99;
        assert_eq!(decode_dsq_id(id), "BUILTIN(99)");
    }

    #[test]
    fn decode_named_value_enforce_true_literal() {
        assert_eq!(decode_named_value("enforce", "true"), "true");
    }

    #[test]
    fn decode_named_value_enforce_zero() {
        assert_eq!(decode_named_value("enforce", "0"), "false");
    }

    #[test]
    fn decode_named_value_enforce_other() {
        assert_eq!(decode_named_value("enforce", "2"), "false");
    }

    #[test]
    fn decode_named_value_dsq_id_hex_prefix() {
        // hex-prefixed dsq_id value
        let hex_val = format!("0x{:x}", (DSQ_TYPE_BUILTIN << DSQ_TYPE_SHIFT) | 1);
        assert_eq!(decode_named_value("dsq_id", &hex_val), "GLOBAL");
    }

    #[test]
    fn decode_named_value_slice_key() {
        assert_eq!(decode_named_value("slice", "5000000"), "5000000");
    }

    #[test]
    fn decode_named_value_vtime_key() {
        assert_eq!(decode_named_value("vtime", "123456789"), "123456789");
    }

    #[test]
    fn decode_named_value_slice_hex_prefix() {
        // Hex-prefixed slice value
        assert_eq!(decode_named_value("slice", "0x4c4b40"), "5000000");
    }

    #[test]
    fn decode_named_value_sticky_cpu_64bit_minus_one() {
        // 64-bit -1 (0xffffffffffffffff) should decode to "-1"
        assert_eq!(
            decode_named_value("sticky_cpu", &0xffffffffffffffffu64.to_string()),
            "-1"
        );
    }

    #[test]
    fn decode_named_value_sticky_cpu_normal() {
        assert_eq!(decode_named_value("sticky_cpu", "7"), "7");
    }

    #[test]
    fn format_raw_arg_two() {
        // val=2 is in 2..=0xff range, should be "int:2"
        assert_eq!(format_raw_arg(2), "int:2");
    }
}
