use super::scx_defs::*;

/// Decode a sched_ext dispatch queue ID into a human-readable name.
///
/// Inspects bits \[63:62\] (`DSQ_TYPE_SHIFT`) to classify:
/// - `DSQ_TYPE_LOCAL_ON` (0b11): `SCX_DSQ_LOCAL_ON|{cpu}`
/// - `DSQ_TYPE_BUILTIN` (0b10): `SCX_DSQ_INVALID`, `SCX_DSQ_GLOBAL`, `SCX_DSQ_LOCAL`, `SCX_DSQ_BYPASS`
/// - Otherwise: `DSQ(0x{id:x})`
pub(crate) fn decode_dsq_id(id: u64) -> String {
    match id >> DSQ_TYPE_SHIFT {
        DSQ_TYPE_LOCAL_ON => format!("SCX_DSQ_LOCAL_ON|{}", id & 0xffffffff),
        DSQ_TYPE_BUILTIN => match (id & 0xffffffff) as u32 {
            DSQ_INVALID => "SCX_DSQ_INVALID".into(),
            DSQ_GLOBAL => "SCX_DSQ_GLOBAL".into(),
            DSQ_LOCAL => "SCX_DSQ_LOCAL".into(),
            DSQ_BYPASS => "SCX_DSQ_BYPASS".into(),
            v => format!("BUILTIN({v})"),
        },
        _ => format!("DSQ(0x{id:x})"),
    }
}

/// Decode a single 64-bit cpumask word into a CPU range string.
/// Handles CPUs 0-63 (one u64 word).
pub(crate) fn decode_cpumask(bits: u64) -> String {
    decode_cpumask_multi(&[bits], None)
}

/// Decode multiple 64-bit cpumask words into a CPU range string.
/// Word 0 covers CPUs 0-63, word 1 covers CPUs 64-127, etc.
///
/// `nr_cpus` bounds valid bits: only CPUs 0..nr_cpus are considered.
/// Bits beyond nr_cpus are ignored (they may be uninitialized kernel
/// memory when the cpumask was read via BPF). Pass `None` to decode
/// all bits.
pub(crate) fn decode_cpumask_multi(words: &[u64], nr_cpus: Option<u32>) -> String {
    let max_cpu = nr_cpus.unwrap_or(words.len() as u32 * 64);
    let mut cpus = Vec::new();
    for (word_idx, &bits) in words.iter().enumerate() {
        let base = word_idx as u32 * 64;
        if base >= max_cpu {
            break;
        }
        let top = (max_cpu - base).min(64);
        for i in 0..top {
            if bits & (1u64 << i) != 0 {
                cpus.push(base + i);
            }
        }
    }
    if cpus.is_empty() {
        return "none".into();
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

/// Decode a named field value based on the struct type and field name.
///
/// `struct_name` is the originating struct (e.g. `"task_struct"`,
/// `"rq"`). SCX-specific decoders only fire for their owning struct
/// to avoid misinterpreting identically-named fields on unrelated types.
/// Pass `""` for scalar params or unknown context.
///
/// Dispatches to specialized decoders: `dsq_id` -> [`decode_dsq_id`],
/// `cpus_ptr`/`cpumask*` -> [`decode_cpumask`], `enq_flags` ->
/// [`decode_enq_flags`], `exit_kind` -> [`decode_exit_kind`],
/// `scx_flags` -> task state/queue flags, etc. Unknown keys pass
/// the value through unchanged.
pub(crate) fn decode_named_value(struct_name: &str, key: &str, val: &str) -> String {
    let as_u64 = || -> u64 {
        if let Some(hex) = val.strip_prefix("0x") {
            u64::from_str_radix(hex, 16).unwrap_or(0)
        } else {
            val.parse().unwrap_or(0)
        }
    };

    match key {
        "dsq_id" | "dsq" => decode_dsq_id(as_u64()),
        "cpus_ptr" | "cpus" | "cpumask" | "cpumask_0" | "cpumask_1" | "cpumask_2" | "cpumask_3" => {
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
        "enq_flags" | "enq" | "enqflags"
            if struct_name.is_empty() || struct_name == "task_struct" =>
        {
            decode_enq_flags(as_u64())
        }
        "exit_kind" if struct_name.is_empty() || struct_name == "scx_exit_info" => {
            decode_exit_kind(as_u64())
        }
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
        "kick_flags" | "kick" => decode_kick_flags(as_u64()),
        "ops_state" | "opss" => decode_ops_state(as_u64()),
        "flags" | "scx_flags" if struct_name.is_empty() || struct_name == "task_struct" => {
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
        _ if key.contains("cpumask") || key.contains("cpus") => {
            let v = as_u64();
            format!("0x{v:x}({cpus})", cpus = decode_cpumask(v))
        }
        _ => val.to_string(),
    }
}

pub(crate) fn decode_kick_flags(flags: u64) -> String {
    decode_bitflags(flags, KICK_FLAG_NAMES)
}

pub(crate) fn decode_ops_state(state: u64) -> String {
    use super::scx_defs::*;
    match state & 0xff {
        OPS_NONE => "NONE".into(),
        OPS_QUEUEING => "QUEUEING".into(),
        OPS_QUEUED => "QUEUED".into(),
        OPS_DISPATCHING => "DISPATCHING".into(),
        v => format!("OPSS({v})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- decode_dsq_id --

    #[test]
    fn decode_dsq_id_zero() {
        assert_eq!(decode_dsq_id(0), "DSQ(0x0)");
    }

    #[test]
    fn decode_dsq_id_global() {
        assert_eq!(decode_dsq_id((1u64 << 63) | 1), "SCX_DSQ_GLOBAL");
    }

    #[test]
    fn decode_dsq_id_local() {
        assert_eq!(decode_dsq_id((1u64 << 63) | 2), "SCX_DSQ_LOCAL");
    }

    #[test]
    fn decode_dsq_id_bypass() {
        assert_eq!(decode_dsq_id((1u64 << 63) | 3), "SCX_DSQ_BYPASS");
    }

    #[test]
    fn decode_dsq_id_invalid() {
        assert_eq!(decode_dsq_id(1u64 << 63), "SCX_DSQ_INVALID");
    }

    #[test]
    fn decode_dsq_id_local_on() {
        let id = (1u64 << 63) | (1u64 << 62) | 7;
        assert_eq!(decode_dsq_id(id), "SCX_DSQ_LOCAL_ON|7");
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
        assert!(out.contains("SCX_DSQ_GLOBAL"), "got: {out}");
    }

    // -- decode_named_value --

    #[test]
    fn decode_named_value_dsq_id() {
        let v = (1u64 << 63) | 2;
        assert_eq!(
            decode_named_value("", "dsq_id", &v.to_string()),
            "SCX_DSQ_LOCAL"
        );
    }

    #[test]
    fn decode_named_value_cpus_ptr() {
        let out = decode_named_value("", "cpus_ptr", "15");
        assert!(out.contains("0-3"), "got: {out}");
    }

    #[test]
    fn decode_named_value_enq_flags() {
        assert_eq!(
            decode_named_value("task_struct", "enq_flags", "1"),
            "WAKEUP"
        );
    }

    #[test]
    fn decode_named_value_exit_kind() {
        assert_eq!(
            decode_named_value("scx_exit_info", "exit_kind", "1024"),
            "ERROR"
        );
    }

    #[test]
    fn decode_named_value_sticky_minus_one() {
        assert_eq!(
            decode_named_value("", "sticky_cpu", &0xffffffffu64.to_string()),
            "-1"
        );
    }

    #[test]
    fn decode_named_value_pid_passthrough() {
        assert_eq!(decode_named_value("", "pid", "1234"), "1234");
    }

    #[test]
    fn decode_named_value_unknown_key() {
        assert_eq!(decode_named_value("", "foobar", "hello"), "hello");
    }

    #[test]
    fn decode_named_value_enforce_true() {
        assert_eq!(decode_named_value("", "enforce", "1"), "true");
    }

    #[test]
    fn decode_named_value_scx_flags() {
        let v = TASK_QUEUED | (TASK_STATE_ENABLED << TASK_STATE_SHIFT);
        assert_eq!(
            decode_named_value("task_struct", "scx_flags", &v.to_string()),
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
        assert_eq!(decode_named_value("", "enforce", "true"), "true");
    }

    #[test]
    fn decode_named_value_enforce_zero() {
        assert_eq!(decode_named_value("", "enforce", "0"), "false");
    }

    #[test]
    fn decode_named_value_enforce_other() {
        assert_eq!(decode_named_value("", "enforce", "2"), "false");
    }

    #[test]
    fn decode_named_value_dsq_id_hex_prefix() {
        // hex-prefixed dsq_id value
        let hex_val = format!("0x{:x}", (DSQ_TYPE_BUILTIN << DSQ_TYPE_SHIFT) | 1);
        assert_eq!(decode_named_value("", "dsq_id", &hex_val), "SCX_DSQ_GLOBAL");
    }

    #[test]
    fn decode_named_value_slice_key() {
        assert_eq!(decode_named_value("", "slice", "5000000"), "5000000");
    }

    #[test]
    fn decode_named_value_vtime_key() {
        assert_eq!(decode_named_value("", "vtime", "123456789"), "123456789");
    }

    #[test]
    fn decode_named_value_slice_hex_prefix() {
        // Hex-prefixed slice value
        assert_eq!(decode_named_value("", "slice", "0x4c4b40"), "5000000");
    }

    #[test]
    fn decode_named_value_sticky_cpu_64bit_minus_one() {
        // 64-bit -1 (0xffffffffffffffff) should decode to "-1"
        assert_eq!(
            decode_named_value("", "sticky_cpu", &0xffffffffffffffffu64.to_string()),
            "-1"
        );
    }

    #[test]
    fn decode_named_value_sticky_cpu_normal() {
        assert_eq!(decode_named_value("", "sticky_cpu", "7"), "7");
    }

    #[test]
    fn format_raw_arg_two() {
        // val=2 is in 2..=0xff range, should be "int:2"
        assert_eq!(format_raw_arg(2), "int:2");
    }

    // -- decode_cpumask_multi --

    #[test]
    fn decode_cpumask_multi_empty() {
        assert_eq!(decode_cpumask_multi(&[0, 0, 0, 0], None), "none");
    }

    #[test]
    fn decode_cpumask_multi_word0_only() {
        assert_eq!(decode_cpumask_multi(&[0xf, 0, 0, 0], None), "0-3");
    }

    #[test]
    fn decode_cpumask_multi_word1() {
        // CPU 64 is bit 0 of word 1.
        assert_eq!(decode_cpumask_multi(&[0, 1, 0, 0], None), "64");
    }

    #[test]
    fn decode_cpumask_multi_span_words() {
        // CPUs 63 and 64: last bit of word 0, first bit of word 1.
        assert_eq!(decode_cpumask_multi(&[1u64 << 63, 1, 0, 0], None), "63-64");
    }

    #[test]
    fn decode_cpumask_multi_all_four_words() {
        // One CPU per word: 0, 64, 128, 192.
        assert_eq!(decode_cpumask_multi(&[1, 1, 1, 1], None), "0,64,128,192");
    }

    #[test]
    fn decode_cpumask_multi_contiguous_across_boundary() {
        // CPUs 62-65: last 2 bits of word 0, first 2 bits of word 1.
        let w0 = (1u64 << 62) | (1u64 << 63);
        let w1 = 0b11u64;
        assert_eq!(decode_cpumask_multi(&[w0, w1, 0, 0], None), "62-65");
    }

    #[test]
    fn decode_cpumask_multi_single_word_compat() {
        // Single-word call should match decode_cpumask.
        assert_eq!(decode_cpumask_multi(&[0x33], None), decode_cpumask(0x33));
    }

    #[test]
    fn decode_cpumask_multi_word3_high() {
        // CPU 255: highest bit of word 3.
        assert_eq!(decode_cpumask_multi(&[0, 0, 0, 1u64 << 63], None), "255");
    }

    #[test]
    fn decode_cpumask_multi_nr_cpus_truncates() {
        // 8-CPU VM: only bits 0-7 are valid.
        // Word 0 has CPUs 0-7 set, word 1 has garbage.
        let w0 = 0xff;
        let w1 = 0xffffffffffffffff; // garbage
        assert_eq!(decode_cpumask_multi(&[w0, w1, 0, 0], Some(8)), "0-7",);
    }

    #[test]
    fn decode_cpumask_multi_nr_cpus_mid_word() {
        // 10 CPUs: bits 0-9 valid. Word 0 has 0xff (CPUs 0-7),
        // plus bits 8-9 in positions that should be included.
        let w0 = 0x3ff; // bits 0-9
        assert_eq!(decode_cpumask_multi(&[w0], Some(10)), "0-9");
        // With garbage beyond bit 9:
        let w0_garbage = 0xffff_ffff_ffff_ffff;
        assert_eq!(decode_cpumask_multi(&[w0_garbage], Some(10)), "0-9");
    }

    #[test]
    fn decode_cpumask_multi_nr_cpus_word_boundary() {
        // Exactly 64 CPUs: full word 0 is valid, word 1 is garbage.
        let w0 = 0xff;
        let w1 = 0xdeadbeef;
        assert_eq!(decode_cpumask_multi(&[w0, w1], Some(64)), "0-7",);
    }

    #[test]
    fn decode_cpumask_multi_nr_cpus_zero() {
        assert_eq!(decode_cpumask_multi(&[0xff], Some(0)), "none");
    }

    // -- decode_kick_flags --

    #[test]
    fn decode_kick_flags_idle() {
        assert_eq!(decode_kick_flags(1), "IDLE");
    }

    #[test]
    fn decode_kick_flags_preempt() {
        assert_eq!(decode_kick_flags(2), "PREEMPT");
    }

    #[test]
    fn decode_kick_flags_wait() {
        assert_eq!(decode_kick_flags(4), "WAIT");
    }

    #[test]
    fn decode_kick_flags_combo() {
        assert_eq!(decode_kick_flags(1 | 2), "IDLE|PREEMPT");
    }

    #[test]
    fn decode_kick_flags_none() {
        assert_eq!(decode_kick_flags(0), "NONE");
    }

    // -- decode_ops_state --

    #[test]
    fn decode_ops_state_none() {
        assert_eq!(decode_ops_state(0), "NONE");
    }

    #[test]
    fn decode_ops_state_queueing() {
        assert_eq!(decode_ops_state(1), "QUEUEING");
    }

    #[test]
    fn decode_ops_state_queued() {
        assert_eq!(decode_ops_state(2), "QUEUED");
    }

    #[test]
    fn decode_ops_state_dispatching() {
        assert_eq!(decode_ops_state(3), "DISPATCHING");
    }

    #[test]
    fn decode_ops_state_unknown() {
        assert_eq!(decode_ops_state(99), "OPSS(99)");
    }

    // -- decode_named_value with cpumask_N keys --

    #[test]
    fn decode_named_value_cpumask_0() {
        let out = decode_named_value("", "cpumask_0", "15");
        assert!(out.contains("0-3"), "got: {out}");
    }

    #[test]
    fn decode_named_value_cpumask_1() {
        let out = decode_named_value("", "cpumask_1", "1");
        assert!(out.contains("0x1"), "got: {out}");
    }

    #[test]
    fn decode_named_value_cpumask_3() {
        let out = decode_named_value("", "cpumask_3", "0");
        assert!(out.contains("0x0"), "got: {out}");
    }

    // -- decode_named_value wildcard cpumask key --

    #[test]
    fn decode_named_value_key_containing_cpumask() {
        let out = decode_named_value("", "my_cpumask_field", "255");
        assert!(out.contains("0-7"), "wildcard cpumask key: {out}");
    }

    // -- decode_named_value struct scoping --

    #[test]
    fn decode_named_value_flags_wrong_struct_passthrough() {
        // "flags" on a non-task_struct should pass through raw, not decode as scx flags.
        assert_eq!(decode_named_value("rq_flags", "flags", "42"), "42");
    }

    #[test]
    fn decode_named_value_flags_task_struct_decodes() {
        let v = TASK_QUEUED;
        assert_eq!(
            decode_named_value("task_struct", "flags", &v.to_string()),
            "QUEUED"
        );
    }

    #[test]
    fn decode_named_value_enq_flags_wrong_struct_passthrough() {
        // "enq_flags" on a non-task_struct should pass through raw.
        assert_eq!(decode_named_value("some_other", "enq_flags", "1"), "1");
    }

    #[test]
    fn decode_named_value_exit_kind_wrong_struct_passthrough() {
        // "exit_kind" on a non-scx_exit_info struct should pass through raw.
        assert_eq!(
            decode_named_value("other_struct", "exit_kind", "1024"),
            "1024"
        );
    }

    #[test]
    fn decode_named_value_enq_flags_empty_struct_decodes() {
        // Empty struct_name (scalar param) should still decode.
        assert_eq!(decode_named_value("", "enq_flags", "1"), "WAKEUP");
    }

    #[test]
    fn decode_named_value_exit_kind_empty_struct_decodes() {
        assert_eq!(decode_named_value("", "exit_kind", "1024"), "ERROR");
    }
}
