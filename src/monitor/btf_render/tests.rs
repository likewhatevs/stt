use super::super::cast_analysis::AddrSpace;
use super::*;

/// Build a Btf instance using the project's standard vmlinux
/// resolver (`find_test_vmlinux`) and BTF loader
/// (`load_btf_from_path`). Both honour the `KTSTR_KERNEL` env var,
/// the local-tree fallbacks, and the BTF sidecar cache that real
/// monitor code relies on, so tests don't drift onto a different
/// resolution path that masks bugs in the production loader.
///
/// Returns `None` only when `find_test_vmlinux` decides to skip;
/// it surfaces a `test_skip` message in that path so the user sees
/// the reason rather than a silent no-op.
fn test_btf() -> Option<Btf> {
    let path = crate::monitor::find_test_vmlinux()?;
    crate::monitor::btf_offsets::load_btf_from_path(&path).ok()
}

#[test]
fn read_uint_le_padding() {
    assert_eq!(read_uint_le(&[0x12, 0x34]), 0x3412);
    assert_eq!(read_uint_le(&[0xff]), 0xff);
    assert_eq!(read_uint_le(&[0xff; 8]), u64::MAX);
}

#[test]
fn sign_extend_basic() {
    // 8-bit -1 => 0xFF; sign-extend to 64-bit gives all-ones.
    assert_eq!(sign_extend(0xFF, 8), u64::MAX);
    // 16-bit -1 => 0xFFFF; sign-extend to 64-bit gives all-ones.
    assert_eq!(sign_extend(0xFFFF, 16), u64::MAX);
    // 8-bit 0x7F is positive (max signed); should stay 0x7F.
    assert_eq!(sign_extend(0x7F, 8), 0x7F);
    // 0-bit and 64-bit are no-ops.
    assert_eq!(sign_extend(123, 0), 123);
    assert_eq!(sign_extend(u64::MAX, 64), u64::MAX);
}

#[test]
fn render_int_truncated() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // `int` is virtually guaranteed present in vmlinux BTF.
    let Ok(ids) = btf.resolve_ids_by_name("int") else {
        crate::report::test_skip("BTF missing 'int' type");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'int' to empty id list");
        return;
    };
    // Empty bytes for a 4-byte int -> Truncated.
    let v = render_value(&btf, id, &[]);
    assert!(matches!(
        v,
        RenderedValue::Truncated {
            needed: 4,
            had: 0,
            ..
        }
    ));
}

#[test]
fn render_truncated_unsigned_int() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // `u32` is a Linux-side typedef; some BTFs may not expose it.
    // A `test_skip` here surfaces a visible reason rather than a
    // silent pass.
    let Ok(ids) = btf.resolve_ids_by_name("u32") else {
        crate::report::test_skip("BTF missing 'u32' typedef");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'u32' to empty id list");
        return;
    };
    // 2 bytes for a 4-byte u32 should yield Truncated.
    let v = render_value(&btf, id, &[0xff, 0xff]);
    assert!(matches!(
        v,
        RenderedValue::Truncated {
            needed: 4,
            had: 2,
            ..
        }
    ));
}

// ---- Display impl coverage --------------------------------------
//
// Display is the human-readable form used in test failure output.
// Variant matrix tests:
//   - scalars (Int / Uint / Bool / Char / Float / Enum / Ptr)
//   - Bytes / Unsupported
//   - Truncated (with various partial shapes)
//   - Array (inline scalar vs block-style nested)
//   - Struct (named, unnamed, empty, nested)

#[test]
fn display_int_uint_bool() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Int {
                bits: 32,
                value: -7
            }
        ),
        "-7"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Uint {
                bits: 64,
                value: 42
            }
        ),
        "42"
    );
    assert_eq!(format!("{}", RenderedValue::Bool { value: true }), "true");
    assert_eq!(format!("{}", RenderedValue::Bool { value: false }), "false");
}

#[test]
fn display_char_printable_and_nonprintable() {
    // Printable ASCII renders as 'x'.
    assert_eq!(format!("{}", RenderedValue::Char { value: b'A' }), "'A'");
    // Non-printable (NUL, control, high-bit) renders as 0xNN.
    assert_eq!(format!("{}", RenderedValue::Char { value: 0x00 }), "0x00");
    assert_eq!(format!("{}", RenderedValue::Char { value: 0x7f }), "0x7f");
    assert_eq!(format!("{}", RenderedValue::Char { value: 0xab }), "0xab");
}

#[test]
fn display_float() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Float {
                bits: 64,
                value: 1.5
            }
        ),
        "1.5"
    );
}

#[test]
fn display_enum_with_and_without_variant() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Enum {
                bits: 32,
                value: 1,
                variant: Some("RUNNING".into()),
            }
        ),
        "RUNNING (1)"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Enum {
                bits: 32,
                value: 99,
                variant: None,
            }
        ),
        "99"
    );
}

#[test]
fn display_ptr_is_lowercase_hex() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Ptr {
                value: 0xffff_8000_1234_5678,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            }
        ),
        "0xffff800012345678"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Ptr {
                value: 0,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            }
        ),
        "0x0"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::CpuList {
                cpus: "0-7".to_string()
            }
        ),
        "cpus={0-7}"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::CpuList {
                cpus: String::new()
            }
        ),
        "cpus={}"
    );
}

#[test]
fn display_bytes_passes_through() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Bytes {
                hex: "12 34 ab".into()
            }
        ),
        "12 34 ab"
    );
}

#[test]
fn display_unsupported_includes_reason() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Unsupported {
                reason: "void".into()
            }
        ),
        "<unsupported: void>"
    );
}

#[test]
fn display_truncated_with_bytes_partial() {
    let v = RenderedValue::Truncated {
        needed: 4,
        had: 2,
        partial: Box::new(RenderedValue::Bytes {
            hex: "12 34".into(),
        }),
    };
    assert_eq!(format!("{v}"), "<truncated needed=4 had=2> 12 34");
}

#[test]
fn display_struct_with_named_members() {
    // Display a typical task-context struct value. The inline
    // form is `TypeName{f=v, f=v}` — `=` separates field name
    // from value, no space before the opening brace, and the
    // `struct` keyword is dropped (the type name stands alone).
    let v = RenderedValue::Struct {
        type_name: Some("task_ctx".into()),
        members: vec![
            RenderedMember {
                name: "weight".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 1024,
                },
            },
            RenderedMember {
                name: "last_runnable_at".into(),
                value: RenderedValue::Uint {
                    bits: 64,
                    value: 12_345_678_901_234,
                },
            },
        ],
    };
    assert_eq!(
        format!("{v}"),
        "task_ctx{weight=1024, last_runnable_at=12345678901234}"
    );
}

#[test]
fn display_struct_anonymous_uses_struct_brace() {
    // Anonymous struct: no type name → inline form is just
    // `{f=v}`.
    let v = RenderedValue::Struct {
        type_name: None,
        members: vec![RenderedMember {
            name: "x".into(),
            value: RenderedValue::Int { bits: 32, value: 7 },
        }],
    };
    assert_eq!(format!("{v}"), "{x=7}");
}

#[test]
fn display_empty_struct_is_one_line() {
    // Empty struct (no members at all) renders inline as
    // `Type{}` (no space between name and brace).
    let v = RenderedValue::Struct {
        type_name: Some("empty".into()),
        members: vec![],
    };
    assert_eq!(format!("{v}"), "empty{}");
}

#[test]
fn display_anonymous_member_uses_anon_marker() {
    // BTF anonymous union/struct members surface with empty
    // name; the inline form marks them with `<anon>=` so the
    // operator knows the position without seeing a bare `=`
    // with no preceding identifier.
    let v = RenderedValue::Struct {
        type_name: Some("u".into()),
        members: vec![RenderedMember {
            name: String::new(),
            value: RenderedValue::Uint { bits: 32, value: 5 },
        }],
    };
    assert_eq!(format!("{v}"), "u{<anon>=5}");
}

#[test]
fn display_nested_struct_renders_inline_when_small() {
    // Outer struct with one nested-Struct field — both small
    // enough to fit inline. Nested-struct value renders via
    // `try_render_inline_string`, packed into the outer's
    // inline form: `outer{child=inner{a=1}}`. No newlines.
    let inner = RenderedValue::Struct {
        type_name: Some("inner".into()),
        members: vec![RenderedMember {
            name: "a".into(),
            value: RenderedValue::Uint { bits: 32, value: 1 },
        }],
    };
    let outer = RenderedValue::Struct {
        type_name: Some("outer".into()),
        members: vec![RenderedMember {
            name: "child".into(),
            value: inner,
        }],
    };
    assert_eq!(format!("{outer}"), "outer{child=inner{a=1}}");
}

#[test]
fn display_nested_struct_breaks_to_multiline_past_inline_budget() {
    // Boundary partner of `display_nested_struct_renders_inline_when_small`:
    // when the inner struct's inline form exceeds
    // STRUCT_INLINE_WIDTH_BUDGET (120), `try_inline_from_rendered`
    // returns None and `write_struct` falls through to the
    // breadcrumb form. The outer wrapping then sees `\n` in the
    // child's pre-rendered string (line ~601 in mod.rs) and also
    // bails to multi-line, so the rendered output must contain
    // newlines.
    //
    // 20 u64 members named `field_NN` with value 3735928559
    // (0xdeadbeef as decimal — `RenderedValue::Uint` renders as
    // decimal regardless of magnitude) produces an inline form
    // around 400+ chars, well past the 120-char budget.
    let inner_members: Vec<RenderedMember> = (0..20)
        .map(|i| RenderedMember {
            name: format!("field_{i:02}"),
            value: RenderedValue::Uint {
                bits: 64,
                value: 0xdeadbeef,
            },
        })
        .collect();
    let inner = RenderedValue::Struct {
        type_name: Some("inner".into()),
        members: inner_members,
    };
    let outer = RenderedValue::Struct {
        type_name: Some("outer".into()),
        members: vec![RenderedMember {
            name: "child".into(),
            value: inner,
        }],
    };
    let rendered = format!("{outer}");
    assert!(
        rendered.contains('\n'),
        "over-budget nested struct must break to multi-line; got: {rendered:?}",
    );
    // The breadcrumb form starts with the outer type name followed
    // by `:` (not `{`) — pin that shape so a regression that
    // re-routes back through the inline path with a wider budget
    // would be caught.
    assert!(
        rendered.starts_with("outer:"),
        "multi-line form must lead with `outer:` breadcrumb, got: {rendered:?}",
    );
    // And the inner field's value must appear at least once so
    // the test isn't satisfied by the breadcrumb header alone.
    assert!(
        rendered.contains("3735928559"),
        "inner-member values must still surface in multi-line form: {rendered:?}",
    );
}

#[test]
fn display_array_scalars_inline() {
    // Use bits:32 to bypass the bits:8 string-detection branch
    // which would route through the C-string render path. With
    // every slot populated and starting at index 0, the array
    // collapses to plain `[v1, v2, v3]` (no run brackets) —
    // run brackets are reserved for sparse / gapped arrays.
    // bits>=32 Uints render as hex via write_array_element.
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 1 },
            RenderedValue::Uint { bits: 32, value: 2 },
            RenderedValue::Uint { bits: 32, value: 3 },
        ],
    };
    assert_eq!(format!("{v}"), "[0x1, 0x2, 0x3]");
}

#[test]
fn display_array_empty() {
    let v = RenderedValue::Array {
        len: 0,
        elements: vec![],
    };
    assert_eq!(format!("{v}"), "[]");
}

#[test]
fn display_array_truncated_marker() {
    // Element list shorter than declared `len` surfaces the
    // truncation in a comment. The 2 elements form a contiguous
    // run from 0; since the run doesn't cover the full declared
    // length (5), the sparse render path applies: `[0..1]={v, v}`
    // with the trailing truncation comment. Use bits:32 to
    // bypass the bits:8 string-detection branch.
    let v = RenderedValue::Array {
        len: 5,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 1 },
            RenderedValue::Uint { bits: 32, value: 2 },
        ],
    };
    assert_eq!(format!("{v}"), "[[0..1]={0x1, 0x2}] /* 2 of 5 shown */");
}

#[test]
fn display_array_of_structs_block_style() {
    // Single-element array of struct: block-style render
    // prefixes the struct with `[0] ` to mark the index. The
    // inline-struct form is `Type{f=v}` (no `struct` keyword,
    // no space before brace, `=` separator).
    let elem = RenderedValue::Struct {
        type_name: Some("e".into()),
        members: vec![RenderedMember {
            name: "v".into(),
            value: RenderedValue::Uint {
                bits: 32,
                value: 10,
            },
        }],
    };
    let v = RenderedValue::Array {
        len: 1,
        elements: vec![elem],
    };
    assert_eq!(format!("{v}"), "[\n  [0] e{v=10}\n]");
}

#[test]
fn display_truncated_with_struct_partial_shows_decoded_members() {
    // The partial-render contract: decoded members survive when
    // the struct's byte slice was short. Display surfaces the
    // partial so test failure output points the operator at the
    // fields that DID decode.
    let partial = RenderedValue::Struct {
        type_name: Some("partial_struct".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: 7 },
            },
            RenderedMember {
                name: "b".into(),
                value: RenderedValue::Truncated {
                    needed: 4,
                    had: 0,
                    partial: Box::new(RenderedValue::Bytes { hex: "".into() }),
                },
            },
        ],
    };
    let v = RenderedValue::Truncated {
        needed: 8,
        had: 4,
        partial: Box::new(partial),
    };
    let out = format!("{v}");
    // Outer truncation marker then breadcrumb form (the
    // Truncated member `b` prevents inline collapsing).
    assert!(
        out.starts_with("<truncated needed=8 had=4> partial_struct:"),
        "expected breadcrumb form, got: {out}"
    );
    assert!(out.contains("a=7"));
    assert!(
        !out.contains("truncated needed=4 had=0"),
        "had=0 truncated fields must be suppressed: {out}"
    );
}

// ---- partial-render contract -------------------------------------
//
// Truncated must carry a `partial: Box<RenderedValue>` rather than
// discarding decoded members. Two fixtures:
//   1. struct truncation -> partial is Struct with decoded
//      members (one of which may itself be Truncated for the
//      member that overran).
//   2. scalar truncation -> partial is Bytes hex of available bytes.

#[test]
fn truncated_int_carries_bytes_partial() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("u32") else {
        crate::report::test_skip("BTF missing 'u32'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'u32' to empty id list");
        return;
    };
    let v = render_value(&btf, id, &[0x12, 0x34]);
    match v {
        RenderedValue::Truncated {
            needed,
            had,
            partial,
        } => {
            assert_eq!(needed, 4);
            assert_eq!(had, 2);
            match *partial {
                RenderedValue::Bytes { hex } => {
                    assert_eq!(hex, "12 34");
                }
                other => panic!("expected Bytes partial, got {other:?}"),
            }
        }
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn truncated_struct_carries_struct_partial_with_decoded_members() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // `task_struct` is the canonical large struct in vmlinux BTF.
    let Ok(ids) = btf.resolve_ids_by_name("task_struct") else {
        crate::report::test_skip("BTF missing 'task_struct'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'task_struct' to empty id list");
        return;
    };
    // Feed only 16 bytes — task_struct is multi-KB. Expect
    // Truncated with a Struct partial whose first members
    // decoded (or are themselves Truncated for any member that
    // straddled the cutoff).
    let v = render_value(&btf, id, &[0u8; 16]);
    match v {
        RenderedValue::Truncated {
            needed,
            had,
            partial,
        } => {
            assert!(needed > 16, "expected struct size > 16, got {needed}");
            assert_eq!(had, 16);
            match *partial {
                RenderedValue::Struct { type_name, members } => {
                    assert_eq!(type_name.as_deref(), Some("task_struct"));
                    assert!(
                        !members.is_empty(),
                        "partial struct must carry SOME decoded members"
                    );
                }
                other => panic!("expected Struct partial, got {other:?}"),
            }
        }
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn truncated_array_element_carries_bytes_partial() {
    // Synthesize a struct containing an array whose backing
    // bytes are short. Use BTF if available; otherwise skip.
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // Find a type that ends in `[]` of int — `cpumask_t.bits`
    // is unsigned long array; struct cpumask exists in vmlinux.
    let Ok(ids) = btf.resolve_ids_by_name("cpumask") else {
        crate::report::test_skip("BTF missing 'cpumask'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'cpumask' to empty id list");
        return;
    };
    // Render with a 1-byte buffer; `bits` is u64[NR_CPUS/64],
    // so the array's first element won't fit, producing a
    // Truncated element somewhere in the partial.
    let v = render_value(&btf, id, &[0u8]);
    // Either the outer struct is Truncated (size > 1) or, if
    // cpumask happens to be 0-byte (kernels with NR_CPUS=0 —
    // not realistic), it would render as Struct. Assert either
    // outcome carries a usable partial / member chain.
    match v {
        RenderedValue::Truncated { partial, .. } => {
            // Partial must be the outer Struct (cpumask), which
            // carries `bits` as either a Truncated array or an
            // array with Truncated elements. Either is correct
            // partial render.
            match *partial {
                RenderedValue::Struct { members, .. } => {
                    // At least one member surfaces partial info.
                    assert!(!members.is_empty());
                }
                other => panic!("expected Struct partial, got {other:?}"),
            }
        }
        // Acceptable fallback: cpumask happens to fit in 1 byte
        // somehow (unlikely in real kernels but not a renderer
        // failure if it does).
        RenderedValue::Struct { .. } => {}
        other => panic!("expected Truncated or Struct, got {other:?}"),
    }
}

// ---- Datasec rendering ------------------------------------------
//
// The renderer recognises `BTF_KIND_DATASEC` (the value type
// libbpf assigns to a global-section ARRAY map like `.bss`) and
// walks its `VarSecinfo` entries to render each variable. Before
// this support landed the renderer returned `Unsupported`, so a
// failure dump's `.bss` map showed an opaque hex dump instead of
// `stall=1, crash=0, ...`.
//
// The probe BPF object built by `build.rs` contains a known
// `.bss` Datasec (declared via the `volatile u32
// ktstr_err_exit_detected = 0;` latch, the per-CPU counter
// array `ktstr_pcpu_counters`, the sticky `ktstr_last_trigger_ts`
// / `ktstr_exit_*` snapshot vars, and the `ktstr_miss_log` /
// `ktstr_miss_log_idx` log buffer in `src/bpf/probe.bpf.c`). The
// tests below load that BTF directly via `load_btf_from_path`
// (which falls back to goblin's `.BTF` ELF section parse for
// non-vmlinux files) and exercise the Datasec render path
// against it. Hard-fail on a missing probe.o because build.rs
// always produces it; a silent skip would hide the regression
// the test is designed to catch.

/// Locate the `.bss` Datasec type id in the probe BTF.
/// `resolve_types_by_name(".bss")` returns a list of types named
/// `.bss`; libbpf normally emits exactly one `BTF_KIND_DATASEC`,
/// but the resolver returns Vec<Type> so we scan for the
/// Datasec variant. Returns `(btf, ds_id)` or panics if the
/// build fixture is missing.
fn load_probe_btf_and_bss_id() -> (Btf, u32) {
    let probe_obj = std::path::PathBuf::from(env!("OUT_DIR")).join("probe.o");
    let btf = crate::monitor::btf_offsets::load_btf_from_path(&probe_obj).unwrap_or_else(|e| {
        panic!(
            "load_btf_from_path({}) failed: {e}. \
                 build.rs always produces probe.o; a missing or \
                 unparseable artifact means the build pipeline is \
                 broken.",
            probe_obj.display()
        )
    });
    let ids = btf
        .resolve_ids_by_name(".bss")
        .expect("probe BTF must carry a `.bss` BTF_KIND_DATASEC");
    // Pick the first id that resolves to a Datasec — there
    // should be exactly one, but we don't blow up if libbpf
    // ever emits something else under the `.bss` name.
    for &id in &ids {
        if let Ok(Type::Datasec(_)) = btf.resolve_type_by_id(id) {
            return (btf, id);
        }
    }
    panic!("probe BTF has `.bss` ids {ids:?} but none resolve to BTF_KIND_DATASEC");
}

#[test]
fn render_datasec_emits_struct_with_named_variables() {
    let (btf, bss_id) = load_probe_btf_and_bss_id();
    // Compute the section size by summing each VarSecinfo's
    // (offset + size) — libbpf-emitted Datasecs aren't laid out
    // contiguously (alignment + ordering), so the section's
    // total size is `max(offset + size)` across all entries.
    // Allocate a zeroed buffer of that size so every variable's
    // slice fits.
    let Type::Datasec(ds) = btf.resolve_type_by_id(bss_id).unwrap() else {
        panic!(".bss id did not resolve to Datasec");
    };
    let section_size = ds
        .variables
        .iter()
        .map(|v| v.offset() as usize + v.size())
        .max()
        .expect("`.bss` Datasec must have at least one variable");
    let bytes = vec![0u8; section_size];
    let rendered = render_value(&btf, bss_id, &bytes);

    // Datasec must render as a Struct (not Unsupported, not
    // Truncated — section_size matches the actual section
    // extent so no variable should overrun).
    let RenderedValue::Struct { type_name, members } = rendered else {
        panic!(
            "expected RenderedValue::Struct for Datasec, got something else \
             — Datasec dispatch in render_value_inner must be reachable"
        );
    };
    assert_eq!(
        type_name.as_deref(),
        Some(".bss"),
        "section name must surface as type_name"
    );
    // Variable names: probe.bpf.c declares
    // `ktstr_err_exit_detected` as a writable global. It MUST
    // appear in the rendered .bss members; the freeze
    // coordinator depends on this exact name being resolvable.
    let names: std::collections::HashSet<&str> = members.iter().map(|m| m.name.as_str()).collect();
    assert!(
        names.contains("ktstr_err_exit_detected"),
        "rendered .bss must contain `ktstr_err_exit_detected` \
         (the freeze latch). Found names: {names:?}"
    );
    // Diagnostic counters and snapshots are writable globals →
    // expected in .bss too. Pin every variable name a downstream
    // consumer relies on so a future addition that lands in
    // .bss without renderer coverage surfaces here. The hot
    // counters live in the `ktstr_pcpu_counters` per-CPU array
    // (host-side reader sums across CPUs); sticky snapshot vars
    // remain individual globals because they're written exactly
    // once per error-class exit.
    for required in [
        // Per-CPU diagnostic counter array — replaces the
        // previous N independent globals
        // (ktstr_trigger_count / ktstr_probe_count / etc.).
        "ktstr_pcpu_counters",
        // Sticky timestamp + scheduler-state snapshots written
        // by the tp_btf/sched_ext_exit handler at the first
        // error-class exit.
        "ktstr_last_trigger_ts",
        // SCX_EV_* counter snapshot taken by `scx_bpf_events`
        // at the first error-class exit. Surfaces the
        // system-wide event totals at fault time.
        "ktstr_exit_event_stats",
    ] {
        assert!(
            names.contains(required),
            "rendered .bss must contain `{required}` \
             diagnostic counter. Found names: {names:?}"
        );
    }
    // Each member's `value` must be a concrete renderable
    // type (Uint, Int, Array of int, etc.) — NOT Unsupported.
    // A zero byte buffer can't be Truncated for variables that
    // fit within section_size, so any Truncated result here
    // would indicate a slicing bug.
    for m in &members {
        assert!(
            !matches!(m.value, RenderedValue::Unsupported { .. }),
            "member {:?} rendered as Unsupported: {:?}",
            m.name,
            m.value
        );
        assert!(
            !matches!(m.value, RenderedValue::Truncated { .. }),
            "member {:?} rendered as Truncated despite section_size \
             buffer: {:?}",
            m.name,
            m.value
        );
    }
    // ktstr_err_exit_detected is `volatile u32 = 0;` — must
    // decode to Uint{bits:32, value:0} given a zero buffer.
    let latch = members
        .iter()
        .find(|m| m.name == "ktstr_err_exit_detected")
        .expect("latch member must be present (asserted above)");
    match &latch.value {
        RenderedValue::Uint { bits, value } => {
            assert_eq!(*bits, 32, "latch is u32 (32 bits)");
            assert_eq!(*value, 0, "latch was zeroed in the buffer");
        }
        other => panic!("expected Uint{{32,0}} for latch, got {other:?}"),
    }
}

#[test]
fn render_datasec_truncates_overrunning_variables() {
    // Feed a byte buffer that's too small to cover every
    // variable in the .bss Datasec. Variables whose
    // (offset + size) extends past the buffer must surface as
    // Truncated members, while variables that fit must render
    // normally. The Struct itself is NOT wrapped in Truncated
    // — the section name and the per-variable partial render
    // both stay intact.
    let (btf, bss_id) = load_probe_btf_and_bss_id();
    let Type::Datasec(ds) = btf.resolve_type_by_id(bss_id).unwrap() else {
        panic!(".bss id did not resolve to Datasec");
    };
    // Buffer that holds only the first variable (smallest
    // offset). Variables with higher offsets become Truncated.
    let min_var = ds
        .variables
        .iter()
        .min_by_key(|v| v.offset())
        .expect("`.bss` must have at least one variable");
    let buf_size = (min_var.offset() as usize) + min_var.size();
    let bytes = vec![0u8; buf_size];
    let rendered = render_value(&btf, bss_id, &bytes);

    let RenderedValue::Struct { type_name, members } = rendered else {
        panic!("expected RenderedValue::Struct even with short buffer");
    };
    assert_eq!(type_name.as_deref(), Some(".bss"));
    // At least one member must be Truncated (one of the
    // higher-offset variables). At least one member must be
    // non-Truncated (the variable at min_var's offset).
    let truncated_count = members
        .iter()
        .filter(|m| matches!(m.value, RenderedValue::Truncated { .. }))
        .count();
    let decoded_count = members.len() - truncated_count;
    assert!(
        decoded_count >= 1,
        "at least one member must decode (the variable at the smallest offset, \
         which fits in buf_size={buf_size})"
    );
    // If there is more than one variable in .bss (probe.bpf.c
    // declares several), the short buffer must produce at
    // least one Truncated. A single-variable .bss would have
    // truncated_count == 0, but our probe has multiple — so
    // assert > 0.
    if members.len() > 1 {
        assert!(
            truncated_count >= 1,
            "multi-variable .bss with short buffer must produce >= 1 \
             Truncated member; got 0 from {members:?}"
        );
    }
}

#[test]
fn render_datasec_empty_buffer_yields_struct_with_truncated_members() {
    // Edge case: zero-byte buffer for a non-empty Datasec.
    // Every variable must surface as Truncated rather than
    // crashing the renderer or returning the legacy
    // Unsupported.
    let (btf, bss_id) = load_probe_btf_and_bss_id();
    let rendered = render_value(&btf, bss_id, &[]);
    let RenderedValue::Struct { members, .. } = rendered else {
        panic!("expected Struct render even with empty buffer");
    };
    assert!(!members.is_empty(), "probe `.bss` Datasec is non-empty");
    for m in &members {
        // Every variable should report Truncated{ needed: var
        // size, had: 0, partial: ... }.
        assert!(
            matches!(m.value, RenderedValue::Truncated { had: 0, .. }),
            "member {:?} should be Truncated{{had:0}} for empty buffer, got {:?}",
            m.name,
            m.value
        );
    }
}

// ---- format_cpu_list -------------------------------------------
//
// Range-collapses a sorted CPU id list into a compact string.
// Pin every shape: empty, single, single-range, sparse (gaps),
// multiple ranges + singletons. Collapse rule: a run of >= 2
// consecutive ids renders as `start-end`; a singleton renders as
// just the id.

#[test]
fn format_cpu_list_empty_is_empty_string() {
    assert_eq!(format_cpu_list(&[]), "");
}

#[test]
fn format_cpu_list_single_element() {
    assert_eq!(format_cpu_list(&[5]), "5");
}

#[test]
fn format_cpu_list_contiguous_range() {
    assert_eq!(format_cpu_list(&[0, 1, 2, 3, 4]), "0-4");
}

#[test]
fn format_cpu_list_two_consecutive_collapses_to_range() {
    // The two-element edge: 0,1 must render as "0-1", not "0,1".
    // The end-of-loop flush has its own start==end branch, so a
    // pure-range input exercises the in-loop range emission.
    assert_eq!(format_cpu_list(&[0, 1]), "0-1");
}

#[test]
fn format_cpu_list_gaps_between_ranges() {
    // Mixed: range, singleton, range. Pins the comma-separator
    // and the singleton-formatting path inside the loop.
    assert_eq!(format_cpu_list(&[0, 1, 2, 5, 7, 8, 9]), "0-2,5,7-9");
}

#[test]
fn format_cpu_list_all_singletons() {
    assert_eq!(format_cpu_list(&[0, 2, 4, 6]), "0,2,4,6");
}

#[test]
fn format_cpu_list_first_range_then_singleton() {
    // Trailing singleton — covers the post-loop flush after a
    // mid-list range when the final cpu is alone.
    assert_eq!(format_cpu_list(&[0, 1, 5]), "0-1,5");
}

#[test]
fn format_cpu_list_singleton_then_trailing_range() {
    // Leading singleton followed by a closing range — the
    // post-loop flush emits the trailing range.
    assert_eq!(format_cpu_list(&[0, 3, 4, 5]), "0,3-5");
}

// ---- try_render_cpumask_bits -----------------------------------
//
// Reads u64 LE words from the byte slice and renders set bits as
// a CpuList. The function:
//   - returns None when fewer than 8 bytes are supplied
//     (insufficient for a single u64 word)
//   - returns Some(CpuList { "" }) when bytes are zeroed
//   - extracts bits across multiple words at offset = word*64+bit

#[test]
fn try_render_cpumask_bits_too_short_returns_none() {
    // Strictly less than 8 bytes can't form a u64 word.
    assert!(try_render_cpumask_bits(&[], u32::MAX).is_none());
    assert!(try_render_cpumask_bits(&[0u8; 1], u32::MAX).is_none());
    assert!(try_render_cpumask_bits(&[0u8; 7], u32::MAX).is_none());
}

#[test]
fn try_render_cpumask_bits_all_zero_yields_empty_list() {
    // 8 zeroed bytes: Some(CpuList { cpus: "" }) — the loop sees
    // word=0 and skips it; format_cpu_list of an empty Vec is "".
    let v = try_render_cpumask_bits(&[0u8; 8], u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(cpus, "", "all-zero bytes must produce empty cpu list");
        }
        other => panic!("expected Some(CpuList), got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_single_word_low_bits() {
    // Single word with bits 0,1,2 set → "0-2".
    let bits: u64 = 0b111;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-2"),
        other => panic!("expected CpuList with 0-2, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_single_word_high_bit() {
    // Bit 63 set → cpu 63.
    let bits: u64 = 1u64 << 63;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "63"),
        other => panic!("expected CpuList with 63, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_caps_at_nr_cpu_ids() {
    // 8 CPUs: bits 0..=7 are real, bits 8..=63 are slab padding
    // / freelist garbage. With max_cpus=8, only bits 0..=7
    // should appear in the rendered list even when the word
    // has additional bits set higher up. Pins the fix:
    // an 8-CPU guest must not render bits 64..4035 from a
    // 1024-byte cpumask slab.
    let bits: u64 = 0xFFFF_FFFF_FFFF_FFFF; // every bit set
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, 8);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(cpus, "0-7", "max_cpus=8 must cap at cpu 7, got {cpus}");
        }
        other => panic!("expected CpuList with 0-7, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_caps_across_word_boundary() {
    // Two words, all bits set. max_cpus=8 must stop walking
    // immediately in word 0 (cap is partial-word). max_cpus=
    // 64 must walk word 0 fully and stop at the start of word
    // 1 (cap is whole-word).
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());

    // Partial-word cap inside word 0.
    let v = try_render_cpumask_bits(&bytes, 8);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-7"),
        other => panic!("expected CpuList 0-7, got {other:?}"),
    }

    // Whole-word cap: 64 means stop at start of word 1.
    // (Word 0 contains bits 0..=63, all 64 of them set.)
    let v = try_render_cpumask_bits(&bytes, 64);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-63"),
        other => panic!("expected CpuList 0-63, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_multi_word_offsets() {
    // Two words: word[0] bit 0 → cpu 0, word[1] bit 0 → cpu 64,
    // word[1] bit 1 → cpu 65. Pins the word-index to cpu-id
    // arithmetic (word*64 + bit).
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&1u64.to_le_bytes());
    let w1: u64 = 0b11;
    bytes[8..16].copy_from_slice(&w1.to_le_bytes());
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0,64-65"),
        other => panic!("expected CpuList with 0,64-65, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_partial_trailing_bytes_ignored() {
    // 12 bytes (1.5 words) — only the first complete word
    // (8 bytes) parses. The trailing 4 bytes are ignored
    // because n_words = 12/8 = 1.
    let mut bytes = [0u8; 12];
    bytes[0..8].copy_from_slice(&1u64.to_le_bytes());
    // Trailing 4 bytes hold bit 0 (cpu 32 if read as a word)
    // but should NOT be parsed.
    bytes[8] = 0xff;
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0"),
        other => panic!("expected CpuList with 0, got {other:?}"),
    }
}

/// Garbage cpumask data with a sane nr_cpu_ids cap produces
/// capped output rather than enumerating phantom CPUs from
/// slab-padding / freelist bytes. 16 words (128 bytes) all-FF
/// would otherwise surface 1024 bit-positions; with max_cpus=4
/// only cpus 0..=3 must appear. Pins the backstop: the
/// nr_cpu_ids cap protects the renderer when SLAB_FREELIST_
/// HARDENED XOR-encoding defeats the top-byte heuristic.
#[test]
fn try_render_cpumask_bits_garbage_capped_at_max_cpus() {
    // 128 bytes = 16 u64 words, every bit set. Without the
    // cap this would render 0-1023; with max_cpus=4 the walker
    // stops after bit 3 of word 0.
    let bytes = vec![0xFFu8; 128];
    let v = try_render_cpumask_bits(&bytes, 4);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(
                cpus, "0-3",
                "max_cpus=4 must clip 1024-bit garbage to cpus 0-3, got: {cpus}",
            );
        }
        other => panic!("expected CpuList 0-3, got {other:?}"),
    }
}

/// `max_cpus = 0` produces an empty cpu list. The whole-word
/// gate (`word_first_cpu >= max_cpus as u64`) fires immediately
/// at word 0. Defensive: a malformed reader returning
/// `nr_cpu_ids = 0` (not the trait default `u32::MAX`) would
/// otherwise expose the per-bit `cpu >= max_cpus` check to a
/// loop entry; verify both paths converge on empty output.
#[test]
fn try_render_cpumask_bits_max_cpus_zero_yields_empty_list() {
    let bits: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, 0);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(cpus, "", "max_cpus=0 must produce empty list, got: {cpus}");
        }
        other => panic!("expected empty CpuList, got {other:?}"),
    }
}

/// Cap matches the actual mask width (max_cpus=64, all 64 bits
/// set in word 0): every bit must surface. A regression that
/// used `>=` for the per-bit cap (clipping the last bit) would
/// produce "0-62" instead of "0-63". Pins the upper-edge
/// off-by-one.
#[test]
fn try_render_cpumask_bits_max_cpus_matches_word_width_keeps_all_bits() {
    let bits: u64 = u64::MAX;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, 64);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(
                cpus, "0-63",
                "max_cpus=64 must surface all 64 bits, got: {cpus}",
            );
        }
        other => panic!("expected CpuList 0-63, got {other:?}"),
    }
}

/// `MemReader` trait default `nr_cpu_ids` is `u32::MAX`.
/// Callers that don't override produce no cap, preserving the
/// pre-fix behavior (every set bit reported). A regression
/// that flipped the default to `0` would silently empty every
/// cpumask render.
#[test]
fn mem_reader_default_nr_cpu_ids_is_u32_max() {
    struct DefaultReader;
    impl MemReader for DefaultReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
    }
    let r = DefaultReader;
    assert_eq!(
        r.nr_cpu_ids(),
        u32::MAX,
        "default nr_cpu_ids must be u32::MAX",
    );
}

/// A custom `MemReader` impl overrides `nr_cpu_ids`. Pin the
/// override path so a regression that ignored the override
/// (always returning the default) is caught.
#[test]
fn mem_reader_custom_nr_cpu_ids_returns_overridden_value() {
    struct CustomReader {
        cpu_count: u32,
    }
    impl MemReader for CustomReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn nr_cpu_ids(&self) -> u32 {
            self.cpu_count
        }
    }
    let r = CustomReader { cpu_count: 16 };
    assert_eq!(r.nr_cpu_ids(), 16);
}

/// `render_struct` consults `MemReader::nr_cpu_ids` when
/// rendering a cpumask-family struct. With max_cpus=8 supplied
/// by the reader, garbage bits beyond cpu 7 are dropped —
/// the `let max_cpus = mem.map(|m| m.nr_cpu_ids())`
/// in `render_struct` (btf_render.rs) wires the reader value
/// through to `try_render_cpumask_bits`. A regression that
/// passed `u32::MAX` instead would surface phantom cpus 8..63.
#[test]
fn render_value_with_mem_caps_cpumask_at_reader_nr_cpu_ids() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("cpumask") else {
        crate::report::test_skip("BTF missing 'cpumask' struct");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'cpumask' to empty id list");
        return;
    };
    // Resolve the underlying struct so we can size the buffer.
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel cpumask modifiers");
        return;
    };
    let size = match type_size(&btf, &ty) {
        Some(n) if n >= 8 => n,
        _ => {
            crate::report::test_skip("cpumask size unresolved or < 8");
            return;
        }
    };
    // Fill the entire cpumask buffer with all-FFs garbage.
    let bytes = vec![0xFFu8; size];
    struct EightCpuReader;
    impl MemReader for EightCpuReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn nr_cpu_ids(&self) -> u32 {
            8
        }
    }
    let reader = EightCpuReader;
    let v = render_value_with_mem(&btf, id, &bytes, &reader);
    match v {
        RenderedValue::CpuList { cpus } => {
            assert_eq!(
                cpus, "0-7",
                "render_struct must propagate reader.nr_cpu_ids=8 to cpu-list \
                 rendering; got: {cpus}",
            );
        }
        other => panic!("expected CpuList from cpumask render, got {other:?}"),
    }
}

/// Without a `MemReader` (the `render_value` entry point passes
/// `None`), the cpumask renderer falls back to `u32::MAX` cap
/// — every set bit surfaces. Pins the `mem.map(...).unwrap_or(u32::MAX)`
/// fallback in `render_struct` for the no-reader code path.
#[test]
fn render_value_without_mem_uses_u32_max_cap() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("cpumask") else {
        crate::report::test_skip("BTF missing 'cpumask' struct");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'cpumask' to empty id list");
        return;
    };
    // 8 bytes (one word) of all-FFs: every bit 0..63 set.
    // No-cap (u32::MAX) means all 64 bits surface as cpus 0-63.
    let bytes = [0xFFu8; 8];
    let v = render_value(&btf, id, &bytes);
    match v {
        RenderedValue::CpuList { cpus } => {
            assert_eq!(
                cpus, "0-63",
                "no-reader cpumask must use u32::MAX cap (all 64 bits), got: {cpus}",
            );
        }
        other => panic!("expected CpuList, got {other:?}"),
    }
}

// ---- is_text_byte -----------------------------------------------
//
// Module-private helper; pinned via direct call to defend the
// string-detection contract: NUL terminator, \n, and printable
// ASCII (0x20..=0x7e) are "text". \t and \r are explicitly NOT
// accepted (binary arrays starting with those bytes were
// misclassified as strings). Non-ASCII (>= 0x80) and other ASCII
// control chars are not text.

#[test]
fn is_text_byte_accepts_nul() {
    assert!(is_text_byte(0x00), "NUL is the C string terminator");
}

#[test]
fn is_text_byte_accepts_newline() {
    assert!(is_text_byte(b'\n'));
}

#[test]
fn is_text_byte_rejects_tab_and_cr() {
    // \t and \r in BPF data are almost always binary; accepting
    // them produced false-positive string classification.
    assert!(!is_text_byte(b'\t'));
    assert!(!is_text_byte(b'\r'));
}

#[test]
fn is_text_byte_accepts_printable_ascii() {
    // Boundaries of the printable range.
    assert!(is_text_byte(0x20)); // space
    assert!(is_text_byte(b'A'));
    assert!(is_text_byte(0x7e)); // ~
}

#[test]
fn is_text_byte_rejects_other_control_chars() {
    // Non-newline control chars.
    assert!(!is_text_byte(0x01));
    assert!(!is_text_byte(0x07)); // BEL
    assert!(!is_text_byte(0x1f));
}

#[test]
fn is_text_byte_rejects_high_bit_bytes() {
    assert!(!is_text_byte(0x7f)); // DEL is just past printable
    assert!(!is_text_byte(0x80));
    assert!(!is_text_byte(0xff));
}

// ---- is_string_value --------------------------------------------
//
// Detects whether a RenderedValue::Array represents a printable
// C-string-like byte array. Used by the Display path to suppress
// bpf_printk format strings nested inside Structs.

#[test]
fn is_string_value_accepts_8bit_int_array() {
    let v = RenderedValue::Array {
        len: 4,
        elements: vec![
            RenderedValue::Int {
                bits: 8,
                value: b'h' as i64,
            },
            RenderedValue::Int {
                bits: 8,
                value: b'i' as i64,
            },
            RenderedValue::Int {
                bits: 8,
                value: b'\n' as i64,
            },
            RenderedValue::Int { bits: 8, value: 0 },
        ],
    };
    assert!(is_string_value(&v));
}

#[test]
fn is_string_value_accepts_8bit_uint_array() {
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 8,
                value: b'a' as u64,
            },
            RenderedValue::Uint {
                bits: 8,
                value: b'b' as u64,
            },
        ],
    };
    assert!(is_string_value(&v));
}

#[test]
fn is_string_value_accepts_char_array() {
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Char { value: b'X' },
            RenderedValue::Char { value: 0 },
        ],
    };
    assert!(is_string_value(&v));
}

#[test]
fn is_string_value_rejects_too_short_array() {
    // Single-element arrays don't qualify (length floor of 2).
    let v = RenderedValue::Array {
        len: 1,
        elements: vec![RenderedValue::Char { value: b'X' }],
    };
    assert!(!is_string_value(&v));
}

#[test]
fn is_string_value_rejects_non_text_byte() {
    // 0x80 is non-text → array doesn't qualify as string.
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 8,
                value: b'a' as u64,
            },
            RenderedValue::Uint {
                bits: 8,
                value: 0x80,
            },
        ],
    };
    assert!(!is_string_value(&v));
}

#[test]
fn is_string_value_rejects_wider_int() {
    // u32 elements aren't bytes — even with text-looking values.
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 32,
                value: b'a' as u64,
            },
            RenderedValue::Uint {
                bits: 32,
                value: b'b' as u64,
            },
        ],
    };
    assert!(!is_string_value(&v));
}

#[test]
fn is_string_value_rejects_non_array() {
    // Only Array is a candidate; everything else is not.
    assert!(!is_string_value(&RenderedValue::Bytes { hex: "00".into() }));
    assert!(!is_string_value(&RenderedValue::Uint {
        bits: 8,
        value: b'a' as u64
    }));
    assert!(!is_string_value(&RenderedValue::Struct {
        type_name: None,
        members: vec![],
    }));
}

// ---- is_zero ----------------------------------------------------
//
// Module-public: callers (dump/display.rs Display paths) suppress
// zeroed scalars in struct/array rendering. Pin every variant's
// zero-detection.

#[test]
fn is_zero_int_uint_bool_char() {
    assert!(is_zero(&RenderedValue::Int { bits: 32, value: 0 }));
    assert!(!is_zero(&RenderedValue::Int {
        bits: 32,
        value: -1
    }));
    assert!(is_zero(&RenderedValue::Uint { bits: 64, value: 0 }));
    assert!(!is_zero(&RenderedValue::Uint { bits: 64, value: 1 }));
    assert!(is_zero(&RenderedValue::Bool { value: false }));
    assert!(!is_zero(&RenderedValue::Bool { value: true }));
    assert!(is_zero(&RenderedValue::Char { value: 0 }));
    assert!(!is_zero(&RenderedValue::Char { value: b'a' }));
}

#[test]
fn is_zero_float() {
    assert!(is_zero(&RenderedValue::Float {
        bits: 64,
        value: 0.0
    }));
    assert!(!is_zero(&RenderedValue::Float {
        bits: 64,
        value: 1.0
    }));
    // -0.0 is bit-distinct from 0.0 but is_zero compares with ==,
    // so both register as zero per IEEE-754.
    assert!(is_zero(&RenderedValue::Float {
        bits: 64,
        value: -0.0
    }));
}

#[test]
fn is_zero_enum() {
    assert!(is_zero(&RenderedValue::Enum {
        bits: 32,
        value: 0,
        variant: None
    }));
    assert!(!is_zero(&RenderedValue::Enum {
        bits: 32,
        value: 1,
        variant: Some("RUNNING".into())
    }));
}

#[test]
fn is_zero_cpulist_empty_vs_populated() {
    assert!(
        is_zero(&RenderedValue::CpuList {
            cpus: String::new()
        }),
        "empty cpu list is zero"
    );
    assert!(
        !is_zero(&RenderedValue::CpuList { cpus: "0-7".into() }),
        "populated cpu list is non-zero"
    );
}

#[test]
fn is_zero_ptr() {
    assert!(is_zero(&RenderedValue::Ptr {
        value: 0,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
    assert!(!is_zero(&RenderedValue::Ptr {
        value: 0xffff_8000_dead_beef,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
    // A pointer with a deref but value=0 is still zero per the
    // is_zero contract — only `value` matters.
    assert!(is_zero(&RenderedValue::Ptr {
        value: 0,
        deref: Some(Box::new(RenderedValue::Uint { bits: 32, value: 5 })),
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
}

#[test]
fn is_zero_compound_always_false() {
    // Per the impl, Struct / Array / Bytes / Truncated /
    // Unsupported short-circuit to false. Pin that to lock in the
    // "compounds are always rendered" decision.
    assert!(!is_zero(&RenderedValue::Struct {
        type_name: None,
        members: vec![],
    }));
    assert!(!is_zero(&RenderedValue::Array {
        len: 0,
        elements: vec![],
    }));
    assert!(!is_zero(&RenderedValue::Bytes { hex: "".into() }));
    assert!(!is_zero(&RenderedValue::Unsupported { reason: "x".into() }));
}

// ---- is_inline_scalar -------------------------------------------
//
// Determines whether an array element renders inline (single
// line) vs block-style. Scalars + Bytes + Unsupported are inline;
// composite types (Struct, Array, CpuList, Truncated) are block.

#[test]
fn is_inline_scalar_accepts_scalars() {
    assert!(is_inline_scalar(&RenderedValue::Int { bits: 32, value: 0 }));
    assert!(is_inline_scalar(&RenderedValue::Uint {
        bits: 64,
        value: 1
    }));
    assert!(is_inline_scalar(&RenderedValue::Bool { value: false }));
    assert!(is_inline_scalar(&RenderedValue::Char { value: b'x' }));
    assert!(is_inline_scalar(&RenderedValue::Float {
        bits: 64,
        value: 0.0
    }));
    assert!(is_inline_scalar(&RenderedValue::Enum {
        bits: 32,
        value: 0,
        variant: None,
    }));
    assert!(is_inline_scalar(&RenderedValue::Ptr {
        value: 0,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
    assert!(is_inline_scalar(&RenderedValue::Bytes { hex: "00".into() }));
    assert!(is_inline_scalar(&RenderedValue::Unsupported {
        reason: "void".into(),
    }));
}

#[test]
fn is_inline_scalar_rejects_composites() {
    assert!(!is_inline_scalar(&RenderedValue::Struct {
        type_name: None,
        members: vec![],
    }));
    assert!(!is_inline_scalar(&RenderedValue::Array {
        len: 0,
        elements: vec![],
    }));
    assert!(!is_inline_scalar(&RenderedValue::CpuList {
        cpus: "0".into(),
    }));
    assert!(!is_inline_scalar(&RenderedValue::Truncated {
        needed: 4,
        had: 0,
        partial: Box::new(RenderedValue::Bytes { hex: "".into() }),
    }));
    assert!(!is_inline_scalar(&RenderedValue::Ptr {
        value: 0x1000,
        deref: Some(Box::new(RenderedValue::Struct {
            type_name: Some("scx_cgroup_llc_ctx".into()),
            members: vec![],
        })),
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
}

// ---- Ptr Display with deref → arrow notation -------------------
//
// A Ptr with `deref: Some(...)` renders as `0x<hex> → <inner>`.
// The arrow ("→", U+2192) is the resolved-pointer indicator.
// Tests cover: deref to a scalar, deref to a CpuList (the
// common cpumask kptr chase), deref to a struct.

#[test]
fn display_ptr_with_scalar_deref_uses_arrow() {
    let v = RenderedValue::Ptr {
        value: 0xffff_8000_1234_5678,
        deref: Some(Box::new(RenderedValue::Uint {
            bits: 32,
            value: 42,
        })),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(
        out.contains(" → "),
        "Display must include arrow separator: {out}"
    );
    assert!(out.starts_with("0xffff800012345678"));
    assert!(out.ends_with("42"));
}

#[test]
fn display_ptr_with_cpulist_deref_renders_inline() {
    // Mirrors the cpumask-kptr chase: pointer hex, arrow, then
    // the rendered cpu list.
    let v = RenderedValue::Ptr {
        value: 0xffff_8888_aaaa_bbbb,
        deref: Some(Box::new(RenderedValue::CpuList { cpus: "0-3".into() })),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    assert_eq!(format!("{v}"), "0xffff8888aaaabbbb → cpus={0-3}");
}

#[test]
fn display_ptr_with_struct_deref_indents_correctly() {
    // The deref payload is a Struct → render follows the arrow.
    // The inline form is `inner{v=7}` (no `struct` keyword,
    // no space before brace, `=` separator).
    let inner = RenderedValue::Struct {
        type_name: Some("inner".into()),
        members: vec![RenderedMember {
            name: "v".into(),
            value: RenderedValue::Uint { bits: 32, value: 7 },
        }],
    };
    let v = RenderedValue::Ptr {
        value: 0xdead_beef,
        deref: Some(Box::new(inner)),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(out.contains("0xdeadbeef → inner{"));
    assert!(out.contains("v=7"));
}

#[test]
fn display_ptr_without_deref_no_arrow() {
    // Negative control: deref=None AND no skip reason must NOT
    // emit the arrow.
    let v = RenderedValue::Ptr {
        value: 0xff,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(
        !out.contains("→"),
        "no-deref Ptr must not have arrow: {out}"
    );
    assert_eq!(out, "0xff");
}

/// Ptr with deref_skipped_reason but no deref renders the
/// reason inline in `[chase: ...]` notation. Pins the
/// fix: a chase that failed for a known cause (cross-page,
/// 4 KiB cap, plausibility gate) must surface that cause in
/// the operator-visible Display, not silently look identical
/// to "no chase attempted."
#[test]
fn display_ptr_with_skip_reason_surfaces_inline() {
    let v = RenderedValue::Ptr {
        value: 0x7fff_aaaa_0000,
        deref: None,
        deref_skipped_reason: Some(
            "arena read failed (cross-page boundary or unmapped page)".to_string(),
        ),
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(
        out.contains("[chase: arena read failed"),
        "skip reason must be surfaced in [chase: ...] form: {out}"
    );
    assert!(
        out.starts_with("0x7fffaaaa0000"),
        "pointer hex must come first: {out}"
    );
    assert!(
        !out.contains("→"),
        "skip reason render must NOT emit arrow (no actual deref): {out}"
    );
}

// ---- Struct template grouping (try_write_struct_template) -----
//
// Block-style array Display has a "template" optimization for
// arrays of similar structs: when 3+ consecutive single-element
// groups are structs of the same shape with < 8 differing
// fields, they collapse into one `[start-end] struct {}` block
// showing common fields once and varying fields in a per-index
// table. Below the threshold (< 3 structs OR 0 or > 3 varying
// fields), it falls back to the default block render.

#[test]
fn array_of_3_similar_structs_uses_template_block() {
    // Three structs differing only in one field → template
    // collapse: common field shown once, varying field in
    // per-index lines.
    let mk = |x: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "common".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 100,
                },
            },
            RenderedMember {
                name: "x".into(),
                value: RenderedValue::Uint { bits: 32, value: x },
            },
        ],
    };
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![mk(1), mk(2), mk(3)],
    };
    let out = format!("{v}");
    // Template indicator: `[start-end] TypeName:` breadcrumb form.
    assert!(
        out.contains("[0-2] s:"),
        "must surface template index range header: {out}"
    );
    // Common field shown once with `=` assignment.
    assert!(out.contains("common=100"), "common field once: {out}");
    // Varying field rendered as per-index list — `name:`
    // introduces the list (multiple values per row), distinct
    // from the `name=value` scalar form.
    assert!(out.contains("x: "), "varying field name present: {out}");
    assert!(out.contains("[0]="), "per-index marker for first: {out}");
    assert!(out.contains("[2]="), "per-index marker for last: {out}");
}

#[test]
fn array_of_2_similar_structs_renders_per_element() {
    // Regression: the inline-template path checks
    // `structs.len() < 3` and returns Ok(false) without writing
    // anything. The caller previously discarded that signal
    // (`let _ = try_write_struct_template(...)`) and skipped
    // past both groups, producing an empty `[\n]`. Fix surfaces
    // the false return and falls through to per-element render.
    // The inline form drops the `struct` keyword and uses
    // `Type{f=v}` notation.
    let mk = |x: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![RenderedMember {
            name: "x".into(),
            value: RenderedValue::Uint { bits: 32, value: x },
        }],
    };
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![mk(1), mk(2)],
    };
    let out = format!("{v}");
    // Template header NOT present (group merge requires >= 3).
    assert!(
        !out.contains("[0-1]"),
        "two-element array must not use template: {out}"
    );
    // Both elements must surface in per-element render; the
    // pre-fix code dropped them entirely.
    assert!(out.contains("[0] s{"), "missing [0]: {out}");
    assert!(out.contains("[1] s{"), "missing [1]: {out}");
    assert!(out.contains("x=1"), "missing x=1: {out}");
    assert!(out.contains("x=2"), "missing x=2: {out}");
}

#[test]
fn array_with_too_many_varying_fields_falls_back() {
    // > 3 varying fields → template not used; falls back to
    // per-element block render. Pre-fix the same `let _ =
    // try_write_struct_template` bug also dropped these three
    // elements silently; assert that they all surface.
    let mk = |a: u64, b: u64, c: u64, d: u64, e: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: a },
            },
            RenderedMember {
                name: "b".into(),
                value: RenderedValue::Uint { bits: 32, value: b },
            },
            RenderedMember {
                name: "c".into(),
                value: RenderedValue::Uint { bits: 32, value: c },
            },
            RenderedMember {
                name: "d".into(),
                value: RenderedValue::Uint { bits: 32, value: d },
            },
            RenderedMember {
                name: "e".into(),
                value: RenderedValue::Uint { bits: 32, value: e },
            },
        ],
    };
    // All 5 fields differ across the 3 elements → varying.len() > 3.
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![mk(1, 1, 1, 1, 1), mk(2, 2, 2, 2, 2), mk(3, 3, 3, 3, 3)],
    };
    let out = format!("{v}");
    assert!(
        !out.contains("[0-2]"),
        ">3 varying fields must skip template, falls back to per-element: {out}",
    );
    // Per-element fallback must surface all three. Each
    // element renders inline as `[N] s{a=v, b=v, ...}` because
    // the rendered single-line form fits within the inline
    // width budget; per-element prefix `[N] ` is added by the
    // array's per-element block path.
    assert!(out.contains("[0] s{"), "missing [0]: {out}");
    assert!(out.contains("[1] s{"), "missing [1]: {out}");
    assert!(out.contains("[2] s{"), "missing [2]: {out}");
}

#[test]
fn array_of_identical_structs_groups_via_run() {
    // All-identical structs: the leading group walker collapses
    // them into one `[start-end]` group rather than the template
    // (template requires varying fields). Pins that the
    // ConsecutiveSimilar detection routes correctly. The
    // grouped struct itself renders inline so the marker is
    // `[0-2] s{x=5}`.
    let s = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![RenderedMember {
            name: "x".into(),
            value: RenderedValue::Uint { bits: 32, value: 5 },
        }],
    };
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![s.clone(), s.clone(), s],
    };
    let out = format!("{v}");
    // Identical-element grouping shows `[0-2]` followed by
    // the inline-struct render.
    assert!(out.contains("[0-2] s{"), "must group identical: {out}");
}

#[test]
fn array_inline_sparse_runs() {
    // Inline scalar arrays with gaps render each contiguous
    // non-zero run as `[idx]=value` (single) or
    // `[start..end]={v, v}` (multi-element). Zero gaps are
    // implicit from the run brackets — no `(N zero)` suffix.
    // Use bits:32 to avoid the bits:8 string-detection branch
    // which routes to the C-string render path.
    let v = RenderedValue::Array {
        len: 5,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 1 },
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 2 },
        ],
    };
    // Two single-element runs at indices 1 and 4. The gap
    // between them (indices 2, 3) is implicit from the index
    // brackets.
    assert_eq!(format!("{v}"), "[[1]=0x1  [4]=0x2]");
}

#[test]
fn array_inline_all_zero_collapses() {
    // All-zero inline array renders the special "all N zero"
    // collapse marker. Use bits:32 since bits:8 all-zero
    // arrays trip the string-detection NUL-string branch
    // (rendered as `""`) — see the "[all N zero]"
    // short-circuit at the head of the Array Display arm.
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 0 },
        ],
    };
    assert_eq!(format!("{v}"), "[all 3 zero]");
}

#[test]
fn array_block_all_zero_collapses() {
    // Block-style (non-inline) all-zero collapse: when every
    // element is a zero-rendering compound proxy. Since the
    // is_zero check skips compound types, this only triggers
    // when elements are inline scalars wrapped in something
    // else — or, more reliably, when the elements pass the
    // inline check AND are zero. Use Ptr with value=0 (inline,
    // is_zero=true).
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Ptr {
                value: 0,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            },
            RenderedValue::Ptr {
                value: 0,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            },
        ],
    };
    let out = format!("{v}");
    // Inline scalars (Ptr passes is_inline_scalar), so this hits
    // the inline branch with all-zero collapse.
    assert!(
        out.contains("all 2 zero"),
        "inline all-zero collapse: {out}"
    );
}

#[test]
fn struct_zero_field_suppression_drops_silently() {
    // Struct Display suppresses zero fields silently — no
    // `(N fields zero)` summary appears anywhere. The
    // operator infers from the rendered (and absent) fields
    // that the rest are zero; an explicit count line adds
    // overhead without insight.
    let v = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "shown".into(),
                value: RenderedValue::Uint { bits: 32, value: 5 },
            },
            RenderedMember {
                name: "zero1".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
            RenderedMember {
                name: "zero2".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
        ],
    };
    let out = format!("{v}");
    assert!(out.contains("shown=5"), "non-zero field shown: {out}");
    assert!(!out.contains("zero1"), "zero fields suppressed: {out}");
    assert!(
        !out.contains("fields zero"),
        "no `(N fields zero)` summary in any form: {out}",
    );
}

#[test]
fn struct_all_zero_emits_empty_inline_form() {
    // All-zero struct: every field is suppressed by deeply-
    // zero collapse, leaving an empty visible set. The inline
    // form emits a bare brace pair `Type{}` — no count
    // summary. The empty body is self-explanatory.
    let v = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
            RenderedMember {
                name: "b".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
        ],
    };
    let out = format!("{v}");
    assert_eq!(
        out, "s{}",
        "all-zero struct collapses to empty inline form: {out}",
    );
}

#[test]
fn struct_bpf_printk_format_strings_collapsed() {
    // Members whose names contain "___fmt" / "____fmt" AND whose
    // values are string-shaped Arrays get suppressed (compile-
    // time constants, not runtime state). With inline rendering
    // active for ≤ 3 visible fields, the summary line
    // "(N bpf_printk format strings)" is dropped — at this
    // density the suppression is implicit.
    let fmt_string_value = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Char { value: b'h' },
            RenderedValue::Char { value: b'i' },
            RenderedValue::Char { value: 0 },
        ],
    };
    let v = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "real_field".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 42,
                },
            },
            RenderedMember {
                name: "ktstr___fmt_blah".into(),
                value: fmt_string_value.clone(),
            },
            RenderedMember {
                name: "____fmt_other".into(),
                value: fmt_string_value,
            },
        ],
    };
    let out = format!("{v}");
    assert!(out.contains("real_field=42"));
    assert!(
        !out.contains("ktstr___fmt_blah"),
        "fmt string suppressed: {out}"
    );
    assert!(
        !out.contains("____fmt_other"),
        "fmt string suppressed: {out}"
    );
}

// ---- Array string-mode rendering -------------------------------
//
// 8-bit Int/Uint/Char arrays containing printable bytes render
// as a quoted string (single-line) or block (multi-line). Pins
// both branches.

#[test]
fn array_renders_as_quoted_string_when_printable() {
    let v = RenderedValue::Array {
        len: 6,
        elements: vec![
            RenderedValue::Char { value: b'h' },
            RenderedValue::Char { value: b'e' },
            RenderedValue::Char { value: b'l' },
            RenderedValue::Char { value: b'l' },
            RenderedValue::Char { value: b'o' },
            RenderedValue::Char { value: 0 },
        ],
    };
    let out = format!("{v}");
    assert_eq!(out, "\"hello\"");
}

#[test]
fn array_renders_multiline_string_with_pipe() {
    // A string with embedded newlines uses the `|` block scalar
    // marker and indents per line.
    let v = RenderedValue::Array {
        len: 8,
        elements: vec![
            RenderedValue::Char { value: b'a' },
            RenderedValue::Char { value: b'\n' },
            RenderedValue::Char { value: b'b' },
            RenderedValue::Char { value: b'\n' },
            RenderedValue::Char { value: b'c' },
            RenderedValue::Char { value: 0 },
            RenderedValue::Char { value: 0 },
            RenderedValue::Char { value: 0 },
        ],
    };
    let out = format!("{v}");
    assert!(
        out.starts_with("|\n"),
        "must start with pipe + newline: {out}"
    );
    assert!(out.contains("a"), "must contain first segment: {out}");
    assert!(out.contains("b"), "must contain second segment: {out}");
}

#[test]
fn write_array_element_uint_wide_renders_hex() {
    // write_array_element formats Uint with bits>=32 as hex,
    // bits<32 stays decimal. Indirect coverage via array
    // Display.
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 32,
                value: 255,
            },
            RenderedValue::Uint {
                bits: 64,
                value: 0xdead_beef,
            },
        ],
    };
    let out = format!("{v}");
    // 32+ bit Uints render as hex, separated and indexed.
    assert!(out.contains("0xff"), "32-bit uint hex: {out}");
    assert!(out.contains("0xdeadbeef"), "64-bit uint hex: {out}");
}

// ---- Cycle detection in pointer chase --------------------------
//
// A `Type::Ptr` whose deref contains a back-pointer to an
// already-visited address must not recurse through the cycle.
// Without the visited-set check, the renderer recurses until
// `MAX_RENDER_DEPTH` (32) fires, producing a wall of identical
// nested structs in the failure dump. With the check, the
// pointer surfaces a `[cycle]` marker after its hex value and
// stops.

/// Stub MemReader that returns canned bytes for specific arena
/// addresses. Used to construct synthetic cycles in pointer
/// chases. `bytes_by_addr` maps arena address → backing bytes;
/// `arena_range` defines `is_arena_addr` accept set.
struct CycleArenaReader {
    bytes_by_addr: std::collections::HashMap<u64, Vec<u8>>,
    arena_start: u64,
    arena_end: u64,
}
impl MemReader for CycleArenaReader {
    fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
        None
    }
    fn is_arena_addr(&self, addr: u64) -> bool {
        addr >= self.arena_start && addr < self.arena_end
    }
    fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let bytes = self.bytes_by_addr.get(&addr)?;
        if bytes.len() < len {
            return None;
        }
        Some(bytes[..len].to_vec())
    }
}

/// Self-pointing cycle: a `struct list_head` whose `next` field
/// points to its own arena address. The renderer must surface
/// the cycle on the inner pointer rather than recursing 32
/// levels deep.
#[test]
fn ptr_cycle_self_pointing_surfaces_cycle_reason() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("list_head") else {
        crate::report::test_skip("BTF missing 'list_head'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'list_head' to empty id list");
        return;
    };
    // Verify list_head is the expected shape (Struct with two
    // pointer fields). Skip if the BTF carries a different
    // type under the same name.
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel list_head modifiers");
        return;
    };
    let Type::Struct(_) = ty else {
        crate::report::test_skip("BTF 'list_head' is not a Struct");
        return;
    };
    let Some(size) = type_size(&btf, &ty) else {
        crate::report::test_skip("list_head size unresolved");
        return;
    };
    // Arena addresses for the cycle.
    const ARENA_START: u64 = 0x10_0000_0000;
    const ARENA_END: u64 = 0x10_0001_0000;
    const NODE_A: u64 = 0x10_0000_1000;
    // Bytes: list_head { next = NODE_A, prev = NODE_A } —
    // both fields point back at this same node. The first
    // ptr-chase visits NODE_A; the recursive render of
    // NODE_A's content sees `next` pointing at NODE_A again
    // and the visited-set check fires.
    let mut node_bytes = vec![0u8; size];
    node_bytes[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    node_bytes[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let mut bytes_by_addr = std::collections::HashMap::new();
    bytes_by_addr.insert(NODE_A, node_bytes);
    let reader = CycleArenaReader {
        bytes_by_addr,
        arena_start: ARENA_START,
        arena_end: ARENA_END,
    };

    // Wrap NODE_A in a parent ptr buffer (8 bytes pointing at
    // NODE_A) and render against `list_head *`. Find a
    // `list_head *` type id by scanning all types — if the
    // BTF doesn't expose a typed pointer to list_head as a
    // top-level type id, we exercise the cycle through Struct
    // member rendering instead.
    //
    // Simpler alternative: render the Struct directly with
    // bytes containing back-pointers to the arena. The
    // renderer recurses through the `next`/`prev` fields, both
    // arena-typed pointers, and exercises the cycle path.
    // Render the struct with bytes that point its `next` at
    // an arena address whose stored value points back at the
    // same address. Visit NODE_A from inside the rendered
    // struct.

    // Build outer buffer: a struct list_head whose next/prev
    // both point to NODE_A. The renderer will chase NODE_A
    // (visiting it the first time, inserting into visited),
    // recurse into the rendered struct, and on rendering the
    // inner `next` pointer, see NODE_A is already visited.
    let mut outer = vec![0u8; size];
    outer[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    outer[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let v = render_value_with_mem(&btf, id, &outer, &reader);
    let out = format!("{v}");

    // The output must contain a `[cycle]` marker for at least
    // one pointer in the rendered tree. The exact placement
    // depends on traversal order but the marker must appear.
    assert!(
        out.contains("[cycle]"),
        "rendered output must surface cycle marker for a self-pointing list_head: {out}",
    );
    // The output must NOT recurse 32 levels deep — verify by
    // counting `0x{NODE_A:x}` occurrences. Without cycle
    // detection, the renderer would emit NODE_A's hex many
    // times (once per recursion frame). With detection, the
    // address appears once per pointer site (no duplicate hex
    // inside the `[cycle]` marker, which is now a bare
    // diagnostic with no embedded address). Outer's 2 fields
    // each chase NODE_A once and emit a `[cycle]` for each
    // inner field — total ≤ 6 NODE_A occurrences. Cap at 10
    // to leave a margin without admitting a 32-deep runaway.
    let node_hex = format!("0x{NODE_A:x}");
    let occurrences = out.matches(&node_hex).count();
    assert!(
        occurrences < 10,
        "cycle detection must bound recursion; saw {occurrences} \
         occurrences of {node_hex}: {out}",
    );
}

/// Two-node cycle: NODE_A's `next` → NODE_B; NODE_B's `next` →
/// NODE_A. The renderer chases A, then B, then sees A in the
/// visited set and surfaces the cycle.
#[test]
fn ptr_cycle_two_node_loop_surfaces_cycle_reason() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("list_head") else {
        crate::report::test_skip("BTF missing 'list_head'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'list_head' to empty id list");
        return;
    };
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel list_head modifiers");
        return;
    };
    let Type::Struct(_) = ty else {
        crate::report::test_skip("BTF 'list_head' is not a Struct");
        return;
    };
    let Some(size) = type_size(&btf, &ty) else {
        crate::report::test_skip("list_head size unresolved");
        return;
    };

    const ARENA_START: u64 = 0x10_0000_0000;
    const ARENA_END: u64 = 0x10_0001_0000;
    const NODE_A: u64 = 0x10_0000_1000;
    const NODE_B: u64 = 0x10_0000_2000;

    // NODE_A: next=NODE_B, prev=NODE_B.
    let mut a_bytes = vec![0u8; size];
    a_bytes[0..8].copy_from_slice(&NODE_B.to_le_bytes());
    a_bytes[8..16].copy_from_slice(&NODE_B.to_le_bytes());
    // NODE_B: next=NODE_A, prev=NODE_A.
    let mut b_bytes = vec![0u8; size];
    b_bytes[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    b_bytes[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let mut bytes_by_addr = std::collections::HashMap::new();
    bytes_by_addr.insert(NODE_A, a_bytes);
    bytes_by_addr.insert(NODE_B, b_bytes);
    let reader = CycleArenaReader {
        bytes_by_addr,
        arena_start: ARENA_START,
        arena_end: ARENA_END,
    };

    // Render starting with bytes matching NODE_A's content.
    let mut outer = vec![0u8; size];
    outer[0..8].copy_from_slice(&NODE_B.to_le_bytes());
    outer[8..16].copy_from_slice(&NODE_B.to_le_bytes());

    let v = render_value_with_mem(&btf, id, &outer, &reader);
    let out = format!("{v}");

    // Must surface the cycle marker.
    assert!(
        out.contains("[cycle]"),
        "two-node cycle must surface cycle marker: {out}",
    );
}

/// `render_value_with_mem` constructs a fresh empty visited set
/// for each call. Two independent renders with the same arena
/// reader must each detect their own cycle independently — a
/// stale visited entry from a prior call must not poison a
/// later one.
#[test]
fn ptr_cycle_visited_set_does_not_leak_across_calls() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("list_head") else {
        crate::report::test_skip("BTF missing 'list_head'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'list_head' to empty id list");
        return;
    };
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel list_head modifiers");
        return;
    };
    let Type::Struct(_) = ty else {
        crate::report::test_skip("BTF 'list_head' is not a Struct");
        return;
    };
    let Some(size) = type_size(&btf, &ty) else {
        crate::report::test_skip("list_head size unresolved");
        return;
    };

    const ARENA_START: u64 = 0x10_0000_0000;
    const ARENA_END: u64 = 0x10_0001_0000;
    const NODE_A: u64 = 0x10_0000_1000;

    let mut node_bytes = vec![0u8; size];
    node_bytes[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    node_bytes[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let mut bytes_by_addr = std::collections::HashMap::new();
    bytes_by_addr.insert(NODE_A, node_bytes);
    let reader = CycleArenaReader {
        bytes_by_addr,
        arena_start: ARENA_START,
        arena_end: ARENA_END,
    };

    let mut outer = vec![0u8; size];
    outer[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    outer[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    // Two back-to-back renders. Each must succeed and surface
    // a cycle marker. A leaking visited set from the first
    // call would prevent the second call from chasing NODE_A
    // at all (it would surface the cycle on the OUTER
    // pointer, not the inner one — wrong semantics).
    let v1 = render_value_with_mem(&btf, id, &outer, &reader);
    let out1 = format!("{v1}");
    assert!(out1.contains("[cycle]"), "call 1 cycle: {out1}");

    let v2 = render_value_with_mem(&btf, id, &outer, &reader);
    let out2 = format!("{v2}");
    assert!(out2.contains("[cycle]"), "call 2 cycle: {out2}");

    // Both renders must produce identical output (visited set
    // cleared between them).
    assert_eq!(out1, out2, "fresh visited set per call: outputs must match",);
}

// ---- cast_annotation_for static-string mapping ------------------
//
// `cast_annotation_for` is the single source of truth for the
// operator-visible cast tag emitted on `RenderedValue::Ptr`. A
// 2x2 match over `(AddrSpace, sdt_alloc_resolved)` returns one of
// four `&'static str` literals; the renderer borrows these via
// `Cow::Borrowed` so the annotation costs zero per-chase
// allocations. Because every cast-recovered `Ptr` consumer
// (Display, JSON serializer, downstream operator tooling) keys
// off these exact bytes, drift in any of the four cells is a
// silent operator-visible behavior change.
//
// This test pins all four mappings directly. It is co-located
// with the Cast intercept section below because the integration
// tests assert the SAME strings via `cast_annotation.as_deref()`
// — when one of those tests fails, this one localises the
// regression to the mapping table itself rather than the chase
// pipeline that calls it.

/// Direct-call coverage of every `(AddrSpace, sdt_alloc_resolved)`
/// pair handled by [`cast_annotation_for`]. Asserts the exact
/// `&'static str` returned for each of the four cells:
///
/// - `(Arena, false)` → `"cast→arena"`
/// - `(Arena, true)`  → `"cast→arena (sdt_alloc)"`
/// - `(Kernel, false)`→ `"cast→kernel"`
/// - `(Kernel, true)` → `"cast→kernel (sdt_alloc)"`
///
/// `AddrSpace` is `Copy` so the same enum value is reused for
/// the two `sdt_alloc_resolved` polarities. The match in
/// `cast_annotation_for` is exhaustive over `AddrSpace`, so a
/// new variant fails compilation here AND in production —
/// keeping the operator-visible tag set in lockstep with the
/// analyzer's address-space taxonomy.
#[test]
fn cast_annotation_for_all_four_cells() {
    assert_eq!(
        cast_annotation_for(AddrSpace::Arena, false),
        "cast→arena",
        "(Arena, false) annotation drift",
    );
    assert_eq!(
        cast_annotation_for(AddrSpace::Arena, true),
        "cast→arena (sdt_alloc)",
        "(Arena, true) annotation drift",
    );
    assert_eq!(
        cast_annotation_for(AddrSpace::Kernel, false),
        "cast→kernel",
        "(Kernel, false) annotation drift",
    );
    assert_eq!(
        cast_annotation_for(AddrSpace::Kernel, true),
        "cast→kernel (sdt_alloc)",
        "(Kernel, true) annotation drift",
    );
}

// ---- Cast intercept (render_cast_pointer) ----------------------
//
// `render_member`'s cast intercept fires when:
//   - the parent BTF id is known (we are inside `render_struct`),
//   - a [`MemReader`] is plumbed,
//   - the member peels to a plain unsigned 8-byte Int (not signed,
//     bool, char, or any other size),
//   - [`MemReader::cast_lookup`] returns `Some(hit)` for
//     (parent_btf_id, member_byte_offset),
//   - the parent_bytes slice covers the full 8-byte u64 field.
//
// On hit, [`render_cast_pointer`] dispatches by [`AddrSpace`]:
// arena reads through `read_arena` after `is_arena_addr`; kernel
// reads through `read_kva` and applies the freed-slab plausibility
// gate (top-byte 0xff on the first qword).
//
// The tests below build minimal synthetic BTF blobs (mirroring
// `cast_analysis::tests::build_btf` but pared down to only the
// kinds the renderer's cast path uses: BTF_KIND_INT and
// BTF_KIND_STRUCT) and parse them via `Btf::from_bytes`. Each test
// supplies a stub `MemReader` that returns the canned `CastHit`
// the scenario exercises and observes the resulting
// `RenderedValue` tree directly. Synthetic BTF + stub reader
// keeps these tests independent of vmlinux availability and pins
// the exact intercept gate without any real kernel BTF noise.

const CAST_BTF_MAGIC: u16 = 0xEB9F;
const CAST_BTF_VERSION: u8 = 1;
const CAST_BTF_HEADER_LEN: u32 = 24;
const CAST_BTF_KIND_INT: u32 = 1;
/// `BTF_KIND_PTR` per `btf-rs::obj::resolve` — kind 2 maps to
/// `Type::Ptr`. Used by the Fwd-pointee chase tests so the Type::Ptr
/// arm hits a forward-declared pointee.
const CAST_BTF_KIND_PTR: u32 = 2;
const CAST_BTF_KIND_STRUCT: u32 = 4;
/// `BTF_KIND_FWD` per `btf-rs::obj::resolve` — kind 7 maps to
/// `Type::Fwd`. Used by the Fwd-pointee chase tests; libbpf emits
/// this for structs whose body lives in a separate BTF (e.g.
/// `struct sdt_data` defined in the sdt_alloc library and referenced
/// from a scheduler that doesn't include the full body).
const CAST_BTF_KIND_FWD: u32 = 7;
/// `BTF_KIND_TYPEDEF` per `btf-rs::obj::resolve` — kind 8 maps to
/// `Type::Typedef`. Used by the modifier-chain integration test.
const CAST_BTF_KIND_TYPEDEF: u32 = 8;
/// `BTF_KIND_CONST` per `btf-rs::obj::resolve` — kind 10 maps to
/// `Type::Const`. Used by the modifier-chain integration test.
const CAST_BTF_KIND_CONST: u32 = 10;

/// Build a minimal BTF blob containing `types` (id=1..) and a
/// string-section payload `strings` (must start with `\0`). The
/// header layout matches `cast_analysis::tests::build_btf`:
/// 24-byte header, type section, string section. Only the kinds
/// the renderer's cast intercept exercises (Int, Struct) are
/// supported here.
fn cast_build_btf(types: &[CastSynType], strings: &[u8]) -> Vec<u8> {
    let mut type_section = Vec::new();
    for ty in types {
        match ty {
            CastSynType::Int {
                name_off,
                size,
                encoding,
                offset,
                bits,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_INT << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                type_section.extend_from_slice(&int_data.to_le_bytes());
            }
            CastSynType::Struct {
                name_off,
                size,
                members,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let info = ((CAST_BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    let bit_off = m.byte_offset * 8;
                    type_section.extend_from_slice(&bit_off.to_le_bytes());
                }
            }
            CastSynType::Typedef { name_off, type_id } => {
                // BTF_KIND_TYPEDEF wire layout: name_off (4) + info (4)
                // + size_type (4) where size_type holds the wrapped
                // type id. Per `cbtf::btf_type::kind`, the kind is
                // bits 24..29 of `info`; vlen is 0 for Typedef.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_TYPEDEF << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            CastSynType::Const { type_id } => {
                // BTF_KIND_CONST wire layout: name_off (4, always 0) +
                // info (4) + size_type (4, the wrapped type id). Per
                // the BTF spec, Const types are anonymous so name_off
                // is unused.
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_CONST << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            CastSynType::Ptr { type_id } => {
                // BTF_KIND_PTR wire layout: name_off (4, always 0) +
                // info (4) + size_type (4, the pointee type id). Ptr
                // types are anonymous per the BTF spec.
                let name_off: u32 = 0;
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (CAST_BTF_KIND_PTR << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            CastSynType::Fwd { name_off, is_union } => {
                // BTF_KIND_FWD wire layout: name_off (4) + info (4) +
                // size_type (4, unused — emit 0). Per
                // `btf-rs::Fwd::is_union`, the kind_flag (bit 31 of
                // info) selects struct (0) vs union (1) for the
                // forward declaration's referent.
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let kind_flag = if *is_union { 1u32 << 31 } else { 0 };
                let info = ((CAST_BTF_KIND_FWD << 24) & 0x1f00_0000) | kind_flag;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&0u32.to_le_bytes());
            }
        }
    }

    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;

    let mut blob = Vec::new();
    // Header (24 bytes): magic (2) + version (1) + flags (1)
    // + hdr_len (4) + type_off (4) + type_len (4)
    // + str_off (4) + str_len (4).
    blob.extend_from_slice(&CAST_BTF_MAGIC.to_le_bytes());
    blob.push(CAST_BTF_VERSION);
    blob.push(0); // flags
    blob.extend_from_slice(&CAST_BTF_HEADER_LEN.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes()); // type_off
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes()); // str_off = type_len
    blob.extend_from_slice(&str_len.to_le_bytes());
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(strings);
    blob
}

#[derive(Clone, Copy)]
struct CastSynMember {
    name_off: u32,
    type_id: u32,
    byte_offset: u32,
}

enum CastSynType {
    /// `BTF_KIND_INT`. encoding=0 = plain unsigned (not signed,
    /// not char, not bool — the gate the cast intercept requires).
    Int {
        name_off: u32,
        size: u32,
        encoding: u32,
        offset: u32,
        bits: u32,
    },
    Struct {
        name_off: u32,
        size: u32,
        members: Vec<CastSynMember>,
    },
    /// `BTF_KIND_TYPEDEF` (kind=8). Wraps another type id with a
    /// name. The renderer's [`peel_modifiers_with_id`] peels through
    /// it; the analyzer's [`super::super::bpf_map::resolve_to_struct_id`]
    /// peels through it too. Used by the modifier-chain integration
    /// test to verify both peel paths agree on the underlying
    /// struct id the [`CastMap`] keys on.
    Typedef { name_off: u32, type_id: u32 },
    /// `BTF_KIND_CONST` (kind=10). Anonymous wrapper around another
    /// type id. Same renderer / analyzer peel treatment as Typedef.
    /// `name_off` is always 0 per the BTF spec (Const types are
    /// anonymous), but the field is still emitted for wire-format
    /// completeness.
    Const { type_id: u32 },
    /// `BTF_KIND_PTR` (kind=2). Anonymous pointer-to-`type_id`. Used
    /// to model a Type::Ptr field whose pointee is a forward-
    /// declared aggregate (the scenario the Fwd chase test exercises).
    Ptr { type_id: u32 },
    /// `BTF_KIND_FWD` (kind=7). Forward declaration of a struct
    /// (`is_union: false`) or union (`is_union: true`). Carries a
    /// name but no body — `type_size` returns `None`. Models the
    /// scenario where a scheduler library defines the struct (e.g.
    /// `struct sdt_data` in the sdt_alloc library) and the using
    /// program only references it via pointer; the program's BTF
    /// then carries `Fwd` rather than the full `Struct`.
    Fwd { name_off: u32, is_union: bool },
}

/// Helper: build a string section + name offsets for the names
/// used across cast tests. Returns `(strings, n_int_name, n_t,
/// n_q, n_f, n_x)` where `n_*` are the byte offsets of each name
/// inside the string section.
fn cast_strings_for_t_q() -> (Vec<u8>, u32, u32, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_q = push(&mut strings, "Q");
    let n_f = push(&mut strings, "f");
    let n_x = push(&mut strings, "x");
    (strings, n_int, n_t, n_q, n_f, n_x)
}

/// Build a BTF blob with: id=1 plain-unsigned u64 (size=8,bits=64),
/// id=2 struct T { u64 f at offset 0; } size=8, id=3 struct Q
/// { u64 x at offset 0; } size=8. T_id=2, Q_id=3.
fn cast_btf_t_and_q() -> (Vec<u8>, u32, u32) {
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 2, 3)
}

/// Build a BTF blob where T's intercepted member is u32 (size=4)
/// instead of u64. Used to verify the intercept's size==8 gate
/// rejects sub-u64 fields. id=1: u32 (size=4,bits=32),
/// id=2: struct T { u32 f at offset 0; } size=4. T_id=2 — Q
/// (the cast target) is unused since the gate fires before
/// `cast_lookup`, but we still emit a valid u64 + Q so the
/// fixture covers a hit returned for a hypothetical reader that
/// returns Some despite the size mismatch.
fn cast_btf_t_with_u32() -> (Vec<u8>, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u32 = push(&mut strings, "u32");
    let n_u64 = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_q = push(&mut strings, "Q");
    let n_f = push(&mut strings, "f");
    let n_x = push(&mut strings, "x");

    let types = vec![
        // id 1: u32 plain unsigned.
        CastSynType::Int {
            name_off: n_u32,
            size: 4,
            encoding: 0,
            offset: 0,
            bits: 32,
        },
        // id 2: u64 plain unsigned (Q's field type).
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 3: struct T { u32 f at offset 0; } size=4.
        CastSynType::Struct {
            name_off: n_t,
            size: 4,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: struct Q { u64 x at offset 0; } size=8.
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 2,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 3, 4)
}

/// Stub `MemReader` for cast-intercept tests. Two cast-lookup
/// modes:
///
/// - `cast_map = Some(map)` — looks up
///   `(parent_type_id, member_byte_offset)` in a real
///   [`super::super::cast_analysis::CastMap`] (typically produced by
///   [`super::super::cast_analysis::analyze_casts`]) and returns
///   the matching [`CastHit`]. The integration tests use this mode
///   to wire actual analyzer output into the renderer.
/// - `cast_map = None` — returns the fixed `hit` (or `None` when
///   `hit` is `None`) for every query. The unit tests for the
///   intercept gate use this mode because they only need the
///   intercept to fire or not fire on a single (parent, offset)
///   pair.
///
/// `arena_bytes_at` and `kva_bytes_at` drive the address-space
/// dispatch; tests that don't exercise reads leave the maps empty.
///
/// `arena_type_at` carries the sdt_alloc bridge entries the
/// renderer's [`MemReader::resolve_arena_type`] override consults
/// — `addr → btf_type_id`. Mirrors the production
/// `AccessorMemReader::resolve_arena_type` shape (the `dump/render_map.rs`
/// override masks the address with `0xFFFF_FFFF` and looks up in
/// the per-pass index); the stub keeps full addresses keyed
/// directly so tests can use the actual chased value.
#[derive(Default)]
struct CastStubReader {
    /// Fixed [`CastHit`] returned by `cast_lookup` when `cast_map`
    /// is `None`. Universal-match avoids hand-keying the same
    /// (id, offset) pair into every gate-focused test.
    hit: Option<CastHit>,
    /// Real cast map consulted when `Some`. The
    /// `(parent_type_id, member_byte_offset)` lookup mirrors
    /// `AccessorMemReader::cast_lookup` in `dump/render_map.rs`
    /// (the production path), so the integration tests cover the
    /// same shape.
    cast_map: Option<super::super::cast_analysis::CastMap>,
    arena_window: Option<(u64, u64)>,
    arena_bytes_at: std::collections::HashMap<u64, Vec<u8>>,
    kva_bytes_at: std::collections::HashMap<u64, Vec<u8>>,
    /// `addr → ArenaResolveHit` lookup the stub returns from
    /// [`MemReader::resolve_arena_type`]. Empty by default — the
    /// trait method then surfaces the trait-default `None` for
    /// every query, matching every existing test that does not
    /// exercise the sdt_alloc bridge. The
    /// [`ArenaResolveHit::header_skip`] field is the byte count
    /// the chase must skip from `addr` before the payload struct
    /// begins (0 for payload-start chases, the slot's header size
    /// for slot-start chases) — see
    /// [`MemReader::resolve_arena_type`] for the production
    /// contract.
    arena_type_at: std::collections::HashMap<u64, ArenaResolveHit>,
    /// Owned BTFs the stub holds for cross-BTF Fwd resolution.
    /// `cross_btf_resolve_fwd` returns a borrow into this vec when
    /// `cross_btf_index` has a hit. None / empty disables the
    /// trait method's response (default `None`).
    cross_btf_btfs: Vec<std::sync::Arc<Btf>>,
    /// `name -> (cross_btf_btfs index, type_id, want_struct)` for
    /// cross-BTF Fwd resolution. The `bool` is the
    /// aggregate-kind flag the trait gates on
    /// (`true = Type::Struct`, `false = Type::Union`); a
    /// stored entry only fires when the query's `kind`
    /// matches.
    cross_btf_index: std::collections::HashMap<String, (usize, u32, bool)>,
    /// Set of low-32 windowed slot starts the dump pre-pass would
    /// have already rendered. The
    /// [`MemReader::is_already_rendered`] override returns `true`
    /// when `addr as u32` lies in this set so the chase
    /// short-circuits to a `deref: None` Ptr with the "already
    /// rendered" reason — mirrors the production
    /// `AccessorMemReader` dedup wired through the
    /// `rendered_slot_addrs` field. Empty by default so existing
    /// tests stay untouched.
    rendered_slot_addrs: std::collections::HashSet<u32>,
}

impl MemReader for CastStubReader {
    fn read_kva(&self, kva: u64, len: usize) -> Option<Vec<u8>> {
        let bytes = self.kva_bytes_at.get(&kva)?;
        if bytes.len() < len {
            return None;
        }
        Some(bytes[..len].to_vec())
    }
    fn is_arena_addr(&self, addr: u64) -> bool {
        match self.arena_window {
            Some((lo, hi)) => addr >= lo && addr < hi,
            None => false,
        }
    }
    fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let bytes = self.arena_bytes_at.get(&addr)?;
        if bytes.len() < len {
            return None;
        }
        Some(bytes[..len].to_vec())
    }
    fn cast_lookup(&self, parent_type_id: u32, member_byte_offset: u32) -> Option<CastHit> {
        // CastMap mode: look up (parent, offset) in the analyzer's
        // output. Mirrors the production `AccessorMemReader::cast_lookup`
        // so the integration tests cover the same key/value shape.
        if let Some(map) = &self.cast_map {
            return map.get(&(parent_type_id, member_byte_offset)).copied();
        }
        // Fixed-hit mode (default): return the canned hit
        // regardless of (parent, offset). Used by gate-focused
        // unit tests above.
        self.hit
    }
    fn resolve_arena_type(&self, addr: u64) -> Option<ArenaResolveHit> {
        self.arena_type_at.get(&addr).copied()
    }
    fn cross_btf_resolve_fwd(
        &self,
        name: &str,
        kind: super::FwdKind,
    ) -> Option<super::CrossBtfRef<'_>> {
        let &(idx, type_id, idx_is_struct) = self.cross_btf_index.get(name)?;
        let idx_kind = super::FwdKind::from_is_struct(idx_is_struct);
        if idx_kind != kind {
            return None;
        }
        let btf = self.cross_btf_btfs.get(idx)?;
        Some(super::CrossBtfRef {
            btf: btf.as_ref(),
            type_id,
        })
    }
    fn is_already_rendered(&self, addr: u64) -> bool {
        self.rendered_slot_addrs.contains(&(addr as u32))
    }
}

/// Arena cast hit on a u64 member: render_cast_pointer chases
/// the value through `read_arena` and surfaces a Ptr whose
/// `deref` carries the rendered target struct. The outer
/// rendered Struct member is `Ptr{ value, deref: Some(...) }` —
/// not `Uint`, which is the no-intercept default.
#[test]
fn cast_intercept_u64_renders_as_ptr_with_chase() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Outer T bytes: u64 at offset 0 = 0x10_0000_1000 (an arena
    // address inside the configured window). Inner Q bytes at
    // that arena address: u64 at offset 0 = 0x42 (a plain
    // counter-shaped value — passes the kernel plausibility
    // gate, though that gate is irrelevant here since the
    // address space is Arena).
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected Struct render, got {v:?}");
    };
    assert_eq!(type_name.as_deref(), Some("T"));
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "f");
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(
        value, TARGET_ADDR,
        "Ptr value must be the loaded u64 (arena address)"
    );
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase: no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("chase succeeded → deref must be Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the rendered Q Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "inner deref Struct must carry the target's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "x");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42);
}

/// Null-pointer guard: `value == 0` short-circuits before any
/// `is_arena_addr` / `read_arena` call. Output is
/// `Ptr{ value: 0, deref: None, deref_skipped_reason: None }` —
/// no chase attempted, no skip reason emitted (matching the
/// `Type::Ptr` arm's null handling).
#[test]
fn cast_intercept_null_value_no_crash() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Outer T bytes: u64 at offset 0 = 0. The reader is
    // configured with no arena window and no canned bytes; if
    // the renderer ever called is_arena_addr or read_arena it
    // would short-circuit on `false` / `None`, but the null
    // guard fires first and neither is reached.
    let outer_bytes = 0u64.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "null intercept must still surface as Ptr (matches Type::Ptr arm); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, 0);
    assert!(deref.is_none(), "null Ptr has no deref");
    assert!(
        deref_skipped_reason.is_none(),
        "null Ptr must NOT carry a skip reason: a chase was never attempted"
    );
}

/// Size gate: a u32 (size=4) member with `cast_lookup` returning
/// `Some(hit)` is NOT intercepted. The gate
/// `int.size() != 8` fires before `cast_lookup` is called and
/// the renderer falls through to the normal Int render, producing
/// `RenderedValue::Uint{ bits: 32, .. }`. This is the structural
/// invariant the cast intercept relies on: BPF stores recovered
/// typed pointers in u64 slots only.
#[test]
fn cast_intercept_non_u64_field_not_intercepted() {
    let (blob, t_id, q_id) = cast_btf_t_with_u32();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    let outer_bytes = 0xCAFEu32.to_le_bytes().to_vec();
    // Reader returns Some(hit) for any (parent, offset) — the
    // gate must reject before this is consulted.
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "u32 field with size==4 must render as Uint, NOT Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(bits, 32, "u32 surfaces as 32-bit Uint");
    assert_eq!(value, 0xCAFE);
}

/// `MemReader::cast_lookup` returns `None` for every
/// `(parent, off)`; the cast intercept short-circuits on the
/// `cast_lookup` gate and the renderer takes the pre-existing
/// path. A u64 member surfaces as `Uint{ bits: 64, value }` —
/// same as before the cast intercept landed. A regression that
/// fired the intercept for None-returning readers would surface
/// here as a Ptr render, breaking every reader that doesn't
/// override `cast_lookup`.
#[test]
fn cast_intercept_no_hit_renders_uint() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    let outer_bytes = 0x12345678u64.to_le_bytes().to_vec();
    // CastStubReader::default() leaves `hit` as None — its
    // `cast_lookup` returns None for every (parent, offset). This
    // mirrors the "no override" default trait method behavior.
    let reader = CastStubReader::default();

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "no cast_lookup hit must yield plain Uint, got {:?}",
            members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x12345678);
}

/// Self-cycle through cast-recovered pointers: T at the outer
/// bytes contains a u64 whose cast hit chases another T at the
/// SAME arena address. `render_cast_pointer` inserts the value
/// into the visited set, recurses into the inner T render, and
/// the inner u64's cast intercept hits the visited check and
/// surfaces `deref_skipped_reason` containing "cycle".
///
/// Without the visited check, the chase would recurse until
/// `MAX_RENDER_DEPTH` (32), producing a deep nest of Ptr -> Ptr
/// -> ... in the failure dump.
#[test]
fn cast_chase_cycle_detection() {
    // Use T with a self-cycle: T contains a u64 at offset 0 that
    // points to a T-shaped instance whose own u64 at offset 0
    // points back to itself.
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const SELF_ADDR: u64 = 0x10_0000_1000;
    // Outer T bytes: u64 at offset 0 = SELF_ADDR.
    let outer_bytes = SELF_ADDR.to_le_bytes().to_vec();
    // Bytes at SELF_ADDR: u64 at offset 0 = SELF_ADDR (loop).
    let self_bytes = SELF_ADDR.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(SELF_ADDR, self_bytes);
    let reader = CastStubReader {
        // Target T (id=2) so the inner render recurses into the
        // same shape and the inner u64 also gets the cast
        // intercept (cast_lookup returns the same hit
        // regardless of parent id).
        hit: Some(CastHit { alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    // (outer_bytes is little-endian per the renderer's wire format
    // assumption; SELF_ADDR.to_le_bytes() above produces those.)
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    // Outer Ptr: chase succeeded once (visited was empty), so
    // deref is Some(inner T struct), no skip reason.
    let RenderedValue::Ptr {
        value: outer_value,
        deref: ref outer_deref,
        deref_skipped_reason: ref outer_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "outer chase must surface as Ptr, got {:?}",
            members[0].value
        );
    };
    assert_eq!(outer_value, SELF_ADDR);
    assert!(
        outer_reason.is_none(),
        "outer chase succeeded; no skip reason expected, got {outer_reason:?}"
    );
    let inner = outer_deref.as_deref().expect("outer chase deref Some");
    // Inner is the rendered T at SELF_ADDR. Its `f` member must
    // surface the cycle (visited contains SELF_ADDR by the time
    // the inner render reaches it).
    let RenderedValue::Struct {
        members: ref inner_members,
        ..
    } = *inner
    else {
        panic!("inner deref must be a Struct, got {inner:?}");
    };
    let RenderedValue::Ptr {
        value: inner_value,
        deref: ref inner_deref,
        deref_skipped_reason: ref inner_reason,
        ..
    } = inner_members[0].value
    else {
        panic!(
            "inner u64 cast intercept must surface as Ptr, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(inner_value, SELF_ADDR);
    assert!(
        inner_deref.is_none(),
        "cycle detection must NOT recurse into the deref payload"
    );
    let reason = inner_reason
        .as_deref()
        .expect("cycle detection must populate deref_skipped_reason");
    assert!(
        reason.contains("cycle"),
        "skip reason must mention cycle, got: {reason}"
    );
}

/// Kernel cast hit whose `read_kva` returns 8 bytes whose first
/// qword has top byte 0xff — the freed-slab freelist-pointer
/// signature on x86_64 / aarch64. `render_cast_pointer`'s
/// plausibility gate rejects the read and surfaces
/// `deref_skipped_reason` mentioning "plausibility". The rendered
/// Ptr's `deref` is `None` (chase attempted but rejected).
#[test]
fn cast_chase_kernel_plausibility_rejects_freed_slab() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const KVA: u64 = 0xffff_8000_dead_beef;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Bytes at KVA: first qword has top byte 0xff (matches the
    // SLAB_FREELIST_HARDENED-defeating heuristic that the
    // plausibility gate enforces). Use a value whose top byte is
    // 0xff but isn't structurally a real kernel address; the gate
    // doesn't care about the lower bits, only the top byte.
    let stale_bytes: Vec<u8> = 0xff00_0000_0000_0001u64.to_le_bytes().to_vec();

    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, stale_bytes);
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        }),
        kva_bytes_at: kva_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "kernel cast intercept must surface as Ptr, got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "plausibility-rejected chase must NOT carry a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("plausibility rejection must populate skip reason");
    assert!(
        reason.contains("plausibility"),
        "skip reason must mention plausibility, got: {reason}"
    );
}

/// Runtime dispatch wins over the analyzer's address-space hint:
/// a `CastHit` whose `addr_space` is `Kernel` but whose value
/// falls inside the arena window must still chase via the arena
/// reader (not `read_kva`), and the `cast_annotation` must
/// reflect the path actually taken (`"cast→arena"`).
///
/// Per [`render_cast_pointer`]'s runtime address-space dispatch:
/// "Address-space dispatch is RUNTIME-driven: `is_arena_addr` is
/// consulted on the actual pointer value to decide whether to
/// chase via `read_arena` (in-window) or `read_kva` (out-of-window).
/// The `CastHit::addr_space` tag from the analyzer is treated as
/// a hint only — runtime evidence from the pointer value is
/// authoritative." This test pins that contract: a Kernel-hinted
/// hit with an arena-window value goes through the arena path.
#[test]
fn cast_intercept_kernel_hint_arena_value_dispatches_to_arena_reader() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Configure: arena window covers [ARENA_LO, ARENA_HI); the
    // pointer value (TARGET_ADDR) falls inside that window even
    // though the analyzer's hint says Kernel. The arena reader
    // has bytes for TARGET_ADDR; the kva reader is intentionally
    // empty so a wrongly-routed kernel chase would surface as a
    // skip reason rather than a successful chase.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    // CastMap mode (key-specific) so the inner Q.x render
    // does NOT re-trigger the cast intercept on a (Q, 0) lookup —
    // only the outer (T, 0) key has an entry. The hit-on-every-
    // query mode would chase Q.x through the kernel path because
    // 0x42 is not in the arena window, surfacing a spurious
    // failure that has nothing to do with the runtime-vs-hint
    // dispatch this test pins.
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            // Kernel hint — but value is an arena address. Runtime
            // detection is_arena_addr(value) returns true → arena
            // reader fires regardless of the hint.
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        // kva_bytes_at is intentionally empty — read_kva would
        // return None for any address. If runtime dispatch ever
        // routed a Kernel-hinted but in-arena value through the
        // kernel path, this test would surface a skip reason.
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "Kernel-hint + arena-value cast must surface as Ptr, got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR, "Ptr value is the loaded u64");
    assert!(
        deref_skipped_reason.is_none(),
        "arena dispatch chose the arena reader → no skip reason; got {deref_skipped_reason:?}",
    );
    let inner = deref
        .as_deref()
        .expect("arena reader returned Some bytes → deref payload populated");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!(
            "deref payload must be the rendered Q struct (proves arena chase, \
             not kernel chase, did the read), got {inner:?}",
        );
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "inner deref carries Q's name → render_value_inner(target_type_id) succeeded",
    );
    assert_eq!(inner_members.len(), 1, "Q has one u64 member");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!(
            "Q.x must render as Uint (was rendered through arena reader bytes), got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42, "arena reader returned 0x42 at TARGET_ADDR");
    // The annotation must reflect the path actually taken (arena),
    // not the analyzer's hint (Kernel). A regression that emitted
    // "cast→kernel" here would mean the renderer fell back to the
    // kernel arm but somehow succeeded, which is impossible without
    // canned kva bytes — but the annotation pins the dispatch
    // outcome explicitly so the contract isn't merely inferred.
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "runtime dispatch chose arena → annotation is cast→arena, NOT cast→kernel; \
         got {cast_annotation:?}",
    );
}

// ---- Cast pipeline integration ---------------------------------
//
// The tests above stub `cast_lookup` with a fixed [`CastHit`]. The
// integration tests below run real
// [`super::super::cast_analysis::analyze_casts`] over a synthetic
// BPF program, feed the resulting [`super::super::cast_analysis::CastMap`]
// into [`CastStubReader::cast_map`], and render a struct against
// the analyzer's actual output. The point: verify the
// `(parent_type_id, member_byte_offset)` keys the analyzer emits
// match the keys [`render_member`]'s cast intercept queries via
// [`MemReader::cast_lookup`]. A drift between analyzer and
// renderer (e.g. analyzer keys on a typedef-wrapped surface id,
// renderer queries with the peeled struct id) would surface as a
// missed intercept here even though both halves work in
// isolation.
//
// The fixtures use the same [`cast_build_btf`] /
// [`CastSynType`] machinery the gate-focused tests above use,
// extended with `Typedef` and `Const` for the modifier-chain case.
// BPF instruction encoding goes through [`BpfInsn::new`]; opcode
// constants come from `libbpf_rs::libbpf_sys` (the same source
// the analyzer's own private constants use, so the test
// instruction stream stays in lock-step with the analyzer's
// decode tables).

use super::super::cast_analysis::{BpfInsn, CastMap, InitialReg, analyze_casts};

/// `BPF_LDX | BPF_DW | BPF_MEM` in the [`BpfInsn::code`] byte. The
/// arena-cast LDX shape the analyzer's `handle_ldx` matches: load
/// 8 bytes through a typed pointer base. Constants pulled from
/// `libbpf_rs::libbpf_sys` so the test encoding stays in lock-step
/// with the analyzer's own decode tables.
fn cast_ldx_dw_mem_code() -> u8 {
    use libbpf_rs::libbpf_sys as bs;
    (bs::BPF_LDX | bs::BPF_DW | bs::BPF_MEM) as u8
}

/// `BPF_JMP | BPF_EXIT` in the [`BpfInsn::code`] byte. Terminator
/// for synthetic programs.
fn cast_exit_code() -> u8 {
    use libbpf_rs::libbpf_sys as bs;
    (bs::BPF_JMP | bs::BPF_EXIT) as u8
}

/// One-shot helper: emit `r{dst} = *(u64 *)(r{src} + off)` as a
/// single [`BpfInsn`] for cast-integration tests.
fn cast_ldx_dw(dst: u8, src: u8, off: i16) -> BpfInsn {
    BpfInsn::new(cast_ldx_dw_mem_code(), dst, src, off, 0)
}

/// One-shot helper: emit `exit` as a single [`BpfInsn`].
fn cast_exit() -> BpfInsn {
    BpfInsn::new(cast_exit_code(), 0, 0, 0, 0)
}

/// One-shot helper: emit `BPF_ADDR_SPACE_CAST` (ALU64 | MOV | X
/// with `off=1`). The analyzer treats `imm=1` as the as(1)→as(0)
/// cast (arena→kernel), which adds the source's `(struct,
/// field_offset)` to `arena_confirmed` — the F1 mitigation
/// prerequisite for shape-inference findings.
fn cast_addr_space_cast(dst: u8, src: u8, imm: i32) -> BpfInsn {
    use libbpf_rs::libbpf_sys as bs;
    let code = (bs::BPF_ALU64 | bs::BPF_MOV | bs::BPF_X) as u8;
    BpfInsn::new(code, dst, src, 1, imm)
}

/// Build a BTF blob shaped to drive [`analyze_casts`] to a unique
/// (T, 8) → (Q, Arena) finding when run against the canonical
/// `r2 = T.f; r3 = *r2` pair. Layout:
///
///   id=1: u64 (size 8, plain unsigned)
///   id=2: struct T { u64 f @ offset 8 }, size 16
///   id=3: struct Q { u64 x @ offset 0 }, size 8
///
/// T's u64 lives at offset 8 (not 0), so the access pattern
/// `(offset=0, size=8)` from `*r2` only matches Q in the layout
/// index — the analyzer's source-removal step doesn't fire (T is
/// not in the candidate set), so the single-candidate condition
/// emits the entry.
fn cast_btf_t_at_offset_8_q_at_offset_0() -> (Vec<u8>, u32, u32) {
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        // id 1: u64 plain unsigned.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 f @ 8 }, size 16. Offset 8 keeps T
        // out of the (offset=0, size=8) layout-index bucket, so the
        // analyzer's intersection collapses to {Q} cleanly.
        CastSynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: struct Q { u64 x @ 0 }, size 8.
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 2, 3)
}

/// First true integration test: real [`analyze_casts`] output drives
/// the renderer's cast intercept end-to-end. Verifies that the
/// `(parent_type_id, member_byte_offset)` keys the analyzer emits
/// match the keys [`render_member`] queries via
/// [`MemReader::cast_lookup`], and that the renderer chases the
/// recovered target through [`render_cast_pointer`] to produce a
/// `Ptr` with a populated `deref`.
///
/// A drift between analyzer key format and renderer query format
/// would surface here as a missed intercept (the u64 field
/// rendered as `Uint` instead of `Ptr`), even though both halves
/// work in isolation.
#[test]
fn cast_pipeline_analyzer_output_drives_renderer_intercept() {
    let (blob, t_id, q_id) = cast_btf_t_at_offset_8_q_at_offset_0();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // BPF program:
    //   r2 = *(u64 *)(r1 + 8)   ; load T.f → r2 = LoadedU64Field{T, 8}
    //   r2 = arena_cast(r2)     ; arena_confirmed evidence (F1)
    //   r3 = *(u64 *)(r2 + 0)   ; deref @0 records access (0, 8) under (T, 8)
    //   exit
    let insns = vec![
        cast_ldx_dw(2, 1, 8),
        cast_addr_space_cast(2, 2, 1),
        cast_ldx_dw(3, 2, 0),
        cast_exit(),
    ];
    let cast_map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    // Analyzer must produce exactly one finding: (T, 8) → (Q, Arena).
    assert_eq!(
        cast_map.get(&(t_id, 8)),
        Some(&CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "analyzer must emit (T, 8) → (Q, Arena); got: {cast_map:?}"
    );

    // Render T's bytes (16 bytes: 8 padding + arena address at
    // offset 8) with the analyzer's CastMap as the lookup source.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let mut outer_bytes = vec![0u8; 16];
    outer_bytes[8..16].copy_from_slice(&TARGET_ADDR.to_le_bytes());
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected outer Struct render, got {v:?}");
    };
    assert_eq!(type_name.as_deref(), Some("T"));
    assert_eq!(members.len(), 1, "T has a single u64 member at offset 8");
    assert_eq!(members[0].name, "f");
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "(T, 8) cast hit must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR, "Ptr value must be the loaded u64");
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase must carry no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("chase succeeded → deref must be Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the rendered Q struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "deref payload must carry Q's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "x");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42);
}

/// Modifier-chain integration: render a `Typedef → Const → Struct(T)`
/// surface type. The renderer's [`peel_modifiers_with_id`] must
/// peel both wrappers and emit `parent_type_id = T_id` (the
/// underlying struct) so [`MemReader::cast_lookup`] queries with
/// the same id the analyzer keyed on. Catches the fragile coupling
/// where a future change to one peel path (renderer or analyzer)
/// drifts away from the other.
#[test]
fn cast_pipeline_modifier_chain_renderer_peels_to_analyzer_struct_id() {
    // Layout extends `cast_btf_t_at_offset_8_q_at_offset_0` with two
    // modifier wrappers: id=4 = Const(T), id=5 = Typedef(Const(T)).
    // The analyzer still keys on T_id=2 (struct id) because both
    // `bpf_map::resolve_to_struct_id` (used by InitialReg seeding)
    // and `peel_modifiers_with_id` (used by render_struct) collapse
    // the wrapper chain.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    // Add a typedef name distinct from the existing strings so the
    // wrapper carries an identifiable name on the wire (the
    // renderer doesn't surface it because peel collapses the
    // wrapper, but the BTF blob round-trips correctly).
    let mut strings = strings;
    let n_typedef = push(&mut strings, "T_alias");
    let types = vec![
        // id 1..3 mirror cast_btf_t_at_offset_8_q_at_offset_0.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: const(T) — wraps T_id=2.
        CastSynType::Const { type_id: 2 },
        // id 5: typedef T_alias = const(T) — wraps id 4. Render via
        // this id; peel_modifiers_with_id collapses 5→4→2.
        CastSynType::Typedef {
            name_off: n_typedef,
            type_id: 4,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;
    let typedef_id: u32 = 5;

    // Analyzer is run against the SAME BTF; the wrappers are inert
    // for analyze_casts (they're not Struct/Union, so the layout
    // index skips them). InitialReg seeds with the typedef wrapper
    // to verify the analyzer's `resolve_to_struct_id` peels through
    // the same chain the renderer does.
    // F1 mitigation: include arena_space_cast on r2 to
    // populate arena_confirmed for (T, 8) so the shape-inference
    // finding emits.
    let insns = vec![
        cast_ldx_dw(2, 1, 8),
        cast_addr_space_cast(2, 2, 1),
        cast_ldx_dw(3, 2, 0),
        cast_exit(),
    ];
    let cast_map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: typedef_id,
        }],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        cast_map.get(&(t_id, 8)),
        Some(&CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena
        }),
        "analyzer must peel typedef→const→struct and key on T_id={t_id}; got: {cast_map:?}"
    );

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let mut outer_bytes = vec![0u8; 16];
    outer_bytes[8..16].copy_from_slice(&TARGET_ADDR.to_le_bytes());
    let inner_bytes = 0x99u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    // Render via the typedef id — the renderer's peel must produce
    // parent_type_id=T_id so the cast lookup hits.
    let v = render_value_with_mem(&btf, typedef_id, &outer_bytes, &reader);
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected Struct render after peel, got {v:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("T"),
        "renderer must collapse typedef/const wrappers to underlying T name"
    );
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "modifier-chain peel must reach the cast intercept; got {:?}. \
             A failure here means the renderer's peel diverges from the \
             analyzer's — the integration is broken.",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(deref_skipped_reason.is_none());
    let inner = deref.as_deref().expect("chase deref Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        ..
    } = *inner
    else {
        panic!("deref payload must be Q struct, got {inner:?}");
    };
    assert_eq!(inner_name.as_deref(), Some("Q"));
}

/// Multi-field integration: a struct with three u64 fields where
/// only two are flagged by the analyzer must render the flagged
/// fields as `Ptr` and the third as `Uint`. The point: per-member
/// cast lookup is independent — a hit on one offset must not
/// promote unrelated u64 fields.
#[test]
fn cast_pipeline_multi_field_only_flagged_offsets_render_as_ptr() {
    // Layout: T has u64 fields at offsets 0, 8, 16. Q is a generic
    // 8-byte target (u64 @ 0).
    //   id=1: u64
    //   id=2: struct T { u64 f0 @ 0; u64 f1 @ 8; u64 f2 @ 16 }, size 24
    //   id=3: struct Q { u64 x @ 0 }, size 8
    let (strings, n_int, n_t, n_q, _n_f, n_x) = cast_strings_for_t_q();
    // Add three field names distinct from `f`/`x`.
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_f0 = push(&mut strings, "f0");
    let n_f1 = push(&mut strings, "f1");
    let n_f2 = push(&mut strings, "f2");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 24,
            members: vec![
                CastSynMember {
                    name_off: n_f0,
                    type_id: 1,
                    byte_offset: 0,
                },
                CastSynMember {
                    name_off: n_f1,
                    type_id: 1,
                    byte_offset: 8,
                },
                CastSynMember {
                    name_off: n_f2,
                    type_id: 1,
                    byte_offset: 16,
                },
            ],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    // Build a CastMap by hand with entries for offsets 0 and 8
    // ONLY (NOT 16). This mirrors a partial-coverage analyzer
    // result: some u64 fields recovered, others missed (false
    // negatives are the safe direction for the analyzer).
    let mut cast_map: CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    cast_map.insert(
        (t_id, 8),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );

    // Outer T bytes: arena addresses at offsets 0 and 8, plain
    // u64 counter at offset 16. The renderer must NOT chase the
    // counter even though it could be misinterpreted as an arena
    // address (it falls in the arena window for stress purposes).
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const ADDR_F0: u64 = 0x10_0000_1000;
    const ADDR_F1: u64 = 0x10_0000_2000;
    const COUNTER_F2: u64 = 0x10_0000_3000;
    let mut outer_bytes = vec![0u8; 24];
    outer_bytes[0..8].copy_from_slice(&ADDR_F0.to_le_bytes());
    outer_bytes[8..16].copy_from_slice(&ADDR_F1.to_le_bytes());
    outer_bytes[16..24].copy_from_slice(&COUNTER_F2.to_le_bytes());

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(ADDR_F0, 0xAAu64.to_le_bytes().to_vec());
    arena_bytes.insert(ADDR_F1, 0xBBu64.to_le_bytes().to_vec());
    // Note: deliberately NO entry at COUNTER_F2 — even if the
    // intercept were buggy and fired for f2, the chase would
    // surface a `read_arena returned None` skip reason rather
    // than a deref payload. The test's primary check is that f2
    // renders as Uint (no intercept attempted).
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    assert_eq!(members.len(), 3, "T has three u64 members");
    assert_eq!(members[0].name, "f0");
    assert_eq!(members[1].name, "f1");
    assert_eq!(members[2].name, "f2");

    // f0 (cast hit at offset 0): Ptr with deref Some.
    let RenderedValue::Ptr {
        value: f0_value,
        ref deref,
        ..
    } = members[0].value
    else {
        panic!(
            "f0 (offset 0) must render as Ptr (cast map hit); got {:?}",
            members[0].value
        );
    };
    assert_eq!(f0_value, ADDR_F0);
    assert!(deref.is_some(), "f0 chase must succeed (deref Some)");

    // f1 (cast hit at offset 8): Ptr with deref Some.
    let RenderedValue::Ptr {
        value: f1_value,
        ref deref,
        ..
    } = members[1].value
    else {
        panic!(
            "f1 (offset 8) must render as Ptr (cast map hit); got {:?}",
            members[1].value
        );
    };
    assert_eq!(f1_value, ADDR_F1);
    assert!(deref.is_some(), "f1 chase must succeed (deref Some)");

    // f2 (NO cast entry at offset 16): plain Uint counter.
    let RenderedValue::Uint {
        bits: f2_bits,
        value: f2_value,
    } = members[2].value
    else {
        panic!(
            "f2 (offset 16) must render as Uint (no cast map entry); \
             got {:?}. A failure here means a hit on one offset is \
             contaminating unrelated offsets in the same struct.",
            members[2].value
        );
    };
    assert_eq!(f2_bits, 64);
    assert_eq!(f2_value, COUNTER_F2);
}

/// Empty-CastMap integration: a [`CastMap`] with no entries (the
/// no-cast case for a scheduler whose program contains zero
/// recovered casts) must leave every u64 field rendering as a
/// plain `Uint`. Verifies the lookup miss path through the real
/// `BTreeMap::get` returns `None` exactly as the trait-default
/// `cast_lookup` does, so deploying an empty analyzer result is
/// behaviorally indistinguishable from no analyzer at all.
#[test]
fn cast_pipeline_empty_cast_map_renders_uint() {
    let (blob, t_id, _q_id) = cast_btf_t_at_offset_8_q_at_offset_0();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Empty CastMap. cast_lookup must return None for every query.
    let cast_map: CastMap = std::collections::BTreeMap::new();
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        ..Default::default()
    };

    // T has u64 at offset 8; render with a counter-shaped value
    // there. With no cast entries, the field must render as a
    // plain Uint regardless of the value's numeric range.
    let mut outer_bytes = vec![0u8; 16];
    outer_bytes[8..16].copy_from_slice(&0xCAFE_F00Du64.to_le_bytes());
    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "empty cast map must leave u64 as Uint; got {:?}. A \
             failure here means an empty BTreeMap is being treated \
             as 'wildcard hit' instead of 'no hits' — a regression \
             that would promote every u64 to a phantom pointer.",
            members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0xCAFE_F00D);
}

/// Wrong-struct integration: a [`CastMap`] entry keyed on a
/// struct id DIFFERENT from the one being rendered must NOT
/// fire. The renderer queries `cast_lookup` with the parent
/// struct's id; an entry for an unrelated struct (even at the
/// same byte offset, with the same target type) is a miss.
/// Catches the regression where the lookup ignored
/// `parent_type_id` and matched on offset alone.
#[test]
fn cast_pipeline_wrong_struct_id_does_not_intercept() {
    // Layout: two distinct structs with identical shape (u64 @ 0).
    // The CastMap entry is keyed on U; the test renders T. The
    // intercept must NOT fire on T even though T's offset-0 u64
    // is structurally identical to U's.
    //   id=1: u64
    //   id=2: struct T { u64 f @ 0 }, size 8
    //   id=3: struct Q { u64 x @ 0 }, size 8 (cast target)
    //   id=4: struct U { u64 g @ 0 }, size 8 (the "wrong" parent)
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_u = push(&mut strings, "U");
    let n_g = push(&mut strings, "g");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: T { u64 f @ 0 }
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: Q { u64 x @ 0 } (cast target — present so the
        // CastMap entry references a real id, even though the
        // intercept must not fire for this test).
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: U { u64 g @ 0 } — the unrelated parent the cast
        // map is keyed on.
        CastSynType::Struct {
            name_off: n_u,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_g,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;
    let u_id: u32 = 4;

    // CastMap entry keyed on (U, 0) — NOT T. Rendering T must
    // miss the lookup.
    let mut cast_map: CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (u_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );

    // Reader is configured with an arena window and bytes for the
    // value, so a buggy renderer that ignored parent_type_id and
    // intercepted anyway would chase successfully — surfacing the
    // bug as a Ptr render. The correct renderer treats the field
    // as Uint.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const VAL: u64 = 0x10_0000_1000;
    let outer_bytes = VAL.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(VAL, 0x77u64.to_le_bytes().to_vec());
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "(T, 0) must miss the (U, 0) cast entry → render as \
             Uint; got {:?}. A failure here means cast_lookup is \
             ignoring parent_type_id, which would promote every \
             u64 at the entry's offset across every struct in the \
             scheduler.",
            members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, VAL);
}

// ---- render_cast_pointer runtime address-space dispatch --------
//
// `render_cast_pointer` dispatches on [`MemReader::is_arena_addr`]
// against the actual pointer value, not the analyzer's `AddrSpace`
// hint: an in-window value enters [`chase_arena_pointer`], an
// out-of-window value enters the kernel arm regardless of the hint
// (the hint is preserved in `cast_annotation` only when the chase
// could not be performed). These tests pin the runtime fallthrough
// so a regression that re-introduced an early "outside arena
// window" skip would surface here.

/// Arena cast hit whose value falls OUTSIDE the configured arena
/// window: the runtime `is_arena_addr` check in
/// [`render_cast_pointer`] is false, so the renderer falls through
/// to the kernel arm via runtime address-space detection. With no
/// kva entry configured, `read_kva` returns None and the renderer
/// surfaces `Ptr{ deref: None, deref_skipped_reason:
/// Some("kernel read_kva failed at 0x...") }` with
/// `cast_annotation: Some("cast→kernel")`. Pins the runtime
/// fallthrough — a regression that re-introduced an early "outside
/// arena window" skip would block legitimate kernel chases
/// whenever the analyzer's `AddrSpace` tag drifted.
#[test]
fn cast_chase_arena_hint_with_non_arena_value_falls_through_to_kernel_arm() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    // Value below the arena window; `is_arena_addr` returns false,
    // dispatching to the kernel arm. No kva entry → `read_kva`
    // fails, surfacing as a labelled skip with cast→kernel
    // annotation.
    const OUT_OF_WINDOW: u64 = 0x0F_FFFF_FFFF;
    let outer_bytes = OUT_OF_WINDOW.to_le_bytes().to_vec();
    let mut cast_map: super::super::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };
    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!("must surface as Ptr; got {:?}", members[0].value);
    };
    assert_eq!(value, OUT_OF_WINDOW);
    assert!(deref.is_none());
    let reason = deref_skipped_reason.as_deref().expect("skip reason");
    // Runtime fallthrough → kernel arm; read_kva has no entry, so
    // the kernel-read-failed reason fires. The reason includes the
    // "(cast analysis may have flagged a non-pointer field)" suffix
    // because the original hint was Arena, indicating the analyzer
    // may have mis-tagged the slot.
    assert!(
        reason.contains("read_kva failed"),
        "skip reason must mention 'read_kva failed' (kernel arm); got: {reason}"
    );
    assert!(
        reason.contains("cast analysis may have flagged"),
        "Arena→kernel runtime dispatch must annotate suffix; got: {reason}"
    );
    // cast_annotation reflects the runtime decision (kernel), not
    // the analyzer's hint (Arena).
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→kernel"),
        "runtime kernel dispatch must produce cast→kernel annotation"
    );
}

// ---- render_cast_pointer kernel-arm skip-reason paths ----------
//
// `render_cast_pointer`'s kernel arm walks four pre-read gates:
// (1) target type peels (via `peel_modifiers_resolving_fwd`), (2)
// `type_size` returns Some, (3) size != 0, (4) `read_kva` returns
// bytes. Each failure surfaces a distinctly
// worded `deref_skipped_reason`. A regression that collapsed two
// gates into one — or skipped a gate entirely — would produce a
// chase against an unresolved or zero-sized target, with the
// rendered output silently degrading to garbage. These tests pin
// each reason string so every failure path stays distinguishable.

/// Kernel cast target whose `target_type_id` does not resolve to any
/// type in the BTF — `peel_modifiers` returns `None`. The renderer
/// surfaces `Ptr{ deref: None, deref_skipped_reason: Some("kernel
/// cast target type id N unresolvable") }` from
/// [`render_cast_pointer`]'s kernel-arm peel gate (the
/// `peel_modifiers_resolving_fwd` call that precedes the
/// `try_sdt_alloc_bridge` and size resolution).
/// Without this guard, a corrupt or stale CastMap entry pointing at
/// a freed type id would propagate `None` further down and surface
/// as the same "kernel read_kva failed" reason that genuine read
/// failures use — collapsing two distinct failure modes into one.
#[test]
fn cast_chase_kernel_target_type_id_unresolvable() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Use an id beyond every type emitted by `cast_btf_t_and_q` (it
    // produces ids 1..=3, so 9999 is safely out of range and
    // `btf.resolve_type_by_id` errors → peel_modifiers → None).
    const UNRESOLVABLE: u32 = 9999;
    const KVA: u64 = 0xffff_8000_0000_1000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: UNRESOLVABLE,
            addr_space: AddrSpace::Kernel,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "unresolvable target must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "unresolvable target must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("peel_modifiers failure must populate skip reason");
    assert!(
        reason.contains("unresolvable"),
        "skip reason must mention 'unresolvable'; got: {reason}"
    );
    assert!(
        reason.contains(&UNRESOLVABLE.to_string()),
        "skip reason must include the offending type id; got: {reason}"
    );
}

/// Kernel cast target whose BTF size is 0 (a struct with `size_type
/// = 0` — represents an incomplete forward declaration the BPF
/// compiler emitted without a definition). The renderer surfaces
/// `Ptr{ deref: None, deref_skipped_reason: Some("...BTF size is 0
/// (incomplete type)") }` from [`render_cast_pointer`]'s
/// kernel-arm `if btf_size == 0` gate. This guard
/// prevents a zero-byte `read_kva` from succeeding spuriously and
/// rendering an empty struct as if the chase had landed.
#[test]
fn cast_chase_kernel_target_btf_size_zero() {
    // Build a BTF blob where the cast target is a zero-sized
    // struct. Layout:
    //   id=1: u64
    //   id=2: struct T { u64 f @ 0 }, size 8
    //   id=3: struct Q {}, size 0  (the zero-sized cast target)
    //
    // T_id=2, Q_id=3. The kernel arm's `if btf_size == 0` check at
    // [`render_cast_pointer`]'s kernel-arm `if btf_size == 0`
    // gate is what we are exercising; `type_size`
    // returns `Some(0)` for a zero-sized Struct so the prior guard
    // (None case) does not fire.
    let (strings, n_int, n_t, n_q, n_f, _n_x) = cast_strings_for_t_q();
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // Q with size 0 and no members: the BTF wire format permits
        // it (vlen=0, size_type=0), and `type_size` returns Some(0).
        CastSynType::Struct {
            name_off: n_q,
            size: 0,
            members: vec![],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    const KVA: u64 = 0xffff_8000_0000_1000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "zero-sized target must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(deref.is_none());
    let reason = deref_skipped_reason
        .as_deref()
        .expect("zero-sized target must populate skip reason");
    assert!(
        reason.contains("BTF size is 0"),
        "skip reason must say 'BTF size is 0'; got: {reason}"
    );
    assert!(
        reason.contains("incomplete type"),
        "skip reason must mention 'incomplete type'; got: {reason}"
    );
}

/// Kernel cast hit whose target peels to a `BTF_KIND_FWD` (forward
/// declaration). `type_size` returns `None` for `Type::Fwd` because
/// a forward declaration carries no body in this BTF, so the chase
/// has no BTF-declared size to bound the read. The renderer surfaces
/// a `Ptr{ deref: None, deref_skipped_reason: Some("kernel cast
/// target struct sdt_data (type id N) is a forward declaration;
/// body not in this BTF") }` via [`unsizable_chase_reason`].
///
/// Without this case-specific path, the `type_size` failure would
/// have surfaced as the generic "has unresolvable size" message
/// (the legacy fall-through), which gives no operator the cause —
/// they would not know whether the BTF was malformed, the analyzer
/// emitted a stale id, or the chase landed on a valid forward
/// declaration whose body lives in a sibling BTF.
#[test]
fn cast_chase_kernel_target_fwd_struct() {
    // Build a BTF blob where the cast target is a forward-declared
    // struct named "sdt_data". Layout:
    //   id=1: u64
    //   id=2: struct T { u64 f @ 0 }, size 8
    //   id=3: BTF_KIND_FWD struct sdt_data (no body)
    //
    // The cast analyzer's production output never hits this — it
    // only emits Struct/Union ids — but a future analyzer change or
    // a manual cast_map mutation should still surface a clear
    // diagnostic rather than the generic "unresolvable size".
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_f = push(&mut strings, "f");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let fwd_id: u32 = 3;

    const KVA: u64 = 0xffff_8000_0000_3000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: fwd_id,
            addr_space: AddrSpace::Kernel,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "Fwd target must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "Fwd target must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd target must populate skip reason");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must mention 'forward declaration'; got: {reason}"
    );
    assert!(
        reason.contains("body not in this BTF"),
        "skip reason must mention body absence; got: {reason}"
    );
    assert!(
        reason.contains("sdt_data"),
        "skip reason must include the Fwd type's name; got: {reason}"
    );
    assert!(
        reason.contains("struct"),
        "skip reason must say 'struct' (not 'union') for is_struct() Fwd; got: {reason}"
    );
    assert!(
        reason.contains(&fwd_id.to_string()),
        "skip reason must include the type id; got: {reason}"
    );
    // The legacy fall-through message must NOT appear; if it does,
    // the dispatch in `unsizable_chase_reason` did not catch the
    // Type::Fwd arm and we regressed to the generic path.
    assert!(
        !reason.contains("has unresolvable size"),
        "Fwd targets must not surface the generic fall-through; got: {reason}"
    );
}

/// Same scenario as [`cast_chase_kernel_target_fwd_struct`] but the
/// forward declaration is a union (`is_union: true`). The Fwd
/// kind_flag bit selects struct vs union; the renderer surfaces
/// "union" in the reason so operators see the correct aggregate
/// kind.
#[test]
fn cast_chase_kernel_target_fwd_union() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "my_union");
    let n_f = push(&mut strings, "f");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: true,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let fwd_id: u32 = 3;

    const KVA: u64 = 0xffff_8000_0000_4000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: fwd_id,
            addr_space: AddrSpace::Kernel,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "Fwd union target must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd union target must populate skip reason");
    assert!(
        reason.contains("union my_union"),
        "skip reason must surface 'union my_union'; got: {reason}"
    );
    assert!(
        !reason.contains("struct my_union"),
        "Fwd union must not be labelled 'struct'; got: {reason}"
    );
}

/// Arena chase whose pointee BTF type is a `BTF_KIND_FWD`. The
/// real-world trigger: a `struct sdt_chunk` union member declared
/// as `struct sdt_data __arena *`, where `struct sdt_data`'s body
/// lives in the sdt_alloc library's BTF and the using scheduler's
/// own program BTF carries only a forward declaration. The
/// [`btf_rs::Type::Ptr`] arm calls `chase_arena_pointer` with the
/// pointee type id; before this fix `type_size` returned `None`
/// and surfaced as "arena chase target type id N has unresolvable
/// size", giving the operator no signal that the cause was a
/// forward declaration.
///
/// The fix routes `type_size` failures through
/// [`unsizable_chase_reason`], which inspects the peeled type and
/// emits a Fwd-specific message. This test mirrors the production
/// trigger: outer struct holds a `Ptr` to a `Fwd`, the pointer
/// lands in the arena window, and the renderer surfaces the
/// renamed reason.
#[test]
fn arena_chase_pointee_fwd_surfaces_descriptive_reason() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "sdt_chunk");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_data = push(&mut strings, "data");
    // BTF layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD struct sdt_data (no body — emulates the
    //         scheduler-side view of the library struct)
    //   id=3: BTF_KIND_PTR -> id=2 (the `struct sdt_data *` field)
    //   id=4: struct sdt_chunk { struct sdt_data *data @ 0 }, size 8
    //
    // The Type::Ptr arm in render_value_inner reads the u64 at
    // offset 0, recognises the value as an arena address (via
    // `is_arena_addr`), and calls `chase_arena_pointer(btf,
    // pointee_type_id=2, ...)`. The pointee peels to Type::Fwd,
    // type_size returns None, and `unsizable_chase_reason`
    // composes the descriptive message under test.
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
        CastSynType::Ptr { type_id: 2 },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let chunk_id: u32 = 4;
    let fwd_id: u32 = 2;

    // Arena window 0x10_0000_0000 .. 0x10_0001_0000; the address
    // 0x10_0000_1000 lands inside.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, chunk_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "data");
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "data field must render as Ptr (BTF Type::Ptr arm); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        cast_annotation.is_none(),
        "BTF-typed pointers must leave cast_annotation None; got {cast_annotation:?}"
    );
    assert!(
        deref.is_none(),
        "Fwd pointee chase must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd pointee must populate skip reason");
    assert!(
        reason.starts_with("arena chase"),
        "BTF Ptr arm must use 'arena chase' label; got: {reason}"
    );
    assert!(
        reason.contains("forward declaration"),
        "skip reason must mention 'forward declaration'; got: {reason}"
    );
    assert!(
        reason.contains("body not in this BTF"),
        "skip reason must mention body absence; got: {reason}"
    );
    assert!(
        reason.contains("sdt_data"),
        "skip reason must include the Fwd type's name; got: {reason}"
    );
    assert!(
        reason.contains("struct"),
        "skip reason must say 'struct' (kind_flag=0); got: {reason}"
    );
    assert!(
        reason.contains(&fwd_id.to_string()),
        "skip reason must include the Fwd type id; got: {reason}"
    );
    assert!(
        !reason.contains("has unresolvable size"),
        "Fwd targets must not surface the legacy generic message; got: {reason}"
    );
}

/// Anonymous Fwd: a forward declaration with `name_off = 0` (an
/// unnamed forward — uncommon but legal in BTF). The reason text
/// must indicate "anonymous" and still record the type id and
/// aggregate kind so an operator can correlate.
#[test]
fn arena_chase_pointee_fwd_anonymous() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "wrap");
    let n_data = push(&mut strings, "data");
    // BTF layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD anonymous (name_off=0) struct
    //   id=3: BTF_KIND_PTR -> id=2
    //   id=4: struct wrap { void *data @ 0 }, size 8
    //
    // Anonymous Fwd nodes appear in BTF when a struct is forward-
    // declared inside a function or unnamed scope. The chase
    // reason path must still produce a useful message — names
    // shouldn't be load-bearing.
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: 0,
            is_union: false,
        },
        CastSynType::Ptr { type_id: 2 },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let chunk_id: u32 = 4;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, chunk_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    let reason = deref_skipped_reason
        .as_deref()
        .expect("anonymous Fwd must populate skip reason");
    assert!(
        reason.contains("anonymous"),
        "anonymous Fwd reason must say 'anonymous'; got: {reason}"
    );
    assert!(
        reason.contains("struct forward declaration"),
        "anonymous Fwd reason must mention the aggregate kind; got: {reason}"
    );
}

// ---- Fwd → complete-sibling resolution ------------------------
//
// `peel_modifiers_resolving_fwd` looks up a [`Type::Fwd`] terminal
// by name in the same BTF and prefers a complete
// [`Type::Struct`] / [`Type::Union`] sibling of matching aggregate
// kind. The chase pipeline uses this so a forward-declared
// pointee whose body lives one BTF id away (a routine outcome of
// concatenated BPF object files where one .bpf.c only declares
// the type while another defines it) renders as a chased struct
// rather than skipping with "forward declaration; body not in
// this BTF". Tests below exercise:
//   - Fwd + complete Struct siblings of the same name → chase
//     succeeds, deref carries the Struct render.
//   - Fwd with no sibling → chase still skips with the
//     descriptive reason from `unsizable_chase_reason`.
//   - Fwd-struct vs Union same-name → renderer must NOT collapse
//     across aggregate kinds (BTF wire format permits same-name
//     struct + union; collapsing would mis-render the wrong
//     layout).
//   - Anonymous Fwd → no resolution attempted (no name to look
//     up); descriptive reason still surfaces.

/// Arena chase: Type::Ptr arm. The pointee is a [`Type::Fwd`]
/// `task_ctx`; the SAME BTF carries a complete [`Type::Struct`]
/// `task_ctx` at a different id. Without the Fwd shortcut,
/// `chase_arena_pointer` would skip with "forward declaration;
/// body not in this BTF". With the shortcut, it lands on the
/// complete struct and renders the member values from arena
/// bytes.
#[test]
fn arena_chase_pointee_fwd_resolves_to_complete_struct_sibling() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "scx_task_map_val");
    let n_data = push(&mut strings, "data");
    let n_task_ctx = push(&mut strings, "task_ctx");
    let n_field = push(&mut strings, "field");
    // BTF layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD struct task_ctx (no body — emulates
    //         what clang emits when only a pointer to task_ctx
    //         is referenced in the unit; the body is in a
    //         sibling unit's BTF)
    //   id=3: BTF_KIND_PTR -> id=2
    //   id=4: struct scx_task_map_val { struct task_ctx *data @ 0 }
    //   id=5: BTF_KIND_STRUCT task_ctx { u64 field @ 0 } size=8
    //         — the COMPLETE shape `peel_modifiers_resolving_fwd`
    //         lands on after looking up "task_ctx" in the BTF.
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: n_task_ctx,
            is_union: false,
        },
        CastSynType::Ptr { type_id: 2 },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
        CastSynType::Struct {
            name_off: n_task_ctx,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let outer_id: u32 = 4;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x77u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "data field must render as Ptr (BTF Type::Ptr arm); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "Fwd-resolved chase must succeed; got skip reason: {deref_skipped_reason:?}"
    );
    let payload = deref
        .as_deref()
        .expect("Fwd-resolved chase must produce a deref payload");
    let RenderedValue::Struct {
        ref type_name,
        members: ref inner_members,
    } = *payload
    else {
        panic!("deref must be Struct render; got {payload:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("task_ctx"),
        "deref must carry the resolved Struct name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "field");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!(
            "inner field must decode as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x77);
}

/// Cast intercept arena arm: a [`CastHit`] whose `target_type_id`
/// resolves to a [`Type::Fwd`] with a complete [`Type::Struct`]
/// sibling. Mirrors the BTF Type::Ptr test above but exercises
/// `render_cast_pointer`'s arena arm via a u64-typed parent
/// member and a `cast_lookup` hit. Also covers the cast analyzer
/// scenario where a u64 slot's recovered target id happens to
/// resolve to a Fwd at runtime (e.g. cross-BTF id mismatch under
/// libbpf BTF dedup).
#[test]
fn cast_chase_arena_target_fwd_resolves_to_complete_struct_sibling() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_t = push(&mut strings, "scx_task_map_val");
    let n_data = push(&mut strings, "data");
    let n_target = push(&mut strings, "task_ctx");
    let n_field = push(&mut strings, "field");
    // Layout:
    //   id=1: u64
    //   id=2: struct scx_task_map_val { u64 data @ 0 } size=8
    //   id=3: BTF_KIND_FWD task_ctx (struct)
    //   id=4: BTF_KIND_STRUCT task_ctx { u64 field @ 0 } size=8
    //
    // The CastMap entry points at id=3 (the Fwd). Without the
    // Fwd shortcut, the chase would skip; with it, it lands on
    // id=4 and renders the field.
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_target,
            is_union: false,
        },
        CastSynType::Struct {
            name_off: n_target,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let parent_id: u32 = 2;
    let fwd_target_id: u32 = 3;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0xABCDu64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (parent_id, 0),
        CastHit { alloc_size: None,
            target_type_id: fwd_target_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, parent_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "data field must render as cast-recovered Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "cast intercept must annotate the arena chase"
    );
    assert!(
        deref_skipped_reason.is_none(),
        "Fwd-resolved cast chase must not skip; got: {deref_skipped_reason:?}"
    );
    let payload = deref
        .as_deref()
        .expect("Fwd-resolved cast chase must produce deref payload");
    let RenderedValue::Struct {
        ref type_name,
        members: ref inner_members,
    } = *payload
    else {
        panic!("deref must be Struct render; got {payload:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("task_ctx"),
        "deref must carry the resolved Struct name"
    );
    assert_eq!(inner_members[0].name, "field");
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "inner field must decode as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0xABCD);
}

/// Cast intercept kernel arm: same scenario as the arena test
/// above but with `AddrSpace::Kernel`. The kernel arm of
/// `render_cast_pointer` also calls `peel_modifiers_resolving_fwd`
/// so the resolution shortcut applies symmetrically across
/// address spaces.
#[test]
fn cast_chase_kernel_target_fwd_resolves_to_complete_struct_sibling() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_t = push(&mut strings, "parent");
    let n_data = push(&mut strings, "data");
    let n_target = push(&mut strings, "kernel_target");
    let n_field = push(&mut strings, "field");
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_target,
            is_union: false,
        },
        CastSynType::Struct {
            name_off: n_target,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let parent_id: u32 = 2;
    let fwd_target_id: u32 = 3;

    // Use a non-arena KVA so the kernel arm fires.
    const KVA: u64 = 0xffff_8000_0000_3000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let inner_bytes = 0xDEADBEEFu64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (parent_id, 0),
        CastHit { alloc_size: None,
            target_type_id: fwd_target_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, parent_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "data field must render as cast-recovered Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert_eq!(cast_annotation.as_deref(), Some("cast→kernel"));
    assert!(
        deref_skipped_reason.is_none(),
        "Fwd-resolved kernel cast chase must not skip; got: {deref_skipped_reason:?}"
    );
    let payload = deref
        .as_deref()
        .expect("Fwd-resolved kernel cast chase must produce deref payload");
    let RenderedValue::Struct {
        ref type_name,
        members: ref inner_members,
    } = *payload
    else {
        panic!("deref must be Struct render; got {payload:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("kernel_target"),
        "deref must carry the resolved Struct name"
    );
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "inner field must decode as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0xDEADBEEF);
}

/// Aggregate-kind mismatch: a [`Type::Fwd`] declared as
/// `struct foo` must NOT resolve to a [`Type::Union`] of the same
/// name in the same BTF. The wire format permits same-name
/// struct/union declarations (rare but legal); collapsing across
/// aggregate kinds would mis-render the wrong layout. Verifies
/// the Fwd shortcut respects [`btf_rs::Fwd::is_struct`] /
/// [`is_union`].
#[test]
fn fwd_shortcut_rejects_aggregate_kind_mismatch() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_wrap = push(&mut strings, "wrap");
    let n_data = push(&mut strings, "data");
    let n_foo = push(&mut strings, "foo");
    let n_x = push(&mut strings, "x");
    // Layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD struct foo (kind_flag=0)
    //   id=3: BTF_KIND_PTR -> id=2
    //   id=4: struct wrap { struct foo *data @ 0 } size=8
    //   id=5: BTF_KIND_UNION foo { u64 x @ 0 } size=8
    //         — the same name as the Fwd, but the wrong aggregate
    //         kind (Union, not Struct). The shortcut must NOT
    //         resolve to it.
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: n_foo,
            is_union: false,
        },
        CastSynType::Ptr { type_id: 2 },
        CastSynType::Struct {
            name_off: n_wrap,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
        // Encode a Union via the synthetic builder — emit a
        // BTF_KIND_UNION (wire kind=5) by hand since the cast test
        // helper does not yet include a Union variant. Using the
        // raw byte writer keeps the helper API minimal.
        CastSynType::Struct {
            // intentionally still Struct here as a placeholder —
            // see below for the union manual emission.
            name_off: n_foo,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    // The synthetic builder above emits Struct foo at id=5, NOT a
    // Union. To verify aggregate-kind mismatch we need a Union.
    // Manually patch the kind nibble of id=5's `info` u32 from
    // BTF_KIND_STRUCT (4) to BTF_KIND_UNION (5) in the produced
    // blob. The synthetic types vector above is only for
    // documentation: the actual Union emission happens inline
    // below.
    let mut blob = cast_build_btf(&types, &strings);
    // id=5 starts after the header (24 bytes) plus id=1..=4. Compute
    // the byte offset of id=5's info u32 (`name_off` + 4 bytes).
    // id=1 (Int): 16 bytes total
    // id=2 (Fwd): 12 bytes
    // id=3 (Ptr): 12 bytes
    // id=4 (Struct, 1 member): 12 + 12 = 24 bytes
    // id=5 starts at offset 24 + 16 + 12 + 12 + 24 = 88.
    let id5_info_off: usize = 24 + 16 + 12 + 12 + 24 + 4;
    // Read existing info u32, mask out kind nibble (bits 24..28),
    // write BTF_KIND_UNION (5).
    let info = u32::from_le_bytes(blob[id5_info_off..id5_info_off + 4].try_into().unwrap());
    let new_info = (info & !(0x1f << 24)) | (5u32 << 24);
    blob[id5_info_off..id5_info_off + 4].copy_from_slice(&new_info.to_le_bytes());

    let btf = Btf::from_bytes(&blob).expect("synthetic BTF with union parses");
    let wrap_id: u32 = 4;
    let fwd_id: u32 = 2;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, wrap_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref.is_none(),
        "aggregate-kind mismatch must NOT resolve the Fwd; chase must skip"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("aggregate-kind mismatch must populate skip reason (Fwd unresolved)");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must report the Fwd; got: {reason}"
    );
    assert!(
        reason.contains("foo"),
        "skip reason must include the Fwd's name; got: {reason}"
    );
    assert!(
        reason.contains(&fwd_id.to_string()),
        "skip reason must include the Fwd's id; got: {reason}"
    );
}

/// Unit test: `peel_modifiers_resolving_fwd` returns the Fwd
/// unchanged when no complete sibling exists, so the
/// chase-pipeline skip path remains intact for the unrecoverable
/// case (e.g. `struct sdt_data` referenced by lavd: the body lives
/// only in the sdt_alloc library, never in the program's own
/// BTF).
#[test]
fn peel_modifiers_resolving_fwd_no_sibling_returns_fwd() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_fwd = push(&mut strings, "lonely_fwd");
    // BTF: id=1 u64, id=2 Fwd 'lonely_fwd' (struct, no sibling).
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let (peeled, peeled_id) =
        peel_modifiers_resolving_fwd(&btf, 2).expect("Fwd resolves through helper");
    assert!(
        matches!(peeled, Type::Fwd(_)),
        "no-sibling lookup must return the original Fwd; got {peeled:?}"
    );
    assert_eq!(peeled_id, 2);
}

/// Unit test: anonymous Fwd (name_off=0) cannot be looked up by
/// name; helper returns the Fwd unchanged.
#[test]
fn peel_modifiers_resolving_fwd_anonymous_fwd_returns_fwd() {
    let strings: Vec<u8> = vec![0];
    let types = vec![
        CastSynType::Int {
            name_off: 0,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: 0,
            is_union: false,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let (peeled, peeled_id) =
        peel_modifiers_resolving_fwd(&btf, 2).expect("anonymous Fwd resolves through helper");
    assert!(
        matches!(peeled, Type::Fwd(_)),
        "anonymous Fwd must remain Fwd; got {peeled:?}"
    );
    assert_eq!(peeled_id, 2);
}

/// Unit test: a Typedef chain ending in a Fwd is peeled through
/// the modifier chain AND the Fwd is then resolved to a complete
/// Struct sibling. Mirrors the typical `typedef struct foo foo;`
/// plus `struct foo;` pattern clang emits when a header
/// forward-declares a typedef and a separate compilation unit
/// defines the underlying struct.
#[test]
fn peel_modifiers_resolving_fwd_through_typedef_chain() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_alias = push(&mut strings, "alias");
    let n_target = push(&mut strings, "target");
    let n_field = push(&mut strings, "field");
    // BTF:
    //   id=1: u64
    //   id=2: BTF_KIND_TYPEDEF alias -> id=3
    //   id=3: BTF_KIND_FWD target (struct)
    //   id=4: BTF_KIND_STRUCT target { u64 field @ 0 } size=8
    //
    // Calling peel_modifiers_resolving_fwd(&btf, 2) must:
    //  1. Peel Typedef -> Fwd (peel_modifiers_with_id terminates
    //     at the Fwd).
    //  2. Resolve Fwd 'target' to the sibling Struct at id=4.
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Typedef {
            name_off: n_alias,
            type_id: 3,
        },
        CastSynType::Fwd {
            name_off: n_target,
            is_union: false,
        },
        CastSynType::Struct {
            name_off: n_target,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let (peeled, peeled_id) =
        peel_modifiers_resolving_fwd(&btf, 2).expect("Typedef→Fwd→Struct chain resolves");
    assert!(
        matches!(peeled, Type::Struct(_)),
        "Typedef→Fwd chain must land on the complete Struct; got {peeled:?}"
    );
    assert_eq!(
        peeled_id, 4,
        "resolved id must be the complete Struct's id, not the Typedef or Fwd id"
    );
}

/// Kernel cast hit where `read_kva` returns `None` (the page is
/// unmapped or the page-table walk failed). The renderer surfaces
/// `Ptr{ deref: None, deref_skipped_reason: Some("kernel read_kva
/// failed at 0x... (unmapped page or no PTE); needed N bytes") }`
/// from [`render_cast_pointer`]'s kernel-arm `read_kva`
/// failure branch. Without this gate, a `None` from
/// `read_kva` would propagate to the unwrap downstream and either
/// crash or render whatever default the inner type produces from
/// zero bytes — both worse than the labelled skip.
#[test]
fn cast_chase_kernel_read_kva_failure() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Q is an 8-byte struct (size=8), so `read_size` will be 8.
    // Reader carries NO `kva_bytes_at` entries, so `read_kva` returns
    // `None` for every address — the failure path under test.
    const KVA: u64 = 0xffff_8000_0000_2000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "read_kva failure must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "read_kva failure must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("read_kva failure must populate skip reason");
    assert!(
        reason.contains("read_kva failed"),
        "skip reason must say 'read_kva failed'; got: {reason}"
    );
    assert!(
        reason.contains(&format!("0x{KVA:x}")),
        "skip reason must include the failing address in hex; got: {reason}"
    );
    assert!(
        reason.contains("needed"),
        "skip reason must include the requested byte count; got: {reason}"
    );
}

/// Kernel cast hit where the requested size exceeds the bytes
/// remaining in the current 4 KiB page. `read_size = btf_size
/// .min(POINTER_CHASE_CAP).min(page_remaining)` clamp inside
/// [`render_cast_pointer`]'s kernel arm caps the read at the page
/// edge so `read_kva` never crosses an allocation boundary; the
/// resulting `truncated_at_cap` flag wraps the inner render in
/// `RenderedValue::Truncated{needed: btf_size, had: page_remaining,
/// partial: ...}` at the kernel arm's `truncated_at_cap` branch.
/// This
/// test pins the page-edge clipping AND the Truncated wrapper.
#[test]
fn cast_chase_kernel_page_edge_truncation() {
    // Build a target with size > page_remaining. Q's size is set to
    // 100 bytes (a single u64 at offset 0 plus 92 bytes of trailing
    // padding, recorded only in `size_type`). The kernel value KVA
    // = page_base + 4080 leaves 16 bytes in the current page
    // (4096 - (4080 % 4096) = 16). So:
    //   btf_size       = 100
    //   page_remaining = 16
    //   POINTER_CHASE_CAP = 4096
    //   read_size      = min(100, 4096, 16) = 16
    //   truncated_at_cap = (100 > 16) = true
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // Q size=100, single u64 member at offset 0. The struct's
        // declared size is what `type_size` reports — not the sum
        // of member sizes. Tail bytes 8..100 are unaccounted for in
        // BTF members, modelling a struct with padding or members
        // the test doesn't care about.
        CastSynType::Struct {
            name_off: n_q,
            size: 100,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    // KVA leaves exactly 16 bytes remaining in the page.
    const KVA: u64 = 0xffff_8000_0000_0ff0;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Provide 16 bytes at KVA so `read_kva(KVA, 16)` succeeds. First
    // 8 bytes carry Q.x = 0xCAFE (low value, top byte is 0x00 — the
    // plausibility gate (top-byte-0xff freelist heuristic) in
    // [`render_cast_pointer`]'s kernel arm accepts it).
    let mut target_bytes = vec![0u8; 16];
    target_bytes[0..8].copy_from_slice(&0xCAFEu64.to_le_bytes());
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, target_bytes);
    // Use cast_map mode so only the outer T.f → Q intercept fires;
    // Q.x at (q_id, 0) has no entry, so the inner u64 surfaces as
    // a plain Uint render rather than recursing into another chase.
    let mut cast_map: super::super::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "page-edge clipped chase must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref_skipped_reason.is_none(),
        "successful (clipped) read must carry no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("read succeeded → deref must be Some");
    // Outer wrap: Truncated{needed: 100, had: 16, partial: ...}.
    let RenderedValue::Truncated {
        needed,
        had,
        ref partial,
    } = *inner
    else {
        panic!("btf_size > read_size must wrap deref payload in Truncated; got {inner:?}");
    };
    assert_eq!(needed, 100, "Truncated.needed must be the BTF size");
    assert_eq!(
        had, 16,
        "Truncated.had must be the page-edge-clipped read size"
    );
    // Partial render: the inner Struct{Q} also wraps as Truncated
    // because 16 bytes < Q.size=100, with its own partial: Struct.
    // Walk through both wrappers to reach Q's members.
    let inner_struct = match &**partial {
        RenderedValue::Struct { .. } => partial.as_ref(),
        RenderedValue::Truncated {
            partial: deeper, ..
        } => deeper.as_ref(),
        other => panic!(
            "partial render must reach a Q struct (possibly via inner Truncated); got {other:?}"
        ),
    };
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner_struct
    else {
        panic!("expected inner Struct render, got {inner_struct:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "inner struct must carry Q's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "x");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0xCAFE, "first 8 bytes of clipped read must decode");
}

/// Kernel cast hit whose `read_kva` returns plausible bytes (top
/// byte != 0xff) and whose target peels + sizes correctly: the
/// chase succeeds and the rendered struct's members are surfaced
/// in the `deref` payload. Complement to
/// [`cast_chase_kernel_plausibility_rejects_freed_slab`] — same
/// path but the plausibility gate ALLOWS the read instead of
/// rejecting it. Without this test, a regression that flipped the
/// plausibility gate's polarity (`if first_qword >> 56 != 0xff`
/// rejects, instead of accepts) would only show up as a missing
/// deref on every kernel chase, with no test catching the inversion.
#[test]
fn cast_chase_kernel_successful_chase_top_byte_non_ff() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // KVA in the kernel direct-map range; the value passed to the
    // plausibility gate is the FIRST QWORD OF target_bytes (Q.x's
    // value), not KVA itself. Choose target bytes whose first qword
    // top byte is 0x00 (a plain counter) so the gate at
    // the plausibility gate in [`render_cast_pointer`]'s kernel
    // arm (top-byte-0xff freelist heuristic) accepts.
    const KVA: u64 = 0xffff_8000_dead_b000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Q is 8 bytes (the existing fixture). Provide exactly 8 bytes
    // at KVA whose first qword is a low counter value (0x42).
    let inner_bytes: Vec<u8> = 0x42u64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    // Use cast_map mode so only the outer T.f → Q intercept fires;
    // Q.x at (q_id, 0) has no entry, so the inner u64 surfaces as
    // a plain Uint render rather than recursing into another chase.
    let mut cast_map: super::super::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "kernel chase must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase carries no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("plausibility-allowed chase → deref must be Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the rendered Q struct; got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "deref payload must carry Q's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "x");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42, "Q.x must reflect the bytes read_kva returned");
}

// ---- POINTER_CHASE_CAP and parent-byte-boundary edge cases -----
//
// These tests pin the cap-and-boundary behavior that protects the
// renderer from unbounded reads (POINTER_CHASE_CAP) and from
// out-of-range slicing (parent_bytes boundary). Both produce
// `RenderedValue::Truncated` wrappers when the rendered subtree is
// partial, so the consumer can tell the rendered output is
// incomplete.

/// Arena cast target whose BTF size exceeds [`POINTER_CHASE_CAP`]
/// (4096 bytes). [`chase_arena_pointer`]'s
/// `let read_size = btf_size.min(POINTER_CHASE_CAP)` clamp limits
/// the read to 4096 bytes and sets `truncated_at_cap = true`, then
/// wraps the rendered struct in `RenderedValue::Truncated{needed:
/// btf_size, had: 4096, partial: ...}` at its `truncated_at_cap`
/// payload-wrap branch. Without
/// the cap, an analyzer that emits a 1 MiB struct would force the
/// renderer to allocate (and read) a megabyte from the arena snapshot
/// for a single failure dump — pulling the dump through
/// O(num_recovered_pointers * pointee_size) memory pressure.
#[test]
fn cast_chase_arena_pointee_exceeds_cap_wraps_in_truncated() {
    // Q is sized 5000 (above POINTER_CHASE_CAP=4096) so the cap
    // clamps the read. Single u64 member at offset 0 covers the
    // first 8 bytes; the remaining 4992 bytes are unaccounted-for
    // padding in the BTF wire format.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 5000,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // Provide exactly 4096 bytes (POINTER_CHASE_CAP) at TARGET_ADDR
    // so `read_arena(TARGET_ADDR, 4096)` succeeds. Bytes 0..8 carry
    // Q.x = 0x77.
    let mut target_bytes = vec![0u8; 4096];
    target_bytes[0..8].copy_from_slice(&0x77u64.to_le_bytes());
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, target_bytes);
    // Use cast_map mode (NOT the universal `hit` field) so only the
    // outer T.f → Q intercept fires; Q.x at (q_id, 0) has no entry,
    // so the inner u64 falls through to the plain Uint render and
    // doesn't recurse into a phantom kernel chase.
    let mut cast_map: super::super::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "cap-clamped chase must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "cap-clamped read is a SUCCESS; no skip reason expected; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("read succeeded → deref must be Some");
    // Outer wrap: Truncated{needed: 5000, had: 4096, partial: ...}.
    let RenderedValue::Truncated {
        needed,
        had,
        ref partial,
    } = *inner
    else {
        panic!("btf_size > POINTER_CHASE_CAP must wrap deref in Truncated; got {inner:?}");
    };
    assert_eq!(needed, 5000, "Truncated.needed must be Q's BTF size");
    assert_eq!(
        had, 4096,
        "Truncated.had must equal POINTER_CHASE_CAP (4096)"
    );
    // `render_struct` itself emitted a Truncated wrapper because
    // 4096 < Q.size=5000; walk through it to reach the inner Struct.
    let inner_struct = match &**partial {
        RenderedValue::Struct { .. } => partial.as_ref(),
        RenderedValue::Truncated {
            partial: deeper, ..
        } => deeper.as_ref(),
        other => panic!(
            "partial render must reach a Q struct (possibly via inner Truncated); got {other:?}"
        ),
    };
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner_struct
    else {
        panic!("expected inner Struct render, got {inner_struct:?}");
    };
    assert_eq!(inner_name.as_deref(), Some("Q"));
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(
        value, 0x77,
        "first 8 bytes of cap-clamped read must decode correctly"
    );
}

/// Cast intercept on a u64 member where `byte_off + 8 >
/// parent_bytes.len()` — the `field_bytes.get(..8)?` boundary
/// guard in [`try_cast_intercept`] returns `None`, the intercept
/// does not fire, and execution falls through to the existing
/// partial-decode path in [`render_member`] (the `byte_off + size
/// <= parent_bytes.len()` check that wraps a short member in
/// `RenderedValue::Truncated`) which
/// emits a `RenderedValue::Truncated` wrapping whatever the inner
/// renderer salvaged. Without this guard, the intercept would slice
/// `parent_bytes[byte_off..byte_off+8]` past the slice's end and
/// either crash or read uninitialized memory off the end of the
/// allocation — a critical bug given the renderer runs over
/// guest-supplied bytes the analyzer already corrupted somewhere.
#[test]
fn cast_intercept_u64_at_parent_bytes_boundary_falls_through() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // T's u64 member is at offset 0 with size 8 — the full member
    // needs 8 bytes. Provide only 4 bytes so `byte_off + 8 = 8 > 4
    // = parent_bytes.len()` and the intercept short-circuits. The
    // configured cast hit + arena window + bytes are all set up so
    // a buggy intercept that ignored the boundary check would
    // produce a Ptr render — a successful test confirms only Truncated.
    let outer_bytes = vec![0xCA, 0xFE, 0xBA, 0xBE];
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(0xBEBA_FECAu64, vec![0u8; 8]);
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((0, u64::MAX)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    // The outer T struct is 8 bytes but only 4 were supplied, so
    // `render_struct` wraps in `Truncated{needed:8, had:4, partial:
    // Struct{T}}`. The intercept's boundary guard at
    // [`try_cast_intercept`]'s `field_bytes.get(..8)?` boundary
    // guard short-circuits when `byte_off + 8 > parent_bytes.len()`,
    // so T.f also surfaces as a per-member `Truncated`, NOT a `Ptr`.
    let outer_struct = match &v {
        RenderedValue::Struct { .. } => &v,
        RenderedValue::Truncated { partial, .. } => partial.as_ref(),
        other => panic!("expected Struct or Truncated{{Struct}}; got {other:?}"),
    };
    let RenderedValue::Struct { ref members, .. } = *outer_struct else {
        panic!("expected Struct under outer Truncated; got {outer_struct:?}");
    };
    match &members[0].value {
        RenderedValue::Truncated { needed, had, .. } => {
            assert_eq!(*needed, 8, "u64 needs 8 bytes");
            assert_eq!(*had, 4, "supplied bytes for member is 4");
        }
        RenderedValue::Ptr { .. } => panic!(
            "boundary fall-through must NOT produce Ptr — intercept's \
             boundary guard (`field_bytes.get(..8)?` in \
             try_cast_intercept) short-circuits; got {:?}",
            members[0].value
        ),
        other => panic!("boundary fall-through must produce Truncated; got {other:?}"),
    }
}

// ---- Int-encoding gate paths -----------------------------------
//
// The intercept's `int.size() != 8 || int.is_signed() ||
// int.is_bool() || int.is_char()` encoding gate in
// [`try_cast_intercept`] rejects every non-plain-unsigned-u64
// member regardless of the
// reader's `cast_lookup` hit. The size-mismatch case
// (`cast_intercept_non_u64_field_not_intercepted`) is already
// covered above; these tests cover the encoding flags. BPF stores
// recovered typed pointers in plain-unsigned u64 slots only; a
// `_Bool`, `char`, or signed 8-byte field is structurally not the
// analyzer's output shape and must NOT be intercepted.

/// Cast intercept on a `_Bool`-encoded 8-byte field with a fixed
/// hit returned by `cast_lookup`: the intercept's gate at
/// the encoding gate in [`try_cast_intercept`] rejects
/// (int.is_bool() == true) before
/// `cast_lookup` is consulted, and the renderer falls through to
/// the normal Int render — `RenderedValue::Bool{value}` (an 8-byte
/// _Bool encodes as is_bool() at the BTF level; render_int
/// produces Bool when is_bool()).
#[test]
fn cast_intercept_bool_field_not_intercepted() {
    // Build T with a single 8-byte _Bool member at offset 0.
    // Encoding = BTF_INT_BOOL (= 4); size = 8; bits = 64. The Q
    // type is included so the cast hit can reference a real id, but
    // the gate rejects before the hit is consulted.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        // id 1: 8-byte _Bool. encoding=4 = BTF_INT_BOOL.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 4,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { _Bool f @ 0 }, size 8.
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: struct Q { _Bool x @ 0 }, size 8 (cast target;
        // unused since the gate rejects first, but a valid id).
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    // Outer T bytes: a non-zero _Bool value (0x01). Reader returns
    // Some(hit) for any (parent, offset) — the bool gate must
    // reject before this is consulted.
    let outer_bytes = 1u64.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((0, u64::MAX)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    // The bool gate must reject the intercept. Render path produces
    // Bool — the cast intercept must NOT have fired.
    match &members[0].value {
        RenderedValue::Bool { value } => {
            assert!(*value, "bool value 0x01 must render as true");
        }
        RenderedValue::Ptr { .. } => panic!(
            "_Bool field must NOT be intercepted (int.is_bool() gate at \
             encoding gate in try_cast_intercept rejects); got {:?}",
            members[0].value
        ),
        other => panic!("_Bool field must render as Bool; got {other:?}"),
    }
}

/// Cast intercept on a SIGNED 8-byte int member with a fixed hit:
/// the encoding gate in [`try_cast_intercept`] rejects
/// (int.is_signed() == true) before `cast_lookup` is consulted, and
/// the renderer falls through to the normal Int render —
/// `RenderedValue::Int{bits: 64, value: <signed value>}`. Signed
/// 8-byte ints in BPF programs are typically counters / deltas, not
/// recovered pointers; the gate keeps the analyzer-emitted hits
/// from contaminating a counter slot.
#[test]
fn cast_intercept_signed_8byte_int_not_intercepted() {
    // Build T with a single 8-byte SIGNED int member at offset 0.
    // Encoding = BTF_INT_SIGNED (= 1); size = 8; bits = 64.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        // id 1: 8-byte signed int. encoding=1 = BTF_INT_SIGNED.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 1,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: struct Q with the same signed-int field. Cast target
        // is unused since the gate rejects first.
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    // Outer bytes: signed -1 (all-ones u64). Without the gate, the
    // reader would interpret 0xFFFFFFFFFFFFFFFF as a pointer value
    // — the gate must reject the intercept.
    let outer_bytes = (-1i64).to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((0, u64::MAX)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    match &members[0].value {
        RenderedValue::Int { bits, value } => {
            assert_eq!(
                *bits, 64,
                "signed int must render at its declared 64-bit width"
            );
            assert_eq!(*value, -1, "signed -1 must round-trip as Int{{value: -1}}");
        }
        RenderedValue::Ptr { .. } => panic!(
            "signed 8-byte int must NOT be intercepted (int.is_signed() \
             encoding gate in try_cast_intercept rejects); got {:?}",
            members[0].value
        ),
        other => panic!("signed 8-byte int must render as Int; got {other:?}"),
    }
}

// ---- parent_type_id == None path -------------------------------
//
// `render_member`'s `parent_type_id` parameter is `Option<u32>` at
// [`render_member`]'s parameter list. Through the public entry
// points (`render_value_with_mem` → `render_value_inner` →
// `render_struct`'s per-member loop) it is always
// `Some(parent_type_id)`. The `None` case is reachable only by
// calling `render_member` directly. [`render_member`]'s
// `parent_type_id.and_then(|parent| ...)` guard short-circuits
// when `None`, leaving the closure's
// `cast_intercept` as `None` and the renderer falling through to
// the unmodified path. This test pins the no-crash, no-intercept
// behavior.

/// Direct call to `render_member` with `parent_type_id = None` must
/// short-circuit the cast intercept ([`render_member`]'s
/// `parent_type_id.and_then(...)` guard) and produce the
/// same render as the no-cast case — for a u64 field, that is
/// `RenderedValue::Uint{bits: 64, value}`. Establishes the contract
/// for any future entry point that bypasses `render_struct` (e.g.
/// rendering a stand-alone member from a synthesized layout): such
/// callers must explicitly opt in to the cast intercept by passing
/// the parent struct id, never silently inheriting it.
#[test]
fn cast_intercept_parent_type_id_none_does_not_crash() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Resolve T's first member directly so we can hand it to
    // render_member without going through render_struct (which
    // always passes Some(parent_type_id)).
    let Type::Struct(t_struct) = btf.resolve_type_by_id(t_id).expect("T resolves") else {
        panic!("T_id resolves to non-Struct type");
    };
    let m = t_struct.members.first().expect("T has one member");

    // Configure the reader so a buggy renderer that ignored the
    // None short-circuit and called cast_lookup anyway WOULD chase
    // the value. The correct render must produce Uint regardless.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, 0xAAu64.to_le_bytes().to_vec());
    let reader = CastStubReader {
        hit: Some(CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    // Direct call to render_member with parent_type_id = None.
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let v = render_member(
        &btf,
        m,
        None,
        &outer_bytes,
        0,
        Some(&reader as &dyn MemReader),
        &mut visited,
    );

    // None short-circuit MUST keep the intercept off; the u64
    // renders as Uint with the loaded value.
    let RenderedValue::Uint { bits, value } = v else {
        panic!(
            "parent_type_id=None must short-circuit the intercept and \
             render as Uint; got {v:?}. A failure here means \
             render_member's `let parent = parent_type_id?` guard at \
             `parent_type_id.and_then(...)` guard in render_member \
             was bypassed."
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, TARGET_ADDR);
}

// ---- Recursive cast chase + modifier-chain unit test -----------
//
// A cast target whose own struct contains a u64 field flagged by
// the cast analyzer drives the renderer into a recursive chase:
// the outer chase produces a `Ptr{ deref: Some(Struct{ ... }) }`,
// and the inner Struct's u64 field surfaces as another nested
// `Ptr{ deref: Some(Struct{ ... }) }`. The visited set carries the
// outer address through the recursion so genuine cycles still
// surface as `[cycle]` (covered by `cast_chase_cycle_detection`).
// This test pins the non-cycle recursive case where two distinct
// addresses chain.
//
// The modifier-chain unit test renders a `Typedef -> Const ->
// Struct(T)` surface type via `cast_map`-mode lookup keyed on
// `T_id` directly. The renderer's `peel_modifiers_with_id` MUST
// produce the underlying `T_id` so `cast_lookup(T_id, 0)` hits.
// This is a focused unit-only sibling of
// `cast_pipeline_modifier_chain_renderer_peels_to_analyzer_struct_id`
// (which also runs the analyzer); the pure-renderer test isolates
// the peel behavior so a regression in one half (renderer or
// analyzer) is distinguishable from a regression in the other.

/// Target struct whose body itself contains a cast-flagged u64
/// field. The renderer's recursion must:
///   - chase the outer Ptr (T.f → Q struct at TARGET_ADDR),
///   - render Q.x as a NESTED Ptr (cast hit at (Q, 0)),
///   - chase Q.x → R struct at TARGET_ADDR_2 → R.y = Uint(0xBB).
///
/// Both deref payloads must be present (deref: Some) and neither
/// should carry a skip reason.
#[test]
fn cast_chase_recursive_target_with_inner_cast_field() {
    // Build a 4-type BTF: u64, T (8 bytes, u64 @ 0), Q (8 bytes,
    // u64 @ 0), R (8 bytes, u64 @ 0). The CastMap entries are:
    //   (T_id, 0) → (Q_id, Arena)
    //   (Q_id, 0) → (R_id, Arena)
    // R has no flagged member, so the recursion terminates.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_r = push(&mut strings, "R");
    let n_y = push(&mut strings, "y");
    let types = vec![
        // id 1: u64
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: T { u64 f @ 0 }
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: Q { u64 x @ 0 }
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: R { u64 y @ 0 }
        CastSynType::Struct {
            name_off: n_r,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_y,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;
    let r_id: u32 = 4;

    let mut cast_map: super::super::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    cast_map.insert(
        (q_id, 0),
        CastHit { alloc_size: None,
            target_type_id: r_id,
            addr_space: AddrSpace::Arena,
        },
    );

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    const TARGET_ADDR_2: u64 = 0x10_0000_2000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // Q at TARGET_ADDR: Q.x = TARGET_ADDR_2 (the inner cast value).
    let q_bytes: Vec<u8> = TARGET_ADDR_2.to_le_bytes().to_vec();
    // R at TARGET_ADDR_2: R.y = 0xBB (terminal value).
    let r_bytes: Vec<u8> = 0xBBu64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, q_bytes);
    arena_bytes.insert(TARGET_ADDR_2, r_bytes);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    // Outer T.f: Ptr → Q.
    let RenderedValue::Ptr {
        value: outer_value,
        deref: ref outer_deref,
        deref_skipped_reason: ref outer_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "outer cast intercept must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(outer_value, TARGET_ADDR);
    assert!(
        outer_reason.is_none(),
        "outer chase must succeed; got {outer_reason:?}"
    );
    let q_inner = outer_deref
        .as_deref()
        .expect("outer deref Some (chase succeeded)");
    let RenderedValue::Struct {
        type_name: ref q_name,
        members: ref q_members,
    } = *q_inner
    else {
        panic!("outer deref must be Q struct; got {q_inner:?}");
    };
    assert_eq!(q_name.as_deref(), Some("Q"));
    assert_eq!(q_members.len(), 1);
    assert_eq!(q_members[0].name, "x");
    // Inner Q.x: nested Ptr → R (the cast intercept fires
    // recursively because the renderer recurses with parent_type_id
    // = Q_id, and the CastMap has (Q, 0) → (R, Arena)).
    let RenderedValue::Ptr {
        value: inner_value,
        deref: ref inner_deref,
        deref_skipped_reason: ref inner_reason,
        ..
    } = q_members[0].value
    else {
        panic!(
            "inner Q.x must surface as Ptr (recursive cast hit); got {:?}. \
             A failure here means the renderer didn't pass Q_id as \
             parent_type_id when recursing through the deref payload.",
            q_members[0].value
        );
    };
    assert_eq!(inner_value, TARGET_ADDR_2);
    assert!(
        inner_reason.is_none(),
        "inner chase must succeed; got {inner_reason:?}"
    );
    let r_inner = inner_deref
        .as_deref()
        .expect("inner deref Some (chase succeeded)");
    let RenderedValue::Struct {
        type_name: ref r_name,
        members: ref r_members,
    } = *r_inner
    else {
        panic!("inner deref must be R struct; got {r_inner:?}");
    };
    assert_eq!(r_name.as_deref(), Some("R"));
    assert_eq!(r_members.len(), 1);
    assert_eq!(r_members[0].name, "y");
    // Terminal R.y: plain Uint (no cast entry at (R, 0)).
    let RenderedValue::Uint { bits, value } = r_members[0].value else {
        panic!(
            "R.y must terminate as Uint (no recursive cast entry at (R,0)); \
             got {:?}",
            r_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0xBB);
}

/// Modifier-chain unit case: the parent type is rendered via a
/// `Typedef -> Const -> Struct(T)` chain, and the CastMap is keyed
/// on `(T_id, 0)` directly. The renderer's
/// `peel_modifiers_with_id` MUST collapse the wrapper chain to
/// `T_id` before threading it into `render_struct` as
/// `parent_type_id`; the cast intercept then queries
/// `cast_lookup(T_id, 0)` and hits. Pure-renderer test —
/// complementary to
/// `cast_pipeline_modifier_chain_renderer_peels_to_analyzer_struct_id`
/// which couples the analyzer in. Without this isolated test, a
/// regression in just the renderer's peel would only surface
/// indirectly through the integration test (where it'd be
/// indistinguishable from an analyzer regression).
#[test]
fn cast_intercept_modifier_chain_parent_uses_post_peel_id() {
    // Layout:
    //   id=1: u64
    //   id=2: T { u64 f @ 0 }
    //   id=3: Q { u64 x @ 0 } (cast target)
    //   id=4: const(T) — wraps id=2
    //   id=5: typedef T_alias = const(T) — wraps id=4
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_typedef = push(&mut strings, "T_alias");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Const { type_id: 2 },
        CastSynType::Typedef {
            name_off: n_typedef,
            type_id: 4,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;
    let typedef_id: u32 = 5;

    // CastMap keyed on the POST-PEEL id (T_id, 0). If the renderer
    // forwarded the typedef_id (or any non-peeled id) as
    // parent_type_id, this lookup would miss and the field would
    // render as Uint — failing the assertion below.
    let mut cast_map: super::super::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, 0x55u64.to_le_bytes().to_vec());
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    // Render via the typedef wrapper id; the renderer's peel must
    // produce parent_type_id=T_id (the post-peel struct id) so the
    // cast lookup hits.
    let v = render_value_with_mem(&btf, typedef_id, &outer_bytes, &reader);
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected Struct render, got {v:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("T"),
        "renderer must collapse modifier wrappers to underlying T name"
    );
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "modifier-chain rendering must reach the cast intercept (peel \
             must produce T_id as parent_type_id); got {:?}. A failure here \
             means peel_modifiers_with_id forwarded the typedef wrapper id \
             instead of the post-peel struct id when calling \
             render_struct.",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(deref_skipped_reason.is_none());
    let inner = deref.as_deref().expect("chase deref Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be Q struct, got {inner:?}");
    };
    assert_eq!(inner_name.as_deref(), Some("Q"));
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x55);
}

// ---- sdt_alloc arena-type bridge -------------------------------
//
// `MemReader::resolve_arena_type(addr)` lets the renderer recover
// the BTF type id of a chased arena pointer's pointee when the
// program BTF carries only a `BTF_KIND_FWD` (forward declaration —
// body in another BTF). The dump path's `AccessorMemReader`
// populates the lookup from the sdt_alloc pre-pass's allocator
// snapshots; the renderer's [`chase_arena_pointer`] /
// [`render_cast_pointer`] paths consult it after the BTF-only Fwd
// resolve fails.
//
// The tests below cover:
//   - default `MemReader` impl returns `None` (no bridge wiring).
//   - custom `MemReader` impl override returns the configured id.
//   - `chase_arena_pointer` with a Fwd target + matching bridge
//     entry produces a successful chase whose deref renders the
//     resolved struct.
//   - `chase_arena_pointer` with no bridge entry skips with the
//     existing "forward declaration; body not in this BTF" reason.
//   - Type::Ptr arena arm sets `cast_annotation` to "sdt_alloc"
//     when the bridge fires.
//   - `render_cast_pointer` arena arm extends `cast_annotation`
//     to `cast→arena (sdt_alloc)` when the bridge fires.

/// `MemReader` trait default for `resolve_arena_type` returns
/// `None` for every address. Pin the behaviour so a future change
/// that flipped the default would surface here as a test
/// regression rather than silently activating the bridge for every
/// reader.
#[test]
fn mem_reader_default_resolve_arena_type_is_none() {
    struct DefaultReader;
    impl MemReader for DefaultReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
    }
    let r = DefaultReader;
    assert!(
        r.resolve_arena_type(0x10_0000_1000).is_none(),
        "default resolve_arena_type must return None for any address",
    );
    assert!(
        r.resolve_arena_type(0).is_none(),
        "default resolve_arena_type must return None for null too",
    );
    assert!(
        r.resolve_arena_type(u64::MAX).is_none(),
        "default resolve_arena_type must return None for u64::MAX too",
    );
}

/// Custom [`MemReader`] override returns the configured
/// [`ArenaResolveHit`] for known addresses and `None` for
/// everything else. Mirrors the production
/// [`super::super::dump::render_map::AccessorMemReader::resolve_arena_type`]
/// shape. Two distinct seeded entries cover the two production
/// shapes — payload-start chase (`header_skip = 0`) and slot-start
/// chase (`header_skip = header_size`).
#[test]
fn mem_reader_resolve_arena_type_override_returns_configured_hit() {
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start entry: header_skip = 0.
    arena_types.insert(
        0x10_0000_1008u64,
        ArenaResolveHit {
            target_type_id: 7,
            header_skip: 0,
        },
    );
    // Slot-start entry: header_skip = 8 (the size of `union sdt_id`).
    arena_types.insert(
        0x10_0000_2000u64,
        ArenaResolveHit {
            target_type_id: 11,
            header_skip: 8,
        },
    );
    let reader = CastStubReader {
        arena_type_at: arena_types,
        ..Default::default()
    };
    assert_eq!(
        reader.resolve_arena_type(0x10_0000_1008),
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 0,
        }),
    );
    assert_eq!(
        reader.resolve_arena_type(0x10_0000_2000),
        Some(ArenaResolveHit {
            target_type_id: 11,
            header_skip: 8,
        }),
    );
    assert!(
        reader.resolve_arena_type(0x10_0000_3000).is_none(),
        "address not in index must return None",
    );
    assert!(
        reader.resolve_arena_type(0).is_none(),
        "null address must return None",
    );
}

/// Build a synthetic BTF blob for the sdt_alloc bridge tests.
///
/// Layout:
///   - id=1: u64 (size=8, plain unsigned)
///   - id=2: BTF_KIND_FWD struct sdt_data (no body — emulates the
///     scheduler-side forward declaration of the library struct)
///   - id=3: BTF_KIND_PTR -> id=2 (the `struct sdt_data *` field
///     type)
///   - id=4: struct outer { struct sdt_data *data @ 0 }, size=8
///     (the field through which the renderer chases an arena
///     pointer)
///   - id=5: struct task_ctx { u64 weight @ 0 }, size=8 (the real
///     payload type the bridge resolves to — distinct from
///     sdt_data so the renderer must consult the bridge to find it)
///
/// The `struct sdt_data *` field name and pointee Fwd are
/// stand-ins for the bridge's TRIGGER shape, not the production
/// trigger itself. In production the bridge fires on PAYLOAD-START
/// pointers — `cpu_ctx::cached_taskc_raw` is a `u64` storing the
/// return value of `scx_task_data(p)` (which dereferences a
/// `struct sdt_data __arena *` slot and returns `data->payload`,
/// past the 8-byte header). The cast analyzer promotes that `u64`
/// to a typed pointer whose pointee surfaces as a `BTF_KIND_FWD`
/// in the program BTF (the body lives in the sdt_alloc library's
/// BTF). The fixture compresses that into one shape: a Fwd-pointee
/// `Type::Ptr` field whose stored value is the payload-start
/// address used to populate the bridge index.
///
/// `CastStubReader::resolve_arena_type` (the test's MemReader) keys
/// the bridge map on the FULL 64-bit address rather than the low 32
/// bits — production [`AccessorMemReader::resolve_arena_type`] masks
/// `addr & 0xFFFF_FFFF` and looks up in the per-pass index. Tests use
/// full-address keys to avoid the masking concern in the test setup;
/// the masking itself is exercised by
/// [`super::super::dump::tests::accessor_mem_reader_resolve_arena_type_masks_low_32`].
///
/// Returns `(blob, outer_id, fwd_id, task_ctx_id)`.
fn bridge_btf_outer_fwd_taskctx() -> (Vec<u8>, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_outer = push(&mut strings, "outer");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_data = push(&mut strings, "data");
    let n_task = push(&mut strings, "task_ctx");
    let n_weight = push(&mut strings, "weight");
    let types = vec![
        // id 1: u64.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: BTF_KIND_FWD struct sdt_data (no body).
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
        // id 3: struct sdt_data *.
        CastSynType::Ptr { type_id: 2 },
        // id 4: struct outer { struct sdt_data *data @ 0; } size=8.
        CastSynType::Struct {
            name_off: n_outer,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
        // id 5: struct task_ctx { u64 weight @ 0; } size=8.
        CastSynType::Struct {
            name_off: n_task,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_weight,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 4, 2, 5)
}

/// Type::Ptr arena arm with a Fwd pointee: the renderer's BTF-only
/// resolve fails (no complete sibling for the Fwd), but the
/// [`MemReader::resolve_arena_type`] bridge returns the real
/// payload type id. The chase succeeds and renders the pointee
/// against the recovered type. The resulting `Ptr` carries
/// `cast_annotation: Some("sdt_alloc")` to flag the bridge resolve.
#[test]
fn arena_chase_fwd_target_resolved_via_bridge() {
    let (blob, outer_id, _fwd_id, task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    // outer { data: TARGET_ADDR } — the pointer the renderer
    // chases.
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // task_ctx at TARGET_ADDR: weight = 0x42 (u64 LE).
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0 — the renderer reads
    // `btf_size` bytes directly from the chased address.
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: task_ctx_id,
            header_skip: 0,
        },
    );
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "data");
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "data field must render as Ptr (BTF Type::Ptr arm); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "bridge resolve must not surface a skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("bridge resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved task_ctx struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("task_ctx"),
        "bridge must land on the resolved struct's name, not the Fwd's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "weight");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!(
            "task_ctx.weight must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("sdt_alloc"),
        "Type::Ptr arm bridge resolve must surface 'sdt_alloc' annotation",
    );
}

/// Type::Ptr arena arm with a Fwd pointee resolved via the
/// bridge for a SLOT-START chase: the bridge returns
/// `header_skip = header_size`, the chase reads
/// `header_skip + btf_size` bytes from the chased address, slices
/// off the header, and renders the payload struct. Pins the bug
/// fix that surfaced the `data` field in `scx_task_map_val` (a
/// slot-start pointer that did not resolve under the previous
/// payload-start-only key shape).
///
/// Layout: 8-byte header (the `union sdt_id` shape — two
/// arbitrary u32s here, not interpreted by the bridge), followed
/// by the payload struct (`task_ctx { u64 weight }`, 8 bytes).
/// Total elem_size = 16. The chased address points at slot start;
/// the renderer must NOT decode the header bytes as the payload.
#[test]
fn arena_chase_fwd_target_resolved_via_bridge_slot_start_skips_header() {
    let (blob, outer_id, _fwd_id, task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    // Chased address = slot start. The bridge must direct the
    // chase to skip the first 8 bytes of header before rendering.
    const SLOT_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = SLOT_ADDR.to_le_bytes().to_vec();
    // 16 bytes of slot contents at SLOT_ADDR:
    //   [0..8]   header bytes — sentinel pattern, NOT the payload.
    //            If the renderer decoded them as payload, weight
    //            would resolve to 0xDEADBEEFCAFEBABE.
    //   [8..16]  payload (task_ctx.weight = 0x42, LE u64).
    let header_sentinel = 0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes();
    let payload_bytes = 0x42u64.to_le_bytes();
    let mut slot_bytes = Vec::with_capacity(16);
    slot_bytes.extend_from_slice(&header_sentinel);
    slot_bytes.extend_from_slice(&payload_bytes);

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(SLOT_ADDR, slot_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Slot-start chase: header_skip = 8 — the renderer reads
    // `header_skip + btf_size` bytes from the chased address and
    // slices off the first 8 bytes (the header) before rendering.
    arena_types.insert(
        SLOT_ADDR,
        ArenaResolveHit {
            target_type_id: task_ctx_id,
            header_skip: 8,
        },
    );
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    assert_eq!(value, SLOT_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "slot-start bridge resolve must not surface a skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("slot-start bridge resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved task_ctx struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("task_ctx"),
        "bridge must land on the resolved struct's name even with slot-start skip",
    );
    let RenderedValue::Uint {
        bits,
        value: weight,
    } = inner_members[0].value
    else {
        panic!(
            "task_ctx.weight must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    // Critical assertion: weight is the PAYLOAD value (0x42),
    // not the HEADER sentinel (0xDEADBEEFCAFEBABE). If the
    // renderer skipped the header_skip step it would decode the
    // header bytes as the payload struct.
    assert_eq!(
        weight, 0x42,
        "slot-start chase must skip header — weight \
         must be payload value 0x42, not header sentinel \
         0xDEADBEEFCAFEBABE",
    );
    assert_eq!(
        cast_annotation.as_deref(),
        Some("sdt_alloc"),
        "Type::Ptr arm slot-start bridge resolve must surface 'sdt_alloc' annotation",
    );
}

/// Type::Ptr arena arm with a Fwd pointee but no bridge entry for
/// the chased value: the renderer surfaces the existing
/// "forward declaration; body not in this BTF" skip reason. Pin
/// the no-op behaviour so a misconfigured bridge (empty
/// `arena_type_at` / unkeyed addresses) cannot accidentally render
/// against an unrelated type.
#[test]
fn arena_chase_fwd_target_no_bridge_entry_skips() {
    let (blob, outer_id, _fwd_id, _task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // Reader has the arena window configured but NO entry for
    // TARGET_ADDR in arena_type_at. The bridge call returns
    // `None`; the renderer must surface the standard Fwd skip.
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref.is_none(),
        "no-bridge Fwd target must not produce a deref"
    );
    assert!(
        cast_annotation.is_none(),
        "no-bridge resolve must leave cast_annotation None on the Type::Ptr arm; got {cast_annotation:?}",
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd-no-bridge must populate skip reason");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must surface the forward-declaration cause; got: {reason}",
    );
    assert!(
        reason.contains("sdt_data"),
        "skip reason must include the Fwd type name; got: {reason}",
    );
}

/// Cast intercept arena arm with a Fwd target: the cast analyzer
/// produced a hit for a `u64` field but the target type id is a
/// forward declaration. The bridge resolves it, the chase
/// succeeds, and the resulting `Ptr` carries
/// `cast_annotation: Some("cast→arena (sdt_alloc)")`.
#[test]
fn cast_chase_arena_fwd_target_resolved_via_bridge() {
    // The cast intercept fires on a plain u64 field whose cast
    // hit's target is a Fwd; the bridge resolves the Fwd to a
    // real struct. Build a synthetic BTF with that exact shape
    // (the shared `bridge_btf_outer_fwd_taskctx` fixture has a
    // `struct sdt_data *` field instead, which the cast
    // intercept does not exercise).
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_task = push(&mut strings, "task_ctx");
    let n_f = push(&mut strings, "f");
    let n_weight = push(&mut strings, "weight");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 f @ 0; } size=8.
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: BTF_KIND_FWD struct sdt_data (the cast hit's
        // target_type_id — body absent, body not in this BTF).
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
        // id 4: struct task_ctx { u64 weight @ 0; } size=8 (the
        // bridge's resolved id — distinct from sdt_data so the
        // cast_annotation must reflect that the bridge fired).
        CastSynType::Struct {
            name_off: n_task,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_weight,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let local_fwd_id: u32 = 3;
    let local_task_ctx_id: u32 = 4;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x55u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0.
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: local_task_ctx_id,
            header_skip: 0,
        },
    );
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: local_fwd_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase: no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("bridge-resolved cast must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved task_ctx Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("task_ctx"),
        "bridge must land on the resolved struct, not the Fwd"
    );
    assert_eq!(inner_members.len(), 1);
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "task_ctx.weight must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0x55);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena (sdt_alloc)"),
        "cast intercept arena bridge must extend annotation with '(sdt_alloc)'",
    );
}

/// Cast intercept arena arm with a Fwd target but no bridge entry:
/// the bridge returns None, the chase falls through to the
/// existing "forward declaration" skip path. The resulting `Ptr`
/// carries `cast_annotation: Some("cast→arena")` (no `(sdt_alloc)`
/// suffix) — pinning the no-op annotation when the bridge does
/// not fire.
#[test]
fn cast_chase_arena_fwd_target_no_bridge_keeps_plain_annotation() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_f = push(&mut strings, "f");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let fwd_id: u32 = 3;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: fwd_id,
            addr_space: AddrSpace::Arena,
        },
    );
    // Arena window configured, NO arena_type_at entries → bridge
    // returns None for every chase.
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert!(
        deref.is_none(),
        "no-bridge Fwd cast must not produce a deref"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd-no-bridge must populate skip reason");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must surface forward-declaration cause; got: {reason}",
    );
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "no-bridge cast annotation must NOT include '(sdt_alloc)'; got {cast_annotation:?}",
    );
}

/// Cast intercept kernel arm with a Fwd target + bridge entry:
/// the bridge fires (mirrors the arena arm) and the cast
/// annotation extends to `cast→kernel (sdt_alloc)`. The reader's
/// `is_arena_addr` returns false (kernel-shaped value), so the
/// renderer dispatches to the kernel arm; the bridge wiring
/// covers the symmetric resolve there.
#[test]
fn cast_chase_kernel_fwd_target_resolved_via_bridge() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "kern_fwd");
    let n_real = push(&mut strings, "kern_real");
    let n_f = push(&mut strings, "f");
    let n_x = push(&mut strings, "x");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 f @ 0; }
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: BTF_KIND_FWD struct kern_fwd
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
        // id 4: struct kern_real { u64 x @ 0; }
        CastSynType::Struct {
            name_off: n_real,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let fwd_id: u32 = 3;
    let real_id: u32 = 4;

    // KVA outside any arena window — the runtime dispatcher routes
    // to the kernel arm. Use 0xffff_8000_... pattern (the kernel
    // direct-map range) so plausibility makes sense.
    const KVA: u64 = 0xffff_8000_0000_4000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let inner_bytes = 0x77u64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0.
    arena_types.insert(
        KVA,
        ArenaResolveHit {
            target_type_id: real_id,
            header_skip: 0,
        },
    );
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: fwd_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    // No arena_window — `is_arena_addr` returns false for KVA, so
    // the dispatcher routes to the kernel arm.
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!("intercept must produce Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref_skipped_reason.is_none(),
        "kernel arm bridge resolve must succeed; got skip reason {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("kernel arm bridge resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be kern_real Struct, got {inner:?}");
    };
    assert_eq!(inner_name.as_deref(), Some("kern_real"));
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "kern_real.x must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0x77);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→kernel (sdt_alloc)"),
        "kernel arm bridge must extend annotation with '(sdt_alloc)'",
    );
}

/// Cast intercept arena arm with `target_type_id == 0` (the
/// STX-flow analyzer sentinel for "deferred resolve"): the
/// `chase_arena_pointer` special case at the head of the helper
/// consults [`MemReader::resolve_arena_type`] BEFORE the normal
/// peel + Fwd resolve, expecting the bridge to supply the real
/// payload type id. With a populated `arena_type_at` entry the
/// chase succeeds: the bridge returns the resolved struct id, the
/// renderer reads `btf_size` bytes from the chased address (no
/// header skip), renders the payload struct, and `cast_ptr` emits
/// `cast_annotation: "cast→arena (sdt_alloc)"` because
/// `outcome.sdt_alloc_resolved == true` for the deferred-resolve
/// path (line ~3031 in `mod.rs`).
///
/// Pins the new STX-flow renderer path: a regression that broke
/// the deferred-resolve special case would surface as either a
/// miss (skip reason) or a wrong-id chase (rendered struct name
/// reflects the unrelated u64 underlying type).
#[test]
fn cast_chase_arena_target_type_id_zero_resolves_via_resolve_arena_type() {
    // BTF: u64(1), T(2, u64@0 source field), Q(3, u64@0 payload).
    // The CastHit's target_type_id is 0 (deferred); the bridge
    // must supply Q's id at chase time so the rendered subtree
    // names "Q", not "T" or anything else.
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0 — the renderer reads
    // `btf_size` bytes starting at TARGET_ADDR.
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            // STX-flow sentinel: analyzer left the target id
            // unresolved, expecting the bridge to fill it in.
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "deferred-resolve bridge fire must not surface a skip reason; \
         got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("deferred-resolve bridge must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved Q Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "bridge must land on the resolved struct's name (Q), \
         not the analyzer's deferred sentinel",
    );
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(value, 0x42);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena (sdt_alloc)"),
        "deferred-resolve bridge fire must extend annotation with \
         '(sdt_alloc)' since `outcome.sdt_alloc_resolved` is set; \
         got {cast_annotation:?}",
    );
}

/// Cast intercept arena arm with `target_type_id == 0` AND no
/// bridge entry: the `chase_arena_pointer` special case calls
/// [`MemReader::resolve_arena_type`], gets `None`, and surfaces a
/// skip reason mentioning that the analyzer's STX-flow path tagged
/// the slot as Arena with deferred resolve but the bridge had no
/// entry. Pin the skip reason text so an operator reading a
/// failure dump can correlate the analyzer's hint with the
/// missing bridge population.
///
/// Without this gate, a stale or absent allocator pre-pass would
/// fall through to the normal peel + Fwd resolve path with
/// target_type_id=0 — which would either fail or, worse, succeed
/// against an unrelated BTF id 0 if such existed.
#[test]
fn cast_chase_arena_target_type_id_zero_no_bridge_entry_skips() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // arena_window configured (so `is_arena_addr` returns true and
    // the chase enters the arena arm), but `arena_type_at` is
    // empty — the bridge query for TARGET_ADDR returns None.
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref.is_none(),
        "no-bridge deferred-resolve must not produce a deref"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("no-bridge deferred-resolve must populate skip reason");
    assert!(
        reason.contains("STX-flow path tagged slot as Arena"),
        "skip reason must surface the analyzer's STX-flow tag cause; \
         got: {reason}",
    );
    // `outcome.sdt_alloc_resolved` is `false` on the no-bridge
    // path, so the annotation stays at the unprefixed `cast→arena`
    // form (no `(sdt_alloc)` suffix).
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "no-bridge deferred-resolve must NOT include '(sdt_alloc)' suffix; \
         got {cast_annotation:?}",
    );
}

/// G6.1: Dedup short-circuit. When `is_already_rendered` returns
/// true for the chased arena address, `chase_arena_pointer`
/// surfaces a `Ptr` with `deref: None` and the
/// `"already rendered in sdt_allocations"` skip reason — no
/// arena read, no bridge query, no recursive render. The dedup
/// fires BEFORE the deferred-resolve special case so an address
/// pointing at a slot the sdt_alloc pre-pass already rendered
/// short-circuits even when the analyzer would otherwise have
/// supplied a bridge hit.
#[test]
fn cast_chase_already_rendered_short_circuits() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // Seed the dedup set with TARGET_ADDR's low-32 bits — the
    // production [`AccessorMemReader::is_already_rendered`] keys
    // on `addr as u32` (low-32 windowed slot start). Even though
    // arena_type_at would have resolved the address, the dedup
    // takes precedence and skips the chase.
    let mut rendered = std::collections::HashSet::new();
    rendered.insert(TARGET_ADDR as u32);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        rendered_slot_addrs: rendered,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "dedup must still produce Ptr (only the deref is suppressed); \
             got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref.is_none(),
        "dedup short-circuit must suppress the deref"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("dedup must populate the skip reason");
    assert_eq!(
        reason, "already rendered in sdt_allocations",
        "dedup skip reason is wire-stable (operator reads it from \
         RenderedValue::Ptr::deref_skipped_reason); the exact format \
         is part of the dump's machine-checkable contract: got '{reason}'"
    );
}

/// G6.2: Dedup with miss falls through to normal chase. An
/// address NOT in `rendered_slot_addrs` proceeds with the
/// existing chase pipeline (bridge query, peel, read, render).
/// Pins that the dedup gate is per-address and does not blank
/// the chase wholesale when a different slot was rendered.
#[test]
fn cast_chase_already_rendered_miss_proceeds_with_normal_chase() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    const RENDERED_OTHER_ADDR: u64 = 0x10_0000_2000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // Dedup set has a different slot start. The chase target
    // (TARGET_ADDR) is NOT in the set, so dedup misses and the
    // chase proceeds normally through the bridge.
    let mut rendered = std::collections::HashSet::new();
    rendered.insert(RENDERED_OTHER_ADDR as u32);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        rendered_slot_addrs: rendered,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr { ref deref, .. } = members[0].value else {
        panic!("expected Ptr, got {:?}", members[0].value);
    };
    let inner = deref
        .as_deref()
        .expect("dedup-miss path must still produce a deref via the bridge");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        ..
    } = *inner
    else {
        panic!("deref payload must be the resolved Q Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "dedup-miss path must land on Q via the normal chase pipeline",
    );
}

/// G6.3: Default `is_already_rendered` returns false. Readers
/// without a rendered-slot index (the trait default impl)
/// proceed with the chase — pins the no-regression case for
/// every existing renderer that doesn't wire the dedup set.
#[test]
fn cast_chase_default_is_already_rendered_returns_false() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // No rendered_slot_addrs — the field defaults to an empty
    // HashSet, so `is_already_rendered` returns false for every
    // address. The chase must proceed without the short-circuit.
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr { ref deref, .. } = members[0].value else {
        panic!("expected Ptr, got {:?}", members[0].value);
    };
    assert!(
        deref.is_some(),
        "empty rendered_slot_addrs must NOT short-circuit the chase",
    );
}

/// Cast intercept kernel arm with `target_type_id == 0`: the
/// analyzer hinted Arena (the STX-flow sentinel only emits with
/// `addr_space: Arena`) but the runtime value falls outside the
/// arena window so `is_arena_addr` returns false and the kernel
/// arm fires. The kernel arm's special case at line ~3390 of
/// `mod.rs` recognises `target_type_id == 0` as the cgx-bridge
/// sentinel and surfaces a skip reason explaining the
/// analyzer/runtime mismatch — without a BTF id there is no way
/// to size the kernel read.
///
/// Pins the kernel-arm fall-through behaviour: a regression that
/// stripped the special case would attempt to peel type id 0 in
/// the program BTF (which fails) and surface a less useful
/// "kernel cast target type id 0 unresolvable" message.
#[test]
fn cast_chase_kernel_target_type_id_zero_falls_through_with_mismatch_reason() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // KVA pattern (kernel direct-map range) outside ANY arena
    // window — the dispatcher routes to the kernel arm. The reader
    // has no `arena_window` configured, so `is_arena_addr` returns
    // `false` for every value and the kernel arm receives the
    // chase.
    const KVA: u64 = 0xffff_8000_0000_4000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            // STX-flow analyzer sentinel — intentionally `Arena`
            // since that is the only space the analyzer emits with
            // target_type_id=0. Runtime detection sees the value
            // outside the arena window and routes to the kernel
            // arm, exercising the kernel-arm `target_type_id == 0`
            // special case.
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "kernel-arm `target_type_id == 0` special case must skip the chase",
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("kernel-arm `target_type_id == 0` must populate skip reason");
    assert!(
        reason.contains("kernel cast target unresolved"),
        "skip reason must mention `kernel cast target unresolved`; \
         got: {reason}",
    );
    assert!(
        reason.contains("analyzer hinted Arena with deferred resolve"),
        "skip reason must surface the analyzer-hint / runtime-window \
         mismatch; got: {reason}",
    );
    // Kernel-arm path: `cast_ptr` is called with `sdt_alloc_resolved = false`
    // (line ~3401 in mod.rs), so the annotation reflects the actual
    // path taken (kernel) without the sdt_alloc suffix.
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→kernel"),
        "kernel-arm fall-through must use `cast→kernel` annotation \
         (the path actually taken); got {cast_annotation:?}",
    );
}

/// `Type::Ptr` arena arm: an out-of-arena-window pointer must
/// NOT fire the sdt_alloc bridge, even when the
/// `MemReader::resolve_arena_type` table contains a stale entry
/// for that exact address.
///
/// The arm dispatches on `is_arena_addr(value)` BEFORE entering
/// `chase_arena_pointer`. The bridge lives inside the chase
/// helper, so an out-of-window value skips the helper entirely
/// and falls into the kernel-kptr branch (cpumask-name dispatch /
/// Fwd-no-body skip). This test pins the bridge's no-op behaviour
/// for the kptr branch by asserting the rendered Ptr has neither a
/// successful deref nor an `sdt_alloc` annotation.
///
/// The fixture's bridge map keys on the FULL 64-bit address
/// (`CastStubReader` does not implement the production
/// `addr & 0xFFFF_FFFF` mask). That is fine for this test — the
/// gating happens before the lookup runs.
#[test]
fn arena_chase_bridge_address_outside_window_is_no_op() {
    let (blob, outer_id, _fwd_id, task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    // OUT_OF_WINDOW lies BELOW the configured arena window; the
    // BTF Type::Ptr arm dispatches on `is_arena_addr` so it never
    // reaches `chase_arena_pointer` for this value, even though
    // the bridge index has an entry for it. Verifies that an
    // out-of-window address with a stale bridge entry cannot
    // accidentally surface as a chased struct.
    const OUT_OF_WINDOW: u64 = 0x0F_0000_1000;
    let outer_bytes = OUT_OF_WINDOW.to_le_bytes().to_vec();
    let mut arena_types = std::collections::HashMap::new();
    // Stale entry mapped to a payload-start shape — the gate must
    // reject the address before this entry is consulted.
    arena_types.insert(
        OUT_OF_WINDOW,
        ArenaResolveHit {
            target_type_id: task_ctx_id,
            header_skip: 0,
        },
    );
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    // The Type::Ptr arm reaches `chase_arena_pointer` only when
    // `is_arena_addr` returned true. For an out-of-window value,
    // the kernel-kptr branch of the Type::Ptr arm runs; that
    // branch has its own peel/size resolve and bails on a Fwd
    // pointee whose body is missing.
    assert!(
        deref.is_none(),
        "out-of-window pointer must not chase via the bridge"
    );
    assert!(
        cast_annotation.is_none(),
        "BTF Type::Ptr arm must leave cast_annotation None on the kptr branch"
    );
    // Surface either the cpumask-name dispatch reason
    // ("size 0" or absent), the Fwd reason, or the BTF-resolution
    // failure — any of which is correct for the kptr branch with
    // a Fwd pointee. The test only asserts the bridge did NOT
    // fire (no successful deref, no annotation).
    let _ = deref_skipped_reason;
}

/// Cross-BTF Fwd resolution end-to-end: the entry BTF declares
/// `outer { u64 cgx_raw @ 0 }` plus `struct cgx_target;` (a
/// `BTF_KIND_FWD` only — no body). A sibling BTF defines
/// `struct cgx_target { u64 marker @ 0 }` (the body). The cast
/// analyzer recovered `(outer, 0) -> (cgx_target, Arena)` so the
/// chase enters [`render_cast_pointer`] → arena branch →
/// [`chase_arena_pointer`]. Local Fwd resolve fails (no sibling in
/// the entry BTF), the sdt_alloc bridge stays dormant, then
/// [`try_cross_btf_fwd_resolve`] consults
/// [`MemReader::cross_btf_resolve_fwd`] which returns the sibling
/// BTF's `cgx_target` body. The recursion renders against that
/// body and produces `cgx_target { marker = 0xCAFE }`.
///
/// Without the cross-BTF bridge, the chase would have skipped
/// with "forward declaration; body not in this BTF" and the
/// rendered output would be a bare `Ptr` carrying the chased
/// address.
#[test]
fn cross_btf_fwd_resolve_renders_cgx_body_through_sibling_btf() {
    use std::sync::Arc;

    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };

    // Entry BTF: outer (id 2) carrying a u64 cgx_raw @ 0; Fwd of
    // cgx_target (id 3, struct, no body).
    let mut s_a = vec![0u8];
    let n_a_u64 = push(&mut s_a, "u64");
    let n_a_outer = push(&mut s_a, "outer");
    let n_a_field = push(&mut s_a, "cgx_raw");
    let n_a_cgx = push(&mut s_a, "cgx_target");
    let types_a = vec![
        CastSynType::Int {
            name_off: n_a_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_a_outer,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_a_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_a_cgx,
            is_union: false,
        },
    ];
    let blob_a = cast_build_btf(&types_a, &s_a);
    let btf_entry = Btf::from_bytes(&blob_a).expect("entry BTF parses");

    // Sibling BTF: cgx_target as a complete struct (id 2) with
    // `u64 marker @ 0`.
    let mut s_b = vec![0u8];
    let n_b_u64 = push(&mut s_b, "u64");
    let n_b_cgx = push(&mut s_b, "cgx_target");
    let n_b_marker = push(&mut s_b, "marker");
    let types_b = vec![
        CastSynType::Int {
            name_off: n_b_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_b_cgx,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_b_marker,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_b = cast_build_btf(&types_b, &s_b);
    let btf_sibling = Arc::new(Btf::from_bytes(&blob_b).expect("sibling BTF parses"));

    // Configure CastStubReader: outer.cgx_raw u64 maps to a
    // Pointer{cgx_target}; the chased value is an arena address;
    // arena bytes at that address carry the cgx_target body.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_2000;
    let outer_id = 2u32;
    // Cast hint resolves the u64 slot at (outer, 0) to the entry
    // BTF's Fwd `cgx_target` (id 3). The chase then asks the
    // reader to bridge that Fwd via cross-BTF resolution.
    let cgx_fwd_id = 3u32;
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (outer_id, 0),
        CastHit { alloc_size: None,
            target_type_id: cgx_fwd_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    // Sibling cgx_target body: marker = 0xCAFE.
    arena_bytes.insert(TARGET_ADDR, 0xCAFEu64.to_le_bytes().to_vec());
    let mut cross_btf_index = std::collections::HashMap::new();
    cross_btf_index.insert("cgx_target".to_string(), (0usize, 2u32, true));
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        cross_btf_btfs: vec![btf_sibling.clone()],
        cross_btf_index,
        ..Default::default()
    };

    // Render outer; cgx_raw must surface as a Ptr whose deref
    // renders against the SIBLING BTF's cgx_target body — the
    // marker field is 0xCAFE.
    let v = render_value_with_mem(&btf_entry, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr { ref deref, .. } = members[0].value else {
        panic!("cgx_raw must render as Ptr; got {:?}", members[0].value);
    };
    let inner = deref
        .as_ref()
        .expect("cross-BTF Fwd resolve must produce a deref (sibling BTF body), but got None");
    let RenderedValue::Struct {
        ref type_name,
        ref members,
    } = **inner
    else {
        panic!("inner must be Struct (cgx_target body); got {inner:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("cgx_target"),
        "rendered subtree must carry the sibling BTF's struct name"
    );
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "marker");
    let RenderedValue::Uint { value: marker, .. } = members[0].value else {
        panic!("marker must render as Uint; got {:?}", members[0].value);
    };
    assert_eq!(
        marker, 0xCAFE,
        "rendered marker must come from the cross-BTF body's bytes"
    );

    // Sanity: drop the cross-BTF index and re-render — without
    // the bridge, the chase must skip with the Fwd reason and
    // the deref stays None.
    let reader_no_bridge = CastStubReader {
        cast_map: Some({
            let mut m = super::super::cast_analysis::CastMap::new();
            m.insert(
                (outer_id, 0),
                CastHit { alloc_size: None,
                    target_type_id: cgx_fwd_id,
                    addr_space: AddrSpace::Arena,
                },
            );
            m
        }),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: {
            let mut a = std::collections::HashMap::new();
            a.insert(TARGET_ADDR, 0xCAFEu64.to_le_bytes().to_vec());
            a
        },
        ..Default::default()
    };
    let v = render_value_with_mem(&btf_entry, outer_id, &outer_bytes, &reader_no_bridge);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("cgx_raw must render as Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref.is_none(),
        "without cross-BTF bridge, Fwd target must not chase"
    );
    let reason = deref_skipped_reason
        .as_ref()
        .expect("Fwd skip must populate deref_skipped_reason");
    assert!(
        reason.contains("cgx_target") && reason.contains("forward declaration"),
        "skip reason must name the Fwd target: {reason:?}"
    );
}

/// Kernel-arm cross-BTF Fwd resolution: a `CastHit` with
/// `addr_space: Kernel` whose target_type_id resolves to a
/// `BTF_KIND_FWD` in the entry BTF — the renderer's kernel arm
/// dispatches on `is_arena_addr(value)` returning false (no arena
/// window matches), then peels the Fwd target. With the
/// sdt_alloc bridge dormant (no `arena_type_at` entry for the
/// kernel value), `chase_arena_pointer` is NOT invoked but the
/// kernel arm shares the same `try_cross_btf_fwd_resolve` shortcut
/// (line ~3447 in `mod.rs`): a sibling BTF whose
/// [`MemReader::cross_btf_resolve_fwd`] override matches the Fwd
/// name surfaces the body, the kernel read fires against the
/// sibling-BTF's resolved type id, and the rendered subtree
/// names the sibling struct.
///
/// Pin the symmetric kernel-arm wiring against a regression that
/// stripped the cross-BTF probe from the kernel arm (only arena
/// arm honoured it). The shared
/// [`try_cross_btf_fwd_resolve`] call at the kernel-arm
/// fall-through is the only mechanism for kernel-targeted Fwd
/// resolves through a sibling BTF.
#[test]
fn cast_chase_kernel_cross_btf_fwd_resolve_succeeds() {
    use std::sync::Arc;

    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };

    // Entry BTF: T (id 2) with a u64 field at offset 0; Fwd of
    // kern_target (id 3, struct, no body in this BTF).
    let mut s_entry = vec![0u8];
    let n_e_u64 = push(&mut s_entry, "u64");
    let n_e_t = push(&mut s_entry, "T");
    let n_e_f = push(&mut s_entry, "f");
    let n_e_kern = push(&mut s_entry, "kern_target");
    let types_entry = vec![
        CastSynType::Int {
            name_off: n_e_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_e_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_e_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: BTF_KIND_FWD struct kern_target — body lives in
        // the sibling BTF, the renderer must consult the cross-BTF
        // index.
        CastSynType::Fwd {
            name_off: n_e_kern,
            is_union: false,
        },
    ];
    let blob_entry = cast_build_btf(&types_entry, &s_entry);
    let btf_entry = Btf::from_bytes(&blob_entry).expect("entry BTF parses");
    let t_id: u32 = 2;
    let kern_fwd_id: u32 = 3;

    // Sibling BTF: kern_target as a complete struct (id 2) with
    // `u64 marker @ 0`.
    let mut s_sib = vec![0u8];
    let n_s_u64 = push(&mut s_sib, "u64");
    let n_s_kern = push(&mut s_sib, "kern_target");
    let n_s_marker = push(&mut s_sib, "marker");
    let types_sib = vec![
        CastSynType::Int {
            name_off: n_s_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_s_kern,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_s_marker,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_sib = cast_build_btf(&types_sib, &s_sib);
    let btf_sib = Arc::new(Btf::from_bytes(&blob_sib).expect("sibling BTF parses"));

    // Kernel value outside any arena window — the dispatcher
    // routes to the kernel arm. Use the kernel direct-map range
    // so the plausibility-gate sanity holds (top byte 0xff would
    // trigger the freed-slab heuristic).
    const KVA: u64 = 0xffff_8000_0001_2000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Sibling kern_target body: marker = 0xBEEF.
    let inner_bytes = 0xBEEFu64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    let mut cross_btf_index = std::collections::HashMap::new();
    // Cross-BTF index: kern_target -> (sibling BTF index 0, type
    // id 2, want_struct=true).
    cross_btf_index.insert("kern_target".to_string(), (0usize, 2u32, true));
    let mut cast_map = super::super::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit { alloc_size: None,
            target_type_id: kern_fwd_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        // No arena_window — `is_arena_addr` returns false for KVA,
        // dispatcher routes to the kernel arm. No `arena_type_at`
        // — the sdt_alloc bridge stays dormant on the kernel arm
        // so the cross-BTF shortcut fires instead.
        cross_btf_btfs: vec![btf_sib.clone()],
        cross_btf_index,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf_entry, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "kernel cast intercept must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert!(
        deref_skipped_reason.is_none(),
        "kernel-arm cross-BTF Fwd resolve must succeed; \
         got skip reason {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_ref()
        .expect("kernel-arm cross-BTF Fwd resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = **inner
    else {
        panic!("deref payload must be the kern_target body Struct; got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("kern_target"),
        "rendered subtree must carry the sibling BTF's struct name",
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "marker");
    let RenderedValue::Uint { value: marker, .. } = inner_members[0].value else {
        panic!(
            "kern_target.marker must render as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(
        marker, 0xBEEF,
        "rendered marker must come from the kva-side body bytes \
         decoded against the sibling BTF",
    );
}
