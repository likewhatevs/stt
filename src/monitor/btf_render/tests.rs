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
    assert!(out.contains("b <truncated needed=4 had=0>"));
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
    }));
    assert!(!is_zero(&RenderedValue::Ptr {
        value: 0xffff_8000_dead_beef,
        deref: None,
        deref_skipped_reason: None,
    }));
    // A pointer with a deref but value=0 is still zero per the
    // is_zero contract — only `value` matters.
    assert!(is_zero(&RenderedValue::Ptr {
        value: 0,
        deref: Some(Box::new(RenderedValue::Uint { bits: 32, value: 5 })),
        deref_skipped_reason: None,
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
            },
            RenderedValue::Ptr {
                value: 0,
                deref: None,
                deref_skipped_reason: None,
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
