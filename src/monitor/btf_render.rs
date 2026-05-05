//! BTF-driven rendering of raw value bytes into structured output.
//!
//! [`render_value`] takes a BTF type id and a byte slice and produces
//! a [`RenderedValue`] tree that mirrors the type's structure: ints,
//! floats, enums, structs, arrays, pointers. Modifier qualifiers
//! ([`btf_rs::Type::Volatile`], [`btf_rs::Type::Const`],
//! [`btf_rs::Type::Restrict`], [`btf_rs::Type::Typedef`],
//! [`btf_rs::Type::TypeTag`]) are peeled before dispatch.
//!
//! The renderer is total: any type kind it cannot decode (Func, FuncProto,
//! Fwd, Void, or a bytes slice shorter than the type's declared size)
//! yields a [`RenderedValue::Unsupported`] or [`RenderedValue::Truncated`]
//! node so the caller always gets a well-formed tree it can serialize.
//!
//! `BTF_KIND_DATASEC` (the type libbpf assigns as the value type of a
//! global-section ARRAY map like `.bss` / `.data` / `.rodata`) is
//! rendered by walking its `VarSecinfo` entries. Each VarSecinfo points
//! at a `BTF_KIND_VAR`, which in turn references the variable's actual
//! type. The renderer slices the section bytes at
//! `[var_secinfo.offset()..var_secinfo.offset() + var_secinfo.size()]`
//! and recursively renders the variable's type into that slice. The
//! result is a [`RenderedValue::Struct`] whose `type_name` is the
//! section name and whose `members` enumerate the section's variables
//! by their declared names, so a failure dump's `.bss` map shows
//! `stall=1, crash=0, ...` instead of an opaque hex dump.
//!
//! Bitfield handling: when [`btf_rs::Member::bitfield_size`] is `Some(w)`,
//! the renderer reads enough bytes to cover the bitfield's bit range,
//! shifts and masks, and applies sign extension if the underlying int
//! kind is signed.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use btf_rs::{Btf, BtfType, Member, Struct, Type};

use super::dump::hex_dump;

/// Maximum number of qualifier-peel iterations before [`peel_modifiers`]
/// gives up. BTF chains are bounded in practice (struct → typedef →
/// const → volatile is already 4); the cap protects the renderer from
/// a malformed BTF input that introduces a self-referential cycle.
const MAX_MODIFIER_DEPTH: u32 = 32;

/// Maximum array length the renderer expands element-by-element. Larger
/// arrays are truncated with the original length recorded so the caller
/// can serialize a partial view rather than allocating a million
/// [`RenderedValue`] nodes.
const MAX_ARRAY_ELEMS: usize = 4096;

/// Recursion depth limit for nested struct / array rendering. Same
/// motivation as [`MAX_MODIFIER_DEPTH`]: bound output size on
/// pathological BTF.
const MAX_RENDER_DEPTH: u32 = 32;

/// Structured rendering of one BTF-typed value.
///
/// The `kind` tag identifies the variant; field order matches the
/// rendering pipeline (Int / Uint / Bool / Char before Float / Enum /
/// Struct / Array / Ptr, with Bytes / Truncated / Unsupported as the
/// recovery path).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(dead_code)]
pub enum RenderedValue {
    /// Signed integer. `bits` is the BTF-declared width.
    Int { bits: u32, value: i64 },
    /// Unsigned integer. `bits` is the BTF-declared width.
    Uint { bits: u32, value: u64 },
    /// Boolean (BTF int with `is_bool()`).
    Bool { value: bool },
    /// Character (BTF int with `is_char()`). `value` holds the raw
    /// byte for round-tripping; non-printable values are preserved.
    /// Field name matches the other scalar variants (Int / Uint / Bool /
    /// Float / Enum / Ptr) so any field-driven serializer treats them
    /// uniformly.
    Char { value: u8 },
    /// IEEE-754 float (BTF_KIND_FLOAT).
    Float { bits: u32, value: f64 },
    /// Enum value with optional resolved variant name.
    Enum {
        bits: u32,
        value: i64,
        variant: Option<String>,
    },
    /// Aggregate (struct or union). For unions, only the first member
    /// is meaningful — the renderer emits all members each backed by
    /// the same byte range so the caller can pick.
    Struct {
        type_name: Option<String>,
        members: Vec<RenderedMember>,
    },
    /// Array. `elements` is truncated to [`MAX_ARRAY_ELEMS`].
    Array {
        len: usize,
        elements: Vec<RenderedValue>,
    },
    /// CPU bitmask rendered as a range-collapsed list. Produced when
    /// the renderer detects a `cpumask`, `bpf_cpumask`, or `scx_bitmap`
    /// struct by BTF type name.
    ///
    /// Field-name exception: scalar variants (Int / Uint / Bool /
    /// Char / Float / Enum / Ptr) name the inner field `value` for
    /// uniform serialization; CpuList breaks the convention by
    /// using `cpus` because the rendered text is type-specific
    /// (e.g. `"0-2,5"`) and reads more naturally as `cpus` in
    /// JSON consumers — `value` would be misleading for a non-
    /// scalar payload. Char keeps `value` since its payload is a
    /// single byte (the BTF int type is the underlying scalar).
    CpuList { cpus: String },
    /// Pointer value with optional dereferenced content. When `deref`
    /// is Some, the pointer was chased at dump time and the target
    /// struct is rendered inline. `deref_skipped_reason` carries the
    /// cause when the chase was attempted but did not produce a
    /// deref — `None` means no chase was attempted (e.g. null
    /// pointer or no [`MemReader`] supplied), and a non-`None`
    /// reason with `deref: None` means the chase was attempted but
    /// could not complete (cross-page boundary, BTF-size truncated
    /// against the read cap, kernel kptr that failed plausibility
    /// gating, etc.). The reason field enables the consumer to
    /// distinguish "we didn't try" from "we tried and failed for
    /// reason X" without a separate flag.
    Ptr {
        value: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deref: Option<Box<RenderedValue>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deref_skipped_reason: Option<String>,
    },
    /// Fallback hex dump for types the renderer can decode the size of
    /// but not the structure (e.g. `BTF_KIND_FWD`). Hex is lowercase,
    /// space-separated.
    Bytes { hex: String },
    /// The byte slice ended before the type's declared size. `needed`
    /// is the required byte count; `had` is what was supplied.
    /// `partial` carries whatever decoded successfully before the
    /// truncation: a `Struct` with the members that fit (further
    /// truncated members nest as their own [`Truncated`]), an
    /// `Array` with the elements that fit, or a `Bytes` hex dump of
    /// the raw bytes that were available when no structured partial
    /// applied (e.g. a 2-byte slice for a 4-byte int).
    Truncated {
        needed: usize,
        had: usize,
        partial: Box<RenderedValue>,
    },
    /// BTF type kind the renderer does not handle (Func, FuncProto,
    /// Fwd, Void, or a kind beyond the qualifier-peel cap).
    /// `reason` carries the human-readable cause.
    Unsupported { reason: String },
}

/// One member of a [`RenderedValue::Struct`]. `name` is the BTF name;
/// for anonymous union members it is empty. `value` is the recursive
/// rendering of the member's bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedMember {
    pub name: String,
    pub value: RenderedValue,
}

impl std::fmt::Display for RenderedValue {
    /// Human-readable rendering for test-failure output. JSON remains
    /// the programmatic form (via `serde_json`); this Display emits
    /// pretty-printed text suitable for assertion failure messages,
    /// e.g.
    ///
    /// ```text
    /// struct task_ctx {
    ///   weight: 1024
    ///   last_runnable_at: 12345678901234
    /// }
    /// ```
    ///
    /// Nested structs and arrays indent by two spaces per level.
    /// Scalar-only arrays render inline (`[1, 2, 3]`); arrays
    /// containing structs / nested arrays render block-style with
    /// one element per line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_rendered_value(f, self, 0)
    }
}

/// Indentation prefix. Two-space steps match the example in the
/// module-level Display doc.
const INDENT: &str = "  ";

/// Render a [`RenderedValue`] with a caller-supplied starting
/// indentation depth. Wrapper modules (e.g.
/// [`crate::monitor::dump::display`]) use this to nest a renderer
/// output inside their own indented context — passing
/// `depth = 1` produces output indented one level deeper than the
/// default `Display::fmt` path (which always starts at `depth = 0`).
pub(crate) fn write_value_at_depth(
    f: &mut std::fmt::Formatter<'_>,
    v: &RenderedValue,
    depth: usize,
) -> std::fmt::Result {
    write_rendered_value(f, v, depth)
}

/// Recursive Display helper. Tracks current indentation `depth` so
/// nested structs / block-style arrays line up correctly. Direct
/// `Display::fmt` cannot pass extra state, so this is the entry
/// point for every recursive call.
fn write_rendered_value(
    f: &mut std::fmt::Formatter<'_>,
    v: &RenderedValue,
    depth: usize,
) -> std::fmt::Result {
    match v {
        RenderedValue::Int { value, .. } => {
            let mut buf = itoa::Buffer::new();
            f.write_str(buf.format(*value))
        }
        RenderedValue::Uint { value, .. } => {
            // Genuine unsigned integers render as decimal regardless
            // of magnitude. Pointer-typed values are handled by the
            // [`RenderedValue::Ptr`] arm — the BTF type drives the
            // format, not the value. A `u64` counter that happens
            // to land in the kernel-pointer numeric range still
            // renders as decimal because that is what the BPF
            // programmer declared. Pointer-shaped values declared
            // as pointers (Type::Ptr in BTF, possibly through
            // typedefs which `peel_modifiers` collapses) reach the
            // Ptr Display arm below and emit `0x<hex>` there.
            let mut buf = itoa::Buffer::new();
            f.write_str(buf.format(*value))
        }
        RenderedValue::Bool { value } => f.write_str(if *value { "true" } else { "false" }),
        RenderedValue::Char { value } => {
            if (0x20..=0x7e).contains(value) {
                f.write_str("'")?;
                f.write_str(core::str::from_utf8(&[*value]).unwrap_or("?"))?;
                f.write_str("'")
            } else {
                write!(f, "0x{value:02x}")
            }
        }
        RenderedValue::Float { value, .. } => write!(f, "{value}"),
        RenderedValue::Enum { value, variant, .. } => match variant {
            Some(name) => {
                f.write_str(name)?;
                f.write_str(" (")?;
                let mut buf = itoa::Buffer::new();
                f.write_str(buf.format(*value))?;
                f.write_str(")")
            }
            None => {
                let mut buf = itoa::Buffer::new();
                f.write_str(buf.format(*value))
            }
        },
        RenderedValue::CpuList { cpus } => write!(f, "cpus={{{cpus}}}"),
        RenderedValue::Ptr {
            value,
            deref,
            deref_skipped_reason,
        } => {
            write!(f, "0x{value:x}")?;
            if let Some(inner) = deref {
                f.write_str(" → ")?;
                write_rendered_value(f, inner, depth)?;
            } else if let Some(reason) = deref_skipped_reason {
                // Surface the chase-skip cause inline. Cycle markers
                // collapse to the dense `[cycle]` form — operators
                // recognise it without needing the address repeated
                // (the address is already in the pointer's hex
                // value preceding this marker). Other skip reasons
                // (cross-page failure, plausibility gate, etc.)
                // keep the verbose `[chase: ...]` form so the
                // specific cause is visible.
                if reason.starts_with("cycle ") {
                    f.write_str(" [cycle]")?;
                } else {
                    write!(f, " [chase: {reason}]")?;
                }
            }
            Ok(())
        }
        RenderedValue::Bytes { hex } => f.write_str(hex),
        RenderedValue::Truncated {
            needed,
            had,
            partial,
        } => {
            write!(f, "<truncated needed={needed} had={had}> ")?;
            write_rendered_value(f, partial, depth)
        }
        RenderedValue::Unsupported { reason } => write!(f, "<unsupported: {reason}>"),
        RenderedValue::Array { len, elements } => {
            if elements.is_empty() {
                return write!(f, "[]");
            }
            // Detect i8/u8 arrays that are C strings: all elements
            // are 8-bit Int/Uint, mostly printable ASCII or NUL.
            // Detect non-empty C strings: 8-bit arrays starting with
            // a non-NUL printable byte. All-zero arrays and arrays
            // starting with NUL are NOT strings (they're zero data).
            let first_byte = match &elements[0] {
                RenderedValue::Int { bits: 8, value } => Some(*value as u8),
                RenderedValue::Uint { bits: 8, value } => Some(*value as u8),
                RenderedValue::Char { value } => Some(*value),
                _ => None,
            };
            let is_string = first_byte.is_some_and(|b| b != 0 && is_text_byte(b))
                && elements.len() >= 2
                && elements.iter().all(|e| match e {
                    RenderedValue::Int { bits: 8, value } => is_text_byte(*value as u8),
                    RenderedValue::Uint { bits: 8, value } => is_text_byte(*value as u8),
                    RenderedValue::Char { value } => is_text_byte(*value),
                    _ => false,
                });
            if is_string {
                // Build the string to check if it's multi-line.
                let mut s = String::new();
                for e in elements {
                    let ch = match e {
                        RenderedValue::Int { value, .. } => *value as u8,
                        RenderedValue::Uint { value, .. } => *value as u8,
                        RenderedValue::Char { value } => *value,
                        _ => 0,
                    };
                    if ch == 0 {
                        break;
                    }
                    s.push(ch as char);
                }
                if s.contains('\n') {
                    // Multi-line: render with actual newlines, indented.
                    f.write_str("|\n")?;
                    for line in s.split('\n') {
                        if line.is_empty() {
                            continue;
                        }
                        write_indent(f, depth + 1)?;
                        f.write_str(line)?;
                        f.write_str("\n")?;
                    }
                    write_indent(f, depth)?;
                } else {
                    write!(f, "\"{s}\"")?;
                }
                return Ok(());
            }
            let inline = elements.iter().all(is_inline_scalar);
            if inline {
                // Build contiguous runs of non-zero elements. Each
                // run carries (start_idx, end_idx_inclusive,
                // [&values]). Zero elements break a run; the gaps
                // between runs surface implicitly (no `(N zero)`
                // count needed when run brackets carry the index).
                let mut runs: Vec<(usize, usize, Vec<&RenderedValue>)> = Vec::new();
                for (i, e) in elements.iter().enumerate() {
                    if is_zero(e) {
                        continue;
                    }
                    if let Some(last) = runs.last_mut()
                        && last.1 + 1 == i
                    {
                        last.1 = i;
                        last.2.push(e);
                    } else {
                        runs.push((i, i, vec![e]));
                    }
                }

                // All-zero short-circuit. The `[all N zero]` glyph
                // makes "every slot is zero" obvious without
                // listing every index.
                if runs.is_empty() {
                    return write!(f, "[all {len} zero]");
                }

                // Pre-render every element's text form so we can
                // measure widths and pack rows with element-
                // boundary wrapping (no mid-value breaks).
                let render_elem = |e: &RenderedValue| -> String {
                    use std::fmt::Write;
                    let mut s = String::new();
                    match e {
                        RenderedValue::Uint { value, bits } if *bits >= 32 => {
                            let _ = write!(s, "{value:#x}");
                        }
                        _ => {
                            let _ = write!(s, "{e}");
                        }
                    }
                    s
                };

                // Special case: a single run covering the whole
                // array starting at 0 means there are no gaps —
                // emit a plain `[v1, v2, ...]` without index
                // brackets. Long lists wrap at element boundaries
                // with continuation indented to align with the
                // first element after `[`.
                if runs.len() == 1
                    && runs[0].0 == 0
                    && runs[0].1 + 1 == elements.len()
                    && elements.len() == *len
                {
                    let strs: Vec<String> = runs[0].2.iter().map(|e| render_elem(e)).collect();
                    write_inline_list_wrapped(f, "[", "]", &strs, ", ", depth)?;
                    return Ok(());
                }

                // Sparse render: each run as `[start..end]={v, v, ...}`
                // (or `[idx]=v` for single-element runs). Multiple
                // runs separate with two spaces and wrap to a new
                // line at run boundaries when the line would
                // exceed the inline budget.
                let run_strs: Vec<String> = runs
                    .iter()
                    .map(|(start, end, vals)| {
                        if start == end {
                            format!("[{start}]={}", render_elem(vals[0]))
                        } else {
                            let inner: Vec<String> = vals.iter().map(|v| render_elem(v)).collect();
                            format!("[{start}..{end}]={{{}}}", inner.join(", "))
                        }
                    })
                    .collect();
                write_inline_list_wrapped(f, "[", "]", &run_strs, "  ", depth)?;
                if elements.len() < *len {
                    write!(f, " /* {} of {len} shown */", elements.len())?;
                }
                Ok(())
            } else {
                // Group identical elements by content. Show each
                // unique value once with its index range.
                f.write_str("[")?;
                let mut zero_count = 0usize;
                let mut groups: Vec<(usize, usize, &RenderedValue)> = Vec::new();
                for (i, e) in elements.iter().enumerate() {
                    if is_zero(e) {
                        zero_count += 1;
                        continue;
                    }
                    if let Some(g) = groups.last_mut()
                        && g.2 == e
                    {
                        g.1 = i;
                        continue;
                    }
                    groups.push((i, i, e));
                }
                if zero_count == elements.len() {
                    return write!(f, "all {len} zero]");
                }
                // Render groups, merging consecutive similar structs
                // (differ by <8 fields) into a single template with
                // a per-index table for the varying fields.
                let mut i = 0;
                while i < groups.len() {
                    let (start, end, val) = &groups[i];
                    // Try to extend a run of similar singletons.
                    if start == end
                        && let RenderedValue::Struct {
                            members: first_m, ..
                        } = val
                    {
                        let mut run_end = i;
                        'scan: while run_end + 1 < groups.len() {
                            let (ns, ne, nv) = &groups[run_end + 1];
                            if ns != ne {
                                break;
                            }
                            if let RenderedValue::Struct {
                                members: next_m, ..
                            } = nv
                            {
                                if next_m.len() != first_m.len() {
                                    break;
                                }
                                let diffs = first_m
                                    .iter()
                                    .zip(next_m.iter())
                                    .filter(|(a, b)| a.value != b.value)
                                    .count();
                                if diffs >= 8 {
                                    break 'scan;
                                }
                            } else {
                                break;
                            }
                            run_end += 1;
                        }
                        if run_end > i {
                            // Try to merge [i..=run_end] into a template.
                            let run = &groups[i..=run_end];
                            if try_write_struct_template(f, run, depth + 1)? {
                                i = run_end + 1;
                                continue;
                            }
                            // try_write_struct_template returned false
                            // (run too short, no varying fields, or > 3
                            // varying). Wrote nothing. Fall through to
                            // the per-element block below for groups[i];
                            // the loop renders the rest of the run one
                            // group at a time.
                        }
                    }
                    f.write_str("\n")?;
                    write_indent(f, depth + 1)?;
                    if start == end {
                        write!(f, "[{start}] ")?;
                    } else {
                        write!(f, "[{start}-{end}] ")?;
                    }
                    write_rendered_value(f, val, depth + 1)?;
                    i += 1;
                }
                // Zero elements are suppressed silently — the gaps
                // between rendered groups speak for themselves; an
                // explicit count line adds no information the
                // operator needs.
                let _ = zero_count;
                f.write_str("\n")?;
                write_indent(f, depth)?;
                f.write_str("]")?;
                if elements.len() < *len {
                    write!(f, " /* {} of {len} shown */", elements.len())?;
                }
                Ok(())
            }
        }
        RenderedValue::Struct { type_name, members } => {
            write_struct(f, type_name.as_deref(), members, depth)
        }
    }
}

/// Render a Struct (or Union, which shares the same wire shape)
/// using the column-aligned multi-line / inline-with-braces
/// format. Inline takes priority when the rendered form fits on a
/// single line under [`STRUCT_INLINE_WIDTH_BUDGET`]; otherwise the
/// breadcrumb form `TypeName:` is emitted, followed by indented
/// rows of column-aligned scalar fields and per-field lines for
/// compound members.
fn write_struct(
    f: &mut std::fmt::Formatter<'_>,
    type_name: Option<&str>,
    members: &[RenderedMember],
    depth: usize,
) -> std::fmt::Result {
    // Build the anon-overlay sibling-scalar pool once upfront so
    // each anonymous-member dedup check is O(1). Without this,
    // the per-member `anon_overlay_duplicates_siblings` rebuilt
    // the same HashSet for every anonymous overlay — quadratic
    // in the number of overlays.
    let any_anon = members.iter().any(|m| m.name.is_empty());
    let sibling_scalar_pool: Option<std::collections::HashSet<u64>> = if any_anon {
        Some(build_sibling_scalar_pool(members))
    } else {
        None
    };

    // Single-pass filter + pre-render. `visible_rendered` carries
    // tuples of (member, rendered single-line string). The render
    // happens exactly once per visible member; both the inline-fit
    // probe and the final emit reuse the same string. Members
    // whose value renders to multi-line text (an embedded `\n`
    // signals a Struct that broke to its own breadcrumb form) get
    // a `None` rendered string and feed the compound-line path
    // directly — they can't pack into a column row.
    let mut visible_rendered: Vec<(&RenderedMember, Option<String>)> =
        Vec::with_capacity(members.len());
    for m in members {
        if is_deeply_zero(&m.value) {
            continue;
        }
        if (m.name.contains("___fmt") || m.name.contains("____fmt")) && is_string_value(&m.value) {
            continue;
        }
        if m.name.is_empty()
            && let Some(pool) = sibling_scalar_pool.as_ref()
            && anon_duplicates_pool(&m.value, pool)
        {
            continue;
        }
        // Pre-render the value to a single-line string ONLY for
        // flat scalars. Compound members (Struct, Array, Ptr-with-
        // deref, Truncated, CpuList, Unsupported) always produce
        // multi-line output OR carry their own internal layout
        // that the breadcrumb path re-renders directly via
        // `write_rendered_value`. Pre-rendering them here would
        // cost a full format pass whose result is discarded — the
        // inline-fit probe rejects the row at line 922 because
        // `single_line` is `None`, and the multi-line path skips
        // the cached string anyway. Setting `None` for compounds
        // matches the contract `try_inline_from_rendered` already
        // expects (line 922 short-circuits on the first None).
        let single_line = if is_flat_scalar(&m.value) {
            Some(format!("{}", m.value))
        } else {
            None
        };
        visible_rendered.push((m, single_line));
    }

    // Inline-fit probe: assemble `TypeName{name=val, name=val}`
    // by joining the pre-rendered values. Bail to multi-line if
    // any value lacks a single-line render OR the total width
    // exceeds the budget.
    if let Some(inline) = try_inline_from_rendered(type_name, &visible_rendered) {
        return f.write_str(&inline);
    }

    // Multi-line breadcrumb form. Layout when type_name is
    // present:
    //   TypeName:
    //     scalar1=v1   scalar2=v2   scalar3=v3
    //     scalar4=v4
    //     compound1 InnerType:
    //       inner_field=...
    //
    // Anonymous structs (no type_name) drop the breadcrumb name
    // and just emit `:` followed by the indented body — the
    // visual hierarchy is preserved by the indent depth alone.
    if let Some(name) = type_name {
        f.write_str(name)?;
    }
    if visible_rendered.is_empty() {
        // Truly empty struct (no visible fields after suppression):
        // emit `Type{}` (or `{}` for anon). Suppressed zero /
        // fmt-string fields produce no visible artifact, so an
        // all-zero struct lands here as if it had no members at
        // all.
        f.write_str("{}")?;
        return Ok(());
    }
    f.write_str(":")?;

    // Partition visible into flat-scalar cells (Int/Uint/Bool/
    // Char/Float/Enum/Ptr-without-deref) and compound members.
    // Only flat scalars participate in column packing — values
    // with their own internal structure (inline struct braces,
    // pointer chases, arrays, truncated wrappers, cpu lists) get
    // their own full-width lines so the column grid stays
    // visually homogeneous.
    //
    // Reuse the pre-rendered string from `visible_rendered` for
    // each scalar cell. The compound-member path doesn't need
    // the rendered string — it recurses into
    // `write_rendered_value` which will render at the deeper
    // depth. This avoids the third-format pass.
    let mut scalar_cells: Vec<(String, String)> = Vec::new();
    let mut compound_members: Vec<&RenderedMember> = Vec::new();
    for (m, rendered) in &visible_rendered {
        if is_flat_scalar(&m.value) {
            // Flat scalars always have a single-line rendering;
            // unwrap the Option that was Some(_) at filter time.
            let value_str = rendered.clone().expect(
                "is_flat_scalar guarantees a single-line rendering; \
                 visible_rendered must carry Some(string) for flat scalars",
            );
            let name = if m.name.is_empty() {
                "<anon>".to_string()
            } else {
                m.name.clone()
            };
            scalar_cells.push((name, value_str));
        } else {
            compound_members.push(m);
        }
    }

    // Emit scalar rows: 3 per row. Column alignment kicks in
    // only when there are >= 3 rows AND the field-name length
    // variation in a column is significant (>= 4 chars). Below
    // those thresholds, columns just separate with a 3-space
    // gap and `name=value` (no padding, no `=` alignment).
    if !scalar_cells.is_empty() {
        let cells_per_row = 3;
        let n = scalar_cells.len();
        let n_rows = n.div_ceil(cells_per_row);
        // Per-column max / min field-name length. `=` alignment
        // requires both >= 3 rows AND `max - min >= 4`. When
        // name lengths cluster within 3 chars of each other,
        // padding is overhead — the columns read fine without
        // it.
        let mut name_max = vec![0usize; cells_per_row];
        let mut name_min = vec![usize::MAX; cells_per_row];
        for row in 0..n_rows {
            for col in 0..cells_per_row {
                let idx = row * cells_per_row + col;
                if idx >= n {
                    break;
                }
                let nl = scalar_cells[idx].0.len();
                if nl > name_max[col] {
                    name_max[col] = nl;
                }
                if nl < name_min[col] {
                    name_min[col] = nl;
                }
            }
        }
        // Decide per-column whether to pad. Threshold: 3+ rows
        // AND >= 4 char variation. Below either bar, col is
        // unpadded.
        let pad_eq: Vec<bool> = (0..cells_per_row)
            .map(|col| {
                if n_rows < 3 {
                    return false;
                }
                let max = name_max[col];
                let min = name_min[col];
                if min == usize::MAX {
                    return false;
                }
                max.saturating_sub(min) >= 4
            })
            .collect();
        // Per-column cell-width (full `padded_name + sep + value`
        // length) for column-to-column alignment. Built after
        // pad_eq is final.
        let mut cell_widths = vec![0usize; cells_per_row];
        for row in 0..n_rows {
            for col in 0..cells_per_row {
                let idx = row * cells_per_row + col;
                if idx >= n {
                    break;
                }
                let (name, value) = &scalar_cells[idx];
                let cl = if pad_eq[col] {
                    name_max[col] + 3 + value.len() // "name    = value"
                } else {
                    name.len() + 1 + value.len() // "name=value"
                };
                if cl > cell_widths[col] {
                    cell_widths[col] = cl;
                }
            }
        }
        for row in 0..n_rows {
            f.write_str("\n")?;
            write_indent(f, depth + 1)?;
            for col in 0..cells_per_row {
                let idx = row * cells_per_row + col;
                if idx >= n {
                    break;
                }
                let (name, value) = &scalar_cells[idx];
                f.write_str(name)?;
                if pad_eq[col] {
                    // Pad name to column's max so equals signs
                    // line up. Use ` = ` (space-equals-space) for
                    // visual breathing room around the operator.
                    for _ in 0..name_max[col].saturating_sub(name.len()) {
                        f.write_str(" ")?;
                    }
                    f.write_str(" = ")?;
                } else {
                    // Compact form: bare `name=value`, no padding.
                    f.write_str("=")?;
                }
                f.write_str(value)?;
                // Trailing pad to align next column (3-space
                // minimum gap). The last cell on a row needs no
                // trailing pad — the line ends after the value.
                if col + 1 < cells_per_row && (row * cells_per_row + col + 1) < n {
                    let cell_len = if pad_eq[col] {
                        name_max[col] + 3 + value.len()
                    } else {
                        name.len() + 1 + value.len()
                    };
                    let pad = cell_widths[col].saturating_sub(cell_len) + 3;
                    for _ in 0..pad {
                        f.write_str(" ")?;
                    }
                }
            }
        }
    }

    // Emit compound members: each on its own line at depth+1,
    // recursing into write_rendered_value at depth+1 so the
    // nested render picks up correct indentation.
    for m in compound_members {
        f.write_str("\n")?;
        write_indent(f, depth + 1)?;
        if m.name.is_empty() {
            f.write_str("<anon> ")?;
        } else {
            write!(f, "{} ", m.name)?;
        }
        write_rendered_value(f, &m.value, depth + 1)?;
    }

    // Zero fields and bpf_printk format strings are suppressed
    // silently above; nothing further to emit at this level.
    Ok(())
}

/// Width budget for the inline struct form. A struct whose
/// rendered single-line form exceeds this falls through to
/// multi-line. Matches the soft per-line cap in the dump-display
/// layer (so an inline struct that fits here also fits in the
/// failure-dump output column width).
const STRUCT_INLINE_WIDTH_BUDGET: usize = 120;

/// Soft per-line budget for inline list rendering (arrays,
/// sparse runs). When the joined form exceeds this, the helper
/// wraps at element boundaries with continuation indented to the
/// caller's `depth + 1` so the next row aligns under the bracket
/// open. Picked to match the struct inline budget — the two
/// paths share a visual line-width target.
const INLINE_LIST_WRAP_BUDGET: usize = 120;

/// Render `parts` as `<open>part0, part1, ...<close>` joined by
/// `sep`. Single-line form is preferred; when the joined width
/// exceeds [`INLINE_LIST_WRAP_BUDGET`] the helper wraps at
/// element boundaries — never mid-element. Continuation rows
/// indent to `depth + 1` so wrapping aligns with the caller's
/// column.
///
/// Each `parts` entry is a fully-rendered single-line string
/// already (no embedded `\n`). The helper does not re-render or
/// truncate; it only inserts line breaks at part boundaries.
fn write_inline_list_wrapped(
    f: &mut std::fmt::Formatter<'_>,
    open: &str,
    close: &str,
    parts: &[String],
    sep: &str,
    depth: usize,
) -> std::fmt::Result {
    if parts.is_empty() {
        f.write_str(open)?;
        return f.write_str(close);
    }
    // Probe single-line form: open + parts.join(sep) + close.
    let sep_len = sep.len();
    let mut total = open.len() + close.len();
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            total += sep_len;
        }
        total += p.len();
    }
    f.write_str(open)?;
    if total <= INLINE_LIST_WRAP_BUDGET {
        // Fits on one line.
        for (i, p) in parts.iter().enumerate() {
            if i > 0 {
                f.write_str(sep)?;
            }
            f.write_str(p)?;
        }
        return f.write_str(close);
    }
    // Wrap mode: emit parts greedily, breaking to a new line
    // whenever the next part would push the current line past
    // budget. The first part stays on the open-bracket line so
    // the reader's eye doesn't have to track an empty `[`.
    // Continuation lines indent to depth + 1; that's deeper than
    // the bracket itself but matches the multi-line struct's
    // scalar-row indent, keeping the visual hierarchy consistent.
    let indent = INDENT.repeat(depth + 1);
    // Track current-line cursor: starts at length of `open`.
    let mut cursor = open.len();
    for (i, p) in parts.iter().enumerate() {
        if i == 0 {
            f.write_str(p)?;
            cursor += p.len();
            continue;
        }
        // Check whether adding this part (with sep prefix) would
        // exceed the line budget.
        let next_len = sep_len + p.len();
        if cursor + next_len > INLINE_LIST_WRAP_BUDGET {
            // Wrap before this part. The separator's leading
            // characters (e.g. `, ` → `,`) stay on the previous
            // line so the comma reads naturally; here we elide
            // the leading whitespace by starting the new line
            // with just indent + part.
            f.write_str(sep.trim_end())?;
            f.write_str("\n")?;
            f.write_str(&indent)?;
            f.write_str(p)?;
            cursor = indent.len() + p.len();
        } else {
            f.write_str(sep)?;
            f.write_str(p)?;
            cursor += next_len;
        }
    }
    f.write_str(close)?;
    Ok(())
}

/// Inline-fit probe over pre-rendered member values. Returns
/// `Some(joined_string)` when the struct's `TypeName{f=v, f=v}`
/// form fits within [`STRUCT_INLINE_WIDTH_BUDGET`]; `None`
/// otherwise (multi-line path takes over).
///
/// Each `(member, rendered)` pair carries the value's single-line
/// rendering — `None` rendering means the value is multi-line and
/// disqualifies the struct from inline form. Zero / format-string
/// / overlay-dup members are already filtered out by the caller.
///
/// The render-once invariant: this probe builds the inline string
/// by JOINING the pre-rendered values without re-formatting them.
/// The final write either commits the same string (inline path)
/// or discards it and the multi-line path reuses the same Vec
/// (no second render).
fn try_inline_from_rendered(
    type_name: Option<&str>,
    visible_rendered: &[(&RenderedMember, Option<String>)],
) -> Option<String> {
    if visible_rendered.is_empty() {
        // Empty visible set: emit `Type{}` (or `{}` for anon).
        let s = match type_name {
            Some(n) => format!("{n}{{}}"),
            None => "{}".to_string(),
        };
        return if s.len() <= STRUCT_INLINE_WIDTH_BUDGET {
            Some(s)
        } else {
            None
        };
    }
    // Bail to multi-line if any member's value renders multi-line.
    let mut field_strs = Vec::with_capacity(visible_rendered.len());
    for (m, value_str) in visible_rendered {
        let v = value_str.as_deref()?;
        let name = if m.name.is_empty() {
            "<anon>"
        } else {
            m.name.as_str()
        };
        field_strs.push(format!("{name}={v}"));
    }
    let body = field_strs.join(", ");
    let s = match type_name {
        Some(n) => format!("{n}{{{body}}}"),
        None => format!("{{{body}}}"),
    };
    if s.len() <= STRUCT_INLINE_WIDTH_BUDGET {
        Some(s)
    } else {
        None
    }
}

pub fn is_zero(v: &RenderedValue) -> bool {
    match v {
        RenderedValue::Int { value, .. } => *value == 0,
        RenderedValue::Uint { value, .. } => *value == 0,
        RenderedValue::Bool { value } => !*value,
        RenderedValue::Char { value } => *value == 0,
        RenderedValue::Float { value, .. } => *value == 0.0,
        RenderedValue::Enum { value, .. } => *value == 0,
        RenderedValue::CpuList { cpus } => cpus.is_empty(),
        // Ptr zero-detection: only the numeric value matters. A
        // null pointer with a `deref_skipped_reason` (rare but
        // possible if a future code path attaches a reason without
        // a chase) is still zero — the reason is diagnostic, not
        // a value carrier.
        RenderedValue::Ptr { value, .. } => *value == 0,
        // Skip recursive is_zero on compounds — the subtree traversal
        // is O(leaves) and doubles the total rendering cost. Compound
        // types are always rendered; only scalars get zero-suppressed.
        // Use [`is_deeply_zero`] when an all-zero compound should be
        // suppressed alongside scalars (e.g. struct Display arm
        // collapsing all-zero nested aggregates into the "(N fields
        // zero)" summary).
        _ => false,
    }
}

/// Numeric scalar value of a [`RenderedValue`] for cross-member
/// dedup. Returns `Some(u64)` for the scalar variants that carry
/// a numeric value (Int, Uint, Bool, Char, Enum, Ptr); `None` for
/// the rest. Signed Int values are reinterpreted as `u64` bit
/// patterns to allow comparison against unsigned siblings — the
/// dedup heuristic compares wire bit patterns, not arithmetic
/// values.
fn scalar_numeric_value(v: &RenderedValue) -> Option<u64> {
    match v {
        RenderedValue::Int { value, .. } => Some(*value as u64),
        RenderedValue::Uint { value, .. } => Some(*value),
        RenderedValue::Bool { value } => Some(if *value { 1 } else { 0 }),
        RenderedValue::Char { value } => Some(*value as u64),
        RenderedValue::Enum { value, .. } => Some(*value as u64),
        RenderedValue::Ptr { value, .. } => Some(*value),
        _ => None,
    }
}

/// Build the sibling-scalar-value pool used by anonymous-overlay
/// dedup. Walks `members` once, collecting non-zero scalar values
/// from each named sibling AND descending one level into a sibling
/// Struct (the common single-field-struct case
/// e.g. `tid: struct sdt_id { val: u64 }`). Returning the set
/// once lets the caller pass it into [`anon_duplicates_pool`] for
/// each anonymous member without re-walking the sibling list.
fn build_sibling_scalar_pool(members: &[RenderedMember]) -> std::collections::HashSet<u64> {
    let mut sibling_values: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for s in members {
        if let Some(n) = scalar_numeric_value(&s.value) {
            if n != 0 {
                sibling_values.insert(n);
            }
        } else if let RenderedValue::Struct { members: sm, .. } = &s.value {
            for sub in sm {
                if let Some(n) = scalar_numeric_value(&sub.value)
                    && n != 0
                {
                    sibling_values.insert(n);
                }
            }
        }
    }
    sibling_values
}

/// True when the anonymous member `anon` (a Struct overlay from a
/// BTF union) duplicates content already in the sibling-scalar
/// pool — every non-zero scalar leaf in `anon` has a value
/// already present in `pool`. Companion to
/// [`build_sibling_scalar_pool`]: caller builds the pool once,
/// queries it once per anonymous member.
fn anon_duplicates_pool(anon: &RenderedValue, pool: &std::collections::HashSet<u64>) -> bool {
    let RenderedValue::Struct { members, .. } = anon else {
        return false;
    };
    if members.is_empty() || pool.is_empty() {
        return false;
    }
    for m in members {
        match scalar_numeric_value(&m.value) {
            Some(0) => continue, // zero half of a wider scalar
            Some(n) => {
                if !pool.contains(&n) {
                    return false;
                }
            }
            None => return false, // compound sub-member; can't dedup
        }
    }
    true
}

/// Recursive variant of [`is_zero`] that descends into compound
/// types: a `Struct` is deeply zero iff every member's value is
/// deeply zero (an empty member list also qualifies); an `Array`
/// is deeply zero iff every element is deeply zero (an empty
/// elements vec also qualifies). For scalars the result matches
/// [`is_zero`].
///
/// `Bytes`, `Truncated`, and `Unsupported` are treated as NOT
/// deeply zero — they carry diagnostic content (hex bytes, decoded
/// partial, error reason) the consumer needs to see even when the
/// numeric content happens to be all zeros.
///
/// Recursion is capped at depth 16 to bound pathological BTF where
/// rendered nesting exceeds the renderer's own
/// [`MAX_RENDER_DEPTH`] cap. The cap is defense-in-depth: a
/// well-formed render produced by [`render_value_inner`] cannot
/// nest deeper than [`MAX_RENDER_DEPTH`] (32), but the cap here
/// stops the helper from hanging on a malformed externally-supplied
/// `RenderedValue` tree.
pub(crate) fn is_deeply_zero(v: &RenderedValue) -> bool {
    /// Recursion cap. 16 is comfortably below
    /// [`MAX_RENDER_DEPTH`] (32) so a render that came from
    /// [`render_value_inner`] always terminates well within the
    /// cap; the cap protects helper callers that may construct a
    /// `RenderedValue` tree from sources outside the renderer's
    /// depth-limited path.
    const MAX_DEPTH: u32 = 16;
    fn inner(v: &RenderedValue, depth: u32) -> bool {
        if depth >= MAX_DEPTH {
            // Past the cap: refuse to commit. Returning `false`
            // means the caller treats the value as non-zero,
            // surfacing it in Display rather than silently
            // suppressing a deeply nested subtree we couldn't
            // fully verify.
            return false;
        }
        match v {
            RenderedValue::Struct { members, .. } => {
                members.iter().all(|m| inner(&m.value, depth + 1))
            }
            RenderedValue::Array { elements, .. } => elements.iter().all(|e| inner(e, depth + 1)),
            // Bytes carries diagnostic hex; Truncated carries a
            // partial render the operator must see; Unsupported
            // carries an error reason. None of these are
            // suppressible, regardless of the numeric content
            // they may or may not encode.
            RenderedValue::Bytes { .. }
            | RenderedValue::Truncated { .. }
            | RenderedValue::Unsupported { .. } => false,
            // Scalars: match the canonical is_zero.
            _ => is_zero(v),
        }
    }
    inner(v, 0)
}

/// Try to render array-of-structs groups as a template: show the
/// struct once with per-index values for fields that vary. Returns
/// true if template rendering was used.
fn try_write_struct_template(
    f: &mut std::fmt::Formatter<'_>,
    groups: &[(usize, usize, &RenderedValue)],
    depth: usize,
) -> Result<bool, std::fmt::Error> {
    // All groups must be single-element Structs with the same member count.
    let structs: Vec<(usize, &[RenderedMember])> = groups
        .iter()
        .filter_map(|(start, end, val)| {
            if start != end {
                return None;
            }
            match val {
                RenderedValue::Struct { members, .. } => Some((*start, members.as_slice())),
                _ => None,
            }
        })
        .collect();
    if structs.len() != groups.len() || structs.len() < 3 {
        return Ok(false);
    }
    let member_count = structs[0].1.len();
    if structs.iter().any(|(_, m)| m.len() != member_count) {
        return Ok(false);
    }

    // Find which fields vary.
    let first = structs[0].1;
    let mut varying: Vec<usize> = Vec::new();
    for i in 0..member_count {
        if structs[1..]
            .iter()
            .any(|(_, m)| m[i].value != first[i].value)
        {
            varying.push(i);
        }
    }
    if varying.is_empty() || varying.len() > 3 {
        return Ok(false);
    }

    // Validate that template indices are strictly contiguous
    // before emitting `[start-end]` range. Zero-suppression in the
    // caller can drop intermediate indices (e.g. groups [0,2,4] —
    // 0,1,2,3,4 with zeros at 1,3 dropped to keep the rendering
    // compact). The `[0-4]` header would then misleadingly imply
    // every index 0..=4 is in the template. Bail to fallback
    // per-element rendering when indices aren't consecutive.
    if !structs.windows(2).all(|pair| pair[1].0 == pair[0].0 + 1) {
        return Ok(false);
    }

    // Emit template: struct with common fields shown, varying
    // fields as a per-index value table. Header uses the
    // breadcrumb form `[idx-range] TypeName:`. Common fields
    // render as `name=value` rows; varying fields render as
    // `name: [idx]=val [idx]=val ...` (the `:` introduces the
    // per-index list, not a field assignment).
    let type_name = match groups[0].2 {
        RenderedValue::Struct { type_name, .. } => type_name.as_deref(),
        _ => None,
    };
    let idx_range = format!("[{}-{}]", structs[0].0, structs.last().unwrap().0);
    f.write_str("\n")?;
    write_indent(f, depth)?;
    match type_name {
        Some(name) => write!(f, "{idx_range} {name}:")?,
        None => write!(f, "{idx_range}:")?,
    }

    let mut zero_count = 0usize;
    for (i, m) in first.iter().enumerate() {
        if varying.contains(&i) {
            continue;
        }
        // is_deeply_zero so all-zero compound members (e.g. an
        // empty inner struct) suppress alongside scalars in the
        // template's common-fields section. Matches the main
        // `write_struct` filter at line 571 — without this, a
        // template would render an `inner={}` line for the same
        // value that the non-template path collapses silently,
        // producing inconsistent output for callers that flip
        // between template and per-element rendering.
        if is_deeply_zero(&m.value) {
            zero_count += 1;
            continue;
        }
        f.write_str("\n")?;
        write_indent(f, depth + 1)?;
        write!(f, "{}=", m.name)?;
        write_rendered_value(f, &m.value, depth + 1)?;
    }

    // Varying fields as compact per-index lines. The label form
    // `name:` introduces the per-index list — distinct from
    // `name=value` because each row carries multiple values, one
    // per index.
    for &vi in &varying {
        f.write_str("\n")?;
        write_indent(f, depth + 1)?;
        write!(f, "{}: ", first[vi].name)?;
        for (idx, members) in &structs {
            write!(f, "[{idx}]=")?;
            write_rendered_value(f, &members[vi].value, depth + 1)?;
            f.write_str(" ")?;
        }
    }

    // Zero fields are suppressed silently — no count line.
    let _ = zero_count;
    Ok(true)
}

/// Try to render bytes as a cpumask cpu-list. Reads u64 words from
/// the start of `bytes`, extracts set bits, and formats as
/// `cpus={0,2,5-7}`. Returns None if bytes are too short.
///
/// `max_cpus` caps the highest CPU id walked: bits at positions >=
/// `max_cpus` are treated as out-of-range (slab padding / freelist
/// garbage) and stop the walk. The kernel sizes `cpumask_bits[]` to
/// `BITS_TO_LONGS(NR_CPUS)` words but only the first
/// `nr_cpu_ids` bits are meaningful — the bytes between
/// `nr_cpu_ids` and the slab allocation size are uninitialized or
/// recycled freelist data. Callers that don't have `nr_cpu_ids`
/// available pass `u32::MAX` (no cap).
fn try_render_cpumask_bits(bytes: &[u8], max_cpus: u32) -> Option<RenderedValue> {
    if bytes.len() < 8 {
        return None;
    }
    let n_words = bytes.len() / 8;
    let mut set_cpus: Vec<u32> = Vec::new();
    for word_idx in 0..n_words {
        let off = word_idx * 8;
        if off + 8 > bytes.len() {
            break;
        }
        let word_first_cpu = (word_idx * 64) as u64;
        // Once the first cpu id covered by this word is at or
        // beyond `max_cpus`, no further bits in this or later
        // words are meaningful — stop walking.
        if word_first_cpu >= max_cpus as u64 {
            break;
        }
        let word = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        if word == 0 {
            continue;
        }
        // Pointer-shape heuristic: values larger than 2^32 with
        // more than 64 CPUs already collected likely indicate
        // we've walked past the bitmap end into adjacent data
        // (a kernel address is much larger than a sensible
        // CPU-bit pattern). Apply the gate BEFORE pushing this
        // word's bits — pushing then bailing would have already
        // contaminated `set_cpus` with up to 64 garbage entries
        // from the suspect word.
        if word > 0xFFFF_FFFF && set_cpus.len() > 64 {
            break;
        }
        for bit in 0..64 {
            let cpu = (word_idx * 64 + bit) as u32;
            // Per-bit cap: skip bits at or above max_cpus. The
            // outer `word_first_cpu` gate handles whole-word
            // bailout; this catches the partial-word case where
            // max_cpus falls inside the current word (e.g.
            // max_cpus=8 with first word at cpu 0 — bits 8..63
            // are slab padding).
            if cpu >= max_cpus {
                break;
            }
            if word & (1u64 << bit) != 0 {
                set_cpus.push(cpu);
            }
        }
    }
    Some(RenderedValue::CpuList {
        cpus: format_cpu_list(&set_cpus),
    })
}

/// Format a sorted list of CPU IDs as a range-collapsed string.
/// e.g. [0,1,2,5,7,8,9] → "0-2,5,7-9"
///
/// Writes to a single `String` with `fmt::Write` so each range
/// emits at most two integer formats and one comma — half the
/// allocations of the prior `Vec<String>` + `join(",")` approach.
fn format_cpu_list(cpus: &[u32]) -> String {
    use std::fmt::Write;
    if cpus.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut start = cpus[0];
    let mut end = cpus[0];
    let flush = |out: &mut String, start: u32, end: u32| {
        if !out.is_empty() {
            out.push(',');
        }
        if start == end {
            // unwrap is safe: write! to String never fails.
            let _ = write!(out, "{start}");
        } else {
            let _ = write!(out, "{start}-{end}");
        }
    };
    for &cpu in &cpus[1..] {
        if cpu == end + 1 {
            end = cpu;
        } else {
            flush(&mut out, start, end);
            start = cpu;
            end = cpu;
        }
    }
    flush(&mut out, start, end);
    out
}

fn is_text_byte(b: u8) -> bool {
    // Conservative: only NUL (C string terminator), \n, and printable
    // ASCII. \t and \r are excluded — binary BPF arrays starting with
    // those bytes were misclassified as strings.
    b == 0 || b == b'\n' || (0x20..=0x7e).contains(&b)
}

fn is_string_value(v: &RenderedValue) -> bool {
    match v {
        RenderedValue::Array { elements, .. } => {
            elements.len() >= 2
                && elements.iter().all(|e| match e {
                    RenderedValue::Int { bits: 8, value } => is_text_byte(*value as u8),
                    RenderedValue::Uint { bits: 8, value } => is_text_byte(*value as u8),
                    RenderedValue::Char { value } => is_text_byte(*value),
                    _ => false,
                })
        }
        _ => false,
    }
}

/// Whether a value renders as a single line under Display. Used by
/// the array case to pick inline vs block layout, and by
/// `super::dump::display` to decide whether a struct's fields
/// qualify for inline-entry rendering in the FailureDumpEntry /
/// FailureDumpMap table layouts.
pub(crate) fn is_inline_scalar(v: &RenderedValue) -> bool {
    matches!(
        v,
        RenderedValue::Int { .. }
            | RenderedValue::Uint { .. }
            | RenderedValue::Bool { .. }
            | RenderedValue::Char { .. }
            | RenderedValue::Float { .. }
            | RenderedValue::Enum { .. }
            | RenderedValue::Ptr { .. }
            | RenderedValue::Bytes { .. }
            | RenderedValue::Unsupported { .. }
    )
}

/// Whether a value is a "flat scalar" — a primitive carrying a
/// single rendered token without any internal structure. Used by
/// the multi-line struct path to decide which fields participate
/// in column packing: only flat scalars can pack 3-per-row, since
/// compound forms (inline struct braces, pointer derefs, arrays,
/// truncation wrappers, cpu lists) would mismatch the column
/// grid.
///
/// Stricter than [`is_inline_scalar`]: a `Ptr` with `deref:
/// Some(...)` is NOT flat (it carries an arrow plus a nested
/// render), and `Bytes` / `Unsupported` carry diagnostic content
/// that prefers its own line for readability. `CpuList` reads
/// like a structure (`cpus={0-3}`) so it stays out of the column
/// grid too.
pub(crate) fn is_flat_scalar(v: &RenderedValue) -> bool {
    match v {
        RenderedValue::Int { .. }
        | RenderedValue::Uint { .. }
        | RenderedValue::Bool { .. }
        | RenderedValue::Char { .. }
        | RenderedValue::Float { .. }
        | RenderedValue::Enum { .. } => true,
        // Ptr is flat only when there's no deref payload. A
        // pointer rendered as `0xADDR → ...` doesn't fit a
        // narrow column.
        RenderedValue::Ptr {
            deref: None,
            deref_skipped_reason: None,
            ..
        } => true,
        _ => false,
    }
}

fn write_indent(f: &mut std::fmt::Formatter<'_>, depth: usize) -> std::fmt::Result {
    for _ in 0..depth {
        f.write_str(INDENT)?;
    }
    Ok(())
}

/// Render `bytes` according to BTF type `type_id`.
///
/// Total: returns a [`RenderedValue::Unsupported`] or
/// [`RenderedValue::Truncated`] rather than an error when the bytes or
/// type cannot be decoded, so the caller always has something to
/// serialize.
/// Read guest memory at a kernel virtual address or arena address.
pub trait MemReader {
    fn read_kva(&self, kva: u64, len: usize) -> Option<Vec<u8>>;
    /// Check if an address is in the arena range. Arena pointers
    /// resolve into `ArenaSnapshot`'s captured page set, so the
    /// reader has a frozen byte view — chasing them is well-defined.
    /// Kernel kptrs (slab/vmalloc allocations outside the arena
    /// window) are not inherently unsafe, but they MAY be stale
    /// references to objects already freed by the time the freeze
    /// captured them; this trait offers no path to verify
    /// liveness, so the renderer treats them as uncheckable and
    /// declines to chase. Default returns false — pointer chasing
    /// skips arena resolution silently.
    fn is_arena_addr(&self, _addr: u64) -> bool {
        false
    }
    /// Read bytes from the captured arena at a user-space arena
    /// address. Default returns None — the Ptr deref path emits the
    /// raw pointer hex without chasing.
    ///
    /// Returns None if the address is unmapped or the full requested
    /// length cannot be read.
    fn read_arena(&self, _addr: u64, _len: usize) -> Option<Vec<u8>> {
        None
    }
    /// Guest's `nr_cpu_ids` — the number of possible CPUs the
    /// kernel exposes to userspace allocators (`cpumask_size()`,
    /// percpu arrays, etc.). The cpumask renderer caps the bit
    /// walk at this value: the kernel's `cpumask_bits[]` slab
    /// allocation is sized to `BITS_TO_LONGS(NR_CPUS)`, but only
    /// the first `nr_cpu_ids` bits are meaningful — bits beyond
    /// that are slab-internal padding or freelist garbage that
    /// `SLAB_FREELIST_HARDENED` XOR-encoding can mask the top-
    /// byte heuristic from rejecting. Default returns
    /// `u32::MAX` (no cap) so callers without the value still
    /// produce a render.
    fn nr_cpu_ids(&self) -> u32 {
        u32::MAX
    }
}

/// Render a BTF type's bytes into a [`RenderedValue`] without an
/// associated guest-memory reader. Pointer dereferences degrade
/// gracefully — the raw pointer hex is emitted without chasing.
///
/// `dead_code` allow: kept as the public entry point for callers
/// that don't need pointer chasing; the live render paths in
/// failure-dump rendering currently always supply a reader via
/// [`render_value_with_mem`].
#[allow(dead_code)]
pub fn render_value(btf: &Btf, type_id: u32, bytes: &[u8]) -> RenderedValue {
    let mut visited: HashSet<u64> = HashSet::new();
    render_value_inner(btf, type_id, bytes, 0, None::<&dyn MemReader>, &mut visited)
}

pub fn render_value_with_mem(
    btf: &Btf,
    type_id: u32,
    bytes: &[u8],
    mem: &dyn MemReader,
) -> RenderedValue {
    let mut visited: HashSet<u64> = HashSet::new();
    render_value_inner(btf, type_id, bytes, 0, Some(mem), &mut visited)
}

/// `visited` carries the set of pointer addresses already chased on
/// the current traversal path. The `Type::Ptr` arm checks this set
/// before descending: a pointer whose target address is already in
/// `visited` is a cycle (most commonly a linked-list `next` pointer
/// pointing back to a node earlier on the chain). Without the
/// check, a cycle recurses until [`MAX_RENDER_DEPTH`] fires,
/// producing a 32-deep wall of identical nested structs in the
/// failure dump. With the check, the renderer surfaces a `[cycle]`
/// marker after the pointer's hex value and stops.
///
/// `MAX_RENDER_DEPTH` remains as a backstop for non-cycle
/// pathological BTF (deeply nested types without an actual cycle).
fn render_value_inner(
    btf: &Btf,
    type_id: u32,
    bytes: &[u8],
    depth: u32,
    mem: Option<&dyn MemReader>,
    visited: &mut HashSet<u64>,
) -> RenderedValue {
    if depth >= MAX_RENDER_DEPTH {
        return RenderedValue::Unsupported {
            reason: format!("render depth {MAX_RENDER_DEPTH} exceeded"),
        };
    }

    let Some(ty) = peel_modifiers(btf, type_id) else {
        return RenderedValue::Unsupported {
            reason: format!("could not peel modifiers from type id {type_id}"),
        };
    };

    match ty {
        Type::Int(int) => render_int(&int, bytes),
        Type::Float(float) => render_float(float.size(), bytes),
        Type::Enum(e) => {
            // Enum values are 32-bit on the wire (vlen entries each
            // holding `val: u32`), but the underlying storage size
            // comes from `Enum::size()`.
            let needed = e.size();
            if bytes.len() < needed {
                return RenderedValue::Truncated {
                    needed,
                    had: bytes.len(),
                    partial: Box::new(RenderedValue::Bytes {
                        hex: hex_dump(bytes),
                    }),
                };
            }
            let raw = read_uint_le(&bytes[..needed]);
            let signed = e.is_signed();
            let value = if signed {
                sign_extend(raw, needed * 8) as i64
            } else {
                raw as i64
            };
            // BTF_KIND_ENUM members hold 32-bit values
            // (`btf_rs::EnumMember::val()` returns `u32`); mask with
            // `as u32 as u64` so the comparison stays width-truncated
            // even if a future btf-rs API change widens the return
            // type. Without the explicit width mask a signed-typed
            // enum-member value sign-extended to u64 would never match
            // the zero-extended `raw` for negative variants. Resolved
            // variant name is best-effort: an unknown raw value yields
            // `None` rather than failing the render.
            let variant = e
                .members
                .iter()
                .find(|m| m.val() as u64 == raw)
                .and_then(|m| btf.resolve_name(m).ok());
            RenderedValue::Enum {
                bits: (needed * 8) as u32,
                value,
                variant,
            }
        }
        Type::Enum64(e) => {
            let needed = e.size();
            if bytes.len() < needed {
                return RenderedValue::Truncated {
                    needed,
                    had: bytes.len(),
                    partial: Box::new(RenderedValue::Bytes {
                        hex: hex_dump(bytes),
                    }),
                };
            }
            let raw = read_uint_le(&bytes[..needed]);
            let signed = e.is_signed();
            let value = if signed {
                sign_extend(raw, needed * 8) as i64
            } else {
                raw as i64
            };
            let variant = e
                .members
                .iter()
                .find(|m| m.val() == raw)
                .and_then(|m| btf.resolve_name(m).ok());
            RenderedValue::Enum {
                bits: (needed * 8) as u32,
                value,
                variant,
            }
        }
        Type::Ptr(ptr) => {
            if bytes.len() < 8 {
                return RenderedValue::Truncated {
                    needed: 8,
                    had: bytes.len(),
                    partial: Box::new(RenderedValue::Bytes {
                        hex: hex_dump(bytes),
                    }),
                };
            }
            let val = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            // `deref_skipped_reason` carries the cause when chase is
            // attempted but produces no deref (cross-page failure,
            // 4 KiB cap, plausibility gate, cycle). It stays `None`
            // when no chase was attempted (null val, depth cap, no
            // reader). The operator distinguishes "we didn't try"
            // from "we tried and got nothing useful" via this
            // field.
            let mut deref_skipped_reason: Option<String> = None;
            // Cycle detection: if we already chased this address on
            // the current traversal path, the pointer points back
            // into the chain we are walking. Surface the cycle
            // inline and skip the chase. Without this, a
            // linked-list `next` field whose chain loops would
            // recurse until MAX_RENDER_DEPTH fires, producing a
            // wall of identical nested structs in the failure
            // dump. Apply the check before consulting `mem` so
            // null and depth-capped values still take the "no
            // chase attempted" path (deref + reason both None).
            let already_visited = val != 0 && visited.contains(&val);
            if already_visited {
                deref_skipped_reason = Some(format!("cycle → 0x{val:x}"));
            }
            let deref = if val != 0 && depth < MAX_RENDER_DEPTH && !already_visited {
                mem.and_then(|m| {
                    let pointee_type_id = ptr.get_type_id().ok()?;
                    let pointee_ty = peel_modifiers(btf, pointee_type_id)?;
                    let btf_size = type_size(btf, &pointee_ty)?;
                    if btf_size == 0 {
                        deref_skipped_reason =
                            Some("pointee BTF size is 0 (incomplete type)".to_string());
                        return None;
                    }
                    if m.is_arena_addr(val) {
                        // Arena chase. The single-page (4 KiB) cap
                        // matches the arena page granularity
                        // exposed by [`MemReader::read_arena`]: a
                        // pointee larger than 4 KiB renders only
                        // its first page. Cross-page chase would
                        // require splitting the read into per-page
                        // chunks AND stitching them — that's a
                        // future enhancement once a scheduler
                        // actually ships an arena-allocated payload
                        // larger than 4 KiB. Today we surface the
                        // truncation explicitly via
                        // `deref_skipped_reason` when btf_size
                        // exceeds the cap, so the operator sees
                        // that the rendered subtree is partial.
                        const ARENA_CHASE_CAP: usize = 4096;
                        let read_size = btf_size.min(ARENA_CHASE_CAP);
                        let truncated_at_cap = btf_size > ARENA_CHASE_CAP;
                        let Some(target_bytes) = m.read_arena(val, read_size) else {
                            // The MemReader::read_arena contract
                            // returns None when the full requested
                            // length cannot be satisfied — most
                            // commonly because the read crosses a
                            // page boundary in the captured arena
                            // snapshot. Annotate so the consumer
                            // sees why the deref didn't land.
                            deref_skipped_reason = Some(format!(
                                "arena read failed (cross-page boundary or unmapped \
                                 page); needed {read_size} bytes from \
                                 0x{val:x}"
                            ));
                            return None;
                        };
                        // Mark this address visited BEFORE recursing
                        // so any pointer in the pointee that loops
                        // back here is detected as a cycle. Cleared
                        // after the recursion returns (path-based
                        // visited set: a sibling branch that
                        // legitimately points to the same target is
                        // not a cycle on its own path).
                        visited.insert(val);
                        let inner = render_value_inner(
                            btf,
                            pointee_type_id,
                            &target_bytes,
                            depth + 1,
                            Some(m),
                            visited,
                        );
                        visited.remove(&val);
                        if truncated_at_cap {
                            // Partial render: only the first 4 KiB
                            // of a larger struct was read. Wrap so
                            // the consumer can tell the rendered
                            // tree is incomplete even though it
                            // looks structurally sound.
                            return Some(Box::new(RenderedValue::Truncated {
                                needed: btf_size,
                                had: target_bytes.len(),
                                partial: Box::new(inner),
                            }));
                        }
                        return Some(Box::new(inner));
                    }
                    // Kernel kptr: only chase cpumask pointers.
                    // Read up to NR_CPUS / 8 bytes from the bitmap
                    // backing storage and plausibility-gate the
                    // first word against a hardcoded heuristic
                    // (see below).
                    let is_cpumask_ptr = match &pointee_ty {
                        Type::Struct(s) => {
                            let n = btf.resolve_name(s).unwrap_or_default();
                            n == "bpf_cpumask" || n == "cpumask"
                        }
                        _ => false,
                    };
                    if is_cpumask_ptr {
                        // Read enough bytes to cover NR_CPUS up to
                        // 8192 (=1024 bytes = 128 u64 words). The
                        // kernel allocates `cpumask_bits[]` from a
                        // slab cache sized to `cpumask_size()`,
                        // which is `(NR_CPUS + 7) / 8` rounded up
                        // to a multiple of 8 — bounded by NR_CPUS
                        // at config time. 1024 covers every modern
                        // distro kernel; mainline NR_CPUS_DEFAULT
                        // is 8192 for x86_64 / aarch64. The
                        // per-word walker below caps the rendered
                        // bits at the guest's `nr_cpu_ids` so a
                        // small guest (e.g. 8 CPUs) doesn't render
                        // bits 64..8191 from slab padding.
                        const CPUMASK_READ_CAP: usize = 1024;
                        let Some(bits_bytes) = m.read_kva(val, CPUMASK_READ_CAP) else {
                            deref_skipped_reason = Some(format!(
                                "cpumask kptr read_kva failed at 0x{val:x} \
                                 (unmapped page or no PTE)"
                            ));
                            return None;
                        };
                        if bits_bytes.len() < 8 {
                            deref_skipped_reason = Some(format!(
                                "cpumask kptr read returned {} bytes; need at least 8",
                                bits_bytes.len()
                            ));
                            return None;
                        }
                        let max_cpus = m.nr_cpu_ids();
                        let bits0 = u64::from_le_bytes(bits_bytes[..8].try_into().ok()?);
                        // Best-effort plausibility heuristic on
                        // `bits[0]`: a freed slab object's first
                        // qword is often a freelist next pointer,
                        // which on x86_64 / aarch64 typically lands
                        // in the kernel direct-map range
                        // (0xffff800000000000+, top byte 0xff). We
                        // reject reads where the top byte is
                        // exactly 0xff as a probable stale-pointer
                        // pattern. The `nr_cpu_ids` cap below
                        // backstops this: even when
                        // SLAB_FREELIST_HARDENED XOR-encodes the
                        // next pointer (defeating the top-byte
                        // gate), set bits beyond `nr_cpu_ids` are
                        // dropped rather than rendered as phantom
                        // cpu ids. Caveats:
                        //   * False-positive: a fully-online 64-CPU
                        //     mask (0xFFFFFFFFFFFFFFFF in word 0)
                        //     is indistinguishable from the
                        //     0xff... pointer pattern at this gate
                        //     and gets rejected, surfacing as raw
                        //     hex.
                        // The gate is intentionally cheap; a
                        // production-grade detector would walk the
                        // SLUB metadata to confirm liveness.
                        if bits0 >> 56 != 0xff {
                            let mut cpus = Vec::new();
                            // Walk every full u64 in the read.
                            // `bits_bytes.len()` is at least 8
                            // (gated above) and a multiple of 8 in
                            // practice (the read cap is a multiple
                            // of 8 and the read returns a contiguous
                            // bytes vector).
                            'walk: for word_idx in 0..(bits_bytes.len() / 8) {
                                let off = word_idx * 8;
                                let word_first_cpu = (word_idx * 64) as u64;
                                // Cap at the guest's nr_cpu_ids:
                                // bits beyond that are slab
                                // padding, not part of the
                                // kernel-meaningful mask.
                                if word_first_cpu >= max_cpus as u64 {
                                    break;
                                }
                                let word =
                                    u64::from_le_bytes(bits_bytes[off..off + 8].try_into().ok()?);
                                // Per-word pointer-pattern gate.
                                // Slab garbage in trailing words can
                                // appear as a high-bit-set u64 that
                                // would otherwise enumerate phantom
                                // CPU IDs; bail out of the walk when
                                // a later word looks like a kernel
                                // address rather than mask bits.
                                if word >> 56 == 0xff {
                                    break;
                                }
                                for bit in 0..64u32 {
                                    let cpu = (word_idx * 64) as u32 + bit;
                                    // Partial-word cap: max_cpus
                                    // can fall mid-word (e.g.
                                    // nr_cpu_ids=8 means bits
                                    // 8..63 of word 0 are padding).
                                    if cpu >= max_cpus {
                                        break 'walk;
                                    }
                                    if word & (1u64 << bit) != 0 {
                                        cpus.push(cpu);
                                    }
                                }
                            }
                            return Some(Box::new(RenderedValue::CpuList {
                                cpus: format_cpu_list(&cpus),
                            }));
                        } else {
                            deref_skipped_reason = Some(format!(
                                "cpumask kptr plausibility gate rejected: bits[0] top \
                                 byte is 0xff at 0x{val:x} (likely freed slab object)"
                            ));
                        }
                    }
                    None
                })
            } else {
                None
            };
            RenderedValue::Ptr {
                value: val,
                deref,
                deref_skipped_reason,
            }
        }
        Type::Struct(s) | Type::Union(s) => render_struct(btf, &s, bytes, depth, mem, visited),
        Type::Array(arr) => {
            // `Array::get_type_id` returns the element type id
            // directly (`btf_array.r#type`), so resolving a chained
            // type purely to fish for the id is redundant. The
            // element's *Type* (used for size) does need a chained
            // resolve.
            let len = arr.len();
            let Ok(elem_type_id) = arr.get_type_id() else {
                return RenderedValue::Unsupported {
                    reason: "array element type id not resolvable".to_string(),
                };
            };
            let Ok(elem_ty) = btf.resolve_chained_type(&arr) else {
                return RenderedValue::Unsupported {
                    reason: "array element type not resolvable".to_string(),
                };
            };
            let Some(elem_size) = type_size(btf, &elem_ty) else {
                return RenderedValue::Unsupported {
                    reason: "array element size not resolvable".to_string(),
                };
            };
            // Flex array detection: BTF encodes a trailing `T[];` /
            // `T[0]` member as `array.len == 0`. The C-side runtime
            // length lives in a sibling field (e.g. struct topology's
            // `nr_children`), which the renderer doesn't have access
            // to here. Emit Unsupported with an explicit reason
            // rather than silently rendering as `[]` — the operator
            // sees that the array IS a flex array and that runtime
            // population is opaque to the BTF-only renderer. Best-
            // effort element extraction (via a sibling-field read of
            // nr_children, etc.) is out of scope: it requires
            // parent-struct context the array arm doesn't carry.
            //
            // `elem_size > 0` is required so a true zero-element
            // type-id-only array (synthesized by libbpf for empty
            // sections etc.) doesn't accidentally surface as flex.
            if len == 0 && elem_size > 0 && !bytes.is_empty() {
                return RenderedValue::Unsupported {
                    reason: format!(
                        "flex array (BTF len=0); runtime length not \
                         representable in BTF, {} bytes available at site",
                        bytes.len()
                    ),
                };
            }
            let cap = len.min(MAX_ARRAY_ELEMS);
            let mut elements = Vec::with_capacity(cap);
            for i in 0..cap {
                let start = i * elem_size;
                let end = start + elem_size;
                if end > bytes.len() {
                    let avail = &bytes[start.min(bytes.len())..];
                    elements.push(RenderedValue::Truncated {
                        needed: elem_size,
                        had: avail.len(),
                        partial: Box::new(RenderedValue::Bytes {
                            hex: hex_dump(avail),
                        }),
                    });
                    break;
                }
                elements.push(render_value_inner(
                    btf,
                    elem_type_id,
                    &bytes[start..end],
                    depth + 1,
                    mem,
                    visited,
                ));
            }
            RenderedValue::Array { len, elements }
        }
        Type::Fwd(_) => RenderedValue::Unsupported {
            reason: "forward declaration: type body not in BTF".to_string(),
        },
        Type::Func(_) | Type::FuncProto(_) => RenderedValue::Unsupported {
            reason: "function type: no value bytes to render".to_string(),
        },
        Type::Datasec(ds) => render_datasec(btf, &ds, bytes, depth, mem, visited),
        Type::Var(var) => {
            // Standalone Var (i.e. asked to render a Var type id
            // outside a Datasec walk): forward to its underlying
            // type. The Var node carries a name but no storage of
            // its own — render its referent against the supplied
            // bytes. A failed type-id lookup falls back to
            // Unsupported rather than panicking.
            let Ok(inner_id) = var.get_type_id() else {
                return RenderedValue::Unsupported {
                    reason: "var type id not resolvable".to_string(),
                };
            };
            render_value_inner(btf, inner_id, bytes, depth + 1, mem, visited)
        }
        Type::Void => RenderedValue::Unsupported {
            reason: "void: no value bytes to render".to_string(),
        },
        // Modifier types should have been peeled by peel_modifiers.
        // If one slipped through, treat it as unsupported rather than
        // looping forever.
        Type::Volatile(_)
        | Type::Const(_)
        | Type::Restrict(_)
        | Type::Typedef(_)
        | Type::TypeTag(_)
        | Type::DeclTag(_) => RenderedValue::Unsupported {
            reason: "unpeeled modifier (BTF cycle?)".to_string(),
        },
    }
}

fn render_int(int: &btf_rs::Int, bytes: &[u8]) -> RenderedValue {
    let needed = int.size();
    if bytes.len() < needed {
        return RenderedValue::Truncated {
            needed,
            had: bytes.len(),
            partial: Box::new(RenderedValue::Bytes {
                hex: hex_dump(bytes),
            }),
        };
    }
    if int.is_bool() && needed >= 1 {
        // C `_Bool` is canonically 1 byte but BTF can describe wider
        // boolean ints. Truthiness must consider every byte in the
        // declared width: a 4-byte `_Bool` set to 0x00000100 is true,
        // not false. The first-byte-only check predated this fix and
        // would silently miss any non-zero byte above the LSB.
        return RenderedValue::Bool {
            value: bytes[..needed].iter().any(|&b| b != 0),
        };
    }
    if int.is_char() && needed == 1 {
        return RenderedValue::Char { value: bytes[0] };
    }
    // BTF allows ints wider than 8 bytes (e.g. 128-bit __int128).
    // `read_uint_le` caps at 8 bytes, so silently feeding it a wider
    // span would discard the upper bits. Fall back to a Bytes hex
    // dump rather than producing a half-decoded numeric value.
    if needed > 8 {
        return RenderedValue::Bytes {
            hex: hex_dump(&bytes[..needed]),
        };
    }
    let raw = read_uint_le(&bytes[..needed]);
    if int.is_signed() {
        let value = sign_extend(raw, needed * 8) as i64;
        RenderedValue::Int {
            bits: (needed * 8) as u32,
            value,
        }
    } else {
        RenderedValue::Uint {
            bits: (needed * 8) as u32,
            value: raw,
        }
    }
}

// Hex dump of byte slices is the `pub(crate) hex_dump` helper in
// [`super::dump::hex_dump`] — imported above. Single canonical
// implementation; renderer and dump path share the same wire shape.

fn render_float(size: usize, bytes: &[u8]) -> RenderedValue {
    if bytes.len() < size {
        return RenderedValue::Truncated {
            needed: size,
            had: bytes.len(),
            partial: Box::new(RenderedValue::Bytes {
                hex: hex_dump(bytes),
            }),
        };
    }
    let value = match size {
        4 => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
        8 => f64::from_le_bytes(bytes[..8].try_into().unwrap()),
        _ => {
            return RenderedValue::Unsupported {
                reason: format!("unsupported float size {size}"),
            };
        }
    };
    RenderedValue::Float {
        bits: (size * 8) as u32,
        value,
    }
}

fn render_struct(
    btf: &Btf,
    s: &Struct,
    bytes: &[u8],
    depth: u32,
    mem: Option<&dyn MemReader>,
    visited: &mut HashSet<u64>,
) -> RenderedValue {
    let type_name = btf.resolve_name(s).ok().filter(|n| !n.is_empty());

    // Intercept cpumask-family structs: render as cpu-list instead
    // of per-field dump. Detect by struct name. The
    // [`MemReader::nr_cpu_ids`] cap rejects bits past the guest's
    // actual CPU count — kernel cpumask slab allocations are sized
    // to NR_CPUS at config time but only the first `nr_cpu_ids`
    // bits carry meaningful data. Without the cap, slab padding /
    // freelist garbage in trailing words renders as phantom CPU
    // ids (the SLAB_FREELIST_HARDENED top-byte heuristic in the
    // Ptr arm doesn't apply to embedded cpumask_t bytes). When
    // `mem` is None (no reader plumbed), use `u32::MAX` so the
    // pre-existing behavior — every set bit reported — is
    // preserved.
    let max_cpus = mem.map(|m| m.nr_cpu_ids()).unwrap_or(u32::MAX);
    if let Some(ref name) = type_name {
        match name.as_str() {
            "cpumask" | "cpumask_t" => {
                if let Some(cpu_list) = try_render_cpumask_bits(bytes, max_cpus) {
                    return cpu_list;
                }
            }
            "bpf_cpumask" => {
                // bpf_cpumask = { cpumask_t cpumask; refcount_t usage; }
                // cpumask starts at offset 0.
                if let Some(cpu_list) = try_render_cpumask_bits(bytes, max_cpus) {
                    return cpu_list;
                }
            }
            "scx_bitmap" => {
                // scx_bitmap = { sdt_id tid (8 bytes); u64 bits[64]; }
                if bytes.len() >= 16
                    && let Some(cpu_list) = try_render_cpumask_bits(&bytes[8..], max_cpus)
                {
                    return cpu_list;
                }
            }
            _ => {}
        }
    }

    let truncated = bytes.len() < s.size();
    // Render every member regardless of whether the full struct
    // fits: `render_member` already emits a per-member
    // [`RenderedValue::Truncated`] for members that extend past
    // the supplied bytes. This way, the outer Truncated's
    // `partial` field carries the members that DID decode, instead
    // of discarding the whole render. See [`RenderedValue::Truncated`]
    // doc for the partial-render contract.
    let mut members = Vec::with_capacity(s.members.len());
    for m in &s.members {
        let name = btf.resolve_name(m).unwrap_or_default();
        let value = render_member(btf, m, bytes, depth, mem, visited);
        members.push(RenderedMember { name, value });
    }
    let rendered = RenderedValue::Struct { type_name, members };
    if truncated {
        RenderedValue::Truncated {
            needed: s.size(),
            had: bytes.len(),
            partial: Box::new(rendered),
        }
    } else {
        rendered
    }
}

/// Render a `BTF_KIND_DATASEC` (e.g. `.bss`, `.data`, `.rodata`) by
/// walking its `VarSecinfo` entries and rendering each variable into
/// the slice of section bytes its `offset()` and `size()` describe.
///
/// Each VarSecinfo's `get_type_id()` returns a `BTF_KIND_VAR` id; the
/// Var carries the variable's name and its underlying type's id. The
/// renderer slices `bytes[offset..offset+size]` and recursively
/// renders the underlying type into that slice. The result is a
/// [`RenderedValue::Struct`] whose `type_name` is the section name
/// (e.g. `.bss`) and whose `members` are the section's variables —
/// reusing `RenderedValue::Struct` rather than introducing a new
/// variant keeps the existing serde shape (`kind: "struct"`) and
/// Display layout intact, so a failure dump's `.bss` map renders
/// alongside ordinary structs and JSON consumers (the
/// `failure_dump_e2e.rs` fixture among them) iterate the variables via
/// `value.members[]` exactly as they iterate struct members today.
///
/// Truncation: an out-of-range `(offset, size)` for the supplied
/// `bytes` slice surfaces as a per-variable
/// [`RenderedValue::Truncated`] — the variable's name is still
/// recorded under [`RenderedMember::name`], so an operator sees
/// "variable X needed N bytes, had M" rather than the entire
/// section disappearing. Variables with malformed BTF (Var type id
/// fails to resolve, chained type isn't a Var) fall through to
/// [`RenderedValue::Unsupported`] with the reason recorded.
fn render_datasec(
    btf: &Btf,
    ds: &btf_rs::Datasec,
    bytes: &[u8],
    depth: u32,
    mem: Option<&dyn MemReader>,
    visited: &mut HashSet<u64>,
) -> RenderedValue {
    // Section name lives on the Datasec itself
    // (BTF_KIND_DATASEC.name_off via `BtfType::get_name_offset`). An
    // empty / unresolvable name maps to `None`, matching
    // [`RenderedValue::Struct::type_name`]'s contract for anonymous
    // aggregates.
    let type_name = btf.resolve_name(ds).ok().filter(|n| !n.is_empty());
    let mut members = Vec::with_capacity(ds.variables.len());
    for var_info in &ds.variables {
        let offset = var_info.offset() as usize;
        let size = var_info.size();
        // Resolve the chained Var so we can pull the variable's
        // name and its underlying type id. A non-Var here indicates
        // malformed BTF (libbpf always emits Var per VarSecinfo);
        // record the failure as an Unsupported member rather than
        // dropping the slot.
        let chained = match btf.resolve_chained_type(var_info) {
            Ok(t) => t,
            Err(_) => {
                members.push(RenderedMember {
                    name: String::new(),
                    value: RenderedValue::Unsupported {
                        reason: "datasec var type not resolvable".to_string(),
                    },
                });
                continue;
            }
        };
        let var = match chained {
            Type::Var(v) => v,
            other => {
                members.push(RenderedMember {
                    name: String::new(),
                    value: RenderedValue::Unsupported {
                        reason: format!("datasec entry resolved to non-Var ({})", other.name()),
                    },
                });
                continue;
            }
        };
        let var_name = btf.resolve_name(&var).unwrap_or_default();
        let inner_id = match var.get_type_id() {
            Ok(id) => id,
            Err(_) => {
                members.push(RenderedMember {
                    name: var_name,
                    value: RenderedValue::Unsupported {
                        reason: "var underlying type id not resolvable".to_string(),
                    },
                });
                continue;
            }
        };
        // Slice the section bytes for this variable. If the section
        // bytes are shorter than offset+size, emit a per-member
        // Truncated whose `partial` is whatever the inner renderer
        // can decode from the available subset (mirrors
        // `render_member`'s short-bytes behaviour). `checked_add`
        // guards against pathological BTF where `offset + size`
        // would overflow `usize` — without it, a torn VarSecinfo
        // could wrap past `usize::MAX` and the `<= bytes.len()`
        // comparison would silently become true, indexing out of
        // bounds.
        let end = offset.checked_add(size);
        let value = match end {
            Some(end) if end <= bytes.len() => {
                render_value_inner(btf, inner_id, &bytes[offset..end], depth + 1, mem, visited)
            }
            _ => {
                let avail_start = offset.min(bytes.len());
                let avail = &bytes[avail_start..];
                let partial = render_value_inner(btf, inner_id, avail, depth + 1, mem, visited);
                RenderedValue::Truncated {
                    needed: size,
                    had: avail.len(),
                    partial: Box::new(partial),
                }
            }
        };
        members.push(RenderedMember {
            name: var_name,
            value,
        });
    }
    RenderedValue::Struct { type_name, members }
}

fn render_member(
    btf: &Btf,
    m: &Member,
    parent_bytes: &[u8],
    depth: u32,
    mem: Option<&dyn MemReader>,
    visited: &mut HashSet<u64>,
) -> RenderedValue {
    let bit_off = m.bit_offset() as usize;
    let Ok(member_type_id) = m.get_type_id() else {
        return RenderedValue::Unsupported {
            reason: "member has no type id".to_string(),
        };
    };

    if let Some(width) = m.bitfield_size() {
        return render_bitfield(btf, member_type_id, parent_bytes, bit_off, width as usize);
    }

    // Non-bitfield: assume bit_off is byte-aligned. Compute the
    // member's size from its (peeled) type and slice.
    if !bit_off.is_multiple_of(8) {
        return RenderedValue::Unsupported {
            reason: format!("non-bitfield member at non-byte bit offset {bit_off}"),
        };
    }
    let byte_off = bit_off / 8;
    let Some(member_ty) = peel_modifiers(btf, member_type_id) else {
        return RenderedValue::Unsupported {
            reason: "member type modifiers unresolvable".to_string(),
        };
    };
    let Some(size) = type_size(btf, &member_ty) else {
        return RenderedValue::Unsupported {
            reason: "member type size unresolvable".to_string(),
        };
    };
    // `checked_add` guards against pathological BTF where
    // `byte_off + size` would overflow `usize` (a torn member with
    // a wild bit_offset / size pair). Without the check, the wrap
    // would silently make the `> parent_bytes.len()` test false
    // and the slice would index out of bounds.
    let end = byte_off.checked_add(size);
    match end {
        Some(end) if end <= parent_bytes.len() => render_value_inner(
            btf,
            member_type_id,
            &parent_bytes[byte_off..end],
            depth + 1,
            mem,
            visited,
        ),
        _ => {
            // Attempt a partial decode from whatever bytes ARE available
            // for this member: the inner renderer will itself emit a
            // Truncated/Bytes/etc. that carries the recoverable subset.
            // Wrapping that subset in this outer Truncated tells the
            // consumer "the full member needed N bytes, only M survived,
            // here's what we got".
            let avail_start = byte_off.min(parent_bytes.len());
            let avail = &parent_bytes[avail_start..];
            let partial = render_value_inner(btf, member_type_id, avail, depth + 1, mem, visited);
            RenderedValue::Truncated {
                needed: size,
                had: avail.len(),
                partial: Box::new(partial),
            }
        }
    }
}

fn render_bitfield(
    btf: &Btf,
    member_type_id: u32,
    parent_bytes: &[u8],
    bit_off: usize,
    width: usize,
) -> RenderedValue {
    if width == 0 || width > 64 {
        return RenderedValue::Unsupported {
            reason: format!("bitfield width {width} out of range"),
        };
    }
    let byte_start = bit_off / 8;
    let bit_shift = bit_off % 8;
    let bits_needed = bit_shift + width;
    let bytes_needed = bits_needed.div_ceil(8);
    if byte_start + bytes_needed > parent_bytes.len() {
        let avail_start = byte_start.min(parent_bytes.len());
        let avail = &parent_bytes[avail_start..];
        return RenderedValue::Truncated {
            needed: bytes_needed,
            had: avail.len(),
            partial: Box::new(RenderedValue::Bytes {
                hex: hex_dump(avail),
            }),
        };
    }
    // Pull up to 16 bytes (max bitfield is 64 bits + 7 bit shift = 71
    // bits = 9 bytes); use a fixed buffer to avoid heap.
    let mut buf = [0u8; 16];
    buf[..bytes_needed].copy_from_slice(&parent_bytes[byte_start..byte_start + bytes_needed]);
    // Pack into a u128 little-endian, then mask + shift.
    let mut packed: u128 = 0;
    for (i, b) in buf[..bytes_needed].iter().enumerate() {
        packed |= (*b as u128) << (i * 8);
    }
    let raw = ((packed >> bit_shift) & ((1u128 << width) - 1)) as u64;

    let Some(member_ty) = peel_modifiers(btf, member_type_id) else {
        return RenderedValue::Unsupported {
            reason: "bitfield type modifiers unresolvable".to_string(),
        };
    };
    // BTF bitfields can carry signed Int *or* signed Enum / Enum64
    // bases (e.g. `enum scx_exit_kind` declared with negative
    // members). Treat all three as signed for the sign-extension
    // step so a negative-valued bitfield rendered through any of
    // them comes back as a correctly-signed `Int` rather than a
    // raw-bits `Uint`.
    let signed = match &member_ty {
        Type::Int(i) => i.is_signed(),
        Type::Enum(e) => e.is_signed(),
        Type::Enum64(e) => e.is_signed(),
        _ => false,
    };
    if signed {
        let value = sign_extend(raw, width) as i64;
        RenderedValue::Int {
            bits: width as u32,
            value,
        }
    } else {
        RenderedValue::Uint {
            bits: width as u32,
            value: raw,
        }
    }
}

/// Peel pass-through qualifier types (Volatile, Const, Restrict,
/// Typedef, TypeTag, DeclTag) and return the underlying [`Type`].
/// Returns `None` if the chain exceeds [`MAX_MODIFIER_DEPTH`] or fails
/// to resolve.
pub(crate) fn peel_modifiers(btf: &Btf, mut type_id: u32) -> Option<Type> {
    for _ in 0..MAX_MODIFIER_DEPTH {
        let ty = btf.resolve_type_by_id(type_id).ok()?;
        match &ty {
            Type::Volatile(t) => type_id = t.get_type_id().ok()?,
            Type::Const(t) => type_id = t.get_type_id().ok()?,
            Type::Restrict(t) => type_id = t.get_type_id().ok()?,
            Type::Typedef(t) => type_id = t.get_type_id().ok()?,
            Type::TypeTag(t) => type_id = t.get_type_id().ok()?,
            // DeclTag doesn't change the underlying type, just adds
            // metadata; peel through it too.
            Type::DeclTag(t) => type_id = t.get_type_id().ok()?,
            _ => return Some(ty),
        }
    }
    None
}

/// Compute the storage size in bytes of a (peeled) BTF type.
///
/// Returns `None` for types whose size is not resolvable from BTF
/// alone (Func, FuncProto, Datasec, Var, Void) or where the chain
/// requires further resolution that fails.
pub(crate) fn type_size(btf: &Btf, ty: &Type) -> Option<usize> {
    match ty {
        Type::Int(int) => Some(int.size()),
        Type::Float(f) => Some(f.size()),
        Type::Enum(e) => Some(e.size()),
        Type::Enum64(e) => Some(e.size()),
        Type::Struct(s) | Type::Union(s) => Some(s.size()),
        Type::Ptr(_) => Some(8),
        Type::Array(arr) => {
            let len = arr.len();
            let elem_peeled = peel_modifiers(btf, arr.get_type_id().ok()?)?;
            let elem_size = type_size(btf, &elem_peeled)?;
            Some(len * elem_size)
        }
        Type::Volatile(t) | Type::Const(t) | Type::Restrict(t) => {
            let inner = btf.resolve_chained_type(t).ok()?;
            type_size(btf, &inner)
        }
        Type::Typedef(t) | Type::TypeTag(t) => {
            let inner = btf.resolve_chained_type(t).ok()?;
            type_size(btf, &inner)
        }
        // Function types, datasec, var, fwd, void have no value size.
        _ => None,
    }
}

/// Read a little-endian unsigned integer up to 8 bytes wide.
fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_le_bytes(buf)
}

/// Sign-extend `raw` (unsigned, low `bits` populated) to a signed i64-
/// representable value held in u64 bit pattern.
fn sign_extend(raw: u64, bits: usize) -> u64 {
    if bits == 0 || bits >= 64 {
        return raw;
    }
    let shift = 64 - bits;
    ((raw << shift) as i64 >> shift) as u64
}

#[cfg(test)]
mod tests {
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
    fn display_nested_struct_indents_correctly() {
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
        let names: std::collections::HashSet<&str> =
            members.iter().map(|m| m.name.as_str()).collect();
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
        // has additional bits set higher up. Pins the F#7 fix:
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
    /// only cpus 0..=3 must appear. Pins the F#7 backstop: the
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
    /// by the reader, garbage bits beyond cpu 7 are dropped — the
    /// implementer's `let max_cpus = mem.map(|m| m.nr_cpu_ids())`
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
    /// reason inline in `[chase: ...]` notation. Pins the F#15
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
}
