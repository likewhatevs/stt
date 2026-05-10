//! BTF-driven rendering of raw value bytes into structured output.
//!
//! [`render_value`] takes a BTF type id and a byte slice and produces
//! a [`RenderedValue`] tree that mirrors the type's structure: ints,
//! floats, enums, structs, arrays, pointers. Modifier qualifiers
//! ([`btf_rs::Type::Volatile`], [`btf_rs::Type::Const`],
//! [`btf_rs::Type::Restrict`], [`btf_rs::Type::Typedef`],
//! [`btf_rs::Type::TypeTag`], [`btf_rs::Type::DeclTag`]) are peeled
//! before dispatch. [`render_value_with_mem`] is the production
//! entry point that additionally accepts a [`MemReader`] so the
//! [`btf_rs::Type::Ptr`] arm and the cast-intercept path
//! (consulting [`MemReader::cast_lookup`]) chase pointers through
//! arena snapshots, slab/vmalloc reads, the sdt_alloc bridge, and
//! the cross-BTF Fwd resolution index.
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
//! shifts and masks, and applies sign extension when the underlying
//! type is a signed Int, signed Enum, or signed Enum64 — BTF bitfields
//! can carry any of those bases (e.g. `enum scx_exit_kind` declared
//! with negative members).

use std::borrow::Cow;
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
    ///
    /// `cast_annotation` distinguishes cast-recovered pointers
    /// (set by [`render_cast_pointer`] to `"cast→arena"` /
    /// `"cast→kernel"`) from BTF-typed pointers (the
    /// [`Type::Ptr`] arm normally leaves it `None`). Display
    /// surfaces it as a parenthesised tag so operators can tell
    /// at a glance whether the pointer came from native BTF
    /// typing or the cast analyzer's recovery path.
    ///
    /// One [`Type::Ptr`] exception: when the renderer recovers a
    /// `BTF_KIND_FWD` pointee's real struct id via the sdt_alloc
    /// bridge ([`MemReader::resolve_arena_type`]), the arena
    /// branch sets this field to `"sdt_alloc"` so the rendered
    /// subtree is flagged as a recovered chase rather than a
    /// native BTF resolve. Cast-recovered pointers that cleared
    /// the same bridge extend the annotation to
    /// `"cast→{addr_space} (sdt_alloc)"`.
    ///
    /// Storage is [`Cow<'static, str>`] so the renderer's emit
    /// sites — every value the renderer itself produces is one
    /// of a five-element closed set
    /// (`"sdt_alloc"`, `"cast→arena"`, `"cast→kernel"`,
    /// `"cast→arena (sdt_alloc)"`, `"cast→kernel (sdt_alloc)"`)
    /// — borrow `&'static str` literals via [`Cow::Borrowed`]
    /// without per-chase heap allocations. JSON deserialization
    /// produces [`Cow::Owned`] (serde's [`Cow`] impl forwards
    /// to `String`'s deserializer), so existing serialized
    /// snapshots round-trip unchanged.
    Ptr {
        value: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deref: Option<Box<RenderedValue>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deref_skipped_reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cast_annotation: Option<Cow<'static, str>>,
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
    /// task_ctx{weight=1024, last_runnable_at=12345678901234}
    /// ```
    ///
    /// Structs that fit within the inline width budget pack onto one
    /// line as `TypeName{field=value, field=value}`; wider structs
    /// break to a multi-line `TypeName:` breadcrumb form with
    /// indented `field=value` rows. Nested structs and arrays indent
    /// by two spaces per level. Scalar-only arrays render inline
    /// (`[1, 2, 3]`); arrays containing structs / nested arrays
    /// render block-style with one element per line.
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
            cast_annotation,
            ..
        } => {
            write!(f, "0x{value:x}")?;
            if let Some(tag) = cast_annotation {
                write!(f, " ({tag})")?;
            }
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
            if *had == 0 {
                return Ok(());
            }
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
                let mut groups: Vec<(usize, usize, &RenderedValue)> = Vec::new();
                for (i, e) in elements.iter().enumerate() {
                    if is_zero(e) {
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
                // All-zero short-circuit: every element was zero-
                // suppressed, no groups recorded. The full rendering
                // would emit an empty bracket pair, so collapse to
                // the dense "all N zero]" form instead.
                if groups.is_empty() {
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
        // Pre-render the value to a single-line string for flat
        // scalars and for nested Struct values whose own Display
        // happens to fit inline (no embedded `\n`). Other compound
        // members (Array, Ptr-with-deref, Truncated, CpuList,
        // Unsupported) always produce multi-line output OR carry
        // their own internal layout that the breadcrumb path
        // re-renders directly via `write_rendered_value`, so they
        // get `None` here. A `None` rendering causes
        // `try_inline_from_rendered` to bail to the multi-line
        // path. Allowing nested Structs to participate in the
        // outer's inline form lets `outer{child=inner{a=1}}` pack
        // onto one line when both are small enough — without it,
        // any nested Struct would force the breadcrumb form even
        // for trivial two-level cases.
        let single_line = if is_flat_scalar(&m.value) {
            Some(format!("{}", m.value))
        } else if matches!(m.value, RenderedValue::Struct { .. }) {
            let s = format!("{}", m.value);
            if s.contains('\n') { None } else { Some(s) }
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

    for (i, m) in first.iter().enumerate() {
        if varying.contains(&i) {
            continue;
        }
        // is_deeply_zero so all-zero compound members (e.g. an
        // empty inner struct) suppress alongside scalars in the
        // template's common-fields section. Matches the main
        // `write_struct` filter — without this, a template would
        // render an `inner={}` line for the same value that the
        // non-template path collapses silently, producing
        // inconsistent output for callers that flip between
        // template and per-element rendering.
        if is_deeply_zero(&m.value) {
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
    Ok(true)
}

/// Try to render bytes as a cpumask cpu-list. Reads u64 words from
/// the start of `bytes`, extracts set bits, and formats as
/// `cpus={0,2,5-7}`. Returns None if bytes are too short.
///
/// `max_cpus` caps the highest CPU id walked: bits at positions >=
/// `max_cpus` are treated as out-of-range (slab padding / freelist
/// garbage) and stop the walk. The kernel sizes `struct cpumask`'s
/// `bits` array to `BITS_TO_LONGS(NR_CPUS)` words but only the first
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

// Re-export [`CastHit`] so the renderer's [`MemReader::cast_lookup`]
// trait method can name the return type without forcing every
// caller to import the cast_analysis module path. [`AddrSpace`] is
// no longer re-exported — it lives at its canonical home in
// [`super::cast_analysis::AddrSpace`] and the renderer treats the
// hint as runtime-secondary, so callers that need the variant
// import it directly from cast_analysis.
pub use super::cast_analysis::CastHit;

/// Outcome of [`MemReader::resolve_arena_type`]: the BTF type id the
/// chase should render against, paired with the byte count the
/// chase must skip past the chased address before the payload
/// struct begins.
///
/// The production
/// [`super::dump::render_map::AccessorMemReader::resolve_arena_type`]
/// emits exactly two `header_skip` shapes — mid-slot pointers
/// (header-region or mid-payload offsets) return `None`:
///
/// - `header_skip == 0` (payload-start chase): the chased address
///   already lands at the slot's payload start, e.g. the return of
///   `scx_task_data(p)` cached in `cached_taskc_raw`. The renderer
///   reads `btf_size` bytes from the chased address and renders
///   directly against `target_type_id`.
/// - `header_skip == slot.header_size` (slot-start chase): the
///   chased address lands at the slot's first byte, e.g. the
///   `data` field of `scx_task_map_val` storing the raw return of
///   `sdt_alloc()`. The renderer reads `header_skip + btf_size`
///   bytes from the chased address and slices off the leading
///   `header_skip` bytes (the `union sdt_id` header) before
///   rendering the payload struct against `target_type_id`.
///
/// Field name `target_type_id` matches
/// [`CastHit::target_type_id`]'s precedent so the two
/// chase-routing return shapes use the same vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArenaResolveHit {
    /// BTF type id (in the entry BTF's id space) of the payload
    /// struct the chase should render against. Resolved from the
    /// allocator slot's `target_type_id` populated by the
    /// sdt_alloc pre-pass.
    pub target_type_id: u32,
    /// Byte count the chase must skip past the chased address
    /// before the payload struct begins. `0` for a payload-start
    /// chase (the chased address already lands at the payload);
    /// the slot's `header_size` for a slot-start chase (the chase
    /// must skip the leading `union sdt_id` header before
    /// rendering the payload).
    pub header_skip: usize,
}

/// Aggregate kind of a `BTF_KIND_FWD` terminal: `struct foo;` vs
/// `union foo;`. Threaded into [`MemReader::cross_btf_resolve_fwd`]
/// so the resolver only matches a same-name complete body whose
/// aggregate kind agrees — a `Fwd` declared as `struct foo` must
/// NOT resolve to a `union foo` in another BTF (the wire format
/// permits same-name struct + union declarations, rare but legal).
/// Mirrors the gate [`peel_modifiers_resolving_fwd`] applies for
/// in-BTF resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FwdKind {
    /// `BTF_KIND_FWD` declared as `struct foo;`. Matches the
    /// `true` branch of [`btf_rs::Fwd::is_struct`].
    Struct,
    /// `BTF_KIND_FWD` declared as `union foo;`. Matches the
    /// `false` branch of [`btf_rs::Fwd::is_struct`].
    Union,
}

impl FwdKind {
    /// Build a [`FwdKind`] from a [`btf_rs::Fwd`]'s aggregate kind
    /// flag. The two callers
    /// ([`try_cross_btf_fwd_resolve`] and
    /// [`peel_modifiers_resolving_fwd`]) reach the flag via
    /// [`btf_rs::Fwd::is_struct`].
    pub fn from_is_struct(is_struct: bool) -> Self {
        if is_struct {
            FwdKind::Struct
        } else {
            FwdKind::Union
        }
    }
}

/// Reference to a complete struct/union definition in a BTF other
/// than the one the chase entered with. Returned by
/// [`MemReader::cross_btf_resolve_fwd`] when a `BTF_KIND_FWD`
/// terminal in the entry BTF resolves to a body in a sibling
/// object's BTF (the multi-`.bpf.objs` shape: one object declares
/// `struct foo;` (forward), another defines `struct foo { ... }`
/// (full body)).
///
/// Borrowed: the BTF reference is tied to the [`MemReader`]
/// implementation's owned BTF storage (typically `Arc<Btf>` retained
/// across the dump pass). Render code that recurses into `btf`
/// must thread the same [`MemReader`] through so further chases
/// from inside the cross-BTF subtree can also resolve cross-BTF
/// (same-name structs in either direction).
///
/// `type_id` is the resolved struct/union type id WITHIN `btf`'s
/// own id space — distinct from the entry BTF's id space. The
/// chase code switches the rendering BTF to `btf` for the
/// recursion against `type_id`.
///
/// `Copy + Clone`: a borrowed reference plus a `u32` is bitwise
/// copyable. The pair lets the chase paths pass the value by
/// `Copy` rather than by `&` and sidesteps the move-after-borrow
/// snags in any match that would otherwise consume the hit
/// twice. [`Debug`] / [`Hash`] / [`Eq`] are blocked by [`Btf`]
/// in `btf-rs` — it does not derive any of them — so the
/// minimal `Copy + Clone` set is the most we can offer until
/// upstream changes.
#[derive(Copy, Clone)]
pub struct CrossBtfRef<'a> {
    /// Sibling BTF that carries the resolved struct/union body.
    /// Borrowed from the [`MemReader`] implementation's owned BTF
    /// storage (typically `Arc<Btf>` retained across the dump
    /// pass) — the renderer must thread the same [`MemReader`]
    /// through any recursion into this BTF so further chases
    /// from inside the cross-BTF subtree can also resolve
    /// cross-BTF.
    pub btf: &'a Btf,
    /// Resolved struct/union type id WITHIN [`Self::btf`]'s own
    /// id space — distinct from the entry BTF's id space. The
    /// chase code switches the rendering BTF to [`Self::btf`]
    /// and recurses against this id.
    pub type_id: u32,
}

/// # CrossBtfMemReader contract
///
/// [`CrossBtfMemReader`] wraps a `&dyn MemReader` and selectively
/// suppresses id-keyed lookups (cast_lookup, resolve_arena_type)
/// at cross-BTF boundaries. When adding a new method to this trait:
/// check whether the method is id-keyed (operates on BTF type IDs
/// from the entry BTF). If yes, CrossBtfMemReader MUST override it
/// to return `None`/default. If no (raw addresses, string names),
/// CrossBtfMemReader should delegate to inner. Failing to audit
/// causes silent wrong-renders in cross-BTF chase paths.
pub trait MemReader {
    fn read_kva(&self, kva: u64, len: usize) -> Option<Vec<u8>>;
    /// Check if an address is in the arena range. Arena pointers
    /// resolve into `ArenaSnapshot`'s captured page set, so the
    /// reader has a frozen byte view — chasing them is well-defined.
    /// Kernel kptrs (slab/vmalloc allocations outside the arena
    /// window) MAY be stale references to objects already freed by
    /// the time the freeze captured them; the renderer applies a
    /// best-effort plausibility heuristic (top-byte check on the
    /// first qword to reject obvious freelist next-pointer
    /// patterns — see [`render_cast_pointer`] and the cpumask kptr
    /// branch in the [`btf_rs::Type::Ptr`] arm) but cannot verify
    /// liveness. Cast-recovered kernel kptrs ARE chased through
    /// [`MemReader::read_kva`] when [`MemReader::cast_lookup`]
    /// returns an [`AddrSpace::Kernel`] hit, even though slab
    /// liveness is not guaranteed; the heuristic gates and the
    /// `deref_skipped_reason` field on [`RenderedValue::Ptr`]
    /// surface uncertainty without dropping the hit. Default
    /// returns false — pointer chasing skips arena resolution
    /// silently.
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
    /// walk at this value: the kernel's `struct cpumask` `bits`
    /// slab allocation is sized to `BITS_TO_LONGS(NR_CPUS)`, but
    /// only the first `nr_cpu_ids` bits are meaningful — bits
    /// beyond that are slab-internal padding or freelist garbage
    /// that `SLAB_FREELIST_HARDENED` XOR-encoding can mask the
    /// top-byte heuristic from rejecting. Default returns
    /// `u32::MAX` (no cap) so callers without the value still
    /// produce a render.
    fn nr_cpu_ids(&self) -> u32 {
        u32::MAX
    }
    /// Look up a cast finding for `(parent_type_id, member_byte_offset)`.
    /// `parent_type_id` is the BTF type id of the *struct/union* that
    /// owns the member (already peeled through Typedef / Const /
    /// Volatile / Restrict / TypeTag / DeclTag — the cast analyzer
    /// keys on the underlying aggregate, not the modifier-wrapped
    /// surface type).
    /// `member_byte_offset` is the byte offset of the `u64` member
    /// inside that struct.
    ///
    /// Returning `Some(hit)` lets the renderer interpret a `u64`
    /// member as `Ptr(hit.target_type_id)` and chase it through the
    /// reader corresponding to `hit.addr_space`. Returning `None`
    /// (the default) leaves the renderer's existing behavior intact:
    /// the field renders as a plain unsigned integer. Default-`None`
    /// keeps every existing [`MemReader`] impl correct without an
    /// explicit override.
    fn cast_lookup(&self, _parent_type_id: u32, _member_byte_offset: u32) -> Option<CastHit> {
        None
    }
    /// Resolve a chased arena pointer value to the BTF type id of
    /// the payload it points at, plus a `header_skip` byte count
    /// that tells the chase how to land on the payload struct from
    /// the chased address.
    ///
    /// The intended trigger: a [`Type::Ptr`] (or cast-recovered
    /// pointer) whose declared pointee is a [`Type::Fwd`] whose body
    /// lives in a separate BTF object. The scheduler's program BTF
    /// carries only the `BTF_KIND_FWD` forward declaration — there is
    /// no struct body to size against, so the renderer's
    /// [`chase_arena_pointer`] / [`render_cast_pointer`] paths skip
    /// with an "unsizable" reason. The
    /// [`super::sdt_alloc::SdtAllocatorSnapshot`] pre-pass already
    /// resolves the real payload BTF type id via
    /// [`super::sdt_alloc::discover_payload_btf_id`] for every live
    /// allocator. The dump path threads a per-pass index of every
    /// live allocator slot into the renderer's [`MemReader`] so the
    /// chase can recover the real type id and the slot shape when
    /// the BTF-only resolve fails.
    ///
    /// `addr` is the chased pointer value as it appears in guest
    /// memory (i.e. the same value the renderer just read from a
    /// `u64` field or a [`Type::Ptr`] field). Implementations
    /// transform it into the index key shape they store. The
    /// sdt_alloc bridge stores one entry per slot keyed on
    /// `slot_start & 0xFFFF_FFFF`; the production
    /// [`super::dump::render_map::AccessorMemReader`] impl uses a
    /// range lookup to find the slot whose
    /// `[slot_start, slot_start + elem_size)` range contains the
    /// chased address.
    ///
    /// Returns `Some(ArenaResolveHit { target_type_id, header_skip })`
    /// when the address falls inside a known allocator slot at a
    /// position the renderer can chase:
    ///
    /// - **Slot-start pointer** (the chased address equals the slot
    ///   start, e.g. the `data` field of `scx_task_map_val` storing
    ///   the raw return of `sdt_alloc()`): `header_skip` is the
    ///   slot's header size (typically 8 — the size of `union
    ///   sdt_id`). The renderer reads `header_skip + btf_size`
    ///   bytes from `addr`, slices off the first `header_skip`
    ///   bytes (the sdt_id header), and renders the payload struct
    ///   against `target_type_id`.
    /// - **Payload-start pointer** (the chased address equals
    ///   `slot_start + header_size`, e.g. the return of
    ///   `scx_task_data(p)` cached in `cached_taskc_raw`):
    ///   `header_skip == 0`. The renderer reads `btf_size` bytes
    ///   from `addr` and renders directly.
    ///
    /// Returns `None` (the default) when the reader has no index,
    /// the address is outside every known allocator's slot range,
    /// the addressed slot has no resolved payload type, or the
    /// chased pointer landed at an interior slot position
    /// (mid-header or mid-payload) the renderer cannot translate
    /// into a struct render. Default-`None` keeps existing
    /// [`MemReader`] impls correct without an explicit override.
    fn resolve_arena_type(&self, _addr: u64) -> Option<ArenaResolveHit> {
        None
    }
    /// Resolve a `BTF_KIND_FWD` terminal by struct/union name to a
    /// complete definition in a sibling BTF.
    ///
    /// The intended trigger: the renderer's chase paths
    /// ([`chase_arena_pointer`] and [`render_cast_pointer`]) just
    /// peeled the chase target through
    /// [`peel_modifiers_resolving_fwd`] but the local same-BTF
    /// sibling search came up empty. The terminal is still a
    /// `Type::Fwd`, so the BTF-only chase would skip with
    /// "forward declaration; body not in this BTF". This method
    /// asks the reader whether any sibling BTF (built once per
    /// scheduler binary by the cast-analysis pre-pass — see
    /// [`crate::vmm::cast_analysis_load::CastAnalysisOutput::fwd_index`])
    /// carries a complete body for `name` matching the `kind`
    /// aggregate kind ([`FwdKind::Struct`] or [`FwdKind::Union`]).
    ///
    /// Returning `Some(CrossBtfRef { btf, type_id })` lets the
    /// chase switch to `btf` and recurse into `type_id` for the
    /// pointee render. The same [`MemReader`] is threaded into
    /// the inner recursion so chases originating inside the
    /// cross-BTF subtree can also bridge — typical for a struct
    /// whose members are themselves `Fwd` to another BTF.
    ///
    /// Returning `None` (the default) preserves the historical
    /// "forward declaration; body not in this BTF" skip path —
    /// keeps every existing [`MemReader`] impl correct without
    /// an explicit override.
    fn cross_btf_resolve_fwd(&self, _name: &str, _kind: FwdKind) -> Option<CrossBtfRef<'_>> {
        None
    }

    /// Type-gated meta fallback for sdt_alloc bridge. Only called
    /// when `resolve_arena_type` returned None AND the chase target
    /// is a known Fwd type (sdt_data). Returns the first sdt_alloc
    /// meta with a resolved payload type if the address is in the
    /// arena window.
    fn resolve_arena_type_meta_fallback(&self, _addr: u64) -> Option<ArenaResolveHit> {
        None
    }
}

/// Render a BTF type's bytes into a [`RenderedValue`] without an
/// associated guest-memory reader. Pointer dereferences degrade
/// gracefully — the raw pointer hex is emitted without chasing.
#[allow(dead_code)]
pub fn render_value(btf: &Btf, type_id: u32, bytes: &[u8]) -> RenderedValue {
    let mut visited: HashSet<u64> = HashSet::new();
    render_value_inner(btf, type_id, bytes, 0, None::<&dyn MemReader>, &mut visited)
}

/// Render a BTF type's bytes into a [`RenderedValue`] with an
/// associated guest-memory reader for pointer chasing. Identical to
/// [`render_value`] except the supplied [`MemReader`] is threaded
/// through the [`btf_rs::Type::Ptr`] arm and the cast-intercept path
/// in `render_member`, so:
///
/// - BTF-typed pointers ([`btf_rs::Type::Ptr`]) are dereferenced via
///   [`MemReader::read_arena`] (when [`MemReader::is_arena_addr`]
///   matches) or the cpumask kptr chase via [`MemReader::read_kva`].
/// - `u64` fields the cast analyzer flagged via
///   [`MemReader::cast_lookup`] are interpreted as typed pointers and
///   chased through [`render_cast_pointer`] — the same chase path,
///   producing the same [`RenderedValue::Ptr`] shape.
///
/// Total in the same sense as [`render_value`]: any failure (unmapped
/// page, plausibility-gate rejection, cycle, depth cap) surfaces as a
/// `Ptr` with `deref: None` and a populated `deref_skipped_reason`,
/// never an error return.
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

    let Some((ty, peeled_type_id)) = peel_modifiers_with_id(btf, type_id) else {
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
            // `cast_annotation` is normally `None` for BTF-typed
            // pointers (the Type::Ptr arm) — the field is reserved
            // for the cast analyzer's recovered pointers. The one
            // exception is a Fwd-pointee chase the renderer
            // recovered via [`MemReader::resolve_arena_type`] (the
            // sdt_alloc bridge): the chased pointer is structurally
            // BTF-typed but the body lives in another BTF, so the
            // renderer surfaces a `sdt_alloc` annotation to flag the
            // recovered chase even on this arm. Set inside
            // [`chase_arena_pointer`]'s outcome and threaded out via
            // the side-channel below.
            let mut cast_annotation: Option<Cow<'static, str>> = None;
            // [`chase_gate`] applies the null/cycle/depth-cap policy
            // shared with [`render_cast_pointer`]: null and
            // depth-cap take the "no chase attempted" path
            // (`deref` + reason both `None`); a cycle records the
            // `cycle → 0x{val:x}` reason and skips the chase. Only
            // [`ChaseGate::Proceed`] enters the `mem.and_then`
            // closure that performs the actual read.
            let deref = match chase_gate(val, depth, visited) {
                ChaseGate::Skip { reason } => {
                    deref_skipped_reason = reason;
                    None
                }
                ChaseGate::Proceed => mem.and_then(|m| {
                    let pointee_type_id = ptr.get_type_id().ok()?;
                    if m.is_arena_addr(val) {
                        // Arena chase factored into the shared
                        // helper so this arm and
                        // [`render_cast_pointer`]'s arena branch
                        // produce identical [`RenderedValue::Ptr`]
                        // shapes (including the
                        // [`RenderedValue::Truncated`] wrap when
                        // `btf_size > POINTER_CHASE_CAP`). The
                        // helper computes `btf_size` from
                        // `pointee_type_id`, so the local
                        // peel/size resolution that follows runs
                        // only on the kptr path below.
                        let outcome =
                            chase_arena_pointer(btf, pointee_type_id, val, m, depth, visited);
                        if outcome.reason.is_some() {
                            deref_skipped_reason = outcome.reason;
                        }
                        if outcome.sdt_alloc_resolved {
                            cast_annotation = Some(Cow::Borrowed("sdt_alloc"));
                        }
                        return outcome.deref;
                    }
                    // Use the Fwd-resolving peel so a kernel kptr
                    // whose declared pointee is a [`Type::Fwd`] with
                    // a complete sibling in the BTF lands on the
                    // sibling rather than failing the size gate
                    // below. Drops [`effective_type_id`] because
                    // this arm does not recurse into a struct
                    // render — the cpumask-name dispatch below
                    // works against the resolved [`Type`] alone.
                    let (pointee_ty, _) = peel_modifiers_resolving_fwd(btf, pointee_type_id)?;
                    let btf_size = type_size(btf, &pointee_ty)?;
                    if btf_size == 0 {
                        // Sanity gate: an incomplete pointee type
                        // is not safe to chase even on the cpumask
                        // path (BTF reported no bytes — the
                        // underlying allocation may not exist as
                        // declared). Match the historical Type::Ptr
                        // arm behavior.
                        deref_skipped_reason =
                            Some("pointee BTF size is 0 (incomplete type)".to_string());
                        return None;
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
                        // kernel allocates the `struct cpumask`
                        // `bits` storage from a slab cache sized to
                        // `cpumask_size()`, which is `(NR_CPUS + 7)
                        // / 8` rounded up to a multiple of 8 —
                        // bounded by NR_CPUS at config time. 1024
                        // covers every modern distro kernel;
                        // mainline NR_CPUS_DEFAULT is 8192 for
                        // x86_64 / aarch64. The per-word walker
                        // below caps the rendered bits at the
                        // guest's `nr_cpu_ids` so a small guest
                        // (e.g. 8 CPUs) doesn't render bits
                        // 64..8191 from slab padding.
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
                }),
            };
            RenderedValue::Ptr {
                value: val,
                deref,
                deref_skipped_reason,
                cast_annotation,
            }
        }
        Type::Struct(s) | Type::Union(s) => {
            // `peeled_type_id` is the BTF id of `s` after modifier
            // peel — the form [`super::cast_analysis::CastMap`] keys
            // its `(parent_type_id, member_byte_offset)` lookups
            // against. Threaded into `render_struct` so per-member
            // cast intercepts can consult the reader.
            render_struct(btf, &s, peeled_type_id, bytes, depth, mem, visited)
        }
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
        Type::Datasec(ds) => {
            // `peeled_type_id` is the BTF id of the datasec after
            // modifier peel — the form
            // [`super::cast_analysis::CastMap`] keys its
            // `(parent_type_id, member_byte_offset)` lookups
            // against. Threaded into `render_datasec` so per-
            // variable cast intercepts can consult the reader,
            // mirroring the `render_struct` path that handles
            // struct/union members.
            render_datasec(btf, &ds, peeled_type_id, bytes, depth, mem, visited)
        }
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
    parent_type_id: u32,
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
        let bit_off = m.bit_offset() as usize;
        let byte_off = bit_off / 8;
        if byte_off >= bytes.len() && bytes.len() < s.size() {
            continue;
        }
        let name = btf.resolve_name(m).unwrap_or_default();
        let value = render_member(btf, m, Some(parent_type_id), bytes, depth, mem, visited);
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

/// Shared BPF cast intercept gate for [`render_member`] and
/// [`render_datasec`]. Caller passes the (parent_id, byte_off) key
/// the cast analyzer would have seen. Returns `Some(rendered)` when
/// every gate aligns and the field is a cast-recovered typed
/// pointer; `None` otherwise (caller falls through to its standard
/// render path).
///
/// `field_bytes` must be the parent_bytes / section_bytes slice
/// from `byte_off` onward (i.e. the slice already advanced to the
/// field's start). The helper reads the first 8 bytes; a slice
/// shorter than 8 bytes falls through to `None`.
///
/// Gates (in order):
/// - `mem` is `Some` (no chase is possible without a [`MemReader`]).
/// - `peeled` peels to a plain unsigned 8-byte [`Type::Int`] — BPF
///   stores typed pointers in `u64` slots; signed, `_Bool`, `char`,
///   and sub-u64 widths are not the cast analyzer's output shape.
/// - `byte_off` fits in `u32` (datasec / struct offsets exceed
///   `u32::MAX` only in malformed BTF; the analyzer keys offsets as
///   `u32`).
/// - [`MemReader::cast_lookup`] returns a hit for
///   `(parent_type_id, byte_off)` (default `None` keeps every
///   reader correct without an explicit override).
/// - `field_bytes` is at least 8 bytes long (a truncated field
///   falls through to the existing partial-decode path so the
///   consumer still sees whatever survived).
fn try_cast_intercept(
    btf: &Btf,
    cast_key: (u32, usize),
    peeled: &Type,
    field_bytes: &[u8],
    depth: u32,
    mem: Option<&dyn MemReader>,
    visited: &mut HashSet<u64>,
) -> Option<RenderedValue> {
    let (parent_type_id, byte_off) = cast_key;
    let reader = mem?;
    let Type::Int(int) = peeled else {
        return None;
    };
    if int.size() != 8 || int.is_signed() || int.is_bool() || int.is_char() {
        return None;
    }
    let off_u32 = u32::try_from(byte_off).ok()?;
    let hit = reader.cast_lookup(parent_type_id, off_u32)?;
    let head = field_bytes.get(..8)?;
    let value = u64::from_le_bytes(head.try_into().ok()?);
    Some(render_cast_pointer(btf, hit, value, depth, reader, visited))
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
    parent_type_id: u32,
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
        // BPF cast intercept: a `u64` global variable that the cast
        // analyzer recovered as a typed pointer renders as `Ptr`
        // with the recovered target chased through the appropriate
        // address-space reader. Mirrors the gate in
        // [`render_member`] for struct members, but keyed on the
        // datasec id + variable offset (the cast analyzer's
        // `(parent, off)` pair for a BSS / data global). Shared
        // gating logic lives in [`try_cast_intercept`].
        let cast_intercept = peel_modifiers(btf, inner_id).and_then(|inner_ty| {
            let field_bytes = bytes.get(offset..).unwrap_or_default();
            try_cast_intercept(
                btf,
                (parent_type_id, offset),
                &inner_ty,
                field_bytes,
                depth,
                mem,
                visited,
            )
        });
        if let Some(rv) = cast_intercept {
            members.push(RenderedMember {
                name: var_name,
                value: rv,
            });
            continue;
        }
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
    parent_type_id: Option<u32>,
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

    if let Some(width) = m.bitfield_size()
        && width > 0
    {
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

    // BPF cast intercept: a `u64` member that the cast analyzer
    // recovered as a typed pointer is rendered as `Ptr` with the
    // recovered target chased through the appropriate address-space
    // reader. Shared gating logic lives in [`try_cast_intercept`];
    // the additional `parent_type_id?` here ensures we are inside a
    // struct that [`render_struct`] dispatched (a standalone Int
    // render carries no parent and skips the intercept).
    let cast_intercept = parent_type_id.and_then(|parent| {
        let field_bytes = parent_bytes.get(byte_off..).unwrap_or_default();
        try_cast_intercept(
            btf,
            (parent, byte_off),
            &member_ty,
            field_bytes,
            depth,
            mem,
            visited,
        )
    });
    if let Some(rv) = cast_intercept {
        return rv;
    }

    if let Some(parent) = parent_type_id
        && let Type::Array(arr) = &member_ty
        && let (Ok(elem_tid), Some(elem_size)) = (
            arr.get_type_id(),
            peel_modifiers(btf, arr.get_type_id().unwrap_or(0)).and_then(|t| type_size(btf, &t)),
        )
        && elem_size == 8
        && let Some(elem_term) = peel_modifiers(btf, elem_tid)
        && matches!(
            elem_term,
            Type::Int(ref i) if i.size() == 8 && !i.is_signed() && !i.is_bool() && !i.is_char()
        )
    {
        let arr_len = arr.len();
        let has_any_cast = mem.is_some_and(|m| {
            (0..arr_len).any(|i| {
                let elem_off = (byte_off + i * 8) as u32;
                m.cast_lookup(parent, elem_off).is_some()
            })
        });
        if has_any_cast {
            let cap = arr_len.min(MAX_ARRAY_ELEMS);
            let mut elements = Vec::with_capacity(cap);
            for i in 0..cap {
                let elem_off = byte_off + i * 8;
                let elem_bytes = parent_bytes.get(elem_off..elem_off + 8).unwrap_or_default();
                if let Some(rv) = try_cast_intercept(
                    btf,
                    (parent, elem_off),
                    &elem_term,
                    elem_bytes,
                    depth + 1,
                    mem,
                    visited,
                ) {
                    elements.push(rv);
                } else {
                    elements.push(render_value_inner(
                        btf,
                        elem_tid,
                        elem_bytes,
                        depth + 1,
                        mem,
                        visited,
                    ));
                }
            }
            return RenderedValue::Array {
                len: arr_len,
                elements,
            };
        }
    }

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

/// Cap on bytes any pointer chase reads from a target. Shared
/// between the [`Type::Ptr`] arena branch and [`render_cast_pointer`]
/// so a single tunable applies to every chase the renderer performs:
/// a single arena page is 4 KiB, the [`MemReader::read_arena`]
/// contract bails on cross-page reads, and `MemReader::read_kva`
/// callers should avoid pulling many pages of slab content into the
/// dump for a single recovered pointer. Targets larger than the cap
/// surface as a [`RenderedValue::Truncated`] wrapping the partial
/// decode so the consumer can tell the rendered subtree was clipped.
const POINTER_CHASE_CAP: usize = 4096;

/// Outcome of [`chase_gate`]: skip the chase (with optional reason
/// for `deref_skipped_reason`) or proceed with the read+recurse.
///
/// Lets the [`Type::Ptr`] arm and [`render_cast_pointer`] share a
/// single null/cycle/depth-cap policy: null and depth-cap produce
/// `Skip { reason: None }` (no chase attempted, no reason emitted);
/// cycle produces `Skip { reason: Some("cycle → 0x{val:x}") }` so
/// Display shows the cycle marker.
enum ChaseGate {
    /// Skip the chase. `reason` populates `deref_skipped_reason` on
    /// the resulting [`RenderedValue::Ptr`]; `None` means no chase
    /// was attempted and the operator sees an unannotated raw
    /// pointer.
    Skip { reason: Option<String> },
    /// All gates passed; the caller should perform the chase.
    Proceed,
}

/// Pre-chase gate shared between the [`Type::Ptr`] arm and
/// [`render_cast_pointer`]. Returns [`ChaseGate::Skip`] when the
/// renderer must not chase `val` (null, already-visited cycle, or
/// recursion depth cap), [`ChaseGate::Proceed`] otherwise.
///
/// Order of checks matches both call sites' historical behavior:
/// null first, then cycle, then depth. `val == 0` short-circuits
/// before consulting `visited`, so a stray zero entry in the set
/// (which should not occur) does not surface as a phantom cycle.
fn chase_gate(val: u64, depth: u32, visited: &HashSet<u64>) -> ChaseGate {
    if val == 0 {
        return ChaseGate::Skip { reason: None };
    }
    if visited.contains(&val) {
        return ChaseGate::Skip {
            reason: Some(format!("cycle → 0x{val:x}")),
        };
    }
    if depth >= MAX_RENDER_DEPTH {
        return ChaseGate::Skip { reason: None };
    }
    ChaseGate::Proceed
}

/// Compose a `deref_skipped_reason` string for a pointer chase
/// whose peeled target type has no BTF-resolvable storage size.
///
/// [`type_size`] returns `None` for [`Type::Fwd`] (forward-declared
/// struct/union with body in another BTF), [`Type::Func`],
/// [`Type::FuncProto`], [`Type::Datasec`], [`Type::Var`],
/// [`Type::Void`], and [`Type::DeclTag`] (which `peel_modifiers`
/// peels in practice — listed for completeness in case a future
/// analyzer or BTF shape leaks one through). Each variant has a
/// distinct cause; surfacing the variant name plus, when available,
/// the BTF-declared name of the type lets operators correlate the
/// failure with their source layout (e.g. `struct sdt_data` lives
/// in the sdt_alloc library's BTF and surfaces as Fwd in the
/// scheduler's own BTF).
///
/// `kind_label` is the call site's chase prefix (`"arena chase"` /
/// `"kernel cast"`) so the reason matches the existing message
/// style at each site without forcing each caller to thread the
/// label through a `format!`.
fn unsizable_chase_reason(
    btf: &Btf,
    kind_label: &'static str,
    target_type_id: u32,
    target_ty: &Type,
) -> String {
    match target_ty {
        Type::Fwd(fwd) => {
            // A Fwd is `struct X;` or `union X;` with no body in
            // this BTF — typical when a scheduler library defines
            // the struct (e.g. `struct sdt_data` in the sdt_alloc
            // library) and the using program only references it
            // via pointer. The chase has no BTF-declared size to
            // bound the read, so it skips when both the
            // sdt_alloc bridge ([`MemReader::resolve_arena_type`])
            // and the cross-BTF Fwd resolution index
            // ([`MemReader::cross_btf_resolve_fwd`]) have already
            // been consulted and neither produced a hit — i.e.
            // the body lives in some BTF the renderer cannot
            // reach with the available indexes.
            let aggregate = if fwd.is_union() { "union" } else { "struct" };
            let name = btf.resolve_name(fwd).ok().filter(|n| !n.is_empty());
            match name {
                Some(n) => format!(
                    "{kind_label} target {aggregate} {n} (type id \
                     {target_type_id}) is a forward declaration; \
                     body not in this BTF"
                ),
                None => format!(
                    "{kind_label} target type id {target_type_id} \
                     is an anonymous {aggregate} forward declaration; \
                     body not in this BTF"
                ),
            }
        }
        Type::Func(_) => format!(
            "{kind_label} target type id {target_type_id} is a \
             function (BTF_KIND_FUNC); functions have no storage size"
        ),
        Type::FuncProto(_) => format!(
            "{kind_label} target type id {target_type_id} is a \
             function prototype (BTF_KIND_FUNC_PROTO); prototypes \
             have no storage size"
        ),
        Type::Datasec(_) => format!(
            "{kind_label} target type id {target_type_id} is a \
             datasec (BTF_KIND_DATASEC); not a pointer chase target"
        ),
        Type::Var(_) => format!(
            "{kind_label} target type id {target_type_id} is a \
             var (BTF_KIND_VAR); not a pointer chase target"
        ),
        Type::Void => format!(
            "{kind_label} target type id {target_type_id} is void; \
             chasing a void* requires runtime type info"
        ),
        // `peel_modifiers` peels DeclTag in practice, so reaching
        // it here implies a malformed BTF chain — keep the
        // diagnostic explicit rather than collapsing into the
        // generic fall-through.
        Type::DeclTag(_) => format!(
            "{kind_label} target type id {target_type_id} is a \
             decl-tag (BTF_KIND_DECL_TAG); modifiers should have \
             peeled (malformed BTF chain?)"
        ),
        // Defense-in-depth fall-through: every other variant
        // ([`Type::Int`], [`Type::Float`], [`Type::Enum`],
        // [`Type::Enum64`], [`Type::Struct`], [`Type::Union`],
        // [`Type::Ptr`], [`Type::Array`], and the modifier
        // wrappers `peel_modifiers` strips) returns `Some` from
        // [`type_size`], so reaching this arm means a future
        // [`Type`] variant slipped through without a sizing rule.
        // Keep the legacy generic message rather than pretending
        // we know the cause.
        _ => format!(
            "{kind_label} target type id {target_type_id} has \
             unresolvable size"
        ),
    }
}

/// Outcome of an arena pointer chase. Exactly one of `deref` /
/// `reason` carries content: `deref` is the rendered target subtree
/// when the chase succeeded; `reason` populates
/// [`RenderedValue::Ptr::deref_skipped_reason`] when the chase was
/// attempted but did not land. `sdt_alloc_resolved` records whether
/// the renderer recovered the chase target's BTF type id from the
/// [`MemReader::resolve_arena_type`] sdt_alloc bridge instead of the
/// pointer's own BTF declaration — callers surface this through
/// [`RenderedValue::Ptr::cast_annotation`] so operators can
/// distinguish `BTF-typed → Fwd-resolved-via-sdt_alloc` chases from
/// ordinary BTF-typed chases at a glance.
struct ArenaChaseOutcome {
    deref: Option<Box<RenderedValue>>,
    reason: Option<String>,
    sdt_alloc_resolved: bool,
}

/// Outcome of [`try_sdt_alloc_bridge`]. Distinguishes the no-fire
/// case from a fire that adopts a recovered payload type id, with
/// the resolved type, its id, the `header_skip` the caller applies
/// to its arena read, and the payload's `btf_size` so callers
/// don't re-resolve any of them.
///
/// `target_ty` and `effective_type_id` are the post-peel type and
/// type id the caller adopts for its render. The bridge resolves
/// the recovered id through [`peel_modifiers_resolving_fwd`] before
/// producing this struct so the caller can switch its render
/// target without re-running the peel.
///
/// `header_skip` is the byte count the chase must skip past the
/// chased address before the payload struct begins. The
/// [`MemReader::resolve_arena_type`] contract returns this value:
/// 0 when the chased pointer already lands on payload-start (the
/// historical behaviour), or the slot's header size when the
/// chased pointer lands on slot-start (the new path that handles
/// the raw return of `sdt_alloc()` cached in `data` fields).
///
/// Try the sdt_alloc bridge for a `BTF_KIND_FWD` chase target.
///
/// Returns the raw [`ArenaResolveHit`] when the chased address
/// falls in a known sdt_alloc slot. The caller feeds the hit's
/// `target_type_id` through the same peel → cross-BTF → size
/// pipeline it runs for the direct target, so the bridge hit
/// gets the full resolution chain (including cross-BTF Fwd
/// fallback) instead of being silently dropped when the resolved
/// type is also Fwd in the entry BTF.
fn try_sdt_alloc_bridge(
    mem: &dyn MemReader,
    val: u64,
    target_ty: &Type,
) -> Option<ArenaResolveHit> {
    if !matches!(target_ty, Type::Fwd(_)) {
        return None;
    }
    if let Some(hit) = mem.resolve_arena_type(val) {
        return Some(hit);
    }
    // Index miss — the sdt_alloc tree walker may have found 0
    // live allocations (race with scheduler unregistration
    // freeing all slots before the freeze captures bitmaps).
    // Fall back to the sdt_alloc meta when the chased address
    // is in the arena window. This is TYPE-GATED (only fires
    // for Fwd pointee chases from typed Ptr fields, not for
    // arbitrary u64 fields) so it cannot produce the "false
    // positive factory" behavior the blanket meta fallback had.
    mem.resolve_arena_type_meta_fallback(val)
}

/// Slice past `header_skip` bytes when the sdt_alloc bridge fires.
///
/// Returns the byte slice starting at `header_skip` within
/// `raw_bytes`, or `None` when the read returned fewer bytes than
/// the header skip needs (page-tail truncation, short read). Both
/// chase arms — [`chase_arena_pointer`] and [`render_cast_pointer`]
/// — apply this slice after their per-arm read so the callers
/// share a single underrun guard.
fn apply_header_skip(raw_bytes: &[u8], header_skip: usize) -> Option<&[u8]> {
    raw_bytes.get(header_skip..)
}

/// [`MemReader`] adapter that suppresses any lookup whose result
/// is keyed against the entry BTF's id space, while delegating
/// every other method.
///
/// Used by [`chase_arena_pointer`] and [`render_cast_pointer`] when
/// [`try_cross_btf_fwd_resolve`] succeeds and the chase recursion
/// switches from the entry BTF to a sibling BTF.
///
/// Two methods carry entry-BTF-keyed payloads and MUST be
/// suppressed when the chase has crossed a BTF boundary:
///
/// - [`MemReader::cast_lookup`]: the cast analyzer's
///   [`super::cast_analysis::CastMap`] keys on `(parent_type_id,
///   member_byte_offset)` against the entry BTF's id space. Sibling
///   BTFs re-use the same numeric id space, so a cast hit at
///   `(entry_id=N, off=K)` would incorrectly fire when the renderer
///   reaches a sibling BTF's struct that happens to carry id `N`. The
///   renderer would treat a plain `u64` field in the sibling struct
///   as a cast-recovered pointer, producing a phantom `Ptr` render
///   with a "type id unresolvable" skip reason against an id that is
///   only meaningful in the entry BTF.
///
/// - [`MemReader::resolve_arena_type`][]: the
///   [`super::dump::render_map::ArenaTypeIndex`] populates
///   [`ArenaResolveHit::target_type_id`] with BTF type ids resolved
///   against the **entry BTF** at index-build time (the sdt_alloc
///   pre-pass runs against the program BTF, which is the entry BTF
///   for the chase that initiated the dump). When a cross-BTF chase
///   recurses into a sibling BTF and encounters another `Type::Fwd`,
///   [`try_sdt_alloc_bridge`] would call `mem.resolve_arena_type`
///   and receive an entry-BTF id; passing that id to
///   [`peel_modifiers_resolving_fwd`]`(sibling_btf, …)` looks up the
///   wrong id in the sibling BTF's space — silent wrong-render (the
///   sibling BTF carries a different type at that id) or silent skip
///   (the sibling BTF lacks the id). The "no invalid data made"
///   contract requires this to be a hard `None`.
///
/// The remaining [`MemReader`] methods (read_kva / read_arena /
/// is_arena_addr / nr_cpu_ids / cross_btf_resolve_fwd) carry no
/// entry-BTF id payload — they operate on raw addresses and string
/// names — so they stay live. Chases originating inside the
/// cross-BTF subtree still resolve through the same arena snapshot,
/// kernel page-walker, and cross-BTF Fwd index. The suppression is
/// narrowly scoped to the two id-keyed lookups.
struct CrossBtfMemReader<'a> {
    inner: &'a dyn MemReader,
}

impl MemReader for CrossBtfMemReader<'_> {
    fn read_kva(&self, kva: u64, len: usize) -> Option<Vec<u8>> {
        self.inner.read_kva(kva, len)
    }
    fn is_arena_addr(&self, addr: u64) -> bool {
        self.inner.is_arena_addr(addr)
    }
    fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        self.inner.read_arena(addr, len)
    }
    fn nr_cpu_ids(&self) -> u32 {
        self.inner.nr_cpu_ids()
    }
    // cast_lookup intentionally NOT delegated: returns the trait
    // default `None`. See struct doc for why suppression is correct
    // when the chase has crossed BTFs.
    //
    // resolve_arena_type intentionally NOT delegated either:
    // [`ArenaResolveHit::target_type_id`] is keyed against the entry
    // BTF, so a cross-BTF recursion that consulted it would map a
    // sibling-BTF chase onto an entry-BTF id and silently wrong-render
    // (or silently skip with a misleading reason). False negative is
    // the safe direction — the sibling chase can still surface its
    // pointee via [`MemReader::cross_btf_resolve_fwd`] and the
    // ordinary [`Type::Ptr`] arm.
    fn cross_btf_resolve_fwd(&self, name: &str, kind: FwdKind) -> Option<CrossBtfRef<'_>> {
        self.inner.cross_btf_resolve_fwd(name, kind)
    }
}

/// Pre-read state assembled by [`resolve_chase_target`].
///
/// Both chase arms — [`chase_arena_pointer`] and
/// [`render_cast_pointer`]'s kernel arm — share the entire
/// "resolve target type, fire sdt_alloc bridge, fall back to
/// cross-BTF Fwd resolve, settle final size" sequence. That
/// sequence ends right before the per-arm read (`read_arena` vs
/// `read_kva`) — at which point the renderer holds enough
/// state to size the read, slice the header, and recurse. This
/// struct is exactly that "ready to read" snapshot.
///
/// `cross_btf_hit` is preserved by value (the type derives
/// [`Copy`], so threading it through the resolver does not
/// surface borrow lifetime headaches) so the per-arm post-read
/// step can build its own [`CrossBtfMemReader`] wrap on the
/// recursion path.
struct ResolvedTarget<'a> {
    /// Type id of the chase target within [`Self::current_btf`]'s
    /// id space. Threaded into [`render_value_inner`] for the
    /// recursion.
    effective_type_id: u32,
    /// BTF the recursion runs against — the entry BTF when
    /// cross-BTF stayed dormant, the resolved sibling BTF when
    /// the cross-BTF Fwd index returned a hit.
    current_btf: &'a Btf,
    /// Storage size of the resolved payload. The per-arm read
    /// budget is `header_skip + btf_size` clamped at the arm's
    /// cap (4 KiB for arena, 4 KiB AND page-remaining for
    /// kernel).
    btf_size: usize,
    /// Bytes the chase must skip past the chased address before
    /// the payload struct begins. `0` for a payload-start chase;
    /// the slot's header size for a slot-start chase the
    /// sdt_alloc bridge resolved.
    header_skip: usize,
    /// `true` when the resolved payload type id came from the
    /// sdt_alloc bridge ([`MemReader::resolve_arena_type`])
    /// rather than the cast analyzer's declared
    /// `target_type_id`. Surfaces through
    /// [`RenderedValue::Ptr::cast_annotation`] so operators see
    /// the layout came from the bridge.
    sdt_alloc_resolved: bool,
    /// Set when the cross-BTF Fwd index returned a hit (the
    /// chase target's body lives in a sibling BTF). The
    /// post-read recursion wraps `mem` in a
    /// [`CrossBtfMemReader`] in this case so id-keyed lookups
    /// (cast / arena type) cannot fire against the sibling
    /// BTF's id space — see [`CrossBtfMemReader`]'s doc for the
    /// collision rationale.
    cross_btf_hit: Option<CrossBtfRef<'a>>,
}

/// Outcome of [`resolve_chase_target`].
enum ChaseResolve<'a> {
    /// Pre-read sequence completed; chase is ready to read and
    /// recurse. The caller plugs [`Self::Ready`]'s state into its
    /// per-arm reader and post-read gates.
    Ready(ResolvedTarget<'a>),
    /// Pre-read sequence skipped the chase. The reason and
    /// `sdt_alloc_resolved` flag flow into the caller's
    /// per-arm `Ptr` builder so the no-deref render still
    /// surfaces the bridge state when it fired before the skip.
    Skip {
        reason: String,
        sdt_alloc_resolved: bool,
    },
}

/// Shared pre-read resolver for arena and kernel chase arms.
///
/// Encapsulates the steps both [`chase_arena_pointer`] and
/// [`render_cast_pointer`]'s kernel arm execute identically:
///
/// 1. Peel modifiers + resolve [`Type::Fwd`] to a complete
///    same-name sibling within `btf` via
///    [`peel_modifiers_resolving_fwd`].
/// 2. Try the sdt_alloc bridge ([`try_sdt_alloc_bridge`]) when
///    the post-peel terminal is still [`Type::Fwd`]. A bridge
///    fire returns the bridge's resolved `target_ty` /
///    `effective_type_id` plus the slot's `header_skip` and the
///    payload's `btf_size`; this resolver rebinds the local
///    target_ty / effective_type_id from the returned struct.
/// 3. When the bridge stays dormant, try the cross-BTF Fwd
///    index ([`try_cross_btf_fwd_resolve`]). A hit switches the
///    rendering BTF to the resolved sibling and adopts its
///    type id.
/// 4. Resolve the final `btf_size` against `current_btf` (or
///    reuse the bridge's `btf_size` when the bridge fired).
/// 5. Reject `btf_size == 0` payloads (incomplete types whose
///    BTF size resolves but is zero).
///
/// `kind_label` ("arena chase" / "kernel cast") flows into every
/// skip reason so the caller-visible messages still distinguish
/// the two arms; the renderer's test module asserts the prefix on
/// each path.
///
/// On the success path, the [`ChaseResolve::Ready`] payload carries
/// every value the per-arm read+recurse needs:
/// [`ResolvedTarget::effective_type_id`],
/// [`ResolvedTarget::current_btf`], [`ResolvedTarget::btf_size`],
/// [`ResolvedTarget::header_skip`],
/// [`ResolvedTarget::sdt_alloc_resolved`], and
/// [`ResolvedTarget::cross_btf_hit`]. The bridge state is captured
/// `Some(...)` exactly when the bridge fired; the `cross_btf_hit`
/// is captured exactly when the cross-BTF index returned a hit
/// (mutually exclusive with bridge fire).
fn resolve_chase_target<'a>(
    btf: &'a Btf,
    mem: &'a dyn MemReader,
    val: u64,
    target_type_id: u32,
    kind_label: &'static str,
) -> ChaseResolve<'a> {
    // Step 1: peel modifiers AND resolve a Fwd terminal to a
    // complete Struct/Union sibling of the same name when one
    // is in the BTF. Without the Fwd shortcut, `type_size`
    // returns `None` for a `Type::Fwd` terminal and the chase
    // would skip even when the renderer has full layout info
    // one BTF id away. `effective_type_id` is what the render
    // recursion uses to resolve members — the resolved
    // Struct/Union id when a sibling was found, otherwise the
    // post-peel original id.
    let Some((mut target_ty, mut effective_type_id)) =
        peel_modifiers_resolving_fwd(btf, target_type_id)
    else {
        return ChaseResolve::Skip {
            reason: format!("{kind_label} target type id {target_type_id} unresolvable"),
            sdt_alloc_resolved: false,
        };
    };
    // Step 2: when the post-peel terminal is still `Type::Fwd`,
    // ask the sdt_alloc bridge whether the chased address falls
    // inside an allocator slot whose payload type id the
    // pre-pass resolved. The bridge is pure-return: a fire
    // produces a fresh target_ty / effective_type_id which we
    // rebind here, plus the slot's `header_skip` and the
    // resolved payload's `btf_size`.
    let bridge = try_sdt_alloc_bridge(mem, val, &target_ty);
    let (sdt_alloc_resolved, header_skip) = match &bridge {
        Some(hit) => {
            if let Some((resolved_ty, resolved_id)) =
                peel_modifiers_resolving_fwd(btf, hit.target_type_id)
            {
                target_ty = resolved_ty;
                effective_type_id = resolved_id;
            }
            (true, hit.header_skip)
        }
        None => (false, 0usize),
    };
    // Step 3: when the bridge stayed dormant, try the cross-BTF
    // Fwd index. A `Type::Fwd` terminal whose complete body
    // lives in a sibling BTF resolves through the renderer's
    // cross-BTF index — the typical multi-`.bpf.objs` shape
    // (one object declares `struct cgx_target;` forward,
    // another defines the body). When the index returns a hit
    // we switch the rendering BTF to the resolved sibling and
    // adopt its type id. The bridge-fired path skips this
    // probe — its resolved id is in the entry BTF and the
    // recursion doesn't need to switch.
    let cross_btf_hit = if matches!(target_ty, Type::Fwd(_)) {
        try_cross_btf_fwd_resolve(mem, btf, &target_ty)
    } else {
        None
    };
    let current_btf: &Btf = match cross_btf_hit {
        Some(hit) => {
            target_ty = match hit.btf.resolve_type_by_id(hit.type_id) {
                Ok(ty) => ty,
                Err(_) => {
                    return ChaseResolve::Skip {
                        reason: format!(
                            "{kind_label}: cross-BTF Fwd resolve returned \
                             type_id {} but the type does not resolve in \
                             the sibling BTF",
                            hit.type_id
                        ),
                        sdt_alloc_resolved,
                    };
                }
            };
            effective_type_id = hit.type_id;
            hit.btf
        }
        None => btf,
    };
    // Step 4: resolve the final `btf_size`. The bridge fire
    // already paid this resolve; reuse its value when present.
    let btf_size = {
        let Some(sz) = type_size(current_btf, &target_ty) else {
            return ChaseResolve::Skip {
                reason: unsizable_chase_reason(current_btf, kind_label, target_type_id, &target_ty),
                sdt_alloc_resolved,
            };
        };
        sz
    };
    // Step 5: reject zero-size payloads (incomplete types whose
    // BTF size resolves but is zero). The `incomplete type`
    // substring is what the kernel-arm
    // `cast_chase_kernel_target_btf_size_zero` test asserts.
    if btf_size == 0 {
        return ChaseResolve::Skip {
            reason: format!(
                "{kind_label} target type id {target_type_id} BTF size is 0 (incomplete type)"
            ),
            sdt_alloc_resolved,
        };
    }
    ChaseResolve::Ready(ResolvedTarget {
        effective_type_id,
        current_btf,
        btf_size,
        header_skip,
        sdt_alloc_resolved,
        cross_btf_hit,
    })
}

/// Run [`render_value_inner`] against the resolved target and
/// wrap the result for the chase arm.
///
/// The post-read recursion is also identical between
/// [`chase_arena_pointer`] and [`render_cast_pointer`]: both
/// insert the chased value into `visited` before recursing,
/// optionally wrap `mem` in a [`CrossBtfMemReader`] when the
/// chase crossed a BTF boundary, recurse against
/// `effective_type_id` in `current_btf`, remove the value from
/// `visited` after, and box-wrap the rendered value in a
/// [`RenderedValue::Truncated`] when the read was clipped.
///
/// `truncated_at_cap` is the per-arm "the read was clipped"
/// flag — arena's path uses `total_needed > POINTER_CHASE_CAP`
/// (the snapshot is single-page-bound), kernel's path uses
/// `total_needed > read_size` (read_size is bound by both the
/// global cap AND the page-remaining length).
fn recurse_into_target(
    resolved: &ResolvedTarget<'_>,
    target_bytes: &[u8],
    val: u64,
    depth: u32,
    mem: &dyn MemReader,
    visited: &mut HashSet<u64>,
    truncated_at_cap: bool,
) -> Box<RenderedValue> {
    visited.insert(val);
    let cross_btf_wrap = resolved
        .cross_btf_hit
        .as_ref()
        .map(|_| CrossBtfMemReader { inner: mem });
    let recurse_mem: &dyn MemReader = match &cross_btf_wrap {
        Some(w) => w,
        None => mem,
    };
    let inner = render_value_inner(
        resolved.current_btf,
        resolved.effective_type_id,
        target_bytes,
        depth + 1,
        Some(recurse_mem),
        visited,
    );
    visited.remove(&val);
    if truncated_at_cap {
        // Partial render: only the first capped bytes of a
        // larger struct were read. Wrap so the consumer can
        // tell the rendered tree is incomplete even though it
        // looks structurally sound.
        Box::new(RenderedValue::Truncated {
            needed: resolved.btf_size,
            had: target_bytes.len(),
            partial: Box::new(inner),
        })
    } else {
        Box::new(inner)
    }
}

/// Chase an arena pointer and render the target struct.
///
/// See [`ArenaChaseOutcome`] for the return shape. The caller plugs
/// `deref` and `reason` into [`RenderedValue::Ptr`]; when
/// `sdt_alloc_resolved` is `true` the caller adds an
/// `sdt_alloc`-flavoured `cast_annotation` so the recovered chase is
/// distinguishable from a native BTF chase.
///
/// Preconditions the caller must satisfy:
///   * The [`chase_gate`] outcome was [`ChaseGate::Proceed`] for
///     `val` at this `depth` (i.e. `val != 0`, not in `visited`,
///     `depth < MAX_RENDER_DEPTH`).
///   * `mem.is_arena_addr(val)` returned `true`. The cast path
///     wraps a separate "arena cast value outside arena window"
///     reason around an out-of-window value before invoking the
///     helper; the [`Type::Ptr`] arm dispatches on
///     `is_arena_addr` so it only enters this helper when the
///     check passed.
///
/// `visited` bookkeeping is internal: the helper inserts `val`
/// before recursing and removes it after, matching the path-based
/// cycle convention used in both call sites.
fn chase_arena_pointer(
    btf: &Btf,
    target_type_id: u32,
    val: u64,
    mem: &dyn MemReader,
    depth: u32,
    visited: &mut HashSet<u64>,
) -> ArenaChaseOutcome {
    // Special case: `target_type_id == 0` means the cast analyzer's
    // STX-flow path tagged the slot as Arena WITHOUT a resolved
    // BTF type id (the deferred-resolve arena cast path —
    // allocator-return seeds produce findings whose target shape
    // is determined entirely by the
    // [`MemReader::resolve_arena_type`] bridge at chase time).
    // Skip the shared resolver and consult the bridge directly. If
    // the bridge returns a hit, render against the recovered
    // payload type id; otherwise skip with a clear reason so the
    // operator sees why the chase did not land. The deferred path
    // bypasses the cross-BTF probe — the bridge's resolved id is
    // in the entry BTF — by synthesising a `ResolvedTarget` with
    // `cross_btf_hit = None`, which feeds directly into the
    // post-read recursion.
    let resolved = if target_type_id == 0 {
        let Some(hit) = mem.resolve_arena_type(val) else {
            return ArenaChaseOutcome {
                deref: None,
                reason: Some(format!(
                    "arena chase: cast analyzer's STX-flow path tagged \
                     slot as Arena (target_type_id=0, deferred resolve), \
                     but [`MemReader::resolve_arena_type`] had no entry \
                     for 0x{val:x}; allocator pre-pass may not have \
                     populated the index for this allocator"
                )),
                sdt_alloc_resolved: false,
            };
        };
        let Some((resolved_ty, resolved_id)) =
            peel_modifiers_resolving_fwd(btf, hit.target_type_id)
        else {
            return ArenaChaseOutcome {
                deref: None,
                reason: Some(format!(
                    "arena chase: bridge returned target_type_id={} \
                     but the type does not resolve in the program BTF",
                    hit.target_type_id
                )),
                sdt_alloc_resolved: true,
            };
        };
        let Some(btf_size) = type_size(btf, &resolved_ty) else {
            return ArenaChaseOutcome {
                deref: None,
                reason: Some(unsizable_chase_reason(
                    btf,
                    "arena chase",
                    hit.target_type_id,
                    &resolved_ty,
                )),
                sdt_alloc_resolved: true,
            };
        };
        if btf_size == 0 {
            return ArenaChaseOutcome {
                deref: None,
                reason: Some(format!(
                    "arena chase target type id {} BTF size is 0 (incomplete type)",
                    hit.target_type_id
                )),
                sdt_alloc_resolved: true,
            };
        }
        ResolvedTarget {
            effective_type_id: resolved_id,
            current_btf: btf,
            btf_size,
            header_skip: hit.header_skip,
            sdt_alloc_resolved: true,
            cross_btf_hit: None,
        }
    } else {
        match resolve_chase_target(btf, mem, val, target_type_id, "arena chase") {
            ChaseResolve::Ready(r) => r,
            ChaseResolve::Skip {
                reason,
                sdt_alloc_resolved,
            } => {
                return ArenaChaseOutcome {
                    deref: None,
                    reason: Some(reason),
                    sdt_alloc_resolved,
                };
            }
        }
    };
    // The single-page (4 KiB) cap ([`POINTER_CHASE_CAP`]) matches
    // the arena page granularity exposed by
    // [`MemReader::read_arena`]: a pointee larger than 4 KiB
    // renders only its first page. Cross-page chase would require
    // splitting the read into per-page chunks AND stitching them —
    // a future enhancement once a scheduler ships an
    // arena-allocated payload larger than 4 KiB. Today the
    // truncation surfaces explicitly via the
    // [`RenderedValue::Truncated`] wrapper below when btf_size
    // exceeds the cap, so the operator sees the rendered subtree
    // is partial.
    //
    // When the sdt_alloc bridge fired with `header_skip > 0`, the
    // total bytes the chase needs from `val` are
    // `header_skip + btf_size`. The cap still applies to the
    // requested span — slot-start chases of payloads close to the
    // cap may surface as `Truncated` because the header eats into
    // the page-bounded read budget.
    let total_needed = resolved.header_skip.saturating_add(resolved.btf_size);
    let read_size = total_needed.min(POINTER_CHASE_CAP);
    let truncated_at_cap = total_needed > POINTER_CHASE_CAP;
    let Some(raw_bytes) = mem.read_arena(val, read_size) else {
        // [`MemReader::read_arena`] returns `None` when the full
        // requested length cannot be satisfied — most commonly
        // because the read crosses a page boundary in the captured
        // arena snapshot. Annotate so the consumer sees why the
        // deref didn't land.
        return ArenaChaseOutcome {
            deref: None,
            reason: Some(format!(
                "arena read failed (cross-page boundary or unmapped \
                 page); needed {read_size} bytes from 0x{val:x}"
            )),
            sdt_alloc_resolved: resolved.sdt_alloc_resolved,
        };
    };
    // Slot-start bridge fire: skip the header before rendering the
    // payload struct. `read_arena` is contractually
    // single-page-bound so the slice never under-runs unless the
    // page-tail cropped the read; in that pathological case
    // surface a clear skip reason rather than rendering a
    // partial-header-stripped buffer.
    let Some(target_bytes) = apply_header_skip(&raw_bytes, resolved.header_skip) else {
        return ArenaChaseOutcome {
            deref: None,
            reason: Some(format!(
                "arena read at 0x{val:x} returned {} bytes; \
                 sdt_alloc bridge needs at least {} for header skip",
                raw_bytes.len(),
                resolved.header_skip
            )),
            sdt_alloc_resolved: resolved.sdt_alloc_resolved,
        };
    };
    let payload = recurse_into_target(
        &resolved,
        target_bytes,
        val,
        depth,
        mem,
        visited,
        truncated_at_cap,
    );
    ArenaChaseOutcome {
        deref: Some(payload),
        reason: None,
        sdt_alloc_resolved: resolved.sdt_alloc_resolved,
    }
}

/// Try the cross-BTF Fwd resolution bridge for a `BTF_KIND_FWD`
/// chase target. Returns `Some(CrossBtfRef { btf, type_id })` when
/// the [`MemReader::cross_btf_resolve_fwd`] override returns a
/// hit — the renderer's cast-analysis pre-pass populated a name-
/// keyed index over every embedded `.bpf.objs` BTF and the named
/// struct/union has a complete body in a sibling object.
///
/// `entry_btf` is the BTF the chase entered with — used to
/// translate the Fwd's name offset to a string the implementation
/// can key its index against. `target_ty` must be the post-peel
/// terminal of [`peel_modifiers_resolving_fwd`]; the helper
/// itself gates on `matches!(target_ty, Type::Fwd(_))` so calling
/// on a non-Fwd terminal is a no-op and returns `None`. Anonymous
/// Fwds (empty resolved name) likewise return `None` — the index
/// keys on non-empty names.
///
/// The aggregate-kind match ([`btf_rs::Fwd::is_struct`]) is
/// preserved end-to-end: a `Fwd` declared as `struct foo` only
/// resolves to a `struct foo` body in another BTF, never to a
/// `union foo`. Mirrors the same gate
/// [`peel_modifiers_resolving_fwd`] applies for in-BTF resolution.
fn try_cross_btf_fwd_resolve<'a>(
    mem: &'a dyn MemReader,
    entry_btf: &Btf,
    target_ty: &Type,
) -> Option<CrossBtfRef<'a>> {
    let Type::Fwd(fwd) = target_ty else {
        return None;
    };
    let kind = FwdKind::from_is_struct(fwd.is_struct());
    let name = entry_btf.resolve_name(fwd).ok()?;
    if name.is_empty() {
        return None;
    }
    mem.cross_btf_resolve_fwd(&name, kind)
}

/// Build a [`RenderedValue::Ptr`] for a cast-recovered pointer with
/// uniform field assembly.
///
/// Every site in [`render_cast_pointer`] that emits a `Ptr` shares
/// the same four-field shape (`value`, optional `deref`, optional
/// `deref_skipped_reason`, optional `cast_annotation`). The helper
/// resolves the canonical annotation through [`cast_annotation_for`]
/// — a 4-cell match over
/// [`super::cast_analysis::AddrSpace`] × `sdt_alloc_resolved`
/// returning a `&'static str` — so every annotation the renderer
/// emits is a borrow into the binary's read-only string pool, not a
/// per-chase heap allocation. Adding a new address-space variant
/// surfaces as a non-exhaustive match at compile time, keeping the
/// analyzer enum and the operator-visible tag in lockstep.
/// `addr_space` here is the address space the renderer ACTUALLY
/// chased through (runtime decision), not the analyzer's hint, so
/// the annotation reflects the path the chase took.
///
/// `sdt_alloc_resolved` extends the annotation to
/// `cast→{addr_space} (sdt_alloc)` when the chase recovered the
/// target's BTF type id via [`MemReader::resolve_arena_type`]
/// instead of the cast analyzer's declared `target_type_id`.
/// Operators see at a glance that the rendered subtree's layout
/// came from the sdt_alloc bridge rather than the analyzer's own
/// flow-tracked type recovery.
fn cast_ptr(
    value: u64,
    deref: Option<Box<RenderedValue>>,
    reason: Option<String>,
    addr_space: super::cast_analysis::AddrSpace,
    sdt_alloc_resolved: bool,
) -> RenderedValue {
    RenderedValue::Ptr {
        value,
        deref,
        deref_skipped_reason: reason,
        cast_annotation: Some(Cow::Borrowed(cast_annotation_for(
            addr_space,
            sdt_alloc_resolved,
        ))),
    }
}

/// Resolve the canonical cast annotation tag to a `&'static str`.
///
/// The `(addr_space, sdt_alloc_resolved)` pair maps to one of four
/// fixed strings — exhaustively enumerated below so
/// [`super::cast_analysis::AddrSpace`]'s closed variant set drives
/// an exhaustive match. A new variant produces a compile error
/// here, forcing the operator-visible tag to stay in lockstep with
/// the analyzer enum.
///
/// The pre-existing [`super::cast_analysis::AddrSpace`]
/// [`std::fmt::Display`] impl is kept for other call sites
/// (free-form formatting, error messages); the renderer side
/// bypasses `Display` because the closed set lets us hand back
/// static strings instead of allocating a new `String` per
/// chase.
fn cast_annotation_for(
    addr_space: super::cast_analysis::AddrSpace,
    sdt_alloc_resolved: bool,
) -> &'static str {
    use super::cast_analysis::AddrSpace;
    match (addr_space, sdt_alloc_resolved) {
        (AddrSpace::Arena, false) => "cast→arena",
        (AddrSpace::Arena, true) => "cast→arena (sdt_alloc)",
        (AddrSpace::Kernel, false) => "cast→kernel",
        (AddrSpace::Kernel, true) => "cast→kernel (sdt_alloc)",
    }
}

/// Render a cast-recovered typed pointer.
///
/// Builds [`RenderedValue::Ptr`] mirroring the [`Type::Ptr`] arm's
/// shape so consumers (Display, JSON serializer, `is_flat_scalar`
/// classifier) handle cast-recovered pointers and BTF-typed pointers
/// uniformly. Pre-chase gating (null, cycle, depth cap) goes through
/// the shared [`chase_gate`] helper, so a linked-list /
/// parent-pointer cycle in cast-recovered pointers surfaces the
/// same `[cycle]` glyph as the [`Type::Ptr`] arm and a null cast
/// value renders identically.
///
/// Address-space dispatch is RUNTIME-driven: [`MemReader::is_arena_addr`]
/// is consulted on the actual pointer value to decide whether to
/// chase via [`MemReader::read_arena`] (in-window) or
/// [`MemReader::read_kva`] (out-of-window). The
/// [`super::cast_analysis::CastHit::addr_space`] tag from the
/// analyzer is treated as a hint only — runtime evidence from the
/// pointer value is authoritative because the analyzer's
/// flow-insensitive register tracking can mis-classify across
/// branch joins, while `is_arena_addr` is a structural property of
/// the live address space. When both readers might succeed the
/// arena reader wins (its frozen snapshot is more reliable than a
/// live walk through `read_kva`).
///
/// `cast_annotation` on the resulting [`RenderedValue::Ptr`]
/// records which path actually executed (`"cast→arena"` or
/// `"cast→kernel"`) so operators can distinguish cast-recovered
/// pointers from BTF-typed pointers without inspecting the
/// rendered subtree. When [`try_sdt_alloc_bridge`] fired during
/// the chase (the analyzer's `target_type_id` resolved to a
/// `BTF_KIND_FWD` whose real id was recovered via
/// [`MemReader::resolve_arena_type`]), [`cast_ptr`] extends the
/// annotation with a trailing ` (sdt_alloc)` — `"cast→arena
/// (sdt_alloc)"` / `"cast→kernel (sdt_alloc)"` — so operators
/// see the layout came from the bridge rather than the cast
/// analyzer's declared id. The [`Type::Ptr`] arm normally leaves
/// the field `None`, with one exception: when its arena branch
/// also fires the sdt_alloc bridge it sets `cast_annotation` to
/// the unprefixed `"sdt_alloc"` (no `cast→` prefix because the
/// chased pointer is structurally BTF-typed, not analyzer-
/// recovered).
///
/// On read failure (cross-page boundary in the arena snapshot,
/// unmapped page, etc.) the render emits `Ptr` with
/// `deref_skipped_reason` populated and `deref: None` — the chase
/// was attempted, distinguishing it from the no-chase paths above.
fn render_cast_pointer(
    btf: &Btf,
    hit: CastHit,
    value: u64,
    depth: u32,
    mem: &dyn MemReader,
    visited: &mut HashSet<u64>,
) -> RenderedValue {
    // [`chase_gate`] applies the null/cycle/depth-cap policy
    // shared with the [`Type::Ptr`] arm: null and depth-cap take
    // the "no chase attempted" path (`deref` + reason both
    // `None`); a cycle records the `cycle → 0x{value:x}` reason.
    // Only [`ChaseGate::Proceed`] enters the per-arm read+recurse
    // logic below. The cast_annotation on the no-chase path
    // reflects the analyzer's hint so operators still see the
    // pointer was a cast finding rather than a BTF-typed pointer.
    if let ChaseGate::Skip { reason } = chase_gate(value, depth, visited) {
        return cast_ptr(value, None, reason, hit.addr_space, false);
    }
    // Runtime address-space detection: if the value falls in the
    // arena window, chase through `read_arena` (frozen-snapshot
    // backed, no slab-liveness concern). Otherwise chase through
    // `read_kva` with the existing plausibility gate. The
    // analyzer's hint is preserved in `cast_annotation` so a
    // hint→runtime mismatch is visible; the renderer doesn't try
    // to reconcile them — the runtime decision wins.
    if mem.is_arena_addr(value) {
        let outcome = chase_arena_pointer(btf, hit.target_type_id, value, mem, depth, visited);
        return cast_ptr(
            value,
            outcome.deref,
            outcome.reason,
            super::cast_analysis::AddrSpace::Arena,
            outcome.sdt_alloc_resolved,
        );
    }
    // Kernel-arm chase: out-of-arena value, read through the
    // page-table walker. Resolve target type id to a Type so
    // `type_size` can size the read. Failure here is rare in
    // practice — the cast analyzer only emits ids it itself
    // resolved through the same BTF — but the fallthrough emits a
    // labelled skip rather than panicking. Use the Fwd-resolving
    // peel so a [`Type::Fwd`] target with a complete sibling in
    // the BTF lands on the sibling rather than skipping. The
    // arena arm above shares the same shortcut via
    // [`chase_arena_pointer`].
    //
    // Special case: `target_type_id == 0` is the cast analyzer's
    // STX-flow Arena sentinel (the deferred-resolve arena cast
    // path emits this when the target is unresolved at analysis
    // time). The hint was Arena but the runtime value fell
    // outside the arena window, so we landed here. Without a BTF
    // id to resolve, the kernel arm cannot size the read; surface
    // a clear skip reason so the operator sees the analyzer/runtime
    // mismatch.
    if hit.target_type_id == 0 {
        return cast_ptr(
            value,
            None,
            Some(format!(
                "kernel cast target unresolved (analyzer hinted Arena \
                 with deferred resolve, but runtime value 0x{value:x} \
                 fell outside the arena window)"
            )),
            super::cast_analysis::AddrSpace::Kernel,
            false,
        );
    }
    // Pre-read sequence shared with [`chase_arena_pointer`]: peel +
    // Fwd-resolve target, try sdt_alloc bridge, fall back to
    // cross-BTF Fwd index, settle final btf_size, reject zero-size
    // payloads. Returns a [`ResolvedTarget`] ready for the per-arm
    // `read_kva`, or a skip reason that flows into the `cast_ptr`
    // builder so the no-deref render still surfaces the bridge
    // state when it fired before the skip.
    let resolved = match resolve_chase_target(btf, mem, value, hit.target_type_id, "kernel cast") {
        ChaseResolve::Ready(r) => r,
        ChaseResolve::Skip {
            reason,
            sdt_alloc_resolved,
        } => {
            return cast_ptr(
                value,
                None,
                Some(reason),
                super::cast_analysis::AddrSpace::Kernel,
                sdt_alloc_resolved,
            );
        }
    };
    // Kernel reads honour [`POINTER_CHASE_CAP`] and also cap at
    // the remaining bytes in the current 4 KiB page so `read_kva`
    // cannot accidentally cross a page boundary into an unrelated
    // allocation. The kernel direct-map / vmalloc walker returns
    // whatever the page tables resolve, but a struct that
    // straddles a page boundary may have its tail in a freed or
    // unrelated page — bounding the read at the page edge keeps
    // the dump from leaking adjacent slab content.
    //
    // When the sdt_alloc bridge fired with `header_skip > 0`,
    // the total bytes the chase needs from `value` are
    // `header_skip + btf_size` — the header bytes are skipped
    // before the payload starts. The page-bounded cap still
    // applies to the requested span.
    const PAGE_SIZE: u64 = 4096;
    // PAGE_SIZE - (value % PAGE_SIZE) bytes remain in the current
    // 4 KiB page from `value` onward. usize fits the result
    // because PAGE_SIZE is 4096 (well below usize::MAX) and the
    // modulo result is in [0, PAGE_SIZE).
    let page_remaining = (PAGE_SIZE - (value % PAGE_SIZE)) as usize;
    let total_needed = resolved.header_skip.saturating_add(resolved.btf_size);
    let read_size = total_needed.min(POINTER_CHASE_CAP).min(page_remaining);
    let truncated_at_cap = total_needed > read_size;
    let Some(raw_bytes) = mem.read_kva(value, read_size) else {
        // The runtime check rejected the arena window AND the
        // kernel read failed — surface both pieces of evidence so
        // the operator can correlate the analyzer hint with the
        // actual chase outcome. If the analyzer hinted Arena but
        // the value was outside the arena window, this is a
        // structural mismatch: the analyzer may have flagged a
        // non-pointer field.
        let suffix = if matches!(hit.addr_space, super::cast_analysis::AddrSpace::Arena) {
            " (cast analysis may have flagged a non-pointer field)"
        } else {
            ""
        };
        return cast_ptr(
            value,
            None,
            Some(format!(
                "kernel read_kva failed at 0x{value:x} \
                 (unmapped page or no PTE); needed {read_size} bytes{suffix}"
            )),
            super::cast_analysis::AddrSpace::Kernel,
            resolved.sdt_alloc_resolved,
        );
    };
    // Slot-start bridge fire on the kernel arm: skip the header
    // before the plausibility gate and the recursion. Refuse to
    // proceed when the read returned fewer bytes than the header
    // skip needs — surfaces the page-tail truncation as a clear
    // skip reason rather than rendering a corrupted slice.
    let Some(target_bytes) = apply_header_skip(&raw_bytes, resolved.header_skip) else {
        return cast_ptr(
            value,
            None,
            Some(format!(
                "kernel read_kva at 0x{value:x} returned {} bytes; \
                 sdt_alloc bridge needs at least {} for header skip",
                raw_bytes.len(),
                resolved.header_skip
            )),
            super::cast_analysis::AddrSpace::Kernel,
            resolved.sdt_alloc_resolved,
        );
    };
    // Plausibility gate: a freed slab object's first qword is
    // often a freelist next pointer, which on x86_64 / aarch64
    // typically lands in the kernel direct-map range
    // (0xffff800000000000+, top byte 0xff). Reject reads where
    // the first 8 bytes look like that pattern as a probable
    // stale-pointer signature. Same heuristic as the cpumask
    // kptr chase in the `Type::Ptr` arm. Arena reads are exempt
    // (the arena helper above does not apply this gate) — arena
    // pages are caller-controlled allocations whose first bytes
    // are not used for slab freelist metadata. The gate runs
    // against the post-skip payload bytes when the bridge fired.
    if target_bytes.len() >= 8 {
        let first_qword = u64::from_le_bytes(target_bytes[..8].try_into().unwrap());
        if first_qword >> 56 == 0xff {
            return cast_ptr(
                value,
                None,
                Some(format!(
                    "kernel cast plausibility gate rejected: first qword \
                     top byte is 0xff at 0x{value:x} (likely freed slab \
                     object freelist pointer)"
                )),
                super::cast_analysis::AddrSpace::Kernel,
                resolved.sdt_alloc_resolved,
            );
        }
    }
    let deref_payload = recurse_into_target(
        &resolved,
        target_bytes,
        value,
        depth,
        mem,
        visited,
        truncated_at_cap,
    );
    cast_ptr(
        value,
        Some(deref_payload),
        None,
        super::cast_analysis::AddrSpace::Kernel,
        resolved.sdt_alloc_resolved,
    )
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
pub(crate) fn peel_modifiers(btf: &Btf, type_id: u32) -> Option<Type> {
    peel_modifiers_with_id(btf, type_id).map(|(ty, _)| ty)
}

/// Peel modifiers like [`peel_modifiers_with_id`], then if the
/// terminal is a [`Type::Fwd`] resolve it to a complete
/// [`Type::Struct`] / [`Type::Union`] of the same name in the same
/// BTF when one exists.
///
/// `BTF_KIND_FWD` is a forward declaration (`struct foo;` with no
/// body) clang emits when a type is referenced only via pointer in
/// the compilation unit. Concatenated BPF objects routinely include
/// both a `Fwd` and a complete `Struct`/`Union` with the same name —
/// the `Fwd` from a header that only declares the type, the
/// complete shape from a `.bpf.c` that defines it. The chase
/// pipeline calls [`type_size`] right after peeling; `type_size`
/// returns `None` for `Type::Fwd`, which produces "unresolvable
/// size" skips even when the BTF carries a fully-typed sibling
/// the renderer could land against. This helper elides that gap by
/// preferring the complete sibling whenever one is present.
///
/// Returns the original peeled (Type, id) pair when:
/// - the terminal is not a [`Type::Fwd`] (no resolution needed),
/// - the [`Type::Fwd`] has no name (anonymous fwds cannot be
///   keyed for lookup),
/// - no sibling [`Type::Struct`]/[`Type::Union`] of the same name
///   AND matching aggregate kind (struct vs union, per
///   [`btf_rs::Fwd::is_struct`] / [`is_union`]) exists in the BTF.
///
/// The aggregate-kind match is crucial — a `Fwd` declared as
/// `struct foo` must NOT resolve to a `union foo` in the same BTF
/// (the wire format permits same-name struct + union declarations,
/// rare but legal). The renderer would render the wrong layout if
/// we collapsed the two.
///
/// Single-pass resolution: the helper calls
/// [`peel_modifiers_with_id`] once, inspects the terminal, and
/// either returns the original peeled pair or returns the first
/// matching sibling found in the by-name candidate list. There is
/// no re-entry into [`peel_modifiers_resolving_fwd`] from within
/// itself; the bounded modifier-peel cap inside
/// [`peel_modifiers_with_id`] is what protects against malformed
/// `Fwd -> Typedef -> Fwd` chains.
pub(crate) fn peel_modifiers_resolving_fwd(btf: &Btf, type_id: u32) -> Option<(Type, u32)> {
    let (ty, tid) = peel_modifiers_with_id(btf, type_id)?;
    let Type::Fwd(ref fwd) = ty else {
        return Some((ty, tid));
    };
    let kind = FwdKind::from_is_struct(fwd.is_struct());
    let Ok(name) = btf.resolve_name(fwd) else {
        return Some((ty, tid));
    };
    if name.is_empty() {
        return Some((ty, tid));
    }
    let Ok(candidate_ids) = btf.resolve_ids_by_name(&name) else {
        return Some((ty, tid));
    };
    for cid in candidate_ids {
        if cid == tid {
            // The Fwd itself shows up in the by-name list; skip it
            // so the loop searches only siblings.
            continue;
        }
        let Some((candidate_ty, candidate_id)) = peel_modifiers_with_id(btf, cid) else {
            continue;
        };
        match (&candidate_ty, kind) {
            (Type::Struct(_), FwdKind::Struct) => return Some((candidate_ty, candidate_id)),
            (Type::Union(_), FwdKind::Union) => return Some((candidate_ty, candidate_id)),
            _ => continue,
        }
    }
    Some((ty, tid))
}

/// Peel pass-through qualifiers starting from a [`Type`] value
/// rather than a BTF type id. Single shared helper for callers that
/// already hold a resolved [`Type`] and would otherwise re-implement
/// the peel loop. Returns the original `start` type unchanged when
/// it is already non-modifier; returns `None` only on `btf_rs`
/// resolve failure or the [`MAX_MODIFIER_DEPTH`] cap.
pub(crate) fn peel_modifiers_from_type(btf: &Btf, start: Type) -> Option<Type> {
    let mut t = start;
    for _ in 0..MAX_MODIFIER_DEPTH {
        // Each modifier kind binds a different `btf_rs` type
        // (`Volatile`, `Const`, `Restrict`, `Typedef`, `TypeTag`,
        // `DeclTag`), so or-patterns that share an `inner` binding
        // would force the binding to a single type. Use separate
        // arms — `resolve_chained_type` is generic over `BtfType`
        // so each arm reduces to one line.
        let next = match &t {
            Type::Volatile(inner) => btf.resolve_chained_type(inner).ok()?,
            Type::Const(inner) => btf.resolve_chained_type(inner).ok()?,
            Type::Restrict(inner) => btf.resolve_chained_type(inner).ok()?,
            Type::Typedef(inner) => btf.resolve_chained_type(inner).ok()?,
            Type::TypeTag(inner) => btf.resolve_chained_type(inner).ok()?,
            Type::DeclTag(inner) => btf.resolve_chained_type(inner).ok()?,
            _ => return Some(t),
        };
        t = next;
    }
    None
}

/// Same as [`peel_modifiers`] but also returns the BTF type id of the
/// peeled (terminal) type. The cast-intercept path keys
/// [`MemReader::cast_lookup`] on the *struct* type id of the parent
/// aggregate — not the modifier-wrapped surface id — so the
/// post-peel id is what the cast_analysis [`super::cast_analysis::CastMap`]
/// stores. Mirrors [`super::bpf_map::resolve_to_struct_id`]'s
/// modifier-peeling shape; the renderer uses this variant whenever
/// it needs both the resolved Type and its id.
pub(crate) fn peel_modifiers_with_id(btf: &Btf, mut type_id: u32) -> Option<(Type, u32)> {
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
            _ => return Some((ty, type_id)),
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
mod tests;
