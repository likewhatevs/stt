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
//! by their declared names, so a stall dump's `.bss` map shows
//! `stall=1, crash=0, ...` instead of an opaque hex dump.
//!
//! Bitfield handling: when [`btf_rs::Member::bitfield_size`] is `Some(w)`,
//! the renderer reads enough bytes to cover the bitfield's bit range,
//! shifts and masks, and applies sign extension if the underlying int
//! kind is signed.

use serde::{Deserialize, Serialize};

use btf_rs::{Btf, BtfType, Member, Struct, Type};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Raw pointer value (the renderer does not chase pointers — guest
    /// memory translation is the caller's job).
    Ptr { value: u64 },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        RenderedValue::Int { value, .. } => write!(f, "{value}"),
        RenderedValue::Uint { value, .. } => write!(f, "{value}"),
        RenderedValue::Bool { value } => write!(f, "{value}"),
        RenderedValue::Char { value } => {
            // Printable ASCII is shown as `'x'` (matches C char-literal
            // notation an operator would write in source); other bytes
            // fall back to `0xNN` so the raw value is unambiguous.
            if (0x20..=0x7e).contains(value) {
                write!(f, "'{}'", *value as char)
            } else {
                write!(f, "0x{value:02x}")
            }
        }
        RenderedValue::Float { value, .. } => write!(f, "{value}"),
        RenderedValue::Enum { value, variant, .. } => match variant {
            Some(name) => write!(f, "{name} ({value})"),
            None => write!(f, "{value}"),
        },
        RenderedValue::Ptr { value } => write!(f, "0x{value:x}"),
        RenderedValue::Bytes { hex } => write!(f, "{hex}"),
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
            // Inline if every element is scalar (no struct, no array,
            // no truncated-with-struct-partial). Block-style otherwise
            // so nested structs stay readable.
            let inline = elements.iter().all(is_inline_scalar);
            if inline {
                f.write_str("[")?;
                for (i, e) in elements.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write_rendered_value(f, e, depth)?;
                }
                f.write_str("]")?;
                if elements.len() < *len {
                    write!(f, " /* {} of {len} shown */", elements.len())?;
                }
                Ok(())
            } else {
                f.write_str("[")?;
                for e in elements {
                    f.write_str("\n")?;
                    write_indent(f, depth + 1)?;
                    write_rendered_value(f, e, depth + 1)?;
                }
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
            match type_name {
                Some(name) => write!(f, "struct {name} {{")?,
                None => f.write_str("struct {")?,
            }
            if members.is_empty() {
                return f.write_str("}");
            }
            for m in members {
                f.write_str("\n")?;
                write_indent(f, depth + 1)?;
                if m.name.is_empty() {
                    f.write_str("<anon>: ")?;
                } else {
                    write!(f, "{}: ", m.name)?;
                }
                write_rendered_value(f, &m.value, depth + 1)?;
            }
            f.write_str("\n")?;
            write_indent(f, depth)?;
            f.write_str("}")
        }
    }
}

/// Whether a value renders as a single line under Display. Used by
/// the array case to pick inline vs block layout.
fn is_inline_scalar(v: &RenderedValue) -> bool {
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
pub fn render_value(btf: &Btf, type_id: u32, bytes: &[u8]) -> RenderedValue {
    render_value_inner(btf, type_id, bytes, 0)
}

fn render_value_inner(btf: &Btf, type_id: u32, bytes: &[u8], depth: u32) -> RenderedValue {
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
        Type::Ptr(_) => {
            // Pointers are u64 on every architecture ktstr targets
            // (x86_64 + aarch64). When the supplied byte slice is
            // shorter than 8 — e.g. the renderer is asked to decode a
            // pointer-typed bitfield, or the value bytes were
            // truncated upstream — emit Truncated rather than panicking
            // on the slice index.
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
            RenderedValue::Ptr { value: val }
        }
        Type::Struct(s) | Type::Union(s) => render_struct(btf, &s, bytes, depth),
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
        Type::Datasec(ds) => render_datasec(btf, &ds, bytes, depth),
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
            render_value_inner(btf, inner_id, bytes, depth + 1)
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

/// Lowercase, space-separated hex dump of `bytes`. Mirror of the same
/// helper in `dump.rs` to avoid a cross-module dependency from the
/// renderer for a 5-line utility.
fn hex_dump(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "{b:02x}");
    }
    s
}

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

fn render_struct(btf: &Btf, s: &Struct, bytes: &[u8], depth: u32) -> RenderedValue {
    let type_name = btf.resolve_name(s).ok().filter(|n| !n.is_empty());
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
        let value = render_member(btf, m, bytes, depth);
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
/// Display layout intact, so a stall dump's `.bss` map renders
/// alongside ordinary structs and JSON consumers (the
/// `stall_dump_e2e.rs` fixture among them) iterate the variables via
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
fn render_datasec(btf: &Btf, ds: &btf_rs::Datasec, bytes: &[u8], depth: u32) -> RenderedValue {
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
                render_value_inner(btf, inner_id, &bytes[offset..end], depth + 1)
            }
            _ => {
                let avail_start = offset.min(bytes.len());
                let avail = &bytes[avail_start..];
                let partial = render_value_inner(btf, inner_id, avail, depth + 1);
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

fn render_member(btf: &Btf, m: &Member, parent_bytes: &[u8], depth: u32) -> RenderedValue {
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
        Some(end) if end <= parent_bytes.len() => {
            render_value_inner(btf, member_type_id, &parent_bytes[byte_off..end], depth + 1)
        }
        _ => {
            // Attempt a partial decode from whatever bytes ARE available
            // for this member: the inner renderer will itself emit a
            // Truncated/Bytes/etc. that carries the recoverable subset.
            // Wrapping that subset in this outer Truncated tells the
            // consumer "the full member needed N bytes, only M survived,
            // here's what we got".
            let avail_start = byte_off.min(parent_bytes.len());
            let avail = &parent_bytes[avail_start..];
            let partial = render_value_inner(btf, member_type_id, avail, depth + 1);
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
                    value: 0xffff_8000_1234_5678
                }
            ),
            "0xffff800012345678"
        );
        assert_eq!(format!("{}", RenderedValue::Ptr { value: 0 }), "0x0");
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
        // Mirror the team-lead's example.
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
            "struct task_ctx {\n  weight: 1024\n  last_runnable_at: 12345678901234\n}"
        );
    }

    #[test]
    fn display_struct_anonymous_uses_struct_brace() {
        let v = RenderedValue::Struct {
            type_name: None,
            members: vec![RenderedMember {
                name: "x".into(),
                value: RenderedValue::Int { bits: 32, value: 7 },
            }],
        };
        assert_eq!(format!("{v}"), "struct {\n  x: 7\n}");
    }

    #[test]
    fn display_empty_struct_is_one_line() {
        let v = RenderedValue::Struct {
            type_name: Some("empty".into()),
            members: vec![],
        };
        assert_eq!(format!("{v}"), "struct empty {}");
    }

    #[test]
    fn display_anonymous_member_uses_anon_marker() {
        // BTF anonymous union/struct members surface with empty name;
        // Display marks them so the operator knows the position
        // without seeing a `:` with no preceding identifier.
        let v = RenderedValue::Struct {
            type_name: Some("u".into()),
            members: vec![RenderedMember {
                name: String::new(),
                value: RenderedValue::Uint { bits: 32, value: 5 },
            }],
        };
        assert_eq!(format!("{v}"), "struct u {\n  <anon>: 5\n}");
    }

    #[test]
    fn display_nested_struct_indents_correctly() {
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
        assert_eq!(
            format!("{outer}"),
            "struct outer {\n  child: struct inner {\n    a: 1\n  }\n}"
        );
    }

    #[test]
    fn display_array_scalars_inline() {
        let v = RenderedValue::Array {
            len: 3,
            elements: vec![
                RenderedValue::Uint { bits: 8, value: 1 },
                RenderedValue::Uint { bits: 8, value: 2 },
                RenderedValue::Uint { bits: 8, value: 3 },
            ],
        };
        assert_eq!(format!("{v}"), "[1, 2, 3]");
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
        // truncation in a comment.
        let v = RenderedValue::Array {
            len: 5,
            elements: vec![
                RenderedValue::Uint { bits: 8, value: 1 },
                RenderedValue::Uint { bits: 8, value: 2 },
            ],
        };
        assert_eq!(format!("{v}"), "[1, 2] /* 2 of 5 shown */");
    }

    #[test]
    fn display_array_of_structs_block_style() {
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
        assert_eq!(format!("{v}"), "[\n  struct e {\n    v: 10\n  }\n]");
    }

    #[test]
    fn display_truncated_with_struct_partial_shows_decoded_members() {
        // The whole point of #48: decoded members survive when the
        // struct's byte slice was short. Display surfaces the partial
        // so test failure output points the operator at the fields
        // that DID decode.
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
        // Outer truncation marker + partial struct block on the
        // same line (the leading marker is one-line, then the
        // struct's own line breaks follow).
        assert!(out.starts_with("<truncated needed=8 had=4> struct partial_struct {"));
        assert!(out.contains("a: 7"));
        assert!(out.contains("b: <truncated needed=4 had=0>"));
    }

    // ---- #48: partial-render contract --------------------------------
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

    // ---- Datasec rendering (#67) ------------------------------------
    //
    // The fix for #67 is the renderer recognising
    // `BTF_KIND_DATASEC` (the value type libbpf assigns to a
    // global-section ARRAY map like `.bss`) and walking its
    // `VarSecinfo` entries to render each variable. Before the fix
    // the renderer returned `Unsupported`, so a stall dump's `.bss`
    // map showed an opaque hex dump instead of `stall=1, crash=0,
    // ...`.
    //
    // The probe BPF object built by `build.rs` contains a known
    // `.bss` Datasec (declared via the `volatile u32
    // ktstr_err_exit_detected = 0;` and the diagnostic-counter
    // globals `ktstr_trigger_count`, `ktstr_probe_count`,
    // `ktstr_meta_miss`, `ktstr_miss_log_idx` in
    // `src/bpf/probe.bpf.c`). The tests below load that BTF
    // directly via `load_btf_from_path` (which falls back to
    // goblin's `.BTF` ELF section parse for non-vmlinux files) and
    // exercise the Datasec render path against it. Hard-fail on a
    // missing probe.o because build.rs always produces it; a silent
    // skip would hide the regression the test is designed to catch.

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
        // Diagnostic counters are also writable globals → expected
        // in .bss too. Pin one as a smoke test that multiple
        // variables decode (not just the one the freeze coord
        // cares about).
        assert!(
            names.contains("ktstr_trigger_count"),
            "rendered .bss must contain `ktstr_trigger_count` \
             diagnostic counter. Found names: {names:?}"
        );
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
}
