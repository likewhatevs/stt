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
//! Datasec, Var, Fwd, Void, or a bytes slice shorter than the type's
//! declared size) yields a [`RenderedValue::Unsupported`] or
//! [`RenderedValue::Truncated`] node so the caller always gets a
//! well-formed tree it can serialize.
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
    Truncated { needed: usize, had: usize },
    /// BTF type kind the renderer does not handle (Func, FuncProto,
    /// Datasec, Var, Void, or a kind beyond the qualifier-peel cap).
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
                    elements.push(RenderedValue::Truncated {
                        needed: elem_size,
                        had: bytes.len().saturating_sub(start),
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
        Type::Datasec(_) | Type::Var(_) => RenderedValue::Unsupported {
            reason: "datasec/var meta-type: not a value".to_string(),
        },
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
    if bytes.len() < s.size() {
        return RenderedValue::Truncated {
            needed: s.size(),
            had: bytes.len(),
        };
    }
    let mut members = Vec::with_capacity(s.members.len());
    for m in &s.members {
        let name = btf.resolve_name(m).unwrap_or_default();
        let value = render_member(btf, m, bytes, depth);
        members.push(RenderedMember { name, value });
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
    if byte_off + size > parent_bytes.len() {
        return RenderedValue::Truncated {
            needed: size,
            had: parent_bytes.len().saturating_sub(byte_off),
        };
    }
    render_value_inner(
        btf,
        member_type_id,
        &parent_bytes[byte_off..byte_off + size],
        depth + 1,
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
        return RenderedValue::Truncated {
            needed: bytes_needed,
            had: parent_bytes.len().saturating_sub(byte_start),
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
        assert!(matches!(v, RenderedValue::Truncated { needed: 4, had: 0 }));
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
        assert!(matches!(v, RenderedValue::Truncated { needed: 4, had: 2 }));
    }
}
