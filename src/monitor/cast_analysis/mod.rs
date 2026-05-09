//! BPF cast analysis: recover typed pointers from `u64` fields.
//!
//! Schedulers store kernel pointers in BPF map values as raw `u64`
//! (because BTF cannot express a pointer to a per-allocation type),
//! then dereference them later as `struct Q *`. They also stash
//! kernel kptrs (`task_struct *`, `cgroup *`, …) in `u64` slots inside
//! map values that the verifier accepts as plain integers. From BTF
//! alone every such field looks like a counter, so the renderer's
//! native [`btf_rs::Type::Int`] arm has no way to recover the target
//! struct on its own. This module bridges that gap by analysing the
//! BPF program's instruction stream and recording the source / target
//! struct for every `u64` field that is observed to carry a typed
//! pointer. The renderer
//! ([`super::btf_render::render_cast_pointer`]) consumes the resulting
//! [`CastMap`] via [`super::btf_render::MemReader::cast_lookup`] and
//! chases the recovered pointer through the address-space-appropriate
//! reader (arena vs slab/vmalloc) — the same chase shape the
//! [`btf_rs::Type::Ptr`] arm uses for BTF-typed pointers, so cast-
//! recovered and natively-typed pointers render identically.
//!
//! The analysis is intentionally conservative. False negatives (a
//! cast we miss → renderer falls back to raw `u64`, which is the
//! status quo) are acceptable. False positives (a cast we mis-identify
//! → renderer chases garbage and emits structured nonsense) are not.
//!
//! # Algorithm
//!
//! Forward register-state walk over `&[BpfInsn]`. Each register
//! holds one of:
//! - `Unknown`
//! - `Pointer { struct_type_id }` — pointer to BTF struct
//!   `struct_type_id`. Both "address of a struct we will field-access
//!   through" and "kernel kptr being passed around" use this state;
//!   the distinction is the instruction (LDX vs STX) that consumes it.
//! - `LoadedU64Field { source_struct_id, field_offset }` — a 64-bit
//!   value loaded from struct `source_struct_id` at byte
//!   `field_offset`, where the BTF declares the source field as a
//!   plain `u64`. Used for the arena-pointer detection path: a u64
//!   field that is itself dereferenced.
//!
//! Two detection paths emit entries into the [`CastMap`]; a third
//! (BPF_ADDR_SPACE_CAST → arena_confirmed) participates in conflict
//! detection only.
//!
//! 1. **Arena pointer (LDX-side).** On every `BPF_LDX | BPF_MEM`
//!    instruction the destination register is updated according to
//!    the base register's state and the BTF layout at `(struct, off)`.
//!    When the base is a `LoadedU64Field`, the (target_offset,
//!    target_size) access pattern is recorded. After the walk the
//!    recorded patterns are matched against every BTF struct: a
//!    pattern resolves to a unique target struct only if exactly one
//!    struct in the BTF satisfies all observed `(offset, size)` pairs
//!    for that source field. The source struct itself is dropped from
//!    the candidate set before uniqueness is checked — a self-typed
//!    cast (`source.f` → `source*`) would let a self-referential
//!    layout silently win the intersection without any disambiguating
//!    evidence. Tagged with `AddrSpace::Arena`.
//!
//! 2. **Kernel kptr (STX-side).** On every `BPF_STX | BPF_MEM` of
//!    width `BPF_DW` where the destination base is a `Pointer{P}` and
//!    the source register is a `Pointer{T}` AND the BTF declares the
//!    field of `P` at the store offset as a plain `u64`, the
//!    `(P, offset) → T` mapping is recorded directly. Tagged with
//!    `AddrSpace::Kernel`. No BTF-shape inference is needed — `T` is
//!    already known from how the source register became typed (entry
//!    parameter seeded from a FuncProto, propagated through MOV /
//!    stack spill / kfunc return). Self-stores (`P == T`) are
//!    rejected: a typed-pointer aliasing path that resolved
//!    parent and target to the same struct id is almost always the
//!    analyzer's flow-insensitive register tracking confusing two
//!    code paths, and recording a self-store would later resolve to
//!    a chase that loops on the same struct.
//!
//! Stack spill / reload is tracked through `[r10 + neg_off]`: STX
//! through r10 saves the source register's state, LDX through r10
//! restores it. This catches typed pointers that round-trip through
//! the stack across helper calls.
//!
//! Function entry seeding (via [`FuncEntry`]) reseeds R1..R5 with the
//! parameter types from a BTF FuncProto at each function entry PC.
//! The same mechanism handles cross-function jumps inside a single
//! `&[BpfInsn]` slab.
//!
//! Kfunc return values: at every `BPF_CALL` whose `src_reg` is
//! `BPF_PSEUDO_KFUNC_CALL`, `imm` is interpreted as a BTF id; if the
//! kfunc's FuncProto return type peels to `Ptr -> Struct`, R0 is set
//! to `Pointer{struct_type_id}` after the standard R0..R5 clobber.
//!
//! Plain-helper return values: at every `BPF_CALL` whose `src_reg ==
//! 0` (the helper-call form per linux uapi `bpf.h`) AND
//! `imm == BPF_FUNC_map_lookup_elem`, R1's pre-clobber state is
//! consulted. If R1 was [`RegState::DatasecPointer`] into a
//! `BTF_KIND_DATASEC` named `.maps` and the targeted map's BTF
//! declaration carries a `value` member whose type peels to
//! `Ptr -> Struct/Union`, R0 is typed `Pointer{value_struct_id}`
//! after the clobber. Other helper ids leave R0 Unknown — the
//! analyzer keeps a strict per-helper allowlist (currently length 1)
//! to bound false-positive risk. Maps whose value type is a primitive
//! (e.g. stat counters declared `__type(value, u64)`) drop because
//! `Ptr -> u64` does not peel to a Struct/Union.
//!
//! Branches are handled conservatively: on every jump-target PC the
//! pre-pass identifies, register state AND stack-slot state are reset
//! before processing that PC. This drops casts that span branch joins
//! (false negative, acceptable). Function calls clobber `r0..=r5` per
//! the BPF ABI; kfunc and helper return typing happen after the
//! clobber.
//!
//! # Public surface
//!
//! - [`analyze_casts`]: full forward-pass entry point.
//! - [`AddrSpace`]: tag distinguishing arena pointers from kernel
//!   kptrs in the output.
//! - [`CastMap`]: BTreeMap (deterministic iteration order) from
//!   `(source_btf_type_id, field_byte_offset)` to
//!   [`super::btf_render::CastHit`].
//! - [`InitialReg`]: caller-supplied seed register state for entry
//!   parameters / known typed values returned from helpers.
//! - [`FuncEntry`]: function-entry PC + BTF FuncProto id for
//!   automatic R1..R5 seeding from the proto's parameters.
//! - [`SubprogReturn`]: `BPF_PSEUDO_CALL` PC whose resolved subprog
//!   name matches the arena-allocator allowlist; seeds R0 to
//!   [`RegState::ArenaU64FromAlloc`] after the standard R0..=R5
//!   clobber so allocator-return values flow into the STX-flow
//!   arena cast detection path.
//! - [`DatasecPointer`]: caller-supplied annotation pairing a
//!   `BPF_LD_IMM64` PC with its target `BTF_KIND_DATASEC` plus the
//!   byte offset of the referenced global within that section, so
//!   the `BPF_LD_IMM64` arm can set the destination register to
//!   [`RegState::DatasecPointer`] and downstream STX/LDX through
//!   the register fire kptr / arena cast findings against the
//!   datasec's variable layout.
//!
//! # F1 mitigation: arena_confirmed evidence required
//!
//! On aarch64 the 4 GiB arena window catches any 33-bit value as
//! "in arena", so a slot that just happens to hold a 33-bit-shaped
//! counter could be mis-rendered as an arena pointer. Every Arena
//! cast emit therefore requires direct evidence the slot held an
//! arena VA at runtime: either (a) an observed
//! [`BPF_ADDR_SPACE_CAST`] (`ALU64 | MOV | X` with `off=1, imm=1`)
//! on a value loaded from the slot, or (b) an observed STX of an
//! [`RegState::ArenaU64FromAlloc`] value into the slot. Slots with
//! shape-inference evidence ALONE are dropped — the operator can
//! re-enable them by adding either form of direct evidence in the
//! scheduler source.
//!
//! The module does not mutate the BTF object and does not call into
//! libbpf or the kernel — it operates purely on the instruction slice
//! and the parsed BTF. Instructions are represented by [`BpfInsn`], a
//! native Rust struct that mirrors the kernel's on-wire 8-byte layout
//! (`include/uapi/linux/bpf.h struct bpf_insn`). Callers parse program
//! sections out of the raw `.bpf.o` ELF (e.g. via goblin) and feed the
//! resulting byte stream through [`BpfInsn::from_le_bytes`]; the
//! analyzer never invokes `bpf_object__prepare` or any other kernel-
//! side BPF interface. Opcode and register-encoding constants are
//! sourced from `libbpf_rs::libbpf_sys` (the bindgen translation of
//! `linux/include/uapi/linux/bpf.h`) so they track the upstream UAPI
//! without duplicating numeric literals here.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use btf_rs::{Btf, BtfType, Type};
use libbpf_rs::libbpf_sys as bs;

/// One BPF instruction in the kernel's on-wire encoding.
///
/// Mirrors `struct bpf_insn` from linux `include/uapi/linux/bpf.h`:
/// 8 bytes total, where the second byte packs `dst_reg` (low 4 bits)
/// and `src_reg` (high 4 bits). All multi-byte fields are
/// little-endian per the BPF wire format spec.
///
/// Pure host-side data — no kernel interaction, no FFI. Callers obtain
/// a slice of these from raw program bytes in a `.bpf.o` ELF section.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BpfInsn {
    /// Opcode byte: class (low 3 bits) + size/op + mode (high bits).
    pub code: u8,
    /// Packed register byte: `dst_reg | (src_reg << 4)` — 4 bits each
    /// per the wire format. Use [`Self::dst_reg`] / [`Self::src_reg`]
    /// to read. Private so callers cannot bypass the 4-bit packing
    /// invariant; [`Self::new`] and [`Self::from_le_bytes`] are the
    /// only construction paths.
    regs: u8,
    /// Signed 16-bit offset (PC-relative for jumps, byte offset for
    /// mem ops, atomic-op subselect for `BPF_MODE_ATOMIC`).
    pub off: i16,
    /// Signed 32-bit immediate (constant operand, or — for
    /// `BPF_PSEUDO_KFUNC_CALL` — the BTF id of the kfunc).
    pub imm: i32,
}

impl BpfInsn {
    /// Construct an instruction with explicit fields. `dst` and `src`
    /// are 0..=15 register indices (the analyzer rejects 11..=15 at
    /// decode time per `step()`).
    ///
    /// Test-only: production decode uses
    /// [`BpfInsn::from_le_bytes`]; tests construct fixtures directly
    /// to exercise specific opcode/register combinations without a
    /// round-trip through the wire encoder. Gated `#[cfg(test)]` so
    /// release builds do not carry an unused constructor.
    #[cfg(test)]
    pub const fn new(code: u8, dst: u8, src: u8, off: i16, imm: i32) -> Self {
        Self {
            code,
            regs: (dst & 0x0f) | ((src & 0x0f) << 4),
            off,
            imm,
        }
    }

    /// Decode 8 bytes of little-endian wire data into a [`BpfInsn`].
    /// Caller is responsible for chunking program bytes into 8-byte
    /// slots — `BPF_LD_IMM64` consumes two consecutive slots and the
    /// analyzer's `skip_next` flag handles the second one.
    pub fn from_le_bytes(buf: [u8; 8]) -> Self {
        let off = i16::from_le_bytes([buf[2], buf[3]]);
        let imm = i32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        Self {
            code: buf[0],
            regs: buf[1],
            off,
            imm,
        }
    }

    /// Destination register index (low 4 bits of the packed byte).
    #[inline]
    pub const fn dst_reg(&self) -> u8 {
        self.regs & 0x0f
    }

    /// Source register index (high 4 bits of the packed byte).
    #[inline]
    pub const fn src_reg(&self) -> u8 {
        (self.regs >> 4) & 0x0f
    }

    /// Overwrite the high 4 bits of the packed `regs` byte (the
    /// `src_reg` field), preserving `dst_reg`. Used by the host-side
    /// loader's libbpf-style relocation rewrite to flip a clang-
    /// emitted `BPF_PSEUDO_CALL` into a `BPF_PSEUDO_KFUNC_CALL` after
    /// the kfunc BTF id has been resolved. `pub(crate)` rather than
    /// `pub` because the wire-format invariants for the packed byte
    /// are framework-internal — external callers should construct a
    /// fresh [`BpfInsn::new`] instead of mutating.
    #[inline]
    pub(crate) fn set_src_reg(&mut self, src: u8) {
        self.regs = (self.regs & 0x0f) | ((src & 0x0f) << 4);
    }
}

/// Caller-supplied initial state for one BPF register.
///
/// Used to seed entry-parameter typing or the typed return value of
/// a kfunc. Empty seed lists yield no findings — the analysis only
/// produces output along chains rooted in registers it knows are
/// typed pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InitialReg {
    /// Register index, `0..=9`. `r10` (frame pointer) is rejected
    /// during seeding because [`analyze_casts`] never derives kernel
    /// struct accesses through it.
    pub reg: u8,
    /// BTF type id of the struct the register points at. The id is
    /// peeled through `Ptr` / `Const` / `Volatile` / `Restrict` /
    /// `Typedef` / `TypeTag` / `DeclTag` chains until a `Struct` or
    /// `Union` is reached; if no struct is reachable the seed is
    /// silently ignored.
    pub struct_type_id: u32,
}

/// Function-entry PC paired with the BTF type id of its FuncProto.
///
/// Caller obtains these from `bpf_func_info` in `.BTF.ext` (or the
/// kernel `bpf_prog_info` accessor that exposes the same array).
/// At each PC matching `insn_offset`, the analyzer clears ALL
/// registers (R0..R10) and drops every stack slot (the linear walk
/// concatenates subprograms, so stale R6..R9 from an unrelated
/// preceding function must not leak into this entry), then reseeds
/// R1..R5 from the FuncProto's parameter list:
/// parameter `i` (zero-indexed) becomes `R{i+1}`. Parameters that
/// peel to `Ptr -> Struct/Union` produce `Pointer{struct_id}`;
/// everything else (scalar, void, function pointer, …) leaves the
/// register `Unknown`. A variadic sentinel terminates the parameter
/// scan (everything after it is unreachable in the BPF calling
/// convention); parameters past R5 are skipped silently.
///
/// `func_proto_id` must resolve to `Type::FuncProto` in the BTF
/// passed to [`analyze_casts`]. If `Type::Func` is given by mistake,
/// the analyzer peels one level (Func->FuncProto) and proceeds.
/// Anything else silently disables seeding for that entry — false
/// negatives are the safe direction. All registers and the stack
/// are still cleared in that case so the unrecoverable proto
/// cannot retain stale state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FuncEntry {
    /// Instruction index of the function's first instruction.
    pub insn_offset: usize,
    /// BTF id of the function's prototype (`BTF_KIND_FUNC_PROTO`).
    pub func_proto_id: u32,
}

/// Caller-supplied annotation that flags a `BPF_PSEUDO_CALL` to an
/// in-tree subprog whose return value is a u64 carrying an arena
/// virtual address.
///
/// scx schedulers stash arena pointers in `u64` slots after calling
/// helpers like `scx_static_alloc()` / `scx_alloc_internal()` that
/// return a u64 (NOT a typed pointer). BTF declares the destination
/// field as `u64`, so neither the renderer's [`btf_rs::Type::Ptr`] arm
/// nor the cast analyzer's [`Type::Int`] LDX-shape inference fires —
/// the field looks like a counter. The host-side loader walks
/// [`BPF_PSEUDO_CALL`] sites whose resolved subprog name matches the
/// allocator allowlist and emits one [`SubprogReturn`] per call site.
/// The analyzer applies the annotation at the PC immediately AFTER the
/// call (the BPF ABI clobbers R0..=R5 at the call boundary; this seeds
/// R0 to [`RegState::ArenaU64FromAlloc`] AFTER the clobber so the next
/// move/spill/store of R0 carries the tag forward).
///
/// `insn_offset` is the call PC (not PC+1); the analyzer applies the
/// seed inside its [`BPF_OP_CALL`] arm after the standard register
/// clobber, mirroring how [`Self::handle_kfunc_call`] types R0 from
/// the kfunc's FuncProto return type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubprogReturn {
    /// Instruction index of the `BPF_PSEUDO_CALL` site.
    pub insn_offset: usize,
}

/// Caller-supplied annotation that ties a `BPF_LD_IMM64` instruction
/// to its target `BTF_KIND_DATASEC` plus the byte offset of the
/// referenced global within that section.
///
/// Pre-relocation `.bpf.o` bytecode (the input the host-side cast
/// loader sees, before libbpf processes ELF relocations) emits
/// `BPF_LD_IMM64` referencing global variables in `.bss`, `.data`,
/// or `.rodata` with `src_reg == 0` and `imm == 0` — the relocation
/// entry in `.rel.<text>` carries the actual section binding. The
/// host-side loader walks `.rel.<text>`, identifies LD_IMM64 PCs
/// whose target is a datasec section, and emits one `DatasecPointer`
/// per such PC. The analyzer applies the annotation in the
/// `BPF_LD_IMM64` arm to set the destination register state to
/// [`RegState::DatasecPointer { datasec_type_id, base_offset }`],
/// which subsequent STX/LDX through the register treat as a typed
/// pointer into the datasec's variable layout. See linux uapi
/// `bpf.h` `BPF_PSEUDO_MAP_VALUE = 2` for the kernel-side encoding.
///
/// `base_offset` is the byte offset of the referenced global within
/// the datasec. For SHT_REL (the BPF convention — clang emits
/// SHT_REL, not SHT_RELA, for BPF object files), `r_addend` is
/// absent; the offset comes from `LD_IMM64 insn.imm +
/// sym.st_value` and the host-side loader populates `base_offset`
/// from those fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DatasecPointer {
    /// Instruction index of the `BPF_LD_IMM64` to annotate.
    pub insn_offset: usize,
    /// BTF id of the `BTF_KIND_DATASEC` type for the referenced
    /// section.
    pub datasec_type_id: u32,
    /// Byte offset of the referenced global within the datasec.
    pub base_offset: u32,
}

/// Address space of a recovered cast target.
///
/// Distinguishes the two detection paths: arena pointers carry an
/// arena virtual address; kernel kptrs carry a kernel virtual
/// address (slab / vmalloc / per-cpu). Both share the same
/// `(source, offset) -> target` shape. The renderer treats
/// `AddrSpace` as a hint — runtime is-arena-window detection on
/// the actual pointer value is authoritative — so a misclassified
/// finding still chases through the correct reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrSpace {
    /// Arena pointer: a `u64` slot whose stored value is an arena
    /// virtual address. Recovered by tracking LDX-through-LoadedU64.
    Arena,
    /// Kernel kptr: a `u64` slot whose stored value is a kernel
    /// virtual address (slab / vmalloc / per-cpu). Recovered by
    /// tracking STX of a typed `Pointer{T}` register.
    Kernel,
}

impl std::fmt::Display for AddrSpace {
    /// Renders as the lowercase address-space tag (`"arena"` /
    /// `"kernel"`) for free-form formatting (error messages, log
    /// lines). The renderer side bypasses `Display` and uses an
    /// exhaustive `match` over the variant set in
    /// `crate::monitor::btf_render::cast_annotation_for` to hand
    /// back static `&'static str` annotations
    /// (`"cast→arena"`, `"cast→arena (sdt_alloc)"`, `"cast→kernel"`,
    /// `"cast→kernel (sdt_alloc)"`) — so the operator-visible
    /// `cast_annotation` field is allocation-free per chase. A
    /// new `AddrSpace` variant added here must also add a row in
    /// `cast_annotation_for`'s match; the compiler enforces it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AddrSpace::Arena => "arena",
            AddrSpace::Kernel => "kernel",
        };
        f.write_str(s)
    }
}

/// One recovered cast finding, returned by
/// [`super::btf_render::MemReader::cast_lookup`] to tell the
/// renderer that a `u64` field at
/// `(parent_struct_btf_id, member_byte_offset)` actually carries a
/// pointer to a struct whose BTF id is `target_type_id`. The
/// `addr_space` tag is a HINT from the analyzer; the renderer
/// applies runtime detection on the actual pointer value to pick
/// arena vs kernel chasing.
///
/// `Copy` so the renderer can hand it across helper boundaries
/// without lifetime gymnastics; the type is two `Copy` fields and
/// stays small.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastHit {
    /// BTF type id of the recovered target struct/union.
    pub target_type_id: u32,
    /// Address-space hint from the analyzer (arena vs kernel).
    /// The renderer ignores this for dispatch — runtime
    /// is-arena-window detection on the pointer value drives the
    /// choice — but it is preserved as evidence so an operator can
    /// see whether the analyzer's hint matched what the runtime
    /// chase resolved to.
    pub addr_space: AddrSpace,
}

/// Output of [`analyze_casts`].
///
/// Maps `(source_btf_type_id, field_byte_offset)` to the recovered
/// target's [`CastHit`]. The map is `BTreeMap` so iteration order
/// is deterministic, which makes test assertions stable without a
/// sort step at every assertion site.
pub type CastMap = BTreeMap<(u32, u32), CastHit>;

/// Maximum BTF type-id the candidate-search loop probes per pattern.
///
/// `btf_rs` does not expose a "list all type ids" iterator. The
/// matcher walks ids `1..=max_observed_id` instead, where
/// `max_observed_id` is the largest id touched during the forward
/// pass plus this slack. Real ktstr program BTFs top out in the low
/// thousands of types; the slack is generous so a struct that only
/// appears in the BTF (not yet referenced by any instruction we
/// processed) can still match. The hard cap
/// [`super::sdt_alloc::MAX_BTF_ID_PROBE`] backstops a pathological /
/// synthesized BTF.
const CANDIDATE_SEARCH_SLACK: u32 = 65_536;

/// Per-register state during the forward walk.
#[derive(Debug, Clone, Copy)]
enum RegState {
    Unknown,
    /// Register holds a pointer to a known BTF struct.
    Pointer {
        struct_type_id: u32,
    },
    /// Register holds a `u64` value loaded from `(struct,
    /// field_offset)`; the BTF declares the source field as a plain
    /// 8-byte unsigned integer.
    LoadedU64Field {
        source_struct_id: u32,
        field_offset: u32,
    },
    /// Register holds a pointer into a `BTF_KIND_DATASEC` map value
    /// (a `.bss`, `.data`, `.rodata`, or `.data.<name>` global
    /// section), at a known byte offset within that section.
    /// Produced by `BPF_LD_IMM64` with `src_reg ==
    /// BPF_PSEUDO_MAP_VALUE`, where the first instruction slot's
    /// `imm` field carries the byte offset of the referenced global
    /// (added to the section symbol's address, which is 0 for
    /// `STT_SECTION` symbols, so the imm IS the offset).
    ///
    /// STX through this register at insn-offset `N` writes to
    /// `(datasec_type_id, base_offset + N)`. The offset is resolved
    /// against the datasec's `VarSecinfo` entries via
    /// [`struct_member_at`]: each entry spans `[var.offset(),
    /// var.offset() + var.size())` and points at a `BTF_KIND_VAR`
    /// whose underlying type is the global's actual C type. A
    /// hit on a u64-typed Var triggers the kptr finding path
    /// just like a struct member would.
    ///
    /// See linux uapi `bpf.h` `BPF_PSEUDO_MAP_VALUE = 2` and the
    /// libbpf relocation logic in `tools/lib/bpf/libbpf.c`'s
    /// `bpf_program__resolve_map_value_relos` (libbpf rewrites
    /// `R_BPF_64_64` relocations against `STT_SECTION` symbols on
    /// `.bss`/`.data`/`.rodata` into LD_IMM64 instructions with
    /// `src_reg = BPF_PSEUDO_MAP_VALUE`). The host-side cast loader
    /// does not see post-relocation bytecode (the embedded
    /// `.bpf.objs` blob carries the raw `.bpf.o` ELF), so it
    /// reconstructs the same mapping by walking `.rel.<text>`
    /// sections and emitting [`DatasecPointer`] entries that the
    /// analyzer applies at the same insn PC.
    DatasecPointer {
        datasec_type_id: u32,
        base_offset: u32,
    },
    /// Register holds a `u64` value the analyzer believes is an arena
    /// virtual address — either because it came directly from an
    /// allocator-return seed at a [`SubprogReturn::insn_offset`], OR
    /// because it was loaded from a slot the analyzer previously
    /// tagged as Arena via the STX-flow path (alias-set tracking).
    ///
    /// Distinct from [`Self::LoadedU64Field`]: that variant tracks a
    /// generic u64 whose downstream LDX accesses constrain shape
    /// inference. This variant has stronger evidence (the value came
    /// from an allowlisted allocator OR an already-arena-tagged
    /// field), so the STX of this state into a `u64` field of a
    /// typed `Pointer{P}` parent records `(P, off)` as an Arena cast
    /// finding directly — no shape inference required.
    ///
    /// No payload fields: the source slot identity (parent struct +
    /// field offset) is derived at the STX site from the destination
    /// register's `Pointer{P}` state and the store's offset, not
    /// carried in the value register's state.
    ArenaU64FromAlloc,
}

/// Observed `(offset, size_bytes)` access through a `LoadedU64Field`
/// register. Stored in a set per source `(struct, field)` so duplicate
/// patterns coalesce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Access {
    offset: u32,
    size: u32,
}

/// Run the cast analysis over `insns` against `btf`. `initial_regs`
/// seeds the entry / known-pointer registers before the walk starts.
/// `func_entries` identifies function-entry PCs whose R1..Rn should
/// be reseeded from a BTF FuncProto's parameter list before the
/// instruction at that PC executes. `datasec_pointers` annotates
/// `BPF_LD_IMM64` PCs that resolve (after libbpf relocation) to a
/// `BPF_PSEUDO_MAP_VALUE` reference into a `.bss` / `.data` /
/// `.rodata` global section — see [`DatasecPointer`]. `subprog_returns`
/// annotates `BPF_PSEUDO_CALL` sites whose resolved subprog name
/// matches the arena-allocator allowlist (e.g. `scx_static_alloc_internal`,
/// `scx_alloc_internal`, `bpf_arena_alloc_pages`); after the standard
/// R0..=R5 clobber the analyzer seeds R0 to
/// [`RegState::ArenaU64FromAlloc`] so the value flows into STX-side
/// arena cast detection. See [`SubprogReturn`].
///
/// `initial_regs`, `func_entries`, `datasec_pointers`, and
/// `subprog_returns` compose: seeds apply once at PC 0, function-entry
/// reseeding applies at every matching `insn_offset`, datasec
/// annotations apply at every matching `BPF_LD_IMM64` PC, and
/// allocator-return seeds apply at every matching `BPF_PSEUDO_CALL`
/// PC. Reseeding clears ALL registers (R0..R10) and drops every stack
/// slot (subprog entry semantics: the callee's frame is fresh, and
/// stale R6..R9 from linearly-preceding unrelated functions must not
/// leak). R1..R5 are then re-seeded from the FuncProto's parameter
/// types where they resolve to struct pointers.
///
/// The plain-helper return arm in [`Analyzer::step`] does not consume
/// caller-supplied annotations — it derives R0's typing from the
/// analyzer's pre-clobber view of R1 (which the existing datasec
/// annotation pipeline already populates with
/// [`RegState::DatasecPointer`] when the LD_IMM64 of R1 targets the
/// `.maps` BTF datasec). No new caller-side annotation list is
/// needed: a `bpf_map_lookup_elem` call site whose R1 is sourced
/// from a `.maps` LD_IMM64 already has all the evidence the arm
/// requires by the time the call is processed.
///
/// The analysis ignores any [`BpfInsn`] it cannot decode (unknown
/// opcode, malformed encoding) — those manifest as false negatives,
/// the safe direction. Empty input or input that produces no
/// typed-pointer-rooted load or store yields an empty [`CastMap`].
pub fn analyze_casts(
    insns: &[BpfInsn],
    btf: &Btf,
    initial_regs: &[InitialReg],
    func_entries: &[FuncEntry],
    datasec_pointers: &[DatasecPointer],
    subprog_returns: &[SubprogReturn],
) -> CastMap {
    let mut analyzer = Analyzer::new(btf);
    analyzer.seed(initial_regs);
    let targets = jump_targets(insns);
    analyzer.run(
        insns,
        &targets,
        func_entries,
        datasec_pointers,
        subprog_returns,
    );
    analyzer.finalize()
}

struct Analyzer<'a> {
    btf: &'a Btf,
    regs: [RegState; 11],
    /// Per `(source_struct, field_offset)` set of `(target_offset,
    /// target_size)` accesses observed via the arena LDX path.
    patterns: BTreeMap<(u32, u32), BTreeSet<Access>>,
    /// Direct kptr findings keyed by `(source_struct_id,
    /// field_offset)` (the struct that owns the slot, named
    /// "source" for parity with `patterns` above). Populated by the
    /// STX path when a `Pointer{T}` value register is stored into a
    /// `u64` field. The map's value is the inner struct id `T`.
    /// Conflicting writes (same slot, different `T`) collapse to a
    /// sentinel that finalize() drops — ambiguity is a false
    /// negative, never a false positive.
    kptr_findings: BTreeMap<(u32, u32), KptrEntry>,
    /// Stack-slot map keyed by frame-pointer-relative byte offset
    /// (always negative). STX through r10 saves the source register
    /// state; LDX through r10 restores it. Cleared at every
    /// jump-target PC alongside the register file.
    stack_slots: BTreeMap<i16, RegState>,
    /// Fields confirmed as arena pointers by a `BPF_ADDR_SPACE_CAST`
    /// instruction (code=0xBF, off=1, imm=1). Keyed by
    /// `(source_struct_id, field_byte_offset)`.
    ///
    /// Two roles:
    /// 1. Veto a kptr finding when the same slot was also observed as
    ///    the source of an arena cast (the slot cannot simultaneously
    ///    hold an arena VA and a kernel VA — the conflict drop set
    ///    in [`Self::finalize`] uses this).
    /// 2. Gate the shape-inference path: an entry in
    ///    [`Self::patterns`] alone is not enough evidence to emit an
    ///    Arena cast hit (the LDX-shape inference can match
    ///    coincidentally on schedulers whose program BTF carries
    ///    same-shape unrelated structs). `arena_confirmed` is the
    ///    direct evidence that the value held in the slot was an
    ///    arena pointer — required for the shape-inference emit per
    ///    the F1 hostile-input mitigation. The new STX-flow path
    ///    (see [`Self::arena_stx_findings`]) carries its own evidence
    ///    (allocator-return → field) and emits independently.
    arena_confirmed: BTreeSet<(u32, u32)>,
    /// Fields where an [`RegState::ArenaU64FromAlloc`] register was
    /// stored into a `u64` slot of a typed `Pointer{P}` (or
    /// `DatasecPointer`) parent. Keyed by
    /// `(parent_struct_id, field_byte_offset)`.
    ///
    /// Direct evidence the slot holds an arena VA: the value came
    /// from an allocator return (e.g. `scx_static_alloc()`) or
    /// propagated from another already-arena-tagged slot. Conflicting
    /// cross-path observations (a typed `Pointer{T}` STX into the same
    /// slot, indicating a kernel kptr write) are detected by
    /// [`Self::finalize`]'s conflict-drop set, which cross-references
    /// `arena_stx_findings` keys against `kptr_findings` keys and
    /// rejects the slot from BOTH sides — false positive is
    /// unacceptable, false negative is the safe direction. Within
    /// `arena_stx_findings` itself, all current insertions resolve
    /// to [`ArenaStxEntry::Pending`] (see the enum doc for the
    /// `Conflicting` variant's defensive role).
    arena_stx_findings: BTreeMap<(u32, u32), ArenaStxEntry>,
    /// Largest type id touched while resolving sources (struct
    /// pointer types and u64-field source structs). Used to bound
    /// the matcher's id walk below
    /// [`super::sdt_alloc::MAX_BTF_ID_PROBE`].
    max_seen_type_id: u32,
    /// Count of [`SubprogReturn`] / kfunc-allowlist allocator-seed
    /// applications during the forward walk. Incremented every
    /// time the analyzer sets R0 to [`RegState::ArenaU64FromAlloc`]
    /// from either:
    /// - a caller-supplied [`SubprogReturn`] match in the
    ///   `BPF_OP_CALL` arm, OR
    /// - the `ARENA_ALLOC_KFUNC_NAMES` allowlist match in
    ///   [`Self::handle_kfunc_call`].
    ///
    /// Used by [`Self::finalize`] to gate the F4 mitigation warn
    /// (`allocator helpers may need __always_inline`): the warn
    /// must only fire when allocator call sites WERE seen but
    /// produced NO `arena_stx_findings`, which is the actual
    /// "non-inlined helper" signature. Firing when
    /// `arena_stx_findings` is non-empty but `arena_confirmed` is
    /// empty (the prior gate) was too broad — that condition
    /// matches the normal STX-flow path where the allocator IS
    /// inlined and its R0 reaches a STX into a typed slot.
    alloc_seeds_applied: u32,
}

/// Kptr finding state: a single `(parent, offset)` slot may be
/// written by code paths that disagree on the target type. The
/// analyzer collapses the disagreement to `Conflicting` so finalize()
/// can drop it.
#[derive(Debug, Clone, Copy)]
enum KptrEntry {
    /// Single observed target type id. Always non-zero in practice —
    /// `Pointer{T}` source registers carry a real BTF type id and the
    /// `Pointer{T}` STX path is the only insertion site for this
    /// variant. (The arena-STX path uses the sibling
    /// [`ArenaStxEntry`] enum, not this one, so a stale "0 means
    /// deferred resolve" reading does not apply here.)
    Single(u32),
    /// Two or more disjoint target ids observed; drop the slot.
    Conflicting,
}

/// Arena STX finding state for [`Analyzer::arena_stx_findings`]. A
/// single `(parent, offset)` slot may be written by an
/// [`RegState::ArenaU64FromAlloc`] STX (records `Pending`) and, in
/// principle, also by a typed-pointer kptr STX (which records into
/// the sibling [`Analyzer::kptr_findings`], not here).
///
/// The variant set is deliberately distinct from [`KptrEntry`]: the
/// arena STX path has no per-finding payload (the renderer's
/// [`super::btf_render::MemReader::resolve_arena_type`] bridge
/// recovers the actual payload BTF type id at chase time), so reusing
/// `KptrEntry::Single(0)` as a "deferred resolve pending" sentinel
/// would conflate two different concepts at the type level. A
/// dedicated enum makes "this slot saw an arena STX" a single
/// variant ([`Self::Pending`]) and keeps `KptrEntry::Single(0)`'s
/// meaning unambiguous in the kptr path.
///
/// `Conflicting` is preserved for symmetry with [`KptrEntry`] and as
/// a defensive landing pad: today the only insertion path for
/// `arena_stx_findings` is the `StxValueKind::Arena` arm of
/// [`Analyzer::handle_stx`], which only ever inserts
/// [`Self::Pending`]. The arena-STX dedup arm in `handle_stx`
/// therefore matches `Some(Self::Pending)` exhaustively as a no-op
/// and treats `Some(Self::Conflicting)` as `unreachable!()`. If a
/// future code path adds a way to record disagreement on the same
/// slot from the arena side, it can use this variant; finalize's
/// filter will drop it identically to today.
#[derive(Debug, Clone, Copy)]
enum ArenaStxEntry {
    /// Allocator-tagged value observed at the slot. The renderer's
    /// [`super::btf_render::MemReader::resolve_arena_type`] bridge
    /// supplies the actual payload BTF type id at chase time — the
    /// analyzer emits `target_type_id == 0` from finalize and the
    /// bridge fills in the real id.
    Pending,
    /// Two or more disagreeing observations — drop the slot.
    /// Unreachable through today's insertion paths but kept for
    /// symmetry with [`KptrEntry::Conflicting`] and as a defensive
    /// terminal so finalize's filter survives a future enrichment
    /// of the arena-STX path. `#[allow(dead_code)]` because no
    /// current insertion site constructs the variant; the
    /// `unreachable!()` in [`Analyzer::handle_stx`]'s arena dedup
    /// arm and the defensive filter in [`Analyzer::finalize`] both
    /// reference it as a pattern only. Removing the variant would
    /// drop the documented design margin and force a churn of the
    /// match shape if a future enrichment ever needs it.
    #[allow(dead_code)]
    Conflicting,
}

/// Discriminator for the value-side state at a STX site that passed
/// the BTF u64 gate. Used by [`Analyzer::handle_stx`] to dispatch the
/// kptr vs arena finding paths from one match arm rather than
/// re-pattern-matching [`RegState`] inside the recording logic.
#[derive(Debug, Clone, Copy)]
enum StxValueKind {
    /// Source register held [`RegState::Pointer`]: record into
    /// [`Analyzer::kptr_findings`].
    Kptr { target: u32 },
    /// Source register held [`RegState::ArenaU64FromAlloc`]: record
    /// into [`Analyzer::arena_stx_findings`].
    Arena,
}

impl<'a> Analyzer<'a> {
    fn new(btf: &'a Btf) -> Self {
        Self {
            btf,
            regs: [RegState::Unknown; 11],
            patterns: BTreeMap::new(),
            kptr_findings: BTreeMap::new(),
            stack_slots: BTreeMap::new(),
            arena_confirmed: BTreeSet::new(),
            arena_stx_findings: BTreeMap::new(),
            max_seen_type_id: 0,
            alloc_seeds_applied: 0,
        }
    }

    fn seed(&mut self, initial_regs: &[InitialReg]) {
        for seed in initial_regs {
            // r10 is the BPF frame pointer; never a typed pointer to
            // a kernel struct. Reject so the analysis never derives
            // a cast from it. R0..R9 are the typed-value registers
            // per linux uapi `bpf.h` (BPF_REG_0..BPF_REG_9, with
            // BPF_REG_10 as the FP and `MAX_BPF_REG = 11` as the
            // register-file bound). The `>= BPF_REG_R10` gate
            // rejects both r10 itself and any out-of-range index
            // 11..=15 a malformed seed could carry.
            if (seed.reg as usize) >= BPF_REG_R10 {
                continue;
            }
            // Resolve through Ptr/Typedef/etc. to a Struct. If no
            // struct is reachable, drop silently — false negatives
            // here are acceptable.
            let Some(sid) = super::bpf_map::resolve_to_struct_id(self.btf, seed.struct_type_id)
            else {
                continue;
            };
            self.regs[seed.reg as usize] = RegState::Pointer {
                struct_type_id: sid,
            };
            self.note_type_id(sid);
        }
    }

    /// Reseed R1..R5 from a FuncProto's parameter list. Called at
    /// every PC matching a [`FuncEntry::insn_offset`]. Parameters
    /// past R5 are skipped silently — the BPF ABI passes only the
    /// first five in registers and anything beyond is on the stack
    /// (which the analyzer treats as Unknown unless an explicit
    /// spill writes a typed value).
    ///
    /// `func_proto_id` may resolve to either `Type::FuncProto`
    /// directly or to `Type::Func` (in which case it is peeled one
    /// level to its FuncProto). Anything else disables seeding for
    /// this entry.
    ///
    /// All registers and stack slots are cleared. R1..R5 are then
    /// re-seeded from the FuncProto parameters: a parameter that
    /// peels to `Ptr -> Struct/Union` becomes `Pointer{struct_id}`,
    /// everything else (scalar, void, function pointer) leaves the
    /// register `Unknown` — exactly what the BPF ABI provides at a
    /// function entry. The forward walk concatenates subprogram
    /// instruction streams; a function entry reached by fall-through
    /// from a prior EXIT would inherit stale R6..R9 state from an
    /// unrelated function if those registers were preserved (the
    /// BPF ABI's callee-saved property only holds across a real
    /// CALL, not across a textual concatenation), so the unconditional
    /// clear is the correct safe direction.
    fn seed_from_func_proto(&mut self, func_proto_id: u32) {
        // Pre-clear ALL registers and the stack unconditionally. The
        // linear forward walk concatenates subprogram instruction
        // streams; a function entry reached by fall-through from a
        // prior EXIT would inherit stale R6..R9 state from an
        // unrelated function. BPF ABI says R6..R9 are callee-saved
        // (inherited from the CALLER), but in the linear walk there
        // is no real caller — the prior function is unrelated.
        // Preserving R6..R9 would let stale typed pointers leak
        // across function boundaries, risking false positives. The
        // cost: legitimate call-inherited R6..R9 typing is lost
        // (false negative), which is the safe direction. Even if
        // proto resolution fails below, stale typed-pointer state
        // from a prior function must not survive past this entry.
        self.regs = [RegState::Unknown; 11];
        self.stack_slots.clear();
        let proto = match self.btf.resolve_type_by_id(func_proto_id) {
            Ok(Type::FuncProto(fp)) => fp,
            Ok(Type::Func(f)) => match f.get_type_id() {
                Ok(pid) => match self.btf.resolve_type_by_id(pid) {
                    Ok(Type::FuncProto(fp)) => fp,
                    _ => return,
                },
                Err(_) => return,
            },
            _ => return,
        };
        // Cap at R5 — BPF ABI passes args 1..=5 in registers. R0 is
        // the return slot, R6..R9 are callee-saved, and R10 is the
        // read-only frame pointer; the kernel verifier rejects
        // programs that try to pass more than five register-args,
        // so anything beyond that index is dead BTF.
        for (i, param) in proto.parameters.iter().enumerate().take(5) {
            // Variadic sentinel (name_off=0, type=0) terminates the
            // parameter list — `break` rather than `continue` because
            // any subsequent parameter slot is unreachable in the
            // BPF calling convention (the variadic marker is the
            // proto's logical end). A `continue` here would let a
            // parameter following the variadic sentinel reseed a
            // later register, which contradicts the proto's intent.
            if param.is_variadic() {
                break;
            }
            let Ok(tid) = param.get_type_id() else {
                continue;
            };
            // Only `Ptr -> ... -> Struct/Union` parameters become
            // typed registers; scalars and function pointers are
            // not pointer-like for our purposes.
            // `bpf_map::resolve_to_struct_id` walks Ptr / Const /
            // Typedef / TypeTag / DeclTag / Volatile / Restrict —
            // exactly what kernel param decls use.
            if let Some(sid) = super::bpf_map::resolve_to_struct_id(self.btf, tid) {
                let reg_idx = i + 1; // param 0 -> R1, …, param 4 -> R5.
                self.regs[reg_idx] = RegState::Pointer {
                    struct_type_id: sid,
                };
                self.note_type_id(sid);
            }
        }
    }

    fn run(
        &mut self,
        insns: &[BpfInsn],
        jump_targets: &BTreeSet<usize>,
        func_entries: &[FuncEntry],
        datasec_pointers: &[DatasecPointer],
        subprog_returns: &[SubprogReturn],
    ) {
        // BPF_LD_IMM64 is a two-insn pseudo-instruction. The decoder
        // skips its second slot via this flag.
        let mut skip_next = false;

        // Pre-build (insn_offset -> proto_id list) so per-PC lookup
        // is O(1). Multiple FuncEntry records for the same PC are
        // preserved in input order — `seed_from_func_proto` is
        // idempotent enough that running them in sequence has the
        // same effect as a single seed (the last one's parameters
        // overwrite the earlier ones, matching BPF ABI's
        // "last declaration wins" semantics for duplicate
        // FuncProtos).
        let mut entries_by_pc: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::with_capacity(func_entries.len());
        for fe in func_entries {
            entries_by_pc
                .entry(fe.insn_offset)
                .or_default()
                .push(fe.func_proto_id);
        }

        // Pre-build (insn_offset -> (datasec_type_id, base_offset)) so
        // the `BPF_LD_IMM64` arm can apply the annotation in O(1).
        // Duplicates at the same PC keep the last entry's payload —
        // mirrors `entries_by_pc`'s "last write wins" semantics. A
        // genuine collision (two distinct datasecs claiming the same
        // PC) cannot happen with valid input: each LD_IMM64 has at
        // most one relocation, and one relocation resolves to one
        // section symbol. A duplicate annotation entry produced by a
        // caller bug is fail-soft: the analyzer types the register
        // from the last entry and proceeds.
        let mut datasec_by_pc: std::collections::HashMap<usize, (u32, u32)> =
            std::collections::HashMap::with_capacity(datasec_pointers.len());
        for dp in datasec_pointers {
            datasec_by_pc.insert(dp.insn_offset, (dp.datasec_type_id, dp.base_offset));
        }

        // Pre-build the subprog-return seed set so the `BPF_OP_CALL`
        // arm can decide whether to seed R0 to
        // [`RegState::ArenaU64FromAlloc`] in O(1). Duplicates collapse
        // (calling the same allocator at the same PC twice would be
        // physically impossible — one PC is one instruction).
        let mut subprog_returns_by_pc: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(subprog_returns.len());
        for sr in subprog_returns {
            subprog_returns_by_pc.insert(sr.insn_offset);
        }

        for (pc, insn) in insns.iter().enumerate() {
            // Jump-target reset fires BEFORE skip_next so a JMP
            // that lands mid-LD_IMM64 (malformed but parseable)
            // still clears stale state. Without this ordering,
            // pre-jump register state would survive past the
            // skip into the next valid instruction.
            if jump_targets.contains(&pc) {
                self.regs = [RegState::Unknown; 11];
                self.stack_slots.clear();
            }

            if skip_next {
                skip_next = false;
                continue;
            }

            // Function-entry reseeding runs after the jump-target
            // reset (so a func entry that is also a jump target
            // still gets its parameter types restored) and before
            // step() executes the instruction. Multiple matching
            // entries at the same PC are processed in order — the
            // last one wins, matching how the BPF ABI would behave
            // if a duplicate FuncProto were declared.
            if let Some(protos) = entries_by_pc.get(&pc) {
                for proto_id in protos {
                    self.seed_from_func_proto(*proto_id);
                }
            }

            // The `BPF_LD_IMM64` arm consults this entry to type the
            // destination register as `Pointer{datasec_type_id}` plus
            // the per-variable base offset. Read here (not inside
            // `step`) so the lookup table does not have to be threaded
            // through helper layers; the LD arm checks `datasec_hit`
            // and falls through to the default Unknown when None.
            let datasec_hit = datasec_by_pc.get(&pc).copied();

            // Allocator-return seed: the `BPF_OP_CALL` arm consults
            // this flag and, AFTER the standard R0..=R5 clobber, sets
            // R0 to [`RegState::ArenaU64FromAlloc`] when the PC
            // matches a [`SubprogReturn::insn_offset`]. The subsequent
            // STX of R0 (or any propagated copy) into a typed `u64`
            // field of a `Pointer{P}` parent records `(P, off)` as an
            // Arena cast finding.
            let alloc_seed = subprog_returns_by_pc.contains(&pc);

            self.step(*insn, &mut skip_next, datasec_hit, alloc_seed);

            // Dead-code disambiguation barrier: after an EXIT or
            // unconditional JA/gotol, the NEXT linear PC is
            // unreachable along this control-flow path. If the
            // walker continues processing it (because pc+1 is not
            // itself a jump target — e.g. it is part of an
            // unrelated subprogram concatenated after this one),
            // any RegState/stack_slot we leave in place would leak
            // into that unrelated subprogram's analysis and produce
            // false positives. Reset preemptively unless pc+1 is a
            // jump target (in which case the head-of-loop reset at
            // the next iteration already fires).
            let class = insn.code & 0x07;
            let op = insn.code & 0xf0;
            let unconditional_ja =
                (class == BPF_CLASS_JMP || class == BPF_CLASS_JMP32) && op == 0x00;
            let is_exit = class == BPF_CLASS_JMP && op == BPF_OP_EXIT;
            if (is_exit || unconditional_ja) && !jump_targets.contains(&(pc + 1)) {
                self.regs = [RegState::Unknown; 11];
                self.stack_slots.clear();
            }
        }
    }

    fn step(
        &mut self,
        insn: BpfInsn,
        skip_next: &mut bool,
        datasec_hit: Option<(u32, u32)>,
        alloc_seed: bool,
    ) {
        let class = insn.code & 0x07;
        let dst = insn.dst_reg() as usize;
        let src = insn.src_reg() as usize;

        // BPF reg fields are 4-bit (0..=15) per the BpfInsn
        // encoding, but only 0..=10 are valid registers. A
        // malformed instruction stream could carry 11..=15;
        // reject early so subsequent direct array indexing of
        // self.regs[dst] / self.regs[src] cannot panic. Mirrors
        // the bounds gate in `set_reg()`.
        if dst >= self.regs.len() || src >= self.regs.len() {
            return;
        }

        match class {
            BPF_CLASS_LDX => {
                let mode = insn.code & 0xe0;
                let size = insn.code & 0x18;
                // BPF_MEM (0x60) is the plain "load size bytes" mode.
                // BPF_MEMSX is a sign-extended load — does not
                // produce a u64 we care about. BPF_ATOMIC stores
                // through dst, not load. BPF_PROBE_MEM (0x20) and
                // friends are post-verifier markers (see linux
                // include/linux/filter.h) that never appear in
                // pre-verification bytecode the analyzer consumes;
                // treating them as Unknown is the safe direction.
                if mode != BPF_MODE_MEM {
                    self.set_reg(dst, RegState::Unknown);
                    return;
                }
                self.handle_ldx(dst, src, size, insn.off as i32);
            }
            BPF_CLASS_STX => {
                let mode = insn.code & 0xe0;
                let size = insn.code & 0x18;
                // BPF_MEM (0x60): plain store, the spill / kptr
                // path. BPF_ATOMIC (0xc0): read-modify-write — XCHG
                // and CMPXCHG overwrite a register with the OLD
                // memory value, which we model by clobbering the
                // affected register so a stale typed-pointer state
                // does not survive into the post-atomic flow. We do
                // NOT record a kptr finding for atomic ops: their
                // store semantics differ (XCHG returns the prior
                // value into src; CMPXCHG conditionally writes), so
                // attributing a `Pointer{T}` source to the slot is
                // unsafe. BPF_PROBE_* mode bits are post-verifier
                // markers and never appear in pre-verification
                // bytecode (see linux include/linux/filter.h).
                if mode == BPF_MODE_ATOMIC {
                    self.handle_atomic(dst, src, insn.imm, insn.off);
                    return;
                }
                if mode != BPF_MODE_MEM {
                    return;
                }
                self.handle_stx(dst, src, size, insn.off);
            }
            BPF_CLASS_ST => {
                // BPF_ST writes an immediate to memory. The constant
                // is never a typed pointer — but the store may still
                // alias a stack slot we are tracking through STX.
                // Invalidate the slot (write of an immediate
                // overwrites whatever typed value used to live
                // there) so a later LDX r10-relative does not
                // resurrect a stale Pointer state.
                let mode = insn.code & 0xe0;
                if mode == BPF_MODE_MEM && dst == BPF_REG_R10 {
                    self.stack_slots.remove(&insn.off);
                }
            }
            BPF_CLASS_LD => {
                // BPF_LD_IMM64: BPF_LD | BPF_DW | BPF_IMM (0x18).
                // Two-slot instruction: the next BpfInsn carries
                // the upper 32 bits of the 64-bit immediate. The
                // first slot's `imm` is one of: a literal 64-bit
                // constant (src_reg == 0), a map fd
                // (src_reg == BPF_PSEUDO_MAP_FD), or a pointer to
                // a map's value memory at a known offset
                // (src_reg == BPF_PSEUDO_MAP_VALUE) — the case
                // this arm types as `DatasecPointer`. See linux
                // uapi `bpf.h` `BPF_PSEUDO_MAP_*` and `kernel/bpf/
                // syscall.c bpf_check`.
                if insn.code == (BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM) {
                    // BPF_PSEUDO_MAP_VALUE branch: the loaded
                    // value is a pointer into a map's value memory
                    // at a known byte offset. The host-side cast
                    // loader passes a per-PC `(datasec_type_id,
                    // base_offset)` annotation when the map is a
                    // global section (`.bss`/`.data`/`.rodata`/
                    // `.data.<name>`); the LD_IMM64's `imm` field
                    // already carries the per-variable byte offset
                    // (the relocation entry against the section
                    // symbol contributes 0 to the addend, so the
                    // raw imm IS the offset). Type the destination
                    // register as `DatasecPointer` so subsequent
                    // STX/LDX through it can resolve to a specific
                    // global variable via `struct_member_at` over
                    // the datasec's `VarSecinfo` entries.
                    //
                    // The `src_reg == BPF_PSEUDO_MAP_VALUE` gate
                    // matches POST-relocation bytecode — what
                    // libbpf produces in-kernel before the
                    // verifier runs. The host-side cast loader
                    // sees PRE-relocation bytecode (the embedded
                    // `.bpf.objs` blob is the raw `.bpf.o`), where
                    // src_reg == 0 even for map_value references.
                    // The loader carries the relocation evidence
                    // via `datasec_hit` instead, so this arm fires
                    // on either a real PSEUDO_MAP_VALUE src_reg
                    // OR a caller-supplied datasec annotation —
                    // whichever path the input provides.
                    if let Some((datasec_type_id, base_offset)) = datasec_hit {
                        self.set_reg(
                            dst,
                            RegState::DatasecPointer {
                                datasec_type_id,
                                base_offset,
                            },
                        );
                        self.note_type_id(datasec_type_id);
                    } else {
                        // Every other LD_IMM64 shape collapses to
                        // Unknown for the destination register:
                        //   - `src_reg == BPF_PSEUDO_MAP_VALUE`
                        //     without a caller annotation: post-
                        //     relocation bytecode whose map_fd
                        //     alone does not identify a datasec
                        //     (the mapping lives in the loader's
                        //     `.maps` parser). Drop dst — false
                        //     negative is the safe direction.
                        //   - plain LD_IMM64 (literal constant,
                        //     map_fd, BTF id, …): the destination
                        //     receives a 64-bit immediate, never
                        //     a typed kernel pointer the renderer
                        //     needs to chase.
                        self.set_reg(dst, RegState::Unknown);
                    }
                    *skip_next = true;
                } else {
                    // Other BPF_LD modes (BPF_ABS, BPF_IND) load
                    // packet data into r0 — not relevant here.
                    self.set_reg(0, RegState::Unknown);
                }
            }
            BPF_CLASS_ALU64 | BPF_CLASS_ALU => {
                let op = insn.code & 0xf0;
                let src_kind = insn.code & 0x08;
                if op == BPF_OP_MOV && src_kind == BPF_SRC_X {
                    // r_dst = r_src — propagate state. Only
                    // ALU64|MOV preserves a 64-bit value verbatim.
                    // 32-bit MOV on a u64 field would truncate the
                    // pointer; treat 32-bit as Unknown to avoid
                    // false positives.
                    if class == BPF_CLASS_ALU64 {
                        // ALU64|MOV|X reuses the `off` field to
                        // encode sign-extending MOV (off in {8, 16,
                        // 32}) and BPF_ADDR_SPACE_CAST (off == 1)
                        // per linux kernel/bpf/verifier.c
                        // check_alu_op. off == 0 is the plain copy.
                        // R10 is the read-only frame pointer —
                        // never a valid MOV/cast destination.
                        // Reject to maintain the invariant that
                        // regs[10] is always Unknown.
                        if dst == BPF_REG_R10 {
                            return;
                        }
                        match insn.off {
                            0 => {
                                self.regs[dst] = self.regs[src];
                            }
                            1 => {
                                // BPF_ADDR_SPACE_CAST. The verifier
                                // (kernel/bpf/verifier.c
                                // check_alu_op) accepts only
                                // imm == 1 (cast as(1) → as(0):
                                // arena → kernel) and
                                // imm == 1u<<16 (kernel → arena).
                                // For arena → kernel, propagate the
                                // source's RegState so subsequent
                                // dereferences attribute correctly.
                                if insn.imm == 1 {
                                    if let RegState::LoadedU64Field {
                                        source_struct_id,
                                        field_offset,
                                    } = self.regs[src]
                                    {
                                        self.arena_confirmed
                                            .insert((source_struct_id, field_offset));
                                    }
                                    self.regs[dst] = self.regs[src];
                                } else {
                                    // as(0) → as(1) (0x10000,
                                    // cast_user) or reserved imm:
                                    // the result is a user-vma
                                    // arena address that cannot be
                                    // a tracked pointer in our model.
                                    self.set_reg(dst, RegState::Unknown);
                                }
                            }
                            8 | 16 | 32 => {
                                // Sign-extending MOV (s8/s16/s32 →
                                // s64). A typed pointer cannot
                                // survive sign extension; drop dst.
                                self.set_reg(dst, RegState::Unknown);
                            }
                            _ => {
                                // Unknown/reserved off encoding —
                                // be conservative.
                                self.set_reg(dst, RegState::Unknown);
                            }
                        }
                    } else {
                        self.set_reg(dst, RegState::Unknown);
                    }
                } else {
                    // All other ALU ops (add/sub/and/or/lsh/etc.,
                    // including BPF_OP_MOV with BPF_SRC_K) destroy
                    // the typed-pointer property. Drop dst state.
                    self.set_reg(dst, RegState::Unknown);
                }
            }
            BPF_CLASS_JMP | BPF_CLASS_JMP32 => {
                let op = insn.code & 0xf0;
                if op == BPF_OP_CALL {
                    // BPF_CALL clobbers r0..=r5 per the BPF ABI:
                    // r1..r5 are call args (consumed), r0 carries
                    // the return value. Save R1 BEFORE the clobber
                    // so the helper-return arm below can resolve
                    // the map descriptor argument for
                    // `bpf_map_lookup_elem`. Once R0..R5 are
                    // cleared, R1's pre-call state is gone — only
                    // the saved snapshot survives across the
                    // clobber boundary.
                    let pre_call_r1 = self.regs[1];
                    for r in 0..=5 {
                        self.set_reg(r, RegState::Unknown);
                    }
                    let pseudo = insn.src_reg();
                    if pseudo == BPF_PSEUDO_KFUNC_CALL {
                        // kfunc calls (src_reg == BPF_PSEUDO_KFUNC_CALL):
                        // the imm field carries the kernel BTF id of
                        // the kfunc — if its FuncProto return type
                        // peels through Ptr to a Struct/Union, set r0
                        // to a typed pointer. Mutually exclusive with
                        // the plain-helper arm below: kfuncs use a
                        // distinct pseudo selector (see linux uapi
                        // `bpf.h`: `BPF_PSEUDO_KFUNC_CALL = 2`).
                        self.handle_kfunc_call(insn.imm);
                    } else if pseudo == 0 && insn.imm == BPF_FUNC_MAP_LOOKUP_ELEM {
                        // Plain-helper arm. `pseudo == 0` is the
                        // helper-call form (linux uapi `bpf.h`:
                        // `BPF_PSEUDO_CALL = 1` is BPF-to-BPF;
                        // `BPF_PSEUDO_KFUNC_CALL = 2` is kfunc; the
                        // verifier treats `src_reg == 0` as a kernel
                        // helper-id call). `imm` is the helper id
                        // (`BPF_FUNC_*`); the analyzer types R0 only
                        // for `bpf_map_lookup_elem` (helper id 1) —
                        // no other helper has a pointer-to-struct
                        // return shape we can resolve from the BPF
                        // program BTF alone. The map descriptor lives
                        // in R1 at the call site (per
                        // `bpf_map_lookup_elem_proto::arg1_type =
                        // ARG_CONST_MAP_PTR` in linux
                        // `kernel/bpf/helpers.c`); the saved
                        // pre-clobber state above carries the
                        // analyzer's pre-call view.
                        //
                        // Only fires when the saved R1 is a
                        // [`RegState::DatasecPointer`] into a
                        // `BTF_KIND_DATASEC` named `.maps` (the libbpf
                        // user-space BTF map declaration section),
                        // and the map's BTF def carries a `value`
                        // member whose type peels to `Ptr -> Struct/
                        // Union`. Stat-counter maps (`__type(value,
                        // u64)`) drop here — their value type is not
                        // a struct so [`map_value_struct_id`]
                        // returns None. False-negative is the safe
                        // direction.
                        if let RegState::DatasecPointer {
                            datasec_type_id,
                            base_offset,
                        } = pre_call_r1
                            && let Some(sid) =
                                map_value_struct_id(self.btf, datasec_type_id, base_offset)
                        {
                            self.regs[0] = RegState::Pointer {
                                struct_type_id: sid,
                            };
                            self.note_type_id(sid);
                        }
                    }
                    // Allocator-return seed: caller-supplied annotation
                    // identified this `BPF_PSEUDO_CALL` PC as a call to
                    // an arena-allocator subprog (see [`SubprogReturn`]).
                    // After the standard R0..=R5 clobber, type R0 as
                    // [`RegState::ArenaU64FromAlloc`] so the next
                    // STX of R0 (or its propagation through MOV /
                    // stack spill / LDX of an already-arena-tagged
                    // slot) records `(parent, off)` as an Arena cast
                    // finding via [`Self::handle_stx`]. The seed is
                    // applied AFTER the clobber so a same-PC kfunc-
                    // call seed (which sets R0 to a typed `Pointer{T}`)
                    // wins on the rare programs where both annotations
                    // resolve to the same call site — kfunc returns
                    // are stronger evidence than the allocator
                    // allowlist.
                    if alloc_seed && matches!(self.regs[0], RegState::Unknown) {
                        self.regs[0] = RegState::ArenaU64FromAlloc;
                        // F4 telemetry: bump the seed-applied
                        // counter so [`Self::finalize`] can
                        // distinguish "we saw allocator call
                        // sites but no slot got tagged" (the
                        // non-inlined-helper signature) from
                        // "no allocator was ever called". A
                        // saturating add keeps the count bounded
                        // for pathological inputs that loop a
                        // call site (the verifier rejects such
                        // programs but the analyzer must not
                        // panic on them).
                        self.alloc_seeds_applied = self.alloc_seeds_applied.saturating_add(1);
                    }
                }
                // EXIT, JA, conditional jumps: no state change at
                // the current PC. Branch / fall-through joins are
                // handled by the jump-target reset at the head of
                // each PC the pre-pass flagged.
            }
            _ => {
                // Unknown class — drop dst conservatively.
                self.set_reg(dst, RegState::Unknown);
            }
        }
    }

    fn handle_ldx(&mut self, dst: usize, src: usize, size: u8, off: i32) {
        // Bounds: BPF reg fields are 4-bit (0..=15) but only 0..=10
        // are real registers. A malformed instruction stream could
        // carry 11..=15 here; reject before the direct
        // self.regs[src] read on the typed-base path. Mirrors the
        // gate in `step()` but defends `handle_ldx` independently
        // in case it is called from a future caller.
        if dst >= self.regs.len() || src >= self.regs.len() {
            return;
        }
        // R10 is the read-only frame pointer — loading INTO r10
        // violates the BPF ABI (verifier rejects). Guard to
        // maintain the invariant that regs[10] is always Unknown,
        // matching the MOV guard in step().
        if dst == BPF_REG_R10 {
            return;
        }
        let size_bytes = ldx_size_bytes(size);

        // Stack reload: r_dst = *(size *)(r10 + off). Restore the
        // slot's saved RegState. Only BPF_DW reloads carry pointer
        // values intact; sub-word reloads truncate so we mark
        // Unknown. The frame-pointer base is r10; src register here
        // is the address-base, dst is the value receiver. Negative
        // off identifies a spill slot (stack grows down); a
        // non-negative off through r10 is undefined behavior in
        // BPF — drop conservatively rather than guess.
        if src == BPF_REG_R10 {
            if size != BPF_SIZE_DW || off >= 0 {
                self.set_reg(dst, RegState::Unknown);
                return;
            }
            // i32 -> i16: BpfInsn::off is i16 to begin with, the
            // caller widens it to i32. Round-trip to the slot key
            // type. Out-of-range is impossible because the source
            // is i16, but guard anyway.
            let Ok(slot_off) = i16::try_from(off) else {
                self.set_reg(dst, RegState::Unknown);
                return;
            };
            let restored = self.stack_slots.get(&slot_off).copied();
            self.set_reg(dst, restored.unwrap_or(RegState::Unknown));
            return;
        }

        // Compute (parent_btf_id, base_offset) for the load
        // target. Two RegState variants reach the typed-LDX path:
        // `Pointer{struct}` (the kernel-driver case) and
        // `DatasecPointer{datasec, base}` (the BSS / data global
        // case where the LD_IMM64's baked-in offset contributes
        // to the effective field offset). Both share the same
        // member-resolution + u64/Ptr-detection shape; only the
        // parent BTF id selects which layout `struct_member_at`
        // walks.
        let typed_base: Option<(u32, u32)> = match self.regs[src] {
            RegState::Pointer { struct_type_id } => Some((struct_type_id, 0)),
            RegState::DatasecPointer {
                datasec_type_id,
                base_offset,
            } => Some((datasec_type_id, base_offset)),
            _ => None,
        };
        if let Some((parent_btf_id, base_offset)) = typed_base {
            let insn_off = match field_byte_offset(off) {
                Some(o) => o,
                None => {
                    self.set_reg(dst, RegState::Unknown);
                    return;
                }
            };
            let Some(field_off) = base_offset.checked_add(insn_off) else {
                // Overflow on pathological large base + insn off:
                // drop conservatively. False negative is safe.
                self.set_reg(dst, RegState::Unknown);
                return;
            };
            if let Some(member) = struct_member_at(self.btf, parent_btf_id, field_off) {
                let member_type_id = member.member_type_id();
                let resolved = super::btf_render::peel_modifiers(self.btf, member_type_id);
                // Canonical key for the pattern map: for Datasec
                // members, key on the variable's start offset so
                // the renderer's `(parent, member_offset)` lookup
                // matches `VarSecinfo` boundaries. For struct
                // members, the queried offset IS the member start.
                let (canonical_parent, canonical_field_off) = match &member {
                    MemberAt::Struct {
                        resolved_parent_type_id,
                        resolved_member_offset,
                        ..
                    } => (*resolved_parent_type_id, *resolved_member_offset),
                    MemberAt::Datasec {
                        var_byte_offset, ..
                    } => (parent_btf_id, *var_byte_offset),
                };
                match (size_bytes, resolved) {
                    // Ptr field directly -- BTF already typed.
                    (Some(8), Some(Type::Ptr(p))) => {
                        if let Ok(pointee) = p.get_type_id()
                            && let Some(sid) =
                                super::bpf_map::resolve_to_struct_id(self.btf, pointee)
                        {
                            self.set_reg(
                                dst,
                                RegState::Pointer {
                                    struct_type_id: sid,
                                },
                            );
                            self.note_type_id(sid);
                            return;
                        }
                        self.set_reg(dst, RegState::Unknown);
                    }
                    // Plain u64 field -- THIS is the cast target.
                    (Some(8), Some(Type::Int(int))) => {
                        if int.size() == 8 && !int.is_signed() && !int.is_bool() && !int.is_char() {
                            // Alias-set tracking: when LDX reads from
                            // a `(parent, off)` slot the analyzer
                            // previously tagged via the STX-flow
                            // arena path (see
                            // [`Self::arena_stx_findings`]), the
                            // loaded value is itself an arena VA. Set
                            // the destination state to
                            // [`RegState::ArenaU64FromAlloc`] so the
                            // tag propagates through subsequent
                            // moves / spills / stores. Falls through
                            // to the generic `LoadedU64Field` shape
                            // when the slot has not been arena-tagged
                            // yet — the first STX that tags the slot
                            // populates the index, after which later
                            // LDXs through the same slot inherit.
                            //
                            // Using [`BTreeMap::contains_key`]
                            // without inspecting the
                            // [`ArenaStxEntry`] variant is
                            // intentional: any entry — `Pending`
                            // or (today unreachable) `Conflicting`
                            // — proves the slot saw an arena STX
                            // somewhere in the program, which is
                            // the only signal alias-tracking
                            // needs. A future `Conflicting` would
                            // still be arena-shaped (the conflict
                            // would be across paths that all wrote
                            // an arena pointer); finalize would
                            // drop the slot from the cast map but
                            // the LDX value loaded out of it is
                            // still an arena VA.
                            let dst_state = if self
                                .arena_stx_findings
                                .contains_key(&(canonical_parent, canonical_field_off))
                            {
                                RegState::ArenaU64FromAlloc
                            } else {
                                RegState::LoadedU64Field {
                                    source_struct_id: canonical_parent,
                                    field_offset: canonical_field_off,
                                }
                            };
                            self.set_reg(dst, dst_state);
                            self.note_type_id(canonical_parent);
                            self.patterns
                                .entry((canonical_parent, canonical_field_off))
                                .or_default();
                        } else {
                            self.set_reg(dst, RegState::Unknown);
                        }
                    }
                    _ => {
                        // Other field shapes (sub-u64 ints,
                        // structs, unions, enums, arrays, floats,
                        // FuncProto) cannot become a pointer by
                        // load alone. Drop dst.
                        self.set_reg(dst, RegState::Unknown);
                    }
                }
            } else {
                self.set_reg(dst, RegState::Unknown);
            }
            return;
        }
        match self.regs[src] {
            RegState::LoadedU64Field {
                source_struct_id,
                field_offset,
            } => {
                // The interesting case: the loaded u64 is being used
                // as a pointer base. Record the access and mark dst
                // Unknown -- we don't know the resolved target yet
                // (that's the matching phase's job).
                let target_off = match field_byte_offset(off) {
                    Some(o) => o,
                    None => {
                        self.set_reg(dst, RegState::Unknown);
                        return;
                    }
                };
                if let Some(sz) = size_bytes {
                    self.patterns
                        .entry((source_struct_id, field_offset))
                        .or_default()
                        .insert(Access {
                            offset: target_off,
                            size: sz,
                        });
                }
                self.set_reg(dst, RegState::Unknown);
            }
            RegState::ArenaU64FromAlloc => {
                // LDX through an arena pointer reads payload bytes
                // out of an allocator slot. The slot identity is
                // already recorded in
                // [`Self::arena_stx_findings`] via the STX path; the
                // pointer's destination (parent, off) is what
                // matters, not the arbitrary dereference offset.
                // Drop dst to Unknown — we do not run shape inference
                // on dereferences through arena-tagged pointers.
                self.set_reg(dst, RegState::Unknown);
            }
            RegState::Unknown => {
                // Loading through an untyped base never produces a
                // tracked register (we don't speculate the source
                // type from BTF alone).
                self.set_reg(dst, RegState::Unknown);
            }
            // Typed-pointer variants were handled above; the
            // `typed_base` arm always returns to avoid falling
            // through.
            RegState::Pointer { .. } | RegState::DatasecPointer { .. } => unreachable!(),
        }
    }

    /// `STX [r_dst_base + off] = r_src_value`.
    ///
    /// Three roles:
    /// 1. Stack spill — `dst == r10`: save src's RegState in
    ///    `stack_slots[off]`. Sub-DW or non-negative-off stores
    ///    invalidate the slot.
    /// 2. Kptr finding — when both base and value registers are
    ///    typed (Pointer{P} for the base, Pointer{T} for the
    ///    value), and the BTF declares the field of P at the store
    ///    offset as a plain `u64` of width 8, record (P, off) -> T
    ///    in the kptr map. The BTF gate prevents writing to a
    ///    pre-typed Ptr field (the kernel-driver case where BTF
    ///    already knows the target).
    /// 3. Arena STX finding — when the base is typed
    ///    `Pointer{P}` / `DatasecPointer` and the value register
    ///    is [`RegState::ArenaU64FromAlloc`] (allocator-return
    ///    seed or alias-tracked from a previously-arena-tagged
    ///    slot), and the BTF declares the field at the store
    ///    offset as a plain `u64`, record `(P, off)` in
    ///    [`Self::arena_stx_findings`]. The slot now holds an
    ///    arena pointer, even though BTF declared it `u64` — the
    ///    renderer's [`MemReader::resolve_arena_type`] bridge
    ///    resolves the payload type at chase time.
    fn handle_stx(&mut self, dst: usize, src: usize, size: u8, off: i16) {
        // Bounds: BPF reg indices are 0..=10. The decoded fields are
        // 4-bit (0..=15) so a malformed instruction stream could put
        // 11..=15 here; reject without indexing into the array.
        if dst >= self.regs.len() || src >= self.regs.len() {
            return;
        }
        // Spill path runs first — even for Unknown source values,
        // a store through r10 invalidates the slot (the slot now
        // holds an Unknown value rather than its prior typed
        // content). Without invalidation a subsequent reload could
        // resurrect a stale Pointer state and produce a false
        // positive.
        if dst == BPF_REG_R10 {
            if size != BPF_SIZE_DW || off >= 0 {
                // Sub-DW or out-of-spec store: invalidate any
                // existing slot so a later reload can't pick up
                // stale state. The truncating write to a slot that
                // formerly held a typed pointer cannot preserve
                // the pointer.
                self.stack_slots.remove(&off);
                return;
            }
            // Save the source register's state verbatim. Unknown
            // source means an Unknown saved state, which on reload
            // gives an Unknown register. Pointer / LoadedU64Field
            // round-trip preserved.
            self.stack_slots.insert(off, self.regs[src]);
            return;
        }

        // Kptr path: only DW (8-byte) stores can persist a 64-bit
        // pointer. Sub-DW stores are not pointer-valued.
        if size != BPF_SIZE_DW {
            return;
        }
        // Compute the (parent_btf_id, field_byte_offset) for the
        // store target. Two RegState variants reach the kptr path:
        // `Pointer{struct}` (the kernel-driver case where the
        // parent is a struct or union) and
        // `DatasecPointer{datasec, base}` (the BSS / data global
        // case where the parent is a `BTF_KIND_DATASEC` and the
        // base offset baked into the LD_IMM64 contributes to the
        // effective field offset). In both cases the field offset
        // is `(base) + insn.off`; the parent BTF id selects which
        // layout to consult via `struct_member_at`.
        let (parent_btf_id, base_offset) = match self.regs[dst] {
            RegState::Pointer {
                struct_type_id: pid,
            } => (pid, 0u32),
            RegState::DatasecPointer {
                datasec_type_id,
                base_offset,
            } => (datasec_type_id, base_offset),
            _ => return,
        };
        // Two value-side variants reach the cast-finding paths:
        // `Pointer{T}` (kernel kptr STX into a u64 field) and
        // `ArenaU64FromAlloc` (arena pointer from allocator return,
        // or alias-tracked from a previously-arena-tagged slot).
        // Anything else carries no signal.
        let value_state = match self.regs[src] {
            RegState::Pointer {
                struct_type_id: tid,
            } => StxValueKind::Kptr { target: tid },
            RegState::ArenaU64FromAlloc => StxValueKind::Arena,
            _ => return,
        };
        let Some(insn_off) = field_byte_offset(off as i32) else {
            return;
        };
        let Some(field_off) = base_offset.checked_add(insn_off) else {
            // Pathological large base_offset + insn_off overflow:
            // drop conservatively. False negative is the safe
            // direction; a real BPF program never legitimately
            // produces an offset past `u32::MAX`.
            return;
        };
        // BTF gate: the destination field at this offset must be a
        // plain `u64`. A typed Ptr field is the BTF-already-typed
        // case the renderer handles natively; recording a cast
        // there would duplicate work. A non-u64 field (sub-u64
        // int, struct, array) is not a pointer slot at all — the
        // store is undefined behavior we drop conservatively.
        let Some(member) = struct_member_at(self.btf, parent_btf_id, field_off) else {
            return;
        };
        let member_type_id = member.member_type_id();
        let Some(terminal) = super::btf_render::peel_modifiers(self.btf, member_type_id) else {
            return;
        };
        let Type::Int(int) = terminal else { return };
        if int.size() != 8 || int.is_signed() || int.is_bool() || int.is_char() {
            return;
        }
        // The Datasec path stores the variable's start offset
        // (matching `MemberAt::Datasec::var_byte_offset`) as the
        // canonical key, NOT the queried offset. For a plain u64
        // global the two are equal; for a struct global the
        // queried offset can land mid-struct but the cast finding
        // is keyed on the variable's start so the renderer's
        // `(parent, member_offset)` lookup matches the variable
        // boundary. Lookups through the BSS-DATASEC parent then
        // surface the per-variable kptr / arena finding just like
        // a struct member would.
        let (canonical_parent, canonical_field_off) = match &member {
            MemberAt::Struct {
                resolved_parent_type_id,
                resolved_member_offset,
                ..
            } => (*resolved_parent_type_id, *resolved_member_offset),
            MemberAt::Datasec {
                var_byte_offset, ..
            } => (parent_btf_id, *var_byte_offset),
        };
        self.note_type_id(canonical_parent);
        let key = (canonical_parent, canonical_field_off);
        match value_state {
            StxValueKind::Kptr { target } => {
                // Self-store is almost always a structural error
                // (the analyzer concluded `parent == target`
                // because of ambiguous pointer aliasing); reject
                // to keep the false-positive bar high. The Datasec
                // parent path cannot self-store: a datasec id is
                // never the target struct id of a kptr (kptrs
                // target slab structs like task_struct), so this
                // gate fires only on the `Pointer{struct}` case in
                // practice. The unconditional check is the
                // simplest safe form.
                if canonical_parent == target {
                    return;
                }
                self.note_type_id(target);
                match self.kptr_findings.get(&key).copied() {
                    None => {
                        self.kptr_findings.insert(key, KptrEntry::Single(target));
                    }
                    Some(KptrEntry::Single(prev)) if prev == target => {
                        // Same target observed again — keep Single.
                    }
                    Some(_) => {
                        // Different target previously observed at
                        // the same slot, or already collapsed to
                        // Conflicting. The slot is ambiguous;
                        // drop it on finalize.
                        self.kptr_findings.insert(key, KptrEntry::Conflicting);
                    }
                }
            }
            StxValueKind::Arena => {
                // Allocator-return / alias-tracked arena pointer
                // stored into a u64 slot. Record the slot in
                // [`Self::arena_stx_findings`] so finalize emits
                // an Arena cast hit. The target type id is
                // unresolved at analysis time — the renderer's
                // [`super::btf_render::MemReader::resolve_arena_type`]
                // bridge supplies the real payload BTF id at chase
                // time, so the analyzer just records that the slot
                // saw an arena STX via [`ArenaStxEntry::Pending`].
                //
                // Two STX writes to the same slot of the same shape
                // (arena STX after another arena STX) are not a
                // conflict — both observations agree the slot
                // holds an arena pointer. The `Some(Pending)` arm
                // is the dedup no-op.
                //
                // A prior `Pointer{T}` STX into the same slot has
                // already populated `kptr_findings`; the conflict
                // detector in [`Self::finalize`] cross-references
                // both maps and drops the slot from BOTH sides so
                // the resulting CastMap excludes it. That cross-
                // path conflict is detected at finalize, NOT here:
                // `arena_stx_findings` and `kptr_findings` are
                // disjoint maps and this arm only sees prior arena
                // STX state.
                match self.arena_stx_findings.get(&key).copied() {
                    None => {
                        self.arena_stx_findings.insert(key, ArenaStxEntry::Pending);
                    }
                    Some(ArenaStxEntry::Pending) => {
                        // Same arena observation — no-op dedup.
                    }
                    Some(ArenaStxEntry::Conflicting) => {
                        // Unreachable: the only insertion site for
                        // `arena_stx_findings` is THIS arm, and
                        // this arm only inserts
                        // `ArenaStxEntry::Pending`. The
                        // `Conflicting` variant exists for
                        // symmetry with [`KptrEntry::Conflicting`]
                        // and as a defensive landing pad if a
                        // future code path adds disagreement
                        // detection inside the arena STX flow.
                        // Until then, reaching this arm signals a
                        // logic error in the analyzer's insertion
                        // discipline; panicking surfaces it
                        // instead of silently re-inserting and
                        // masking the bug.
                        unreachable!(
                            "arena_stx_findings cannot hold Conflicting: \
                             only the StxValueKind::Arena arm inserts, \
                             and it only inserts Pending"
                        );
                    }
                }
            }
        }
    }

    /// `BPF_STX | BPF_<size> | BPF_ATOMIC` (mode == 0xc0).
    ///
    /// Atomic memory ops carry the specific operation in `imm` (see
    /// linux uapi `bpf.h`). The dispatch is driven by the
    /// `BPF_FETCH` (0x01) bit in `imm`:
    ///
    /// FETCH variants — write a register with the prior memory
    /// value, which the analyzer cannot type:
    /// - `BPF_CMPXCHG = 0xf0 | BPF_FETCH`: `r0 = atomic_cmpxchg(...)`.
    ///   R0 is overwritten with the old memory value — drop R0.
    /// - `BPF_XCHG = 0xe0 | BPF_FETCH`: `src_reg = atomic_xchg(...)`.
    /// - Arithmetic-FETCH (BPF_ADD/AND/OR/XOR with BPF_FETCH set):
    ///   `src_reg` ends up with the prior arithmetic value.
    ///
    /// All FETCH variants drop `src_reg` to Unknown (false negative
    /// on CMPXCHG, which only writes R0 — see `handle_atomic` body).
    ///
    /// Non-FETCH variants (plain BPF_ADD/AND/OR/XOR) read-modify-
    /// write memory but leave every register intact.
    ///
    /// `BPF_LOAD_ACQ = 0x100`: `dst = smp_load_acquire(*src + off)`.
    /// dst receives a memory value the analyzer cannot type — drop
    /// dst. `BPF_STORE_REL = 0x110`: `smp_store_release(*dst + off,
    /// src)` — no register effect; dst is the address base, src is
    /// the value. See linux include/linux/filter.h for the
    /// authoritative semantics.
    ///
    /// Stack-slot invalidation: when `dst == BPF_REG_R10` the atomic
    /// targets a stack slot. Any prior typed RegState parked in
    /// `stack_slots[off]` cannot survive an atomic write — drop the
    /// slot before per-register clobber logic so a later LDX through
    /// r10 cannot resurrect a stale Pointer state. Mirrors the
    /// invalidation in [`Self::handle_stx`] for plain stores and the
    /// `BPF_CLASS_ST` arm in `step()` for immediate stores.
    ///
    /// No kptr finding is recorded on any atomic store. The kptr
    /// path requires the store to publish the value verbatim into
    /// the slot (the kernel's `bpf_kptr_xchg` helper is the proper
    /// kptr-write path); atomic XCHG/CMPXCHG semantics differ
    /// enough that attributing `Pointer{T}` source to the slot is
    /// unsafe.
    fn handle_atomic(&mut self, dst: usize, src: usize, imm: i32, off: i16) {
        // dst / src already bounds-checked at the top of step().
        // Avoid panicking even if a future caller forgets.
        if dst >= self.regs.len() || src >= self.regs.len() {
            return;
        }

        // Stack-slot invalidation: an atomic STORE through r10
        // overwrites the slot's prior content. LOAD_ACQ is the lone
        // exception: dst is the value receiver, NOT the address
        // base (`*src + off` is the address), so dst==r10 on
        // LOAD_ACQ does not write through to the stack — it merely
        // attempts to load INTO r10 (which the verifier rejects)
        // and produces no slot mutation. Skip invalidation in that
        // case so an unrelated stack slot at the same `off` keeps
        // its tracked state.
        if dst == BPF_REG_R10 && imm != BPF_LOAD_ACQ_IMM {
            self.stack_slots.remove(&off);
        }

        // BPF_LOAD_ACQ (0x100): dst register receives memory value;
        // src is the address base. Clobber dst to Unknown — the
        // analyzer does not type loaded values via atomic mode.
        // BPF_STORE_REL (0x110): dst is the address base, src is the
        // value being stored. Stack invalidation above already
        // handles the spill case; no per-register clobber here.
        if imm == BPF_LOAD_ACQ_IMM {
            self.set_reg(dst, RegState::Unknown);
            return;
        }
        if imm == BPF_STORE_REL_IMM {
            return;
        }

        let top = imm & 0xf0;
        let has_fetch = (imm & BPF_FETCH) != 0;

        // BPF_CMPXCHG = 0xf0 | BPF_FETCH = 0xf1. R0 receives the old
        // memory value regardless of whether the compare succeeded.
        if top == BPF_CMPXCHG_TOP && has_fetch {
            self.set_reg(0, RegState::Unknown);
        }
        // BPF_XCHG = 0xe0 | BPF_FETCH = 0xe1. src_reg receives the
        // old memory value. Same direction for any other FETCH
        // variant (BPF_ADD/AND/OR/XOR with BPF_FETCH bit set):
        // src_reg ends up holding a value the analyzer cannot
        // type, so drop it to Unknown.
        //
        // CMPXCHG (0xf1) only writes R0; we conservatively clobber
        // src on all FETCH variants including CMPXCHG — false
        // negative, acceptable.
        if has_fetch {
            self.set_reg(src, RegState::Unknown);
        }
        // Non-fetch atomic ops (plain BPF_ADD/AND/OR/XOR) do not
        // overwrite any register — leave RegState alone.
    }

    /// `BPF_CALL` with `src_reg == BPF_PSEUDO_KFUNC_CALL`.
    ///
    /// `imm` is the BTF id of a `Type::Func` (peeled one level to
    /// its FuncProto) or a `Type::FuncProto` directly. Peel the
    /// return type through Ptr -> Struct and set R0 if the chain
    /// succeeds. Negative or zero `imm` indicates either a
    /// pre-relocation kfunc placeholder (real on-disk `.bpf.o`
    /// files typically have `imm = -1` for kfunc calls before
    /// libbpf resolves the kernel BTF id) or a non-kfunc call;
    /// drop silently in both cases.
    ///
    /// Two distinct R0 typings happen here:
    ///
    /// 1. **Typed-pointer return** (`Ptr -> Struct/Union`): the
    ///    kfunc returns a typed kernel pointer (e.g.
    ///    `bpf_task_acquire`, `bpf_cpumask_first`). R0 becomes
    ///    [`RegState::Pointer`] so the next STX of R0 into a u64
    ///    slot of a typed parent records a kernel kptr finding.
    ///
    /// 2. **Arena-allocator return** (`Ptr -> Void`, allowlisted
    ///    name): the kfunc allocates arena memory and returns a
    ///    raw `void *` whose runtime value is an arena VA (e.g.
    ///    `bpf_arena_alloc_pages`). The Ptr->Void return is
    ///    structurally indistinguishable from a typed pointer at
    ///    the BTF level — neither side carries a `__arena`
    ///    qualifier in the kernel's program-BTF representation —
    ///    so the disambiguator is the kfunc's name. R0 becomes
    ///    [`RegState::ArenaU64FromAlloc`] so the next STX of R0
    ///    into a u64 slot of a typed parent records an arena
    ///    finding via `arena_stx_findings`. Arms 1 and 2 are
    ///    mutually exclusive: arm 1 only fires when the return
    ///    peels to a Struct/Union; arm 2 only fires when the
    ///    return peels to Void AND the name is on the allowlist.
    ///    A kfunc whose name is on the allowlist but whose
    ///    return is NOT Ptr->Void (BTF mismatch — drift between
    ///    kernel source and analyzer's allowlist) drops to no
    ///    typing rather than misclassifying R0.
    fn handle_kfunc_call(&mut self, imm: i32) {
        if imm <= 0 {
            return;
        }
        let func_btf_id = imm as u32;
        // Resolve the kfunc's FuncProto AND retain a handle on the
        // `Type::Func` so we can resolve its name for the
        // allocator-allowlist arm. The two-arm dispatch needs both
        // pieces of evidence (return-type shape + name), so the
        // resolution is unified here rather than running twice.
        let (proto, func_name) = match self.btf.resolve_type_by_id(func_btf_id) {
            Ok(Type::Func(f)) => match f.get_type_id() {
                Ok(pid) => match self.btf.resolve_type_by_id(pid) {
                    Ok(Type::FuncProto(fp)) => {
                        let name = self.btf.resolve_name(&f).ok();
                        (fp, name)
                    }
                    _ => return,
                },
                Err(_) => return,
            },
            Ok(Type::FuncProto(fp)) => (fp, None),
            _ => return,
        };
        let ret_id = proto.return_type_id();
        if ret_id == 0 {
            // Void return at the FuncProto level (return_type_id
            // == 0 marks `void` in BTF). R0 stays Unknown — no
            // arena allocator declares this shape (allocators
            // return `void *`, not `void`).
            return;
        }
        // Arm 1: typed-pointer return.
        if let Some(sid) = super::bpf_map::resolve_to_struct_id(self.btf, ret_id) {
            self.regs[0] = RegState::Pointer {
                struct_type_id: sid,
            };
            self.note_type_id(sid);
            return;
        }
        // Arm 2: arena-allocator return. The allowlist lookup
        // is gated on `Ptr -> Void` to keep the false-positive
        // bar high — a same-named kfunc whose return is NOT
        // Ptr->Void cannot have its R0 typed by this arm. This
        // protects against name collisions between a future
        // arena-returning kfunc and an unrelated kfunc that
        // happens to share a name.
        if return_peels_to_ptr_void(self.btf, ret_id)
            && let Some(name) = func_name.as_deref()
            && ARENA_ALLOC_KFUNC_NAMES.contains(&name)
        {
            self.regs[0] = RegState::ArenaU64FromAlloc;
            // F4 telemetry parity with the SubprogReturn arm:
            // count this as an applied allocator seed so the
            // finalize warn distinguishes "allocator was called
            // but no slot got tagged" from "no allocator was
            // ever called" identically across kfunc and subprog
            // paths.
            self.alloc_seeds_applied = self.alloc_seeds_applied.saturating_add(1);
        }
    }

    fn set_reg(&mut self, idx: usize, state: RegState) {
        // R10 is the read-only frame pointer per BPF ABI; the
        // verifier rejects programs that mutate it. Maintain the
        // invariant that regs[R10] stays Unknown so a later LDX/STX
        // through r10 cannot resurrect a stale typed-pointer state.
        if idx == BPF_REG_R10 {
            return;
        }
        if idx < self.regs.len() {
            self.regs[idx] = state;
        }
    }

    fn note_type_id(&mut self, id: u32) {
        if id > self.max_seen_type_id {
            self.max_seen_type_id = id;
        }
    }

    fn finalize(self) -> CastMap {
        let mut out = CastMap::new();
        let max_id = self
            .max_seen_type_id
            .saturating_add(CANDIDATE_SEARCH_SLACK)
            .min(super::sdt_alloc::MAX_BTF_ID_PROBE);
        // F15 mitigation: warn when the candidate-search slack
        // capped against the hard ceiling. A scheduler whose largest
        // touched id is close to MAX_BTF_ID_PROBE means
        // [`build_layout_index`] cannot probe every type the BTF
        // exposes — shape-inference candidates above the cap are
        // invisible. Surface this as a `warn!` so a future BTF that
        // genuinely exceeds the ceiling shows up rather than silently
        // missing candidates.
        if self.max_seen_type_id.saturating_add(CANDIDATE_SEARCH_SLACK)
            > super::sdt_alloc::MAX_BTF_ID_PROBE
        {
            tracing::warn!(
                max_seen_type_id = self.max_seen_type_id,
                slack = CANDIDATE_SEARCH_SLACK,
                cap = super::sdt_alloc::MAX_BTF_ID_PROBE,
                "cast_analysis: candidate-search slack capped at MAX_BTF_ID_PROBE; \
                 shape-inference candidates above the cap are invisible"
            );
        }

        // Pre-build (offset, size) -> { type_id } so each pattern
        // does not re-walk the entire BTF id space. The walk stops
        // at the first sustained run of unresolved ids -- BTF id
        // tables are dense in practice but tolerate small gaps.
        let layout = build_layout_index(self.btf, max_id);

        // Arena/kptr conflict drop set: any (source, offset) slot
        // observed by BOTH an arena path (`self.patterns` —
        // the slot was loaded as a u64 then dereferenced as a
        // pointer base; OR `self.arena_stx_findings` — an
        // [`RegState::ArenaU64FromAlloc`] value was stored into
        // the slot) AND the kernel STX path (`self.kptr_findings`
        // — a typed `Pointer{T}` was stored into the slot) is
        // ambiguous. The same byte cannot simultaneously hold an
        // arena VA (deref via arena reader) and a kernel VA (deref
        // via slab/vmalloc reader); seeing both is evidence the
        // analyzer's flow-insensitive register tracking confused
        // disjoint code paths against the same slot. False positive
        // is unacceptable, so drop both observations and let the
        // renderer fall back to the raw u64 path. False negative
        // is acceptable. Note that `self.patterns` includes keys
        // with empty access sets (the slot was loaded but never
        // dereferenced); those carry no signal either way and are
        // not treated as arena evidence here.
        let conflicting: BTreeSet<(u32, u32)> = self
            .patterns
            .iter()
            .filter(|(_, accesses)| !accesses.is_empty())
            .map(|(k, _)| *k)
            .chain(self.arena_confirmed.iter().copied())
            .chain(self.arena_stx_findings.keys().copied())
            .filter(|k| self.kptr_findings.contains_key(k))
            .collect();

        // Track keys already emitted as Arena via the STX-flow path
        // so the shape-inference loop below can short-circuit a
        // duplicate emit. Both paths produce
        // `addr_space: AddrSpace::Arena` for the same slot, but
        // shape inference may resolve `target_type_id` to a concrete
        // BTF struct id while the STX-flow path emits `0` (deferred
        // resolve via `MemReader::resolve_arena_type` bridge). The
        // shape-inference target is always at least as informative,
        // so the LATER loop wins by overwriting on the same key.
        // Recording arena-STX hits first, then letting the shape
        // loop overwrite when it has a concrete id, gives the best
        // of both: the bridge fires for slots without shape-derived
        // ids, and concrete ids take precedence when both fire.

        // Arena STX-flow path: directly observed STX of an
        // [`RegState::ArenaU64FromAlloc`] value into a u64 slot.
        // Emit with `target_type_id == 0` — the renderer's
        // [`MemReader::resolve_arena_type`] bridge resolves the
        // payload BTF id at chase time from the live arena snapshot
        // (cross-BTF Fwd resolution). Conflicting slots (also seen
        // as kptr STX) drop here AND on the kptr side.
        //
        // # When the deferred resolve succeeds vs fails at chase
        // time
        //
        // The bridge is backed by
        // [`super::dump::render_map::ArenaTypeIndex`], which the
        // sdt_alloc pre-pass populates by walking
        // [`super::sdt_alloc::SdtAllocatorSnapshot`] for every
        // **per-instance** allocator (`scx_alloc_internal` and
        // friends). The bridge therefore RESOLVES at chase time only
        // when the chased pointer's runtime value falls inside an
        // sdt_alloc slot's `[slot_start, slot_start + elem_size)`
        // range AND lands at either the slot start (header_skip ==
        // header_size) or payload start (header_skip == 0).
        //
        // The bridge does NOT cover bump-allocator allocations from
        // `scx_static_alloc_internal` — that allocator has no
        // per-allocation header and produces a flat arena region
        // with no per-slot metadata the pre-pass can index. A slot
        // whose arena VA was produced by `scx_static_alloc_internal`
        // and whose target BTF type is unique-shape-inferable at
        // analysis time will resolve via the shape-inference loop
        // below (concrete `target_type_id != 0`); a slot whose
        // shape is ambiguous (multiple BTF structs match the access
        // pattern) and whose VA is from `scx_static_alloc_internal`
        // will fall through with `target_type_id == 0` and the
        // bridge will return `None` at chase time, so the chase
        // skips with a clear "no entry for 0x{val:x}" reason.
        // This is the "no invalid data made" contract: ambiguous
        // shape + no per-slot index = fail-closed, no chase, no
        // wrong render.
        for (key, entry) in &self.arena_stx_findings {
            // Filter out `Conflicting` entries defensively: today no
            // insertion path produces them (`handle_stx` only inserts
            // `Pending` and `unreachable!()`s on a `Conflicting`
            // overwrite), but a future enrichment of the arena STX
            // flow could legitimately record disagreement; this gate
            // keeps the drop semantics in one place.
            if !matches!(entry, ArenaStxEntry::Pending) {
                continue;
            }
            if conflicting.contains(key) {
                continue;
            }
            out.insert(
                *key,
                CastHit {
                    target_type_id: 0,
                    addr_space: AddrSpace::Arena,
                },
            );
        }

        // Arena pointer path (shape inference): BTF-shape-inferred
        // targets. Tagged as AddrSpace::Arena because the source
        // u64 field is itself dereferenced and its target struct is
        // recovered by intersecting struct shapes across the
        // observed access pattern.
        //
        // F1 mitigation: require direct evidence the slot held an
        // arena VA before emitting a shape-inference hit. The 4 GiB
        // arena window catches any 33-bit value as "in arena" at
        // chase time, so a slot that just happens to hold a
        // 33-bit-shaped counter could be mis-rendered as an arena
        // pointer. Direct evidence comes from EITHER an
        // observed `BPF_ADDR_SPACE_CAST` on a value loaded from the
        // slot (`self.arena_confirmed`) OR an observed STX of an
        // allocator-tagged value into the slot
        // (`self.arena_stx_findings` — see the STX-flow path above).
        // Slots with neither observation drop here; an operator can
        // re-enable inference for a specific slot by adding either
        // a `bpf_addr_space_cast` site or the STX-flow tag in the
        // scheduler source.
        for ((source, field_off), accesses) in &self.patterns {
            // A field that was loaded but never dereferenced gives
            // no signal. Drop it -- the renderer's existing u64
            // path is the correct fallback.
            if accesses.is_empty() {
                continue;
            }
            // Conflict with kptr path on the same slot: drop both
            // observations (the kptr loop below also skips this key).
            if conflicting.contains(&(*source, *field_off)) {
                continue;
            }
            // F1 gate: shape inference alone is not enough. Require
            // either a `bpf_addr_space_cast` observation
            // (`arena_confirmed`) OR an arena-STX observation
            // (`arena_stx_findings`) on the same slot before we
            // emit a shape-inference target.
            let key = (*source, *field_off);
            let has_direct_evidence =
                self.arena_confirmed.contains(&key) || self.arena_stx_findings.contains_key(&key);
            if !has_direct_evidence {
                tracing::debug!(
                    parent_type_id = source,
                    field_offset = field_off,
                    accesses = accesses.len(),
                    "cast_analysis: shape-inference candidate without direct evidence; dropped (F1 mitigation)"
                );
                continue;
            }
            // Intersection of candidate type ids across every
            // observed (offset, size). The first lookup seeds
            // `candidates` by cloning once; subsequent lookups
            // retain only elements present in the next set.
            let mut iter = accesses.iter();
            let first = iter.next().expect("non-empty checked above");
            let empty = HashSet::new();
            let mut candidates: HashSet<u32> = layout
                .get(&(first.offset, first.size))
                .cloned()
                .unwrap_or_default();
            for acc in iter {
                let next = layout.get(&(acc.offset, acc.size)).unwrap_or(&empty);
                candidates.retain(|c| next.contains(c));
                if candidates.is_empty() {
                    break;
                }
            }
            candidates.remove(source);

            if candidates.len() == 1 {
                let target = candidates.into_iter().next().unwrap();
                // Shape-inference target overwrites any STX-flow
                // hit emitted above for the same key — the concrete
                // id is more informative than the deferred `0`
                // sentinel.
                out.insert(
                    (*source, *field_off),
                    CastHit {
                        target_type_id: target,
                        addr_space: AddrSpace::Arena,
                    },
                );
            }
            // 0 or 2+ candidates -> drop silently. False negative
            // is the safe direction.
        }

        // F4 mitigation: surface allocator call sites that the
        // analyzer saw but could not follow into a typed-slot
        // STX. These manifest when a scheduler does not mark its
        // allocator helpers `__always_inline` — the analyzer sees
        // the helper-call site (one or more allocator seeds applied)
        // but cannot follow the tagged R0 across the call boundary
        // into the caller's frame, so no slot ends up in
        // [`Self::arena_stx_findings`]. Emit one warning per dump
        // pass to keep noise bounded.
        //
        // Gate:
        //   - At least one allocator seed was applied (counted by
        //     [`Self::alloc_seeds_applied`]). Without this, no
        //     allocator was ever called and the warn would be
        //     spurious noise.
        //   - `arena_stx_findings` is empty. A non-empty findings
        //     map means at least one slot DID get tagged; that is
        //     the normal allocator-return seed path's happy shape
        //     the prior gate incorrectly flagged. The gate is now
        //     strict on the specific `__always_inline` failure
        //     mode.
        //
        // The prior gate (`!arena_stx_findings.is_empty() &&
        // arena_confirmed.is_empty()`) fired on the normal
        // allocator-return seed path's happy shape where a
        // scheduler correctly inlines the allocator AND the
        // consumer reads through the slot via STX-flow alone (no
        // `bpf_addr_space_cast` site). The operator received a
        // misleading "may need __always_inline" warning on a
        // working pipeline.
        if self.alloc_seeds_applied > 0 && self.arena_stx_findings.is_empty() {
            tracing::warn!(
                alloc_seeds_applied = self.alloc_seeds_applied,
                "cast_analysis: allocator seeds applied but no slot got an arena \
                 STX tag; allocator helpers may need __always_inline so the \
                 returned R0 reaches a typed-slot STX without crossing a \
                 BPF-to-BPF call boundary"
            );
        }

        // Kernel kptr path: directly observed STX of a typed
        // pointer into a u64 slot. The target type is known
        // exactly from the value register's RegState — no shape
        // inference needed. Conflicting writes to the same slot
        // (different target types) drop. Slots that ALSO appear
        // in any arena path (`conflicting` above) drop on
        // both sides — the analyzer cannot tell which observation
        // is real, and emitting either tag risks a false positive.
        for (key, entry) in self.kptr_findings {
            let KptrEntry::Single(target) = entry else {
                continue;
            };
            if conflicting.contains(&key) {
                continue;
            }
            out.insert(
                key,
                CastHit {
                    target_type_id: target,
                    addr_space: AddrSpace::Kernel,
                },
            );
        }

        out
    }
}

/// Pre-scan the program for jump-target PCs.
///
/// Targets are computed as `pc + 1 + insn.off` for every BPF_JMP /
/// BPF_JMP32 instruction except `EXIT` and `CALL`. Out-of-range
/// targets (negative resolved address, or past `insns.len()`) are
/// dropped.
fn jump_targets(insns: &[BpfInsn]) -> BTreeSet<usize> {
    let mut targets = BTreeSet::new();
    let mut skip_next = false;
    for (pc, insn) in insns.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        let class = insn.code & 0x07;
        if class == BPF_CLASS_LD && insn.code == (BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM) {
            // BPF_LD_IMM64 takes two slots; the second slot's `code`
            // is 0. BPF_JA's op nibble is 0x00 (under BPF_CLASS_JMP =
            // 0x05, the full JA opcode is 0x05); a bare 0 code byte
            // has class 0 (BPF_LD), not 0x05, so it would not match
            // the JMP-class gate below — but skipping the second
            // slot explicitly maintains symmetry with the main pass.
            skip_next = true;
            continue;
        }
        if class != BPF_CLASS_JMP && class != BPF_CLASS_JMP32 {
            continue;
        }
        let op = insn.code & 0xf0;
        if op == BPF_OP_EXIT || op == BPF_OP_CALL {
            continue;
        }
        // JMP32 | JA ("gotol") uses insn.imm for the 32-bit jump
        // offset, not insn.off (which is 16-bit). All other JMP/JMP32
        // instructions use insn.off. See kernel filter.h BPF_JMP32_A.
        let jump_off = if class == BPF_CLASS_JMP32 && op == 0x00 {
            // BPF_JA = 0x00 under JMP32 class = gotol
            insn.imm as i64
        } else {
            insn.off as i64
        };
        let next = pc as i64 + 1 + jump_off;
        if next >= 0 && (next as usize) < insns.len() {
            targets.insert(next as usize);
        }
    }
    targets
}

/// Build a `(offset, size_bytes) -> {type_ids}` index over every
/// BTF struct / union with a non-bitfield member at that location
/// whose member type has the given size. The matching phase
/// intersects sets across observed accesses to collapse to a single
/// candidate when one exists.
fn build_layout_index(btf: &Btf, max_id: u32) -> HashMap<(u32, u32), HashSet<u32>> {
    let mut out: HashMap<(u32, u32), HashSet<u32>> = HashMap::new();
    let mut size_cache: HashMap<u32, Option<u32>> = HashMap::new();
    let mut consecutive_fail: u32 = 0;
    const CONSECUTIVE_FAIL_CAP: u32 = 256;

    let mut tid: u32 = 1;
    while tid <= max_id {
        match btf.resolve_type_by_id(tid) {
            Ok(Type::Struct(s)) | Ok(Type::Union(s)) => {
                consecutive_fail = 0;
                for m in &s.members {
                    let bit_off = m.bit_offset();
                    if bit_off % 8 != 0 {
                        continue;
                    }
                    if matches!(m.bitfield_size(), Some(s) if s > 0) {
                        continue;
                    }
                    let off = bit_off / 8;
                    let size = match cached_member_size(btf, m, &mut size_cache) {
                        Some(sz) => sz,
                        None => continue,
                    };
                    out.entry((off, size)).or_default().insert(tid);
                }
            }
            Ok(_) => {
                consecutive_fail = 0;
            }
            Err(_) => {
                consecutive_fail += 1;
                if consecutive_fail >= CONSECUTIVE_FAIL_CAP {
                    break;
                }
            }
        }
        tid += 1;
    }
    out
}

fn cached_member_size(
    btf: &Btf,
    m: &btf_rs::Member,
    cache: &mut HashMap<u32, Option<u32>>,
) -> Option<u32> {
    let tid = m.get_type_id().ok()?;
    *cache
        .entry(tid)
        .or_insert_with(|| member_size_bytes(btf, m))
}

/// Resolve `bpf_member` to a byte size, peeling Const / Volatile /
/// Restrict / Typedef / TypeTag / DeclTag chains via the renderer's
/// shared [`super::btf_render::peel_modifiers`] and sizing through
/// [`super::btf_render::type_size`]. Returns `None` for shapes the
/// renderer's sizing routine cannot resolve (Func, FuncProto, Var,
/// Datasec, Fwd, Void). For non-byte-multiple ints the BTF-declared
/// size is returned verbatim — a `__int128` member surfaces as
/// `Some(16)` and the matcher simply finds no LDX access of that
/// width to intersect against.
fn member_size_bytes(btf: &Btf, m: &btf_rs::Member) -> Option<u32> {
    let tid = m.get_type_id().ok()?;
    let terminal = super::btf_render::peel_modifiers(btf, tid)?;
    super::btf_render::type_size(btf, &terminal).map(|s| s as u32)
}

/// Resolved member of a parent BTF aggregate at a specific byte
/// offset.
///
/// Models the "what type lives here" answer for both
/// [`Type::Struct`] / [`Type::Union`] (regular C aggregates) and
/// [`Type::Datasec`] (`.bss` / `.data` / `.rodata` global sections,
/// which libbpf encodes as a flat sequence of `VarSecinfo` ->
/// `BTF_KIND_VAR` entries rather than as struct members). The
/// caller looks up `member_type_id`, peels modifiers, and decides
/// whether the location is a u64-typed kptr slot.
///
/// `byte_offset` is the offset returned by the parent's layout —
/// equal to the queried offset for an exact-offset hit on a struct
/// member, or the start-of-variable offset for a Datasec hit when
/// the queried offset lies inside a multi-byte variable's range.
/// The current callers (`handle_ldx` u64 detection and `handle_stx`
/// kptr-finding) require the queried offset to land exactly on a
/// member boundary; the start-of-variable Datasec semantics
/// preserve that invariant for plain u64 globals (the variable
/// starts AT the queried offset) while letting struct globals
/// surface as a Struct-typed member that the LDX path threads
/// through `peel_modifiers` for further analysis.
#[derive(Debug, Clone)]
enum MemberAt {
    /// Hit on a `BTF_KIND_STRUCT` / `BTF_KIND_UNION` member at the
    /// queried byte offset. `member_type_id` is the BTF id of the
    /// member's declared type. `resolved_parent_type_id` is the
    /// BTF id of the struct that directly contains this member —
    /// for nested structs this is the INNERMOST struct, not the
    /// outermost base register's struct. The CastMap keys on this
    /// id so the renderer's per-struct cast_lookup matches.
    Struct {
        member_type_id: u32,
        resolved_parent_type_id: u32,
        resolved_member_offset: u32,
    },
    /// Hit on a `BTF_KIND_DATASEC` `VarSecinfo` whose byte range
    /// contains the queried offset. `var_underlying_type_id` is
    /// the BTF id of the `BTF_KIND_VAR`'s underlying type (the
    /// global variable's actual C type — typically a u64, struct,
    /// or array). `var_byte_offset` is the variable's start
    /// offset within the section. For an exact-offset hit on a
    /// plain u64 global, `var_byte_offset == queried_offset`.
    /// For a struct-typed global, `var_byte_offset <=
    /// queried_offset < var_byte_offset + var_size`.
    Datasec {
        var_underlying_type_id: u32,
        var_byte_offset: u32,
    },
}

impl MemberAt {
    /// BTF id of the member's declared type. The caller peels
    /// modifiers and decides whether the location is a u64-typed
    /// kptr slot.
    fn member_type_id(&self) -> u32 {
        match self {
            Self::Struct { member_type_id, .. } => *member_type_id,
            Self::Datasec {
                var_underlying_type_id,
                ..
            } => *var_underlying_type_id,
        }
    }
}

/// Find the member at `byte_offset` within the parent BTF aggregate
/// identified by `parent_type_id`. Returns `None` for parents the
/// analyzer does not handle (everything other than Struct, Union,
/// or Datasec) and for offsets that do not land on a recognizable
/// member.
///
/// Struct / Union path: matches members at exactly `byte_offset`,
/// skipping bitfields and members at non-byte-aligned bit offsets
/// (the analyzer cannot reason about either as 64-bit pointer
/// slots).
///
/// Datasec path: matches the `VarSecinfo` whose `[offset,
/// offset+size)` range contains `byte_offset`. Datasec entries are
/// laid out flat (no bitfields, no nested layout); each entry's
/// `get_type_id()` resolves to a `BTF_KIND_VAR` whose
/// `get_type_id()` returns the global's underlying C type. The
/// returned [`MemberAt::Datasec`] surfaces the underlying type id
/// so the LDX / STX paths can peel modifiers and check for a
/// plain u64 just like they do for struct members.
fn struct_member_at(btf: &Btf, parent_type_id: u32, byte_offset: u32) -> Option<MemberAt> {
    let (t, parent_type_id) = super::btf_render::peel_modifiers_with_id(btf, parent_type_id)?;
    match t {
        Type::Struct(s) | Type::Union(s) => {
            for m in &s.members {
                if matches!(m.bitfield_size(), Some(s) if s > 0) {
                    continue;
                }
                let bit_off = m.bit_offset();
                if bit_off % 8 != 0 {
                    continue;
                }
                let member_off = bit_off / 8;
                let member_type_id = m.get_type_id().ok()?;
                if member_off == byte_offset {
                    if let Some(terminal) = super::btf_render::peel_modifiers(btf, member_type_id)
                        && matches!(terminal, Type::Struct(_) | Type::Union(_))
                    {
                        return struct_member_at(btf, member_type_id, 0);
                    }
                    return Some(MemberAt::Struct {
                        member_type_id,
                        resolved_parent_type_id: parent_type_id,
                        resolved_member_offset: byte_offset,
                    });
                }
                if member_off < byte_offset
                    && let Some(terminal) = super::btf_render::peel_modifiers(btf, member_type_id)
                {
                    match &terminal {
                        Type::Array(arr) => {
                            let elem_tid = arr.get_type_id().ok()?;
                            let elem_size = super::btf_render::type_size(btf, &{
                                super::btf_render::peel_modifiers(btf, elem_tid)?
                            })? as u32;
                            if elem_size > 0 {
                                let arr_len = arr.len() as u32;
                                let arr_byte_size = elem_size * arr_len;
                                let rel = byte_offset - member_off;
                                if rel < arr_byte_size && rel.is_multiple_of(elem_size) {
                                    return Some(MemberAt::Struct {
                                        member_type_id: elem_tid,
                                        resolved_parent_type_id: parent_type_id,
                                        resolved_member_offset: byte_offset,
                                    });
                                }
                            }
                        }
                        Type::Struct(_) | Type::Union(_) => {
                            let member_size = super::btf_render::type_size(btf, &terminal)? as u32;
                            let rel = byte_offset - member_off;
                            if rel < member_size {
                                return struct_member_at(btf, member_type_id, rel);
                            }
                        }
                        _ => {}
                    }
                }
            }
            None
        }
        Type::Datasec(ds) => {
            for var_info in &ds.variables {
                let off = var_info.offset();
                let size = var_info.size() as u32;
                let end = off.checked_add(size)?;
                if byte_offset < off || byte_offset >= end {
                    continue;
                }
                // Resolve the chained Var so we can pull the
                // underlying type id. A non-Var here indicates
                // malformed BTF (libbpf always emits Var per
                // VarSecinfo); drop silently — false negative is
                // the safe direction. The check on
                // `Type::Var(...)` matches the renderer's
                // `render_datasec` shape so any future Datasec
                // variant added to btf-rs surfaces consistently
                // across both modules.
                let chained = btf.resolve_chained_type(var_info).ok()?;
                let var = match chained {
                    Type::Var(v) => v,
                    _ => return None,
                };
                let var_underlying_type_id = var.get_type_id().ok()?;
                return Some(MemberAt::Datasec {
                    var_underlying_type_id,
                    var_byte_offset: off,
                });
            }
            None
        }
        _ => None,
    }
}

/// Resolve a BTF type id and report whether it peels to
/// `Ptr -> Void`.
///
/// `Ptr` ids whose pointee is `0` (the BTF void marker — same
/// convention as [`FuncProto::return_type_id`] uses) match. The
/// peel walks `Const` / `Volatile` / `Restrict` / `Typedef` /
/// `TypeTag` / `DeclTag` modifiers only — bridging a `Ptr` we
/// would never want, since the result of dereferencing an
/// arbitrary modifier-wrapped type is not a useful "Ptr -> Void"
/// signal for arena-allocator detection.
///
/// Used to gate [`Analyzer::handle_kfunc_call`]'s arena-allocator
/// arm: the allowlisted kfunc names ([`ARENA_ALLOC_KFUNC_NAMES`])
/// only confer a [`RegState::ArenaU64FromAlloc`] tag when the
/// declared return is structurally `void *`. A kfunc whose name
/// drifts onto the allowlist but whose BTF return is not
/// `Ptr -> Void` cannot be misclassified here.
///
/// Returns `false` for any type id that does not resolve, peels
/// to a non-`Ptr` terminal, or whose pointee resolves to a
/// non-void type. Failure is the safe direction — false
/// negatives drop to the existing typed-pointer arm or no-op.
fn return_peels_to_ptr_void(btf: &Btf, ret_id: u32) -> bool {
    // Peel modifiers AROUND the Ptr first — `const void *` and
    // its kin lower to `Const(Ptr)` in BTF. The renderer's
    // [`super::btf_render::peel_modifiers`] handles the same
    // shape; reusing it keeps the semantics aligned with the
    // rest of the analyzer.
    //
    // The peel returns `None` for any type id that does not
    // resolve OR terminates on a non-trivial shape we cannot
    // interpret as Ptr->Void (Func, FuncProto, Var, Datasec).
    // Drop conservatively in those cases — false negatives are
    // the safe direction.
    let Some(peeled) = super::btf_render::peel_modifiers(btf, ret_id) else {
        return false;
    };
    let Type::Ptr(p) = peeled else {
        return false;
    };
    // BTF encodes `void *` with the Ptr's pointee type id == 0.
    // Same convention [`FuncProto::return_type_id`] uses for void
    // returns at the FuncProto level. Anything else is a typed
    // pointer that arm 1 (`resolve_to_struct_id`) already handled
    // — falling through here would let arm 2 mistakenly tag a
    // typed-pointer return as ArenaU64FromAlloc, the very case
    // the strict gate prevents.
    p.get_type_id().map(|id| id == 0).unwrap_or(false)
}

/// Resolve a `bpf_map_lookup_elem` call's R0 value type from the
/// caller-side map descriptor metadata in the program BTF.
///
/// The plain-helper arm of [`Analyzer::step`] looks up R1 in the
/// `.maps` `BTF_KIND_DATASEC` and types R0 as a typed pointer to
/// the map's value struct. This function performs the BTF walk:
///
/// 1. `datasec_id` must resolve to [`Type::Datasec`] whose name is
///    exactly `.maps` — the libbpf-managed user-space BTF map
///    declaration section. A `BTF_KIND_DATASEC` named anything
///    else (e.g. `.bss`, `.data`, `.data.<name>`) is rejected so
///    a non-map struct that happens to carry a `value` member of
///    pointer type cannot drive this arm.
/// 2. The datasec's `VarSecinfo` whose `offset == var_offset` is
///    located. The chained type must be [`Type::Var`] whose
///    underlying type peels through modifiers to a
///    [`Type::Struct`] / [`Type::Union`] — the per-map struct
///    declaration libbpf parses in `parse_btf_map_def`
///    (`tools/lib/bpf/libbpf.c`).
/// 3. The struct's members are scanned for one named `value`
///    (the `__type(value, T)` declaration expanded to
///    `typeof(T) *value` per `tools/lib/bpf/bpf_helpers.h`).
/// 4. The `value` member's type peels to [`Type::Ptr`] — libbpf
///    rejects non-`Ptr` value declarations (`if (!btf_is_ptr(t))`
///    in `parse_btf_map_def`).
/// 5. The `Ptr`'s pointee resolves through
///    [`super::bpf_map::resolve_to_struct_id`] to a
///    [`Type::Struct`] / [`Type::Union`] id. Maps whose value type
///    is a primitive (e.g. `__type(value, u64)` for stat counters)
///    or `void` peel to non-struct terminals; the function returns
///    `None` and the analyzer leaves R0 Unknown.
///
/// Any failure on the walk drops the whole resolution — false
/// negatives are the safe direction. The walk does NOT mutate any
/// analyzer state and does NOT consult `arena_confirmed` /
/// `arena_stx_findings`; the seeded `Pointer{T}` flows into the
/// existing kptr/arena STX paths exactly the same way a kfunc-
/// returned typed pointer does (see [`Analyzer::handle_kfunc_call`]
/// arm 1).
fn map_value_struct_id(btf: &Btf, datasec_id: u32, var_offset: u32) -> Option<u32> {
    // Gate 1: datasec must be `.maps`. Resolve the type, confirm
    // the kind, then resolve the name. Any non-Datasec kind or a
    // name resolution error drops to None — the analyzer's safe
    // direction. A renamed `.maps.foo` section (libbpf does NOT
    // rename `.maps`, but the kernel allows custom section names
    // for non-libbpf-managed BPF objects) would not match here;
    // any future need to broaden this gate must add a corresponding
    // test that proves the broader gate cannot drive a false
    // positive on a non-map datasec.
    let ty = btf.resolve_type_by_id(datasec_id).ok()?;
    let datasec = match ty {
        Type::Datasec(d) => d,
        _ => return None,
    };
    let name = btf.resolve_name(&datasec).ok()?;
    if name != ".maps" {
        return None;
    }

    // Gate 2: locate the per-map VarSecinfo. The verifier guarantees
    // VarSecinfos are non-overlapping per the BTF spec; an exact
    // offset match is the only correct lookup (a partial overlap
    // means the caller's annotation is targeting a struct member,
    // not a map descriptor — the analyzer's R1 must point at the
    // map's struct, never mid-struct, because clang's relocation
    // emission uses the var's start offset).
    let var_info = datasec
        .variables
        .iter()
        .find(|v| v.offset() == var_offset)?;
    let chained = btf.resolve_chained_type(var_info).ok()?;
    let var = match chained {
        Type::Var(v) => v,
        _ => return None,
    };
    let var_type_id = var.get_type_id().ok()?;

    // Gate 3: the var's underlying type must peel to a
    // Struct/Union — that is the map descriptor C struct emitted
    // by clang for `struct { __uint(...); __type(...); ... } name
    // SEC(".maps");`. Modifiers around the struct (Const /
    // Volatile / Typedef / TypeTag / DeclTag / Restrict) are peeled
    // by [`super::btf_render::peel_modifiers`] consistently with
    // the rest of the analyzer.
    let map_def_terminal = super::btf_render::peel_modifiers(btf, var_type_id)?;
    let map_def = match map_def_terminal {
        Type::Struct(s) => s,
        _ => return None,
    };

    // Gate 4: find the `value` member. clang's `__type(value, T)`
    // macro in `tools/lib/bpf/bpf_helpers.h` (`#define __type(name,
    // val) typeof(val) *name`) emits a struct member literally
    // named `value` whose type is `typeof(T) *`. libbpf
    // (`parse_btf_map_def`) keys on `strcmp(name, "value") == 0`
    // for this exact match — the analyzer mirrors the literal name
    // check.
    //
    // A member whose name resolution fails individually does not
    // abort the search: real map decls always have name-resolved
    // members (`type`, `key`, `value`, `max_entries`, …), but a
    // malformed BTF carrying an unnamed member should not poison
    // the lookup. `continue` past the bad name and inspect the
    // next member.
    for member in &map_def.members {
        let Ok(mname) = btf.resolve_name(member) else {
            continue;
        };
        if mname != "value" {
            continue;
        }
        // Gate 5: member type peels to `Ptr -> Struct/Union`.
        // libbpf's `parse_btf_map_def` rejects a non-Ptr `value`
        // member with `-EINVAL`; the analyzer's gate enforces the
        // same shape. A `Ptr -> u64` (stat-counter map) or
        // `Ptr -> Void` peels to a non-struct pointee and
        // `resolve_to_struct_id` returns None, dropping the
        // resolution — the renderer's existing u64 plain-counter
        // path is the correct fallback for stat maps.
        let mtype_id = member.get_type_id().ok()?;
        let mterminal = super::btf_render::peel_modifiers(btf, mtype_id)?;
        let ptr = match mterminal {
            Type::Ptr(p) => p,
            _ => return None,
        };
        let pointee = ptr.get_type_id().ok()?;
        return super::bpf_map::resolve_to_struct_id(btf, pointee);
    }
    // No `value` member: map shapes that omit a value declaration
    // (`BPF_MAP_TYPE_PROG_ARRAY` declared with `__array(values, ...)`
    // for instance) cannot have their R0 typed by this arm. Drop
    // silently.
    None
}

/// Allowlist of kfunc names whose `Ptr -> Void` return is treated
/// as an arena VA seed for [`RegState::ArenaU64FromAlloc`].
///
/// Each entry must be a real kfunc declared in the kernel's
/// `kernel/bpf/arena.c` (or peer kernel arena helpers) AND must
/// return `void *` whose runtime value is a 4 GiB-window arena
/// virtual address. Verified against the kernel source:
/// `bpf_arena_alloc_pages` is declared `__bpf_kfunc void *` per
/// linux `kernel/bpf/arena.c::bpf_arena_alloc_pages`.
///
/// Order is alphabetical for readability — the allowlist is a
/// linear-scan small-N membership test in
/// [`Analyzer::handle_kfunc_call`]. A future arena-returning
/// kfunc is added by appending its name here AND verifying its
/// return type peels to `Ptr -> Void` in the kernel BTF the
/// analyzer consumes; the strict
/// [`return_peels_to_ptr_void`] gate keeps a name-allowlist drift
/// from producing a false positive on a same-named non-arena
/// kfunc.
///
/// Distinct from the `ALLOC_SUBPROG_NAMES` allowlist in
/// [`crate::vmm::cast_analysis_load`]: that list is for in-tree
/// library subprograms (BPF-to-BPF calls with `BPF_PSEUDO_CALL`
/// + symbol resolution against the program ELF); this list is
///   for kernel kfuncs (`BPF_PSEUDO_KFUNC_CALL` + BTF id resolution
///   in [`Analyzer::handle_kfunc_call`]). The kernel kfunc and
///   in-tree subprog code paths are independent — a single name
///   belongs to exactly one of the two allowlists.
pub(crate) const ARENA_ALLOC_KFUNC_NAMES: &[&str] = &[
    // Generic BPF arena page allocator. Returns `void *` per
    // `kernel/bpf/arena.c::bpf_arena_alloc_pages` (`__bpf_kfunc
    // void *bpf_arena_alloc_pages(...)`). The runtime value is
    // either NULL or a user-side arena VA suitable for the
    // STX-flow tagging path.
    "bpf_arena_alloc_pages",
];

/// Convert the `BPF_DW`/`BPF_W`/`BPF_H`/`BPF_B` size bits to a byte
/// count. `None` for unknown encodings.
fn ldx_size_bytes(size_bits: u8) -> Option<u32> {
    match size_bits {
        BPF_SIZE_DW => Some(8),
        BPF_SIZE_W => Some(4),
        BPF_SIZE_H => Some(2),
        BPF_SIZE_B => Some(1),
        _ => None,
    }
}

/// `BpfInsn::off` is `i16`, so a negative value means the load
/// is relative to the base register at a backward offset (e.g.
/// stack-relative loads through r10). The cast pattern only
/// considers non-negative offsets — kernel struct fields never
/// have a negative byte offset relative to the struct base.
fn field_byte_offset(off: i32) -> Option<u32> {
    if off < 0 { None } else { Some(off as u32) }
}

// --- BPF instruction encoding constants --------------------------
//
// Sourced from `libbpf_rs::libbpf_sys` (which re-exports the bindgen
// translation of `linux/include/uapi/linux/bpf.h`). The analyzer
// stores opcodes in `u8` fields per the wire format, so the
// upstream `u32` constants are narrowed to typed locals below.
// Constants not exported by libbpf-sys (the standalone top-nibble
// values for `BPF_XCHG` / `BPF_CMPXCHG`) are derived from the full
// opcodes.

// BPF_CLASS_* — low 3 bits of `code` selecting the instruction
// class. Names retain the `BPF_CLASS_` prefix for parity with the
// kernel's `#define BPF_CLASS(code) ((code) & 0x07)` macro.
const BPF_CLASS_LD: u8 = bs::BPF_LD as u8;
const BPF_CLASS_LDX: u8 = bs::BPF_LDX as u8;
const BPF_CLASS_ST: u8 = bs::BPF_ST as u8;
const BPF_CLASS_STX: u8 = bs::BPF_STX as u8;
const BPF_CLASS_ALU: u8 = bs::BPF_ALU as u8;
const BPF_CLASS_JMP: u8 = bs::BPF_JMP as u8;
const BPF_CLASS_JMP32: u8 = bs::BPF_JMP32 as u8;
const BPF_CLASS_ALU64: u8 = bs::BPF_ALU64 as u8;

// BPF_SIZE_* — bits 3..4 of `code` selecting the access width.
const BPF_SIZE_W: u8 = bs::BPF_W as u8;
const BPF_SIZE_H: u8 = bs::BPF_H as u8;
const BPF_SIZE_B: u8 = bs::BPF_B as u8;
const BPF_SIZE_DW: u8 = bs::BPF_DW as u8;

// BPF_MODE_* — bits 5..7 of `code` selecting the addressing mode.
const BPF_MODE_IMM: u8 = bs::BPF_IMM as u8;
const BPF_MODE_MEM: u8 = bs::BPF_MEM as u8;
/// Atomic memory ops (BPF_STX class). `imm` selects the specific
/// operation (`BPF_XCHG`, `BPF_CMPXCHG`, `BPF_ADD | BPF_FETCH`, …).
/// See linux uapi `bpf.h`: `#define BPF_ATOMIC 0xc0`.
const BPF_MODE_ATOMIC: u8 = bs::BPF_ATOMIC as u8;

// BPF_OP_* — top 4 bits of `code` selecting the ALU / JMP op.
const BPF_OP_MOV: u8 = bs::BPF_MOV as u8;
const BPF_OP_CALL: u8 = bs::BPF_CALL as u8;
const BPF_OP_EXIT: u8 = bs::BPF_EXIT as u8;

/// Source-operand selector. `BPF_X` (== libbpf-sys `BPF_X` == 0x08)
/// signals a register source; `BPF_K` (== 0) signals an immediate.
const BPF_SRC_X: u8 = bs::BPF_X as u8;

/// Atomic-op `imm` field bit set on operations that return the prior
/// memory value. Combined with `BPF_CMPXCHG_TOP` to form `BPF_CMPXCHG`.
/// See linux uapi `bpf.h`: `#define BPF_FETCH 0x01`.
const BPF_FETCH: i32 = bs::BPF_FETCH as i32;

/// Top nibble of the atomic-op `imm` for atomic compare-and-write.
/// Combined with `BPF_FETCH` to form the full opcode. See linux uapi
/// `bpf.h`: `#define BPF_CMPXCHG (0xf0 | BPF_FETCH)`. libbpf-sys
/// exports `BPF_CMPXCHG` (the full 0xf1 opcode); the standalone top
/// nibble is derived by stripping the FETCH bit.
const BPF_CMPXCHG_TOP: i32 = (bs::BPF_CMPXCHG as i32) & !BPF_FETCH;

/// Atomic-op `imm` for `BPF_LOAD_ACQ`: `dst = smp_load_acquire(src
/// + off16)`. See linux include/linux/filter.h.
const BPF_LOAD_ACQ_IMM: i32 = bs::BPF_LOAD_ACQ as i32;
/// Atomic-op `imm` for `BPF_STORE_REL`: `smp_store_release(dst +
/// off16, src)`. See linux include/linux/filter.h.
const BPF_STORE_REL_IMM: i32 = bs::BPF_STORE_REL as i32;

/// Frame-pointer register index. See `BPF_REG_10 = 10` in linux
/// uapi `bpf.h`: r10 is the read-only frame pointer; STX/LDX through
/// it spill / reload the stack frame.
const BPF_REG_R10: usize = bs::BPF_REG_10 as usize;

/// `bpf_call->src_reg == BPF_PSEUDO_KFUNC_CALL` denotes that
/// `bpf_call->imm` is the BTF id of a `BTF_KIND_FUNC` in the running
/// kernel. Defined in linux uapi `bpf.h`.
pub(crate) const BPF_PSEUDO_KFUNC_CALL: u8 = bs::BPF_PSEUDO_KFUNC_CALL as u8;

/// Helper id for `bpf_map_lookup_elem` per linux uapi `bpf.h`
/// (`BPF_FUNC_map_lookup_elem = 1`, the second `bpf_func_id` enum
/// value after `BPF_FUNC_unspec = 0`). Sourced from `libbpf-sys`'s
/// bindgen translation of the same header.
///
/// Helper calls in pre-relocation `.bpf.o` bytecode carry
/// `src_reg == 0` (plain helper, distinct from `BPF_PSEUDO_CALL`
/// for BPF-to-BPF and `BPF_PSEUDO_KFUNC_CALL` for kfuncs) and
/// `imm` set to the helper id. The analyzer's [`BPF_OP_CALL`] arm
/// types R0 only for this single helper id — no other helper has a
/// pointer-to-struct return shape we can recover from the BPF
/// program BTF alone. The kernel's
/// `bpf_map_lookup_elem_proto::ret_type = RET_PTR_TO_MAP_VALUE_OR_NULL`
/// (linux `kernel/bpf/helpers.c`) is the correctness anchor: the
/// returned pointer points at the map's value bytes whose BTF type
/// is the map descriptor's `__type(value, T)` declaration.
const BPF_FUNC_MAP_LOOKUP_ELEM: i32 = bs::BPF_FUNC_map_lookup_elem as i32;

/// `bpf_call->src_reg == BPF_PSEUDO_CALL` denotes a BPF-to-BPF call:
/// `bpf_call->imm` is a pc-relative offset to another BPF function
/// in the same program. Pre-relocation `.bpf.o` files (the production
/// path) emit kfunc call sites with `src_reg = BPF_PSEUDO_CALL` and
/// `imm = -1`; libbpf's RELO_EXTERN_CALL handler rewrites them to
/// `src_reg = BPF_PSEUDO_KFUNC_CALL` + `imm = kfunc_btf_id` at load
/// time. The host-side cast loader mirrors that rewrite via
/// [`crate::vmm::cast_analysis_load`] before invoking
/// [`analyze_casts`], so the analyzer never has to distinguish pre-
/// from post-relocation forms — by the time it runs every kfunc
/// call carries its BTF id.
pub(crate) const BPF_PSEUDO_CALL: u8 = bs::BPF_PSEUDO_CALL as u8;

#[cfg(test)]
mod tests;
