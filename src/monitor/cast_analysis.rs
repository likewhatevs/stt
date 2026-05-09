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
//! Branches are handled conservatively: on every jump-target PC the
//! pre-pass identifies, register state AND stack-slot state are reset
//! before processing that PC. This drops casts that span branch joins
//! (false negative, acceptable). Function calls clobber `r0..=r5` per
//! the BPF ABI; kfunc return typing happens after the clobber.
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

use std::collections::{BTreeMap, BTreeSet};

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
    /// Renders as the lowercase address-space tag the renderer
    /// composes into [`crate::monitor::btf_render::RenderedValue::Ptr::cast_annotation`]
    /// (e.g. `"cast→arena"`, `"cast→kernel"`). Keeps the textual
    /// representation in one place so a new variant cannot drift
    /// between the analyzer enum and the operator-visible
    /// annotation.
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
/// `.rodata` global section — see [`DatasecPointer`].
///
/// `initial_regs`, `func_entries`, and `datasec_pointers` compose:
/// seeds apply once at PC 0, function-entry reseeding applies at
/// every matching `insn_offset`, and datasec annotations apply at
/// every matching `BPF_LD_IMM64` PC. Reseeding clears ALL registers
/// (R0..R10) and drops every stack slot (subprog entry semantics:
/// the callee's frame is fresh, and stale R6..R9 from linearly-
/// preceding unrelated functions must not leak). R1..R5 are then
/// re-seeded from the FuncProto's parameter types where they
/// resolve to struct pointers.
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
) -> CastMap {
    let mut analyzer = Analyzer::new(btf);
    analyzer.seed(initial_regs);
    let targets = jump_targets(insns);
    analyzer.run(insns, &targets, func_entries, datasec_pointers);
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
    /// `(source_struct_id, field_byte_offset)`. Participates in
    /// conflict detection only; does NOT emit standalone entries.
    /// The shape-inference path (`patterns`) and the kptr STX path
    /// (`kptr_findings`) are the only producers of map entries —
    /// arena_confirmed merely vetoes a kptr finding when the same
    /// slot was also observed as the source of an arena cast (the
    /// slot cannot simultaneously hold an arena VA and a kernel
    /// VA).
    arena_confirmed: BTreeSet<(u32, u32)>,
    /// Largest type id touched while resolving sources (struct
    /// pointer types and u64-field source structs). Used to bound
    /// the matcher's id walk below
    /// [`super::sdt_alloc::MAX_BTF_ID_PROBE`].
    max_seen_type_id: u32,
}

/// Kptr finding state: a single `(parent, offset)` slot may be
/// written by code paths that disagree on the target type. The
/// analyzer collapses the disagreement to `Conflicting` so finalize()
/// can drop it.
#[derive(Debug, Clone, Copy)]
enum KptrEntry {
    /// Single observed target type id.
    Single(u32),
    /// Two or more disjoint target ids observed; drop the slot.
    Conflicting,
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
            max_seen_type_id: 0,
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

            self.step(*insn, &mut skip_next, datasec_hit);

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

    fn step(&mut self, insn: BpfInsn, skip_next: &mut bool, datasec_hit: Option<(u32, u32)>) {
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
                    // the return value. Clobber first; for kfunc
                    // calls (src_reg == BPF_PSEUDO_KFUNC_CALL) the
                    // imm field carries the kernel BTF id of the
                    // kfunc — if its FuncProto return type peels
                    // through Ptr to a Struct/Union, set r0 to a
                    // typed pointer. Helper calls and BPF-to-BPF
                    // calls fall through with r0 left Unknown
                    // (false negative, acceptable: we do not have
                    // a typed-helper-return table).
                    for r in 0..=5 {
                        self.set_reg(r, RegState::Unknown);
                    }
                    let pseudo = insn.src_reg();
                    if pseudo == BPF_PSEUDO_KFUNC_CALL {
                        self.handle_kfunc_call(insn.imm);
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
                let canonical_field_off = match &member {
                    MemberAt::Struct { .. } => field_off,
                    MemberAt::Datasec {
                        var_byte_offset, ..
                    } => *var_byte_offset,
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
                            self.set_reg(
                                dst,
                                RegState::LoadedU64Field {
                                    source_struct_id: parent_btf_id,
                                    field_offset: canonical_field_off,
                                },
                            );
                            self.note_type_id(parent_btf_id);
                            self.patterns
                                .entry((parent_btf_id, canonical_field_off))
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
    /// Two roles:
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
        let RegState::Pointer {
            struct_type_id: target_struct_id,
        } = self.regs[src]
        else {
            return;
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
        // case the renderer handles natively; recording a kptr
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
        // Self-store is almost always a structural error (the
        // analyzer concluded `parent == target` because of
        // ambiguous pointer aliasing); reject to keep the false-
        // positive bar high. The Datasec parent path cannot
        // self-store: a datasec id is never the target struct id
        // of a kptr (kptrs target slab structs like task_struct),
        // so this gate fires only on the Pointer{struct} case in
        // practice. The unconditional check is the simplest safe
        // form.
        if parent_btf_id == target_struct_id {
            return;
        }
        // The Datasec path stores the variable's start offset
        // (matching `MemberAt::Datasec::var_byte_offset`) as the
        // canonical key, NOT the queried offset. For a plain u64
        // global the two are equal; for a struct global the
        // queried offset can land mid-struct but the kptr finding
        // is keyed on the variable's start so the renderer's
        // `(parent, member_offset)` lookup matches the variable
        // boundary. Lookups through the BSS-DATASEC parent then
        // surface the per-variable kptr just like a struct member
        // would.
        let canonical_field_off = match member {
            MemberAt::Struct { .. } => field_off,
            MemberAt::Datasec {
                var_byte_offset, ..
            } => var_byte_offset,
        };
        self.note_type_id(parent_btf_id);
        self.note_type_id(target_struct_id);
        let key = (parent_btf_id, canonical_field_off);
        match self.kptr_findings.get(&key).copied() {
            None => {
                self.kptr_findings
                    .insert(key, KptrEntry::Single(target_struct_id));
            }
            Some(KptrEntry::Single(prev)) if prev == target_struct_id => {
                // Same target observed again — keep Single.
            }
            Some(_) => {
                // Different target previously observed at the same
                // slot, or already collapsed to Conflicting. The
                // slot is ambiguous; drop it on finalize.
                self.kptr_findings.insert(key, KptrEntry::Conflicting);
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
    fn handle_kfunc_call(&mut self, imm: i32) {
        if imm <= 0 {
            return;
        }
        let func_btf_id = imm as u32;
        let proto = match self.btf.resolve_type_by_id(func_btf_id) {
            Ok(Type::Func(f)) => match f.get_type_id() {
                Ok(pid) => match self.btf.resolve_type_by_id(pid) {
                    Ok(Type::FuncProto(fp)) => fp,
                    _ => return,
                },
                Err(_) => return,
            },
            Ok(Type::FuncProto(fp)) => fp,
            _ => return,
        };
        let ret_id = proto.return_type_id();
        if ret_id == 0 {
            // Void return — R0 stays Unknown.
            return;
        }
        if let Some(sid) = super::bpf_map::resolve_to_struct_id(self.btf, ret_id) {
            self.regs[0] = RegState::Pointer {
                struct_type_id: sid,
            };
            self.note_type_id(sid);
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

        // Pre-build (offset, size) -> { type_id } so each pattern
        // does not re-walk the entire BTF id space. The walk stops
        // at the first sustained run of unresolved ids -- BTF id
        // tables are dense in practice but tolerate small gaps.
        let layout = build_layout_index(self.btf, max_id);

        // Arena/kptr conflict drop set: any (source, offset) slot
        // observed by BOTH the arena LDX path (`self.patterns` —
        // the slot was loaded as a u64 then dereferenced as a
        // pointer base) AND the kernel STX path (`self.kptr_findings`
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
            .filter(|k| self.kptr_findings.contains_key(k))
            .collect();

        // Arena pointer path: BTF-shape-inferred targets. Tagged as
        // AddrSpace::Arena because the source u64 field is itself
        // dereferenced and its target struct is recovered by
        // intersecting struct shapes across the observed access
        // pattern.
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
            // Intersection of candidate type ids across every
            // observed (offset, size). The first lookup seeds
            // `candidates` by cloning once; subsequent lookups
            // borrow from `layout` and intersect into a fresh set,
            // avoiding a per-access clone of the full candidate
            // BTreeSet.
            let mut iter = accesses.iter();
            let first = iter.next().expect("non-empty checked above");
            let empty = BTreeSet::new();
            let mut candidates: BTreeSet<u32> = layout
                .get(&(first.offset, first.size))
                .cloned()
                .unwrap_or_default();
            for acc in iter {
                let next = layout.get(&(acc.offset, acc.size)).unwrap_or(&empty);
                candidates = candidates.intersection(next).copied().collect();
                if candidates.is_empty() {
                    break;
                }
            }

            // Drop the source struct from candidates — a self-typed
            // cast (source.f → source*) matches tautologically.
            // Remove source and proceed: if exactly one non-source
            // candidate remains, emit it. The source's presence in
            // the candidate set is coincidental (it has a u64 field
            // at the same offset as the target's layout), not
            // evidence of ambiguity.
            candidates.remove(source);

            if candidates.len() == 1 {
                let target = candidates.into_iter().next().unwrap();
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

        // Arena-confirmed path: fields where a BPF_ADDR_SPACE_CAST
        // (off=1, imm=1) was observed on a register loaded from the
        // field. This is authoritative evidence the field holds an
        // arena pointer. However, without a resolved target type id
        // (from the shape-inference path), the renderer cannot chase
        // — emitting target_type_id=0 produces a Ptr with "unresolvable
        // size" skip reason, which is worse than the raw u64 fallback.
        // arena_confirmed participates in conflict detection only:
        // the conflict chain above includes arena_confirmed keys, so
        // a kptr finding on the same slot drops on both sides. It
        // does NOT emit standalone entries.

        // Kernel kptr path: directly observed STX of a typed
        // pointer into a u64 slot. The target type is known
        // exactly from the value register's RegState — no shape
        // inference needed. Conflicting writes to the same slot
        // (different target types) drop. Slots that ALSO appear
        // in the arena LDX path (`conflicting` above) drop on
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
fn build_layout_index(btf: &Btf, max_id: u32) -> BTreeMap<(u32, u32), BTreeSet<u32>> {
    let mut out: BTreeMap<(u32, u32), BTreeSet<u32>> = BTreeMap::new();
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
                    let size = match member_size_bytes(btf, m) {
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
    /// member's declared type.
    Struct { member_type_id: u32 },
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
            Self::Struct { member_type_id } => *member_type_id,
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
    let t = btf.resolve_type_by_id(parent_type_id).ok()?;
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
                if bit_off / 8 == byte_offset {
                    let member_type_id = m.get_type_id().ok()?;
                    return Some(MemberAt::Struct { member_type_id });
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
mod tests {
    use super::*;
    use std::io::Write;

    /// Append a NUL-terminated string to the BTF strings buffer and
    /// return its byte offset. Shared across the cast_analysis test
    /// fixtures so each `let push_name = |s, name| { ... }` closure
    /// stays out of the per-test setup.
    fn push_name(s: &mut Vec<u8>, name: &str) -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    }

    // ----- BTF synthesizers ---------------------------------------
    //
    // The BTF library is read-only; tests build a raw BTF byte blob
    // with a small writer and parse it via Btf::from_bytes. The
    // BTF wire format (header + type section + string section) is
    // documented in linux Documentation/bpf/btf.rst. The helpers
    // below cover only the kinds the cast analyzer needs to see:
    // BTF_KIND_INT (1), BTF_KIND_PTR (2), BTF_KIND_STRUCT (4),
    // BTF_KIND_FUNC (12), BTF_KIND_FUNC_PROTO (13).

    const BTF_MAGIC: u16 = 0xEB9F;
    const BTF_VERSION: u8 = 1;
    const BTF_HEADER_LEN: u32 = 24;

    const BTF_KIND_INT: u32 = 1;
    const BTF_KIND_PTR: u32 = 2;
    const BTF_KIND_STRUCT: u32 = 4;
    const BTF_KIND_UNION: u32 = 5;
    const BTF_KIND_FWD: u32 = 7;
    const BTF_KIND_TYPEDEF: u32 = 8;
    const BTF_KIND_VOLATILE: u32 = 9;
    const BTF_KIND_CONST: u32 = 10;
    const BTF_KIND_FUNC: u32 = 12;
    const BTF_KIND_FUNC_PROTO: u32 = 13;
    /// `BTF_KIND_VAR = 14` — global variable declaration.
    /// References its underlying type via the post-header `type`
    /// u32 and carries a linkage u32 (static / global / extern).
    const BTF_KIND_VAR: u32 = 14;
    /// `BTF_KIND_DATASEC = 15` — global section (`.bss`,
    /// `.data`, `.rodata`, `.data.<name>`). Carries a list of
    /// `VarSecinfo` records that point at `BTF_KIND_VAR` entries.
    const BTF_KIND_DATASEC: u32 = 15;

    /// `info` field bit 31: encodes bitfield-style member offset / vs
    /// regular union for Fwd. Per linux uapi `btf.h` and
    /// `btf-rs::cbtf::btf_type::kind_flag`. Splitting the constant out
    /// keeps the OR expressions in `build_btf` readable when emitting
    /// kind_flag=1 structs / fwd-union.
    const KIND_FLAG_BIT: u32 = 1 << 31;

    /// One member in a synthetic struct.
    #[derive(Clone, Copy)]
    struct SynMember {
        name_off: u32,
        type_id: u32,
        /// Byte offset; converted to bit offset on emit.
        byte_offset: u32,
    }

    /// One parameter in a synthetic FuncProto. `type_id` is the BTF
    /// type id of the parameter's type (0 + name_off=0 marks the
    /// variadic sentinel — tests don't emit it).
    #[derive(Clone, Copy)]
    struct SynParam {
        name_off: u32,
        type_id: u32,
    }

    /// One member in a synthetic struct/union built with kind_flag=1.
    /// Encodes `bit_offset` in the low 24 bits and `bitfield_size_bits`
    /// in the upper 8 bits of the member's `offset` u32 (per linux uapi
    /// `btf.h` and `btf-rs::Member::bitfield_size`). Used by tests that
    /// exercise bitfield handling in `build_layout_index` and
    /// `struct_member_at`.
    #[derive(Clone, Copy)]
    struct SynMemberBits {
        name_off: u32,
        type_id: u32,
        /// Member offset in BITS (NOT bytes). For non-bit-aligned
        /// members the test sets a bit position that is not a multiple
        /// of 8.
        bit_offset: u32,
        /// 0 for a non-bitfield member in a kind_flag=1 struct, > 0
        /// for an actual bitfield. Production skips members with
        /// `bitfield_size > 0` AND members whose `bit_offset % 8 != 0`.
        bitfield_size_bits: u32,
    }

    /// One synthetic BTF type. Tests build a Vec<SynType>; the
    /// writer assigns ids starting at 1 (id 0 is Void) and emits
    /// the type section in order.
    #[allow(dead_code)] // not all variants are used by every test
    enum SynType {
        Int {
            name_off: u32,
            size: u32,
            encoding: u32,
            offset: u32,
            bits: u32,
        },
        Ptr {
            type_id: u32,
        },
        Struct {
            name_off: u32,
            size: u32,
            members: Vec<SynMember>,
        },
        /// `BTF_KIND_UNION` — same wire layout as Struct (members are
        /// `btf_member` records). `btf-rs` aliases `Union = Struct`,
        /// so production code paths walk both via the same
        /// `Type::Struct(s) | Type::Union(s)` arm.
        Union {
            name_off: u32,
            size: u32,
            members: Vec<SynMember>,
        },
        /// `BTF_KIND_STRUCT` with kind_flag=1 — member `offset` u32
        /// packs `bit_offset` in low 24 bits and `bitfield_size` in
        /// upper 8 bits. Real C structs that contain ANY bitfield
        /// member have kind_flag=1 even for non-bitfield members in
        /// the same struct (per linux uapi `btf.h`).
        StructBitfields {
            name_off: u32,
            size: u32,
            members: Vec<SynMemberBits>,
        },
        /// `BTF_KIND_FWD` — forward declaration. No payload after the
        /// `btf_type` header. `kind_flag` selects struct (0) vs union
        /// (1) per `btf-rs::Fwd::is_struct` / `is_union`. Used by
        /// tests that probe `member_size_bytes` for unsupported
        /// terminals via a struct member typed as Fwd.
        Fwd {
            name_off: u32,
            kind_flag: u32,
        },
        /// `BTF_KIND_TYPEDEF` — `typedef X T`. Same wire layout as
        /// `Ptr`: header followed by a `u32 type` referencing the
        /// underlying type id. `peel_modifiers` peels through it.
        Typedef {
            name_off: u32,
            type_id: u32,
        },
        /// `BTF_KIND_VOLATILE` — `volatile T`. Same wire layout as
        /// `Ptr`. `name_off` is 0 per the BTF spec.
        Volatile {
            type_id: u32,
        },
        /// `BTF_KIND_CONST` — `const T`. Same wire layout as `Ptr`.
        /// `name_off` is 0 per the BTF spec.
        Const {
            type_id: u32,
        },
        /// `BTF_KIND_FUNC`. `type_id` is the FuncProto id; `vlen`
        /// encodes BTF_FUNC_STATIC/GLOBAL/EXTERN (0/1/2).
        Func {
            name_off: u32,
            type_id: u32,
            linkage: u32,
        },
        /// `BTF_KIND_FUNC_PROTO`. `return_type_id` is the BTF id
        /// of the return type (0 = void); `params` enumerate the
        /// parameter list. `name_off` is always 0 per the BTF
        /// spec (FuncProto types are anonymous).
        FuncProto {
            return_type_id: u32,
            params: Vec<SynParam>,
        },
        /// `BTF_KIND_VAR` — global variable declaration. Wire
        /// layout: header + `u32 type` + `u32 linkage`. Used as
        /// the entry-pointed-to from a `BTF_KIND_DATASEC`
        /// `VarSecinfo`, since libbpf always emits Datasec
        /// entries pointing at a Var (the Var carries the
        /// variable's name and references its underlying C type).
        /// `linkage` mirrors `BTF_VAR_STATIC=0` /
        /// `BTF_VAR_GLOBAL_ALLOCATED=1` /
        /// `BTF_VAR_GLOBAL_EXTERN=2`.
        Var {
            name_off: u32,
            type_id: u32,
            linkage: u32,
        },
        /// `BTF_KIND_DATASEC` — a global section (`.bss`,
        /// `.data`, `.rodata`, `.data.<name>`). Wire layout:
        /// header + `u32 size` + per-VarSecinfo records of
        /// `{u32 type; u32 offset; u32 size}`. Each VarSecinfo
        /// references a `BTF_KIND_VAR` whose underlying type is
        /// the global's C type.
        Datasec {
            name_off: u32,
            size: u32,
            entries: Vec<SynVarSecinfo>,
        },
    }

    /// One entry in a synthetic `BTF_KIND_DATASEC`. Wire layout:
    /// `{u32 type_id; u32 offset; u32 size}` per VarSecinfo. The
    /// `type_id` references a `BTF_KIND_VAR` whose underlying
    /// type is the global's C type.
    #[derive(Clone, Copy)]
    struct SynVarSecinfo {
        type_id: u32,
        /// Byte offset of the variable within the section.
        offset: u32,
        /// Byte size of the variable's storage (matches the
        /// underlying type's `type_size`).
        size: u32,
    }

    /// Build a minimal BTF byte blob for testing.
    ///
    /// `strings` is the string section payload (must start with
    /// `\0`). Type ids start at 1 and increase in `types` order.
    fn build_btf(types: &[SynType], strings: &[u8]) -> Vec<u8> {
        let mut type_section = Vec::new();
        for ty in types {
            match ty {
                SynType::Int {
                    name_off,
                    size,
                    encoding,
                    offset,
                    bits,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let info = (BTF_KIND_INT << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                    type_section.extend_from_slice(&int_data.to_le_bytes());
                }
                SynType::Ptr { type_id } => {
                    let name_off: u32 = 0;
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let info = (BTF_KIND_PTR << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&type_id.to_le_bytes());
                }
                SynType::Struct {
                    name_off,
                    size,
                    members,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let vlen = members.len() as u32;
                    let info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    for m in members {
                        type_section.extend_from_slice(&m.name_off.to_le_bytes());
                        type_section.extend_from_slice(&m.type_id.to_le_bytes());
                        // Non-bitfield: bit_offset = byte * 8.
                        let bit_off = m.byte_offset * 8;
                        type_section.extend_from_slice(&bit_off.to_le_bytes());
                    }
                }
                SynType::Union {
                    name_off,
                    size,
                    members,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let vlen = members.len() as u32;
                    // Same wire encoding as Struct, only the kind id
                    // differs. kind_flag=0 (regular union, members
                    // carry plain bit_offset).
                    let info = ((BTF_KIND_UNION << 24) & 0x1f00_0000) | (vlen & 0xffff);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    for m in members {
                        type_section.extend_from_slice(&m.name_off.to_le_bytes());
                        type_section.extend_from_slice(&m.type_id.to_le_bytes());
                        // Union members all sit at bit offset 0 in
                        // real C, but the wire format still carries a
                        // u32 offset. Allow tests to set arbitrary
                        // byte_offset values to exercise the
                        // struct_member_at lookup logic; production
                        // matches on byte position regardless of kind.
                        let bit_off = m.byte_offset * 8;
                        type_section.extend_from_slice(&bit_off.to_le_bytes());
                    }
                }
                SynType::StructBitfields {
                    name_off,
                    size,
                    members,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let vlen = members.len() as u32;
                    // kind_flag=1 (bit 31 set): member offset packs
                    // (bitfield_size << 24) | (bit_offset & 0xffffff).
                    let info =
                        (((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff)) | KIND_FLAG_BIT;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    for m in members {
                        type_section.extend_from_slice(&m.name_off.to_le_bytes());
                        type_section.extend_from_slice(&m.type_id.to_le_bytes());
                        let packed =
                            ((m.bitfield_size_bits & 0xff) << 24) | (m.bit_offset & 0x00ff_ffff);
                        type_section.extend_from_slice(&packed.to_le_bytes());
                    }
                }
                SynType::Fwd {
                    name_off,
                    kind_flag,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    // BTF_KIND_FWD: vlen is unused (set to 0); the
                    // kind_flag bit encodes struct (0) vs union (1).
                    // size_type field is also unused per the kernel
                    // wire format but is still 4 bytes long; emit 0.
                    let info = ((BTF_KIND_FWD << 24) & 0x1f00_0000) | ((*kind_flag & 0x1) << 31);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&0u32.to_le_bytes());
                }
                SynType::Typedef { name_off, type_id } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let info = (BTF_KIND_TYPEDEF << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&type_id.to_le_bytes());
                }
                SynType::Volatile { type_id } => {
                    let name_off: u32 = 0;
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let info = (BTF_KIND_VOLATILE << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&type_id.to_le_bytes());
                }
                SynType::Const { type_id } => {
                    let name_off: u32 = 0;
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let info = (BTF_KIND_CONST << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&type_id.to_le_bytes());
                }
                SynType::Func {
                    name_off,
                    type_id,
                    linkage,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    // BTF_KIND_FUNC encodes the linkage in vlen
                    // (0=static, 1=global, 2=extern).
                    let info = ((BTF_KIND_FUNC << 24) & 0x1f00_0000) | (*linkage & 0xffff);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&type_id.to_le_bytes());
                }
                SynType::FuncProto {
                    return_type_id,
                    params,
                } => {
                    let name_off: u32 = 0;
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let vlen = params.len() as u32;
                    let info = ((BTF_KIND_FUNC_PROTO << 24) & 0x1f00_0000) | (vlen & 0xffff);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&return_type_id.to_le_bytes());
                    for p in params {
                        type_section.extend_from_slice(&p.name_off.to_le_bytes());
                        type_section.extend_from_slice(&p.type_id.to_le_bytes());
                    }
                }
                SynType::Var {
                    name_off,
                    type_id,
                    linkage,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    // BTF_KIND_VAR = 14. Header followed by
                    // `u32 type` + `u32 linkage`.
                    let info = (BTF_KIND_VAR << 24) & 0x1f00_0000;
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&type_id.to_le_bytes());
                    type_section.extend_from_slice(&linkage.to_le_bytes());
                }
                SynType::Datasec {
                    name_off,
                    size,
                    entries,
                } => {
                    type_section.extend_from_slice(&name_off.to_le_bytes());
                    let vlen = entries.len() as u32;
                    // BTF_KIND_DATASEC = 15. Header followed by
                    // `u32 size` + per-entry `{u32 type, u32
                    // offset, u32 size}`.
                    let info = ((BTF_KIND_DATASEC << 24) & 0x1f00_0000) | (vlen & 0xffff);
                    type_section.extend_from_slice(&info.to_le_bytes());
                    type_section.extend_from_slice(&size.to_le_bytes());
                    for e in entries {
                        type_section.extend_from_slice(&e.type_id.to_le_bytes());
                        type_section.extend_from_slice(&e.offset.to_le_bytes());
                        type_section.extend_from_slice(&e.size.to_le_bytes());
                    }
                }
            }
        }

        let type_len = type_section.len() as u32;
        let str_len = strings.len() as u32;

        let mut blob = Vec::new();
        // Header (24 bytes).
        blob.write_all(&BTF_MAGIC.to_le_bytes()).unwrap();
        blob.push(BTF_VERSION);
        blob.push(0); // flags
        blob.write_all(&BTF_HEADER_LEN.to_le_bytes()).unwrap();
        blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
        blob.write_all(&type_len.to_le_bytes()).unwrap();
        blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off (= type_len)
        blob.write_all(&str_len.to_le_bytes()).unwrap();
        blob.extend_from_slice(&type_section);
        blob.extend_from_slice(strings);
        blob
    }

    // BTF int encoding flags: signed = 1, char = 2, bool = 4. The
    // synthesizer uses 0 for plain unsigned.

    /// Helper: build a BTF with `task_struct`-like source struct
    /// `T` (id=2) and target struct `Q` (id=3). T has a u64 field
    /// at byte offset `field_off` named `f`. Q has a u32 at byte
    /// offset `target_off`. Returns the byte blob and the (T_id,
    /// Q_id) pair.
    fn btf_with_source_and_target(field_off: u32, target_off: u32) -> (Vec<u8>, u32, u32) {
        // Strings: null + names. Order matters since name_offs
        // index into this byte vector.
        let mut strings: Vec<u8> = vec![0];
        let n_int = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");

        let types = vec![
            // id 1: int u64 (size=8, bits=64).
            SynType::Int {
                name_off: n_int,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: struct T { ... f at field_off ... } size = field_off + 8.
            SynType::Struct {
                name_off: n_t,
                size: field_off + 8,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: field_off,
                }],
            },
            // id 3: struct Q { u64 x at target_off }, size = target_off + 8.
            SynType::Struct {
                name_off: n_q,
                size: target_off + 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: target_off,
                }],
            },
        ];
        (build_btf(&types, &strings), 2, 3)
    }

    /// Small helper to emit a single [`BpfInsn`] with given fields.
    /// Uses `BpfInsn::new` directly so dst/src register packing is
    /// done by the constructor — no `let mut x = X::default(); x.f = …`
    /// clippy footgun, no separate setter calls.
    fn mk_insn(code: u8, dst: u8, src: u8, off: i16, imm: i32) -> BpfInsn {
        BpfInsn::new(code, dst, src, off, imm)
    }

    fn ldx(size: u8, dst: u8, src: u8, off: i16) -> BpfInsn {
        mk_insn(BPF_CLASS_LDX | size | BPF_MODE_MEM, dst, src, off, 0)
    }

    /// `*(size *)(r_dst + off) = r_src`. Plain memory store (BPF_MEM
    /// mode), the spill / kptr-write encoding the analyzer cares
    /// about. dst is the address-base register, src is the value.
    fn stx(size: u8, dst: u8, src: u8, off: i16) -> BpfInsn {
        mk_insn(BPF_CLASS_STX | size | BPF_MODE_MEM, dst, src, off, 0)
    }

    fn mov_x(dst: u8, src: u8) -> BpfInsn {
        mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, dst, src, 0, 0)
    }

    fn mov_k(dst: u8, imm: i32) -> BpfInsn {
        mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV, dst, 0, 0, imm)
    }

    fn call() -> BpfInsn {
        mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, 0, 0, 1)
    }

    /// `BPF_CALL` with `src_reg == BPF_PSEUDO_KFUNC_CALL` and
    /// `imm == kfunc_btf_id`. Models the relocated call form where
    /// the imm carries the BTF id of a `BTF_KIND_FUNC`.
    fn kfunc_call(kfunc_btf_id: u32) -> BpfInsn {
        mk_insn(
            BPF_CLASS_JMP | BPF_OP_CALL,
            0,
            BPF_PSEUDO_KFUNC_CALL,
            0,
            kfunc_btf_id as i32,
        )
    }

    fn exit() -> BpfInsn {
        mk_insn(BPF_CLASS_JMP | BPF_OP_EXIT, 0, 0, 0, 0)
    }

    // ----- Tests --------------------------------------------------

    #[test]
    fn empty_insns_yields_empty_map() {
        let (blob, _t, _q) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let map = analyze_casts(&[], &btf, &[], &[], &[]);
        assert!(map.is_empty());
    }

    #[test]
    fn no_initial_seed_yields_empty_map() {
        // Without seeding any register as a struct pointer, the
        // analyzer cannot identify the source type of an LDX.
        let (blob, _t, _q) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(&insns, &btf, &[], &[], &[]);
        assert!(map.is_empty());
    }

    #[test]
    fn simple_cast_recovers_target() {
        // r1 -> *(T *).
        // r2 = *(u64 *)(r1 + 8)   -- "load u64 at T.f"
        // r3 = *(u64 *)(r2 + 0)   -- "use loaded value as Q*"
        // The unique struct in BTF with a u64 at offset 0 is Q.
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "got: {map:?}"
        );
    }

    #[test]
    fn ambiguous_targets_drop_silently() {
        // Build BTF with two structs having a u64 at offset 0
        // (both Q1 and Q2 match the access pattern). Cast must NOT
        // be recorded because false positives are unacceptable.
        let mut strings: Vec<u8> = vec![0];
        let n_int = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q1 = push_name(&mut strings, "Q1");
        let n_q2 = push_name(&mut strings, "Q2");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");
        let types = vec![
            SynType::Int {
                name_off: n_int,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            SynType::Struct {
                name_off: n_q1,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Struct {
                name_off: n_q2,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: 2,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "ambiguous candidates must drop: {map:?}");
    }

    #[test]
    fn multi_offset_disambiguates_target() {
        // Two Q-shaped structs differ by their second offset:
        //   Q1: { u64 @0; u64 @8 }
        //   Q2: { u64 @0; u32 @8 }
        // When the BPF program reads both Q->@0 (u64) and
        // Q->@8 (u64), only Q1 fits. The intersection-based
        // matcher must converge to Q1.
        let mut strings: Vec<u8> = vec![0];
        let n_u32 = push_name(&mut strings, "u32");
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q1 = push_name(&mut strings, "Q1");
        let n_q2 = push_name(&mut strings, "Q2");
        let n_f = push_name(&mut strings, "f");
        let n_a = push_name(&mut strings, "a");
        let n_b = push_name(&mut strings, "b");
        let types = vec![
            SynType::Int {
                name_off: n_u32,
                size: 4,
                encoding: 0,
                offset: 0,
                bits: 32,
            },
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 2,
                    byte_offset: 8,
                }],
            },
            SynType::Struct {
                name_off: n_q1,
                size: 16,
                members: vec![
                    SynMember {
                        name_off: n_a,
                        type_id: 2,
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_b,
                        type_id: 2,
                        byte_offset: 8,
                    },
                ],
            },
            SynType::Struct {
                name_off: n_q2,
                size: 16,
                members: vec![
                    SynMember {
                        name_off: n_a,
                        type_id: 2,
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 3;
        let q1_id = 4;
        // Sequence: load r1->T.f (offset 8) into r2; then deref r2
        // at offset 0 (8 bytes) and offset 8 (8 bytes).
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            ldx(BPF_SIZE_DW, 4, 2, 8),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q1_id,
                addr_space: AddrSpace::Arena
            }),
            "map: {map:?}"
        );
    }

    #[test]
    fn multiple_distinct_casts_recorded() {
        // T has TWO u64 fields, each loaded and dereferenced as a
        // distinct target struct. Q1 has u64@8 only (no @0). Q2
        // has u64@0 + u32@8. The two cast access patterns each
        // narrow to a single candidate.
        let mut strings: Vec<u8> = vec![0];
        let n_u32 = push_name(&mut strings, "u32");
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q1 = push_name(&mut strings, "Q1");
        let n_q2 = push_name(&mut strings, "Q2");
        let n_f1 = push_name(&mut strings, "f1");
        let n_f2 = push_name(&mut strings, "f2");
        let n_a = push_name(&mut strings, "a");
        let n_b = push_name(&mut strings, "b");
        let types = vec![
            SynType::Int {
                name_off: n_u32,
                size: 4,
                encoding: 0,
                offset: 0,
                bits: 32,
            },
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 24,
                members: vec![
                    SynMember {
                        name_off: n_f1,
                        type_id: 2,
                        byte_offset: 8,
                    },
                    SynMember {
                        name_off: n_f2,
                        type_id: 2,
                        byte_offset: 16,
                    },
                ],
            },
            SynType::Struct {
                name_off: n_q1,
                size: 16,
                members: vec![SynMember {
                    name_off: n_a,
                    type_id: 2,
                    byte_offset: 8,
                }],
            },
            SynType::Struct {
                name_off: n_q2,
                size: 12,
                members: vec![
                    SynMember {
                        name_off: n_a,
                        type_id: 2,
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        // Type ids per `types` order: u32=1, u64=2, T=3, Q1=4, Q2=5.
        let t_id = 3;
        let q1_id = 4;
        let q2_id = 5;

        // Cast 1: T.f1 -> *(Q1*). Read at offset 8 (8 bytes) only;
        // Q1 matches, Q2 has u32@8 (4 bytes) so does not match.
        // Cast 2: T.f2 -> *(Q2*). Read at offset 0 (8 bytes) and
        // offset 8 (4 bytes). Q1 lacks @0 → only Q2 matches.
        let insns = vec![
            // r2 = *(u64 *)(r1 + 8)  -- T.f1 → r2
            ldx(BPF_SIZE_DW, 2, 1, 8),
            // r3 = *(u64 *)(r2 + 8)  -- (T.f1 → Q1).a (offset 8, size 8)
            ldx(BPF_SIZE_DW, 3, 2, 8),
            // Reset r2's loaded-state by overwriting via mov_k.
            mov_k(2, 0),
            // r2 = *(u64 *)(r1 + 16) -- T.f2 → r2
            ldx(BPF_SIZE_DW, 2, 1, 16),
            // r4 = *(u64 *)(r2 + 0)  -- (T.f2 → Q2).a (offset 0, size 8)
            ldx(BPF_SIZE_DW, 4, 2, 0),
            // r5 = *(u32 *)(r2 + 8)  -- (T.f2 → Q2).b (offset 8, size 4)
            ldx(BPF_SIZE_W, 5, 2, 8),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q1_id,
                addr_space: AddrSpace::Arena
            }),
            "f1: {map:?}"
        );
        assert_eq!(
            map.get(&(t_id, 16)),
            Some(&CastHit {
                target_type_id: q2_id,
                addr_space: AddrSpace::Arena
            }),
            "f2: {map:?}"
        );
    }

    #[test]
    fn register_reuse_after_call_clears_state() {
        // Load T.f into r2, then BPF_CALL clobbers r0..r5. The
        // dereference of the post-call r2 must NOT be attributed
        // to the pre-call source.
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8), // r2 = T.f
            call(),                    // clobbers r0..r5
            ldx(BPF_SIZE_DW, 3, 2, 0), // r2 is Unknown now
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "post-call r2 must not retain T.f source: {map:?}"
        );
    }

    #[test]
    fn nondw_load_does_not_track_u64_field() {
        // r2 = *(u32 *)(r1 + 8)  -- not a 64-bit load, cannot carry
        // a pointer. Subsequent deref must not be attributed.
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_W, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "32-bit load must not seed cast: {map:?}");
    }

    #[test]
    fn ptr_field_tracked_as_typed_pointer_not_cast() {
        // T.field is declared as `Q *` in BTF (already typed). The
        // analyzer follows the chain to mark the loaded register
        // as a Q*, but does NOT record a cast (renderer already
        // chases declared Ptr fields).
        let mut strings: Vec<u8> = vec![0];
        let n_int = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");
        let types = vec![
            // id 1: u64
            SynType::Int {
                name_off: n_int,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: struct Q { u64 x @0 }
            SynType::Struct {
                name_off: n_q,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id 3: Q* (pointer to id=2)
            SynType::Ptr { type_id: 2 },
            // id 4: struct T { Q* f @8 }
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 3,
                    byte_offset: 8,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 4;
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8), // r2 = T.f -- typed Q* per BTF
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "typed Ptr field must not be recorded as cast: {map:?}"
        );
    }

    #[test]
    fn null_check_fall_through_preserves_state() {
        // if r2 <COND> 0 goto SKIP; deref r2; SKIP: exit.
        // The deref happens at the FALL-THROUGH after the
        // conditional jump, so the state survives. The analyzer
        // should still record the cast across every supported
        // conditional-jump op-code: per linux uapi `bpf_common.h`
        // and `bpf.h` the JMP class accepts JEQ=0x10, JGT=0x20,
        // JGE=0x30, JNE=0x50, JSGT=0x60, JSGE=0x70, JLT=0xa0,
        // JLE=0xb0, JSLT=0xc0, JSLE=0xd0 (JSET=0x40 also branches
        // but takes a bitmask not a comparison; covered too). Each
        // pairs with BPF_SRC_K (0x00) for an immediate operand; the
        // K and X variants share the same off-relative branch
        // encoding so testing K covers both as far as
        // `jump_targets()` is concerned. JMP32 class mirrors the
        // op codes; covered with BPF_CLASS_JMP32 | BPF_JEQ to verify
        // class-bit independence. None of these touch register
        // state in `step()` (only BPF_OP_CALL clears registers per
        // line ~726), so every variant must preserve the
        // pre-jump LoadedU64Field on the fall-through path.
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_SRC_K is 0; OR with BPF_SRC_X (0x08) for the X-form
        // smoke test on JEQ to confirm src-kind has no effect on
        // jump-target detection or fall-through state preservation.
        let variants: &[(u8, &str)] = &[
            (BPF_CLASS_JMP | 0x10, "JEQ_K"),
            (BPF_CLASS_JMP | 0x10 | BPF_SRC_X, "JEQ_X"),
            (BPF_CLASS_JMP | 0x20, "JGT_K"),
            (BPF_CLASS_JMP | 0x30, "JGE_K"),
            (BPF_CLASS_JMP | 0x40, "JSET_K"),
            (BPF_CLASS_JMP | 0x50, "JNE_K"),
            (BPF_CLASS_JMP | 0x60, "JSGT_K"),
            (BPF_CLASS_JMP | 0x70, "JSGE_K"),
            (BPF_CLASS_JMP | 0xa0, "JLT_K"),
            (BPF_CLASS_JMP | 0xb0, "JLE_K"),
            (BPF_CLASS_JMP | 0xc0, "JSLT_K"),
            (BPF_CLASS_JMP | 0xd0, "JSLE_K"),
            (BPF_CLASS_JMP32 | 0x10, "JEQ32_K"),
        ];
        for (code, label) in variants {
            // pc 0: r2 = T.f
            // pc 1: if r2 <COND> 0 goto +1 (jump to pc=3, skip deref)
            // pc 2: r3 = *r2  (fall-through; r2 still LoadedU64Field)
            // pc 3: exit.
            let jcc = mk_insn(*code, 2, 0, 1, 0);
            let insns = vec![
                ldx(BPF_SIZE_DW, 2, 1, 8),
                jcc,
                ldx(BPF_SIZE_DW, 3, 2, 0),
                exit(),
            ];
            let map = analyze_casts(
                &insns,
                &btf,
                &[InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                }],
                &[],
                &[],
            );
            assert_eq!(
                map.len(),
                1,
                "{label}: exactly one cast expected on fall-through, got: {map:?}"
            );
            assert_eq!(
                map.get(&(t_id, 8)),
                Some(&CastHit {
                    target_type_id: q_id,
                    addr_space: AddrSpace::Arena
                }),
                "{label}: fall-through deref must record: {map:?}"
            );
        }
    }

    #[test]
    fn deref_at_jump_target_is_dropped() {
        // if r2 != 0 goto USE; ... USE: deref r2.
        // The deref is at the branch target, where state is reset
        // by the conservative join handler. False negative is
        // acceptable.
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // pc 0: r2 = T.f
        // pc 1: if r2 != 0 goto +1 (= pc 3, the deref)
        // pc 2: exit (skipped on the taken branch)
        // pc 3: r3 = *r2  -- STATE WAS RESET at pc 3 (target).
        // pc 4: exit.
        let jne = mk_insn(BPF_CLASS_JMP | 0x50, 2, 0, 1, 0); // BPF_JNE_K = 0x50
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            jne,
            exit(),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "deref at branch target must drop: {map:?}");
    }

    #[test]
    fn mov_x_propagates_loaded_state() {
        // r2 = T.f; r4 = r2; deref r4 at offset 0.
        // The MOV r4, r2 propagates LoadedU64Field, so the
        // dereference through r4 records the cast.
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            mov_x(4, 2),
            ldx(BPF_SIZE_DW, 3, 4, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "MOV must propagate: {map:?}"
        );
    }

    #[test]
    fn ld_imm64_skips_second_slot() {
        // BPF_LD_IMM64 is two slots; the second slot's `code` is 0.
        // A bare 0-code insn must not be misinterpreted as anything
        // active. After the two-slot insn, normal flow continues.
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let ld_imm64_lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, 6, 0, 0, 0);
        let ld_imm64_hi = mk_insn(0, 0, 0, 0, 0);
        let insns = vec![
            ld_imm64_lo,
            ld_imm64_hi,
            ldx(BPF_SIZE_DW, 2, 1, 8),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "LD_IMM64 second slot must skip: {map:?}"
        );
    }

    #[test]
    fn r10_seed_rejected() {
        // Seeding the frame pointer is silently dropped — even
        // when the BTF type id is valid. Nothing tracks r10.
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 10, 8),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 10,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "r10 seed must be ignored: {map:?}");
    }

    #[test]
    fn nonu64_field_at_source_offset_not_tracked() {
        // T has a u32 at offset 8 (not u64). Loading from there
        // and treating as a pointer is meaningless — the analyzer
        // must not seed LoadedU64Field.
        let mut strings: Vec<u8> = vec![0];
        let n_u32 = push_name(&mut strings, "u32");
        let n_t = push_name(&mut strings, "T");
        let n_f = push_name(&mut strings, "f");
        let types = vec![
            SynType::Int {
                name_off: n_u32,
                size: 4,
                encoding: 0,
                offset: 0,
                bits: 32,
            },
            SynType::Struct {
                name_off: n_t,
                size: 12,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "u32-typed field must not seed cast: {map:?}"
        );
    }

    // ----- Kptr detection helpers ---------------------------------
    //
    // Kptr tests share a BTF shape: a "task_struct"-like target T,
    // a parent struct P with a `u64 slot @ off` field, and the
    // appropriate FuncProto / Func types when a function-entry
    // seeding test calls them out. The helper keeps each test
    // focused on the instruction sequence under examination.

    /// Build a BTF blob with:
    /// - id 1: u64
    /// - id 2: struct T { u64 x @ 0 }   ("task_struct" stand-in)
    /// - id 3: T*  (pointer to id=2)
    /// - id 4: struct P { u64 slot @ slot_off }
    ///
    /// Returns (blob, T_id, P_id, T_ptr_id). Tests that need a
    /// FuncProto add it on top of this blob.
    fn btf_kptr_base(slot_off: u32) -> (Vec<u8>, u32, u32, u32) {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let types = vec![
            // id 1: u64
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: struct T { u64 x @ 0 }
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id 3: T*
            SynType::Ptr { type_id: 2 },
            // id 4: struct P { u64 slot @ slot_off }
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        (blob, 2, 4, 3)
    }

    #[test]
    fn kptr_from_function_param_stored_to_u64_field() {
        // R1 starts as T* (param[0]).
        // R6 = R1 (preserve across the rest).
        // R2 = some P* (the parent struct holding the kptr slot).
        //   We don't have a separate P param so seed R2 directly.
        // *(u64 *)(R2 + slot_off) = R6
        //   - R2 is Pointer{P}, R6 is Pointer{T}, field is u64 ->
        //     map records (P, slot_off) -> (T, AddrSpace::Kernel).
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![mov_x(6, 1), stx(BPF_SIZE_DW, 2, 6, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 2,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "kptr STX must record kernel-space cast: {map:?}"
        );
    }

    #[test]
    fn kptr_through_stack_spill() {
        // R1 starts as T*; spill to [r10-8]; reload into R3; store
        // R3 into the parent slot. Tests that stack spill / reload
        // preserves the typed-pointer state.
        //
        //   *(u64 *)(r10 - 8) = R1     ; spill T*
        //   R3 = *(u64 *)(r10 - 8)     ; reload as T*
        //   *(u64 *)(R4 + slot_off) = R3
        let slot_off: u32 = 24;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8), // spill R1
            ldx(BPF_SIZE_DW, 3, 10, -8), // reload to R3
            stx(BPF_SIZE_DW, 4, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 4,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "stack spill must preserve typed pointer: {map:?}"
        );
    }

    #[test]
    fn kptr_from_kfunc_return() {
        // BTF layout reused across this test:
        //   id 1: u64
        //   id 2: struct T { u64 x @ 0 }
        //   id 3: T*
        //   id 4: struct P { u64 slot @ 16 }
        //   id 5: FuncProto returning T*  (return_type_id = 3)
        //   id 6: Func("bpf_task_acquire") -> id 5
        //
        // Sequence:
        //   call kfunc id=6
        //   *(u64 *)(R6 + 16) = R0   ; R6 is P*
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let n_kfunc = push_name(&mut strings, "bpf_task_acquire");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            // id 5: FuncProto -> T*
            SynType::FuncProto {
                return_type_id: 3,
                params: vec![],
            },
            // id 6: Func bpf_task_acquire (linkage = global)
            SynType::Func {
                name_off: n_kfunc,
                type_id: 5,
                linkage: 1,
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let kfunc_id = 6;
        let insns = vec![
            kfunc_call(kfunc_id),
            stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: p_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "kfunc-returned T* stored to P.slot must record: {map:?}"
        );
    }

    #[test]
    fn kptr_clobbered_by_call() {
        // R1 starts as T*. A non-kfunc BPF_CALL clobbers R0..R5.
        // The post-call STX of R1 must NOT record a kptr — R1 is
        // Unknown after the helper call.
        //
        //   call helper        ; clobbers R0..R5
        //   *(u64 *)(R6 + 16) = R1   ; R1 was clobbered
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![call(), stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "post-call clobbered R1 must not record kptr: {map:?}"
        );
    }

    #[test]
    fn mixed_arena_and_kptr_in_one_program() {
        // Single BTF, single instruction sequence: trigger BOTH
        // detection paths.
        //
        // BTF:
        //   id 1: u64
        //   id 2: struct T { u64 x @ 0 }     (kernel kptr target)
        //   id 3: T*
        //   id 4: struct A { u64 a0 @ 0; u64 a1 @ 8 }  (arena target)
        //   id 5: struct M {                ; map value
        //           u64 arena_ptr @ 0;     ; carries A*
        //           u64 kptr      @ 16;    ; carries T*
        //         }
        //
        // Instructions:
        //   r1 := M*        (seed via InitialReg)
        //   r6 := T*        (seed via InitialReg, separate value)
        //   r2 = *(u64 *)(r1 + 0)    ; load M.arena_ptr -> r2 = LoadedU64Field
        //   r3 = *(u64 *)(r2 + 0)    ; deref @0 (u64) -> records access
        //   r4 = *(u64 *)(r2 + 8)    ; deref @8 (u64) -> records access
        //                            ;   intersection -> A (unique match)
        //   *(u64 *)(r1 + 16) = r6   ; STX of T* into M.kptr -> Kernel cast
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_a = push_name(&mut strings, "A");
        let n_m = push_name(&mut strings, "M");
        let n_x = push_name(&mut strings, "x");
        let n_a0 = push_name(&mut strings, "a0");
        let n_a1 = push_name(&mut strings, "a1");
        let n_arena_ptr = push_name(&mut strings, "arena_ptr");
        let n_kptr = push_name(&mut strings, "kptr");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Struct {
                name_off: n_a,
                size: 16,
                members: vec![
                    SynMember {
                        name_off: n_a0,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_a1,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            SynType::Struct {
                name_off: n_m,
                size: 24,
                members: vec![
                    SynMember {
                        name_off: n_arena_ptr,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_kptr,
                        type_id: 1,
                        byte_offset: 16,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let a_id = 4;
        let m_id = 5;
        let insns = vec![
            // Arena LDX path.
            ldx(BPF_SIZE_DW, 2, 1, 0),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            ldx(BPF_SIZE_DW, 4, 2, 8),
            // Kernel STX path.
            stx(BPF_SIZE_DW, 1, 6, 16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: m_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: t_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(m_id, 0)),
            Some(&CastHit {
                target_type_id: a_id,
                addr_space: AddrSpace::Arena
            }),
            "arena cast missing: {map:?}"
        );
        assert_eq!(
            map.get(&(m_id, 16)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "kernel kptr missing: {map:?}"
        );
    }

    #[test]
    fn func_entry_seeding_from_btf() {
        // FuncProto with two parameters: param 0 = T* (typed
        // source), param 1 = P* (parent base). FuncEntry seeds
        // R1 = Pointer{T} and R2 = Pointer{P}. R3..R5 must remain
        // Unknown — the FuncProto only describes two parameters,
        // and `seed_from_func_proto()` walks `proto.parameters`
        // (not R3..R5 unconditionally). InitialReg state set
        // before the run does not survive into a function entry
        // at PC 0.
        //
        // The strengthened test verifies BOTH halves:
        //   1. R1 and R2 are typed → STX through R2 records
        //      (P, slot1) -> T at the slot dedicated to the seeded
        //      param.
        //   2. R3, R4, R5 stay Unknown → STX through each into a
        //      distinct u64 slot in P records nothing. If
        //      `seed_from_func_proto()` accidentally typed R3..R5
        //      from leftover state or over-walked the parameter
        //      list, those stores would record (P, slotN) -> T
        //      and the count assertion would fire.
        let slot1: u32 = 16; // store R1 -> P (typed, must record)
        let slot3: u32 = 24; // store R3 -> P (must NOT record)
        let slot4: u32 = 32; // store R4 -> P (must NOT record)
        let slot5: u32 = 40; // store R5 -> P (must NOT record)
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_s1 = push_name(&mut strings, "s1");
        let n_s3 = push_name(&mut strings, "s3");
        let n_s4 = push_name(&mut strings, "s4");
        let n_s5 = push_name(&mut strings, "s5");
        let n_arg_t = push_name(&mut strings, "task");
        let n_arg_p = push_name(&mut strings, "parent");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 }, // id 3: T*
            SynType::Struct {
                name_off: n_p,
                size: slot5 + 8,
                members: vec![
                    SynMember {
                        name_off: n_s1,
                        type_id: 1,
                        byte_offset: slot1,
                    },
                    SynMember {
                        name_off: n_s3,
                        type_id: 1,
                        byte_offset: slot3,
                    },
                    SynMember {
                        name_off: n_s4,
                        type_id: 1,
                        byte_offset: slot4,
                    },
                    SynMember {
                        name_off: n_s5,
                        type_id: 1,
                        byte_offset: slot5,
                    },
                ],
            },
            SynType::Ptr { type_id: 4 }, // id 5: P*
            // id 6: FuncProto(T*, P*) -> void. Only two params, so
            // FuncEntry only seeds R1 and R2 — R3, R4, R5 must
            // stay Unknown.
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![
                    SynParam {
                        name_off: n_arg_t,
                        type_id: 3,
                    },
                    SynParam {
                        name_off: n_arg_p,
                        type_id: 5,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let proto_id = 6;
        // STX *(R2 + slot1) = R1   ; R1=T, R2=P → records (P, slot1) -> T
        // STX *(R2 + slot3) = R3   ; R3=Unknown → no record
        // STX *(R2 + slot4) = R4   ; R4=Unknown → no record
        // STX *(R2 + slot5) = R5   ; R5=Unknown → no record
        let insns = vec![
            stx(BPF_SIZE_DW, 2, 1, slot1 as i16),
            stx(BPF_SIZE_DW, 2, 3, slot3 as i16),
            stx(BPF_SIZE_DW, 2, 4, slot4 as i16),
            stx(BPF_SIZE_DW, 2, 5, slot5 as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[],
            &[FuncEntry {
                insn_offset: 0,
                func_proto_id: proto_id,
            }],
            &[],
        );
        assert_eq!(
            map.len(),
            1,
            "FuncEntry must seed only R1 and R2; R3..R5 stay Unknown so \
             only the R1->slot1 STX records: {map:?}"
        );
        assert_eq!(
            map.get(&(p_id, slot1)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "FuncEntry param seeding must populate R1 and R2: {map:?}"
        );
        // Adversary check: the failure modes we are guarding against
        // would emit (P, slot3/4/5) -> T. Assert each absent.
        assert!(
            !map.contains_key(&(p_id, slot3)),
            "R3 must remain Unknown post-FuncEntry: {map:?}"
        );
        assert!(
            !map.contains_key(&(p_id, slot4)),
            "R4 must remain Unknown post-FuncEntry: {map:?}"
        );
        assert!(
            !map.contains_key(&(p_id, slot5)),
            "R5 must remain Unknown post-FuncEntry: {map:?}"
        );
    }

    // ----- BPF_ADDR_SPACE_CAST tests ------------------------------

    /// `BPF_ADDR_SPACE_CAST` arena -> kernel (`imm == 1`) on a
    /// `LoadedU64Field` source populates `arena_confirmed` but does
    /// NOT produce a standalone map entry when no subsequent deref
    /// refines the target via shape inference. Without a resolved
    /// target type, the renderer cannot chase — emitting a
    /// placeholder would produce worse output than the raw u64
    /// fallback. The cast evidence participates only in conflict
    /// detection (preventing a kptr finding from claiming the slot).
    ///
    /// The "no emit alone" fact must be distinguished from the failure
    /// mode where the cast is silently ignored — both produce an
    /// empty map. The test runs three analyses to nail down which
    /// branch is exercised:
    ///   1. cast alone           → empty (arena_confirmed populated,
    ///      but no deref pattern).
    ///   2. cast + same-slot STX → empty (arena_confirmed conflicts
    ///      with kptr_findings, both drop).
    ///   3. same-slot STX alone  → kptr finding emits.
    ///
    /// (1) + (2) - (3) prove arena_confirmed was populated by the
    /// cast: if it were not, (2) would emit the kptr finding just as
    /// (3) does, contradicting the empty result.
    #[test]
    fn addr_space_cast_arena_alone_does_not_emit() {
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();

        // (1) cast alone — current behavior. arena_confirmed must
        // be populated for (T, 8) but no map entry emitted.
        // r3 = *(u64 *)(r1 + 8)         ; r3 = LoadedU64Field{T, 8}
        // r4 = (cast as(1) -> as(0)) r3 ; arena_confirmed += (T, 8)
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 1);
        let insns_cast_only = vec![ldx(BPF_SIZE_DW, 3, 1, 8), cast, exit()];
        let map_cast_only = analyze_casts(
            &insns_cast_only,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map_cast_only.is_empty(),
            "arena_confirmed alone (no deref pattern) must not emit: {map_cast_only:?}"
        );

        // (2) cast + STX of Pointer{Q} into the same slot. If
        // arena_confirmed for (T, 8) was populated by the cast,
        // the conflict-detection chain in `finalize()` drops both
        // observations and the map stays empty. If the cast did
        // NOT populate arena_confirmed, no conflict and the kptr
        // finding (T, 8) -> Q emits.
        // r3 = *(u64 *)(r1 + 8)            ; LoadedU64Field source
        // r4 = (cast as(1) -> as(0)) r3    ; arena_confirmed += (T, 8)
        // *(u64 *)(r1 + 8) = r5            ; kptr_findings += (T, 8) -> Q
        let insns_cast_plus_kptr = vec![
            ldx(BPF_SIZE_DW, 3, 1, 8),
            cast,
            stx(BPF_SIZE_DW, 1, 5, 8),
            exit(),
        ];
        let map_cast_plus_kptr = analyze_casts(
            &insns_cast_plus_kptr,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 5,
                    struct_type_id: q_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map_cast_plus_kptr.is_empty(),
            "cast + same-slot STX must conflict-drop both observations \
             (proves arena_confirmed was populated): {map_cast_plus_kptr:?}"
        );

        // (3) STX alone — no cast, so arena_confirmed stays empty
        // and the kptr finding emits. Establishes the baseline
        // "STX would have recorded" so that (2)'s empty result is
        // attributable to the conflict, not to a non-functional
        // STX path. Without this baseline (2)'s empty result could
        // be explained by either "conflict dropped" or "STX never
        // recorded for some other reason".
        let insns_kptr_only = vec![stx(BPF_SIZE_DW, 1, 5, 8), exit()];
        let map_kptr_only = analyze_casts(
            &insns_kptr_only,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 5,
                    struct_type_id: q_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map_kptr_only.len(),
            1,
            "STX-only baseline must record exactly one kptr finding: {map_kptr_only:?}"
        );
        assert_eq!(
            map_kptr_only.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Kernel
            }),
            "STX-only baseline records (T, 8) -> (Q, Kernel): {map_kptr_only:?}"
        );
    }

    /// `BPF_ADDR_SPACE_CAST` kernel -> arena (`imm == 0x10000`) drops
    /// the destination register state. Per kernel `verifier.c
    /// check_alu_op` the result is a 32-bit arena address, not a
    /// kernel pointer the analyzer can track. A subsequent LDX
    /// through the cast result must NOT record any access pattern,
    /// so no entry appears in the output map. Production line
    /// (`set_reg(dst, Unknown)` under the kernel->arena branch)
    /// touches ONLY dst — src must retain its prior `LoadedU64Field`
    /// state, otherwise an unrelated deref through src would also
    /// stop recording, masking real cast evidence.
    #[test]
    fn addr_space_cast_kernel_to_arena_drops_dst() {
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r3 = *(u64 *)(r1 + 8)            ; r3 = LoadedU64Field{T, 8}
        // r4 = (cast as(0) -> as(1)) r3    ; r4 = Unknown specifically
        // r5 = *(u64 *)(r4 + 0)            ; r4 Unknown -> no record
        // r6 = *(u64 *)(r3 + 0)            ; r3 retained -> records
        // The trailing deref through r3 distinguishes
        // "dst Unknown, src preserved" (correct) from "both
        // clobbered" (regression where the cast spilled into src).
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 0x10000);
        let insns = vec![
            ldx(BPF_SIZE_DW, 3, 1, 8),
            cast,
            ldx(BPF_SIZE_DW, 5, 4, 0),
            ldx(BPF_SIZE_DW, 6, 3, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        // The deref through r3 unique-resolves to Q (the only
        // BTF struct with a u64 at offset 0). If r3 had been
        // clobbered, the access pattern would never have been
        // recorded and the map would be empty.
        assert_eq!(
            map.len(),
            1,
            "exactly one cast (via preserved r3) expected: {map:?}"
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "cast preserves src LoadedU64Field; dst-only invalidation: {map:?}"
        );
        // Verify the dst-derived deref produced no entry under
        // any other (source, offset) key — only the (T, 8) ->
        // (Q, Arena) finding from r3 is allowed.
        assert!(
            !map.keys().any(|k| *k != (t_id, 8)),
            "no record may originate from r4 (cast-clobbered dst): {map:?}"
        );
    }

    /// Sign-extending MOV (`off in {8, 16, 32}`) destroys the typed-
    /// pointer property — a sign-extended s8/s16/s32 cannot survive
    /// as a 64-bit pointer. Production drops dst to Unknown; a
    /// subsequent deref through the resulting register must not
    /// record any cast.
    #[test]
    fn sign_extend_mov_drops_state() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r3 = *(u64 *)(r1 + 8)         ; r3 = LoadedU64Field{T, 8}
        // r4 = (s32) r3                  ; off=8 sign-extend -> Unknown
        // r5 = *(u64 *)(r4 + 0)         ; r4 Unknown -> no record
        let sxt = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 8, 0);
        let insns = vec![
            ldx(BPF_SIZE_DW, 3, 1, 8),
            sxt,
            ldx(BPF_SIZE_DW, 5, 4, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "sign-extend MOV must drop typed state: {map:?}"
        );
    }

    // ----- BPF_ATOMIC tests ---------------------------------------

    /// Helper: build a `BPF_STX | BPF_DW | BPF_ATOMIC` instruction
    /// with the given atomic-op `imm`. Encoding per kernel uapi
    /// `bpf.h`: `code = STX | DW | ATOMIC = 0x03 | 0x18 | 0xc0 = 0xdb`.
    fn atomic_stx(dst: u8, src: u8, off: i16, imm: i32) -> BpfInsn {
        mk_insn(
            BPF_CLASS_STX | BPF_SIZE_DW | BPF_MODE_ATOMIC,
            dst,
            src,
            off,
            imm,
        )
    }

    /// `BPF_XCHG` (`imm == 0xe0 | BPF_FETCH = 0xe1`) overwrites the
    /// source register with the prior memory value per kernel uapi
    /// `bpf.h`. The analyzer cannot type the prior memory contents,
    /// so the source register's typed state is clobbered. A
    /// subsequent plain STX of that register into a `u64` slot must
    /// NOT produce a kptr finding.
    #[test]
    fn atomic_xchg_clobbers_src() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // R1 = Pointer{T}; XCHG src=R1 to [R2+0] -> R1 = Unknown.
        // Then STX R1 into P.slot must NOT record because R1 is
        // Unknown post-xchg.
        let insns = vec![
            atomic_stx(2, 1, 0, 0xe0 | BPF_FETCH),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "XCHG must clobber src R1 typed state: {map:?}"
        );
    }

    /// `BPF_CMPXCHG` (`imm == 0xf0 | BPF_FETCH = 0xf1`) overwrites R0
    /// with the prior memory value per kernel uapi `bpf.h`,
    /// regardless of whether the compare-and-write succeeded. R0's
    /// typed state must be clobbered. A subsequent STX of R0 into a
    /// `u64` slot must NOT produce a kptr finding.
    #[test]
    fn atomic_cmpxchg_clobbers_r0() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // Seed R0 = Pointer{T} (analyzer accepts InitialReg.reg in
        // 0..=9). CMPXCHG dst=2, src=1 with imm=0xf1 clobbers R0.
        // Subsequent STX of R0 into P.slot must NOT record.
        let insns = vec![
            atomic_stx(2, 1, 0, 0xf0 | BPF_FETCH),
            stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 0,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "CMPXCHG must clobber R0 typed state: {map:?}"
        );
    }

    /// Non-fetch atomic ops (plain `BPF_ADD`/`AND`/`OR`/`XOR` without
    /// the `BPF_FETCH` bit) read-modify-write memory but do not
    /// overwrite any register. Source register typed state must
    /// survive intact, so a subsequent STX of the source into a
    /// `u64` slot still records the kptr finding. Per linux uapi
    /// `bpf_common.h`: BPF_ADD=0x00, BPF_OR=0x40, BPF_AND=0x50,
    /// BPF_XOR=0xa0. All four flavours must round-trip the source
    /// register's `Pointer{T}` state.
    #[test]
    fn atomic_non_fetch_preserves_regs() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // Iterate every non-fetch atomic op to defend against the
        // failure mode where the analyzer accidentally clobbers
        // `src` for some imm encodings but not others (e.g. a future
        // refactor that special-cases BPF_ADD only). All four ops
        // share the kernel verifier's RMW semantics: mutate memory,
        // no register output. Encoding is the bare top nibble — adding
        // BPF_FETCH (0x01) shifts to the clobbering branch tested by
        // `atomic_xchg_clobbers_src`.
        const BPF_ATOMIC_ADD: i32 = 0x00;
        const BPF_ATOMIC_OR: i32 = 0x40;
        const BPF_ATOMIC_AND: i32 = 0x50;
        const BPF_ATOMIC_XOR: i32 = 0xa0;
        for imm in [
            BPF_ATOMIC_ADD,
            BPF_ATOMIC_OR,
            BPF_ATOMIC_AND,
            BPF_ATOMIC_XOR,
        ] {
            let insns = vec![
                atomic_stx(2, 1, 0, imm),
                stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
                exit(),
            ];
            let map = analyze_casts(
                &insns,
                &btf,
                &[
                    InitialReg {
                        reg: 1,
                        struct_type_id: t_id,
                    },
                    InitialReg {
                        reg: 6,
                        struct_type_id: p_id,
                    },
                ],
                &[],
                &[],
            );
            assert_eq!(
                map.len(),
                1,
                "imm=0x{imm:02x}: exactly one kptr finding expected, got: {map:?}"
            );
            assert_eq!(
                map.get(&(p_id, slot_off)),
                Some(&CastHit {
                    target_type_id: t_id,
                    addr_space: AddrSpace::Kernel
                }),
                "imm=0x{imm:02x}: non-fetch ATOMIC must preserve src register: {map:?}"
            );
        }
    }

    /// An `ATOMIC` op targeting `[r10 + neg_off]` mutates a stack
    /// slot the analyzer was tracking from a prior spill. The slot's
    /// saved value is overwritten by the atomic operation, so a
    /// subsequent reload through `r10` must NOT resurrect the
    /// pre-atomic typed-pointer state.
    #[test]
    fn atomic_on_stack_invalidates_slot() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // *(u64 *)(r10 - 8) = R1   ; spill Pointer{T} -> stack[-8]
        // ATOMIC XCHG [r10 - 8] = R2   ; mutates the stack slot
        // R3 = *(u64 *)(r10 - 8)   ; reload must yield Unknown
        // *(u64 *)(R6 + slot_off) = R3 ; R3 Unknown -> no record
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            atomic_stx(10, 2, -8, 0xe0 | BPF_FETCH),
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "ATOMIC on stack slot must invalidate, reload yields Unknown: {map:?}"
        );
    }

    /// `BPF_LOAD_ACQ` (`imm == 0x100`) loads a memory value with
    /// acquire-ordered semantics into `dst` per kernel
    /// `include/linux/filter.h`. The analyzer cannot type a memory
    /// value pulled out via this path, so `dst` is clobbered to
    /// `Unknown`. A subsequent STX of `dst` into a `u64` slot must
    /// NOT record a kptr finding even though `dst` held a typed
    /// pointer prior to the load-acquire.
    #[test]
    fn atomic_load_acq_clobbers_dst() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // R1 = Pointer{T}; LOAD_ACQ targets dst=1, src=2 (address
        // base) -> R1 = Unknown. Then STX R1 into P.slot must NOT
        // record because R1 is Unknown post-load-acquire.
        let insns = vec![
            atomic_stx(1, 2, 0, BPF_LOAD_ACQ_IMM),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "LOAD_ACQ must clobber dst R1 typed state: {map:?}"
        );
    }

    /// `BPF_STORE_REL` (`imm == 0x110`) stores `src` to memory with
    /// release-ordered semantics per kernel
    /// `include/linux/filter.h`. `dst` is the address-base register
    /// and `src` is the value being stored — neither is overwritten.
    /// Both registers' typed-pointer state must survive intact: a
    /// subsequent plain STX of either typed pointer into another
    /// `u64` slot still records the kptr finding.
    #[test]
    fn atomic_store_rel_preserves_src_and_dst() {
        // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@slot_off1, u64@slot_off2).
        // Two distinct slots so we can verify BOTH registers'
        // typed states by storing each into a separate slot.
        let slot_off1: u32 = 16;
        let slot_off2: u32 = 24;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot1 = push_name(&mut strings, "slot1");
        let n_slot2 = push_name(&mut strings, "slot2");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Struct {
                name_off: n_p,
                size: slot_off2 + 8,
                members: vec![
                    SynMember {
                        name_off: n_slot1,
                        type_id: 1,
                        byte_offset: slot_off1,
                    },
                    SynMember {
                        name_off: n_slot2,
                        type_id: 1,
                        byte_offset: slot_off2,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        // R1 = Pointer{T}, R2 = Pointer{T}, R6 = Pointer{P}.
        // STORE_REL dst=R7 src=R1: address-base R7, value R1.
        //   R1 must remain Pointer{T}; R7 is uninvolved here.
        // STX *(R6 + slot1) = R1: records (P, slot1) -> T.
        // STX *(R6 + slot2) = R2: records (P, slot2) -> T.
        // If STORE_REL had clobbered R1, the first kptr write
        // would have dropped — but R2 (the unused-by-STORE_REL
        // typed pointer) would still record, giving a partial map.
        // The two-slot assertion discriminates: both slots present
        // proves STORE_REL left R1 alone; only-slot2 present would
        // catch a regression that clobbers R1 specifically.
        let insns = vec![
            atomic_stx(7, 1, 0, BPF_STORE_REL_IMM),
            stx(BPF_SIZE_DW, 6, 1, slot_off1 as i16),
            stx(BPF_SIZE_DW, 6, 2, slot_off2 as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 2,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off1)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "STORE_REL must preserve src R1 typed state (slot1 missing): {map:?}"
        );
        assert_eq!(
            map.get(&(p_id, slot_off2)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "STORE_REL must not affect uninvolved R2 (slot2 missing): {map:?}"
        );
    }

    /// `BPF_STORE_REL` through `r10` writes the stack slot at
    /// `[r10 + off]`. Even though STORE_REL has no per-register
    /// clobber effect, the stack-slot invalidation arm at the head
    /// of `handle_atomic` runs unconditionally for every atomic
    /// flavor when `dst == r10`. A prior spill of a typed pointer
    /// into the slot is overwritten by the release store, so a
    /// subsequent reload through `r10` must NOT resurrect the
    /// pre-store-release typed-pointer state.
    #[test]
    fn atomic_store_rel_invalidates_stack_slot() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // *(u64 *)(r10 - 8) = R1     ; spill Pointer{T} -> stack[-8]
        // STORE_REL [r10 - 8] = R2   ; release-store overwrites slot
        // R3 = *(u64 *)(r10 - 8)     ; reload must yield Unknown
        // *(u64 *)(R6 + slot_off) = R3 ; R3 Unknown -> no record
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            atomic_stx(10, 2, -8, BPF_STORE_REL_IMM),
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "STORE_REL through r10 must invalidate slot, reload Unknown: {map:?}"
        );
    }

    /// `BPF_ADD | BPF_FETCH` (`imm == 0x01`) is an atomic
    /// fetch-and-add: src receives the prior memory value, memory
    /// receives `memory + src`. Per kernel uapi `bpf.h` and the
    /// `has_fetch` arm in `handle_atomic`, src's typed-pointer state
    /// is dropped to `Unknown`. A subsequent STX of src into a `u64`
    /// slot must NOT record a kptr finding.
    #[test]
    fn atomic_add_fetch_clobbers_src() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // R1 = Pointer{T}; ADD|FETCH src=R1 to [R2+0] -> R1 = Unknown.
        // Then STX R1 into P.slot must NOT record because R1 is
        // Unknown post-fetch-add. BPF_ADD = 0x00 (linux uapi
        // bpf_common.h) | BPF_FETCH = 0x01.
        let insns = vec![
            atomic_stx(2, 1, 0, BPF_FETCH),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "ADD|FETCH must clobber src R1 typed state: {map:?}"
        );
    }

    /// `BPF_AND | BPF_FETCH` (`imm == 0x51`) is an atomic
    /// fetch-and-and: src receives the prior memory value. The
    /// `has_fetch` arm in `handle_atomic` drops src to `Unknown`.
    /// A subsequent STX of src into a `u64` slot must NOT record.
    #[test]
    fn atomic_and_fetch_clobbers_src() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_AND = 0x50 (linux uapi bpf_common.h) | BPF_FETCH = 0x51.
        let insns = vec![
            atomic_stx(2, 1, 0, 0x50 | BPF_FETCH),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "AND|FETCH must clobber src R1 typed state: {map:?}"
        );
    }

    /// `BPF_OR | BPF_FETCH` (`imm == 0x41`) is an atomic
    /// fetch-and-or: src receives the prior memory value. The
    /// `has_fetch` arm in `handle_atomic` drops src to `Unknown`.
    /// A subsequent STX of src into a `u64` slot must NOT record.
    #[test]
    fn atomic_or_fetch_clobbers_src() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_OR = 0x40 (linux uapi bpf_common.h) | BPF_FETCH = 0x41.
        let insns = vec![
            atomic_stx(2, 1, 0, 0x40 | BPF_FETCH),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "OR|FETCH must clobber src R1 typed state: {map:?}"
        );
    }

    /// `BPF_XOR | BPF_FETCH` (`imm == 0xa1`) is an atomic
    /// fetch-and-xor: src receives the prior memory value. The
    /// `has_fetch` arm in `handle_atomic` drops src to `Unknown`.
    /// A subsequent STX of src into a `u64` slot must NOT record.
    #[test]
    fn atomic_xor_fetch_clobbers_src() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_XOR = 0xa0 (linux uapi bpf_common.h) | BPF_FETCH = 0xa1.
        let insns = vec![
            atomic_stx(2, 1, 0, 0xa0 | BPF_FETCH),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "XOR|FETCH must clobber src R1 typed state: {map:?}"
        );
    }

    /// `BPF_ATOMIC` with `BPF_W` (4-byte) size targeting `[r10+off]`
    /// must invalidate the stack slot the same way a DW atomic does.
    /// Per `handle_atomic`, the stack-invalidation arm runs
    /// unconditionally on `dst == r10` regardless of the size bits
    /// in the opcode — a 4-byte atomic write into a slot that
    /// formerly held a 64-bit typed pointer truncates the slot's
    /// content. A subsequent DW reload must NOT resurrect the
    /// pre-atomic typed-pointer state.
    #[test]
    fn atomic_w_size_invalidates_stack_slot() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // *(u64 *)(r10 - 8) = R1                 ; spill Pointer{T} -> stack[-8]
        // ATOMIC<W> XCHG [r10 - 8] = R2          ; W-size atomic on slot
        // R3 = *(u64 *)(r10 - 8)                 ; reload must yield Unknown
        // *(u64 *)(R6 + slot_off) = R3           ; R3 Unknown -> no record
        //
        // Constructed with `mk_insn` directly because the `atomic_stx`
        // helper hard-codes `BPF_SIZE_DW`. Code = STX | W | ATOMIC.
        let atomic_w = mk_insn(
            BPF_CLASS_STX | BPF_SIZE_W | BPF_MODE_ATOMIC,
            10,
            2,
            -8,
            0xe0 | BPF_FETCH,
        );
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            atomic_w,
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "W-size ATOMIC on stack slot must invalidate, reload Unknown: {map:?}"
        );
    }

    /// `BPF_CMPXCHG` (`imm == 0xf1`) overwrites BOTH `R0` (with the
    /// prior memory value) AND `src` (because `BPF_FETCH` is set,
    /// the second-stage `has_fetch` arm in `handle_atomic` runs in
    /// addition to the CMPXCHG-specific R0 clobber). The existing
    /// `atomic_cmpxchg_clobbers_r0` test guards the R0 path; this
    /// test guards the src path. Per kernel uapi `bpf.h` and the
    /// final fall-through `if has_fetch` arm in `handle_atomic`,
    /// src's typed-pointer state is dropped to `Unknown` regardless
    /// of which atomic-op top nibble was used. A subsequent STX of
    /// src into a `u64` slot must NOT record a kptr finding.
    #[test]
    fn atomic_cmpxchg_clobbers_src() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // R1 = Pointer{T}; CMPXCHG dst=R2, src=R1 with imm=0xf1.
        // R0 (the cmpxchg "expected" register, not seeded here) is
        // clobbered to Unknown by the CMPXCHG-specific arm; R1 is
        // clobbered to Unknown by the final has_fetch arm. STX R1
        // into P.slot must NOT record.
        let insns = vec![
            atomic_stx(2, 1, 0, 0xf0 | BPF_FETCH),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "CMPXCHG must clobber src R1 typed state via has_fetch arm: {map:?}"
        );
    }

    // ----- Stack tests --------------------------------------------

    /// A second spill to the same stack slot overwrites the prior
    /// saved state. Reload restores the latest typed pointer, not
    /// the original one. Production line `self.stack_slots.insert`
    /// replaces by key.
    #[test]
    fn stack_spill_overwrite_uses_latest() {
        // BTF: u64(1), T1(2, u64@0), T2(3, u64@0), P(4, u64@slot_off).
        // T1 and T2 are distinguishable by id; the test seeds R1=T1,
        // R2=T2, then spills T2 last so reload should yield T2.
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t1 = push_name(&mut strings, "T1");
        let n_t2 = push_name(&mut strings, "T2");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t1,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Struct {
                name_off: n_t2,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t1_id = 2;
        let t2_id = 3;
        let p_id = 4;
        // Spill R1 (T1*) to [r10-8]
        // Spill R2 (T2*) to [r10-8] -- overwrite
        // Reload to R3 (must be T2*)
        // Store R3 into P.slot
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            stx(BPF_SIZE_DW, 10, 2, -8),
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t1_id,
                },
                InitialReg {
                    reg: 2,
                    struct_type_id: t2_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t2_id,
                addr_space: AddrSpace::Kernel
            }),
            "second spill to same slot must win: {map:?}"
        );
    }

    /// `BPF_CALL` clobbers `r0..r5` per the BPF ABI but does NOT
    /// invalidate stack slots. A typed pointer parked in `[r10-N]`
    /// before a helper call must reload as the same typed pointer
    /// after the call returns.
    #[test]
    fn stack_spill_survives_helper_call() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // Spill R1 (T*) to [r10-8]
        // CALL helper (clobbers R0..R5, R6 untouched)
        // Reload from [r10-8] to R3 (must restore T*)
        // Store R3 into P.slot
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            call(),
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "stack-spilled pointer must survive helper call: {map:?}"
        );
    }

    /// A sub-DW (4-byte W) store through `r10` to a slot previously
    /// holding a typed pointer truncates the stored value. Per
    /// `handle_stx`, the slot is removed so a later DW reload
    /// returns Unknown rather than resurrecting the stale typed
    /// state.
    #[test]
    fn sub_dw_spill_invalidates() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // *(u64 *)(r10 - 8) = R1   ; spill T* to slot
        // *(u32 *)(r10 - 8) = R1   ; sub-DW store, slot removed
        // R3 = *(u64 *)(r10 - 8)   ; reload must yield Unknown
        // *(u64 *)(R6 + slot_off) = R3 ; no record
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            stx(BPF_SIZE_W, 10, 1, -8),
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "sub-DW store must invalidate slot, reload Unknown: {map:?}"
        );
    }

    /// `BPF_ST` (immediate store, `code = BPF_ST | BPF_MEM | BPF_DW
    /// = 0x7A`) writes a constant immediate to memory through `r10`.
    /// The constant is never a typed pointer, but the store overlays
    /// any prior typed value the analyzer was tracking in the
    /// stack slot. Per the `BPF_CLASS_ST` arm in `step()`, the
    /// stack slot is removed when `dst == r10 && mode == BPF_MEM`
    /// so a subsequent reload through `r10` must NOT resurrect the
    /// pre-immediate-store typed-pointer state.
    #[test]
    fn st_imm_invalidates_stack_slot() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // *(u64 *)(r10 - 8) = R1     ; spill Pointer{T} -> stack[-8]
        // *(u64 *)(r10 - 8) = imm 0  ; ST overwrites slot with constant
        // R3 = *(u64 *)(r10 - 8)     ; reload must yield Unknown
        // *(u64 *)(R6 + slot_off) = R3 ; R3 Unknown -> no record
        //
        // Constructed with `mk_insn` directly — there is no
        // helper for BPF_ST class instructions. Code per linux
        // uapi `bpf_common.h` and `bpf.h`: BPF_ST | BPF_MEM | BPF_DW
        // = 0x02 | 0x60 | 0x18 = 0x7A. dst=r10 (frame pointer),
        // src is unused (encoded as 0), off=-8 (slot key matching
        // the prior spill), imm=0 (constant value, irrelevant —
        // any constant overwrites the typed slot the same way).
        let st_imm_dw = mk_insn(BPF_CLASS_ST | BPF_MODE_MEM | BPF_SIZE_DW, 10, 0, -8, 0);
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            st_imm_dw,
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "BPF_ST imm to stack slot must invalidate, reload Unknown: {map:?}"
        );
    }

    // ----- Conflict tests -----------------------------------------

    /// A single `(struct, offset)` slot observed by BOTH the arena
    /// LDX path (loaded then dereferenced) AND the kernel STX path
    /// (typed `Pointer{T}` stored) is ambiguous. The same byte cannot
    /// simultaneously hold an arena VA and a kernel VA. Per
    /// `finalize`, both observations drop and the slot does not
    /// appear in the output map.
    #[test]
    fn arena_and_kptr_same_field_drops_both() {
        // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@8).
        // T is the unique candidate for shape pattern (offset=0, size=8).
        // Arena: load P.u64@8 -> r2, deref r2+0 -> patterns[(P,8)]={(0,8)}.
        // Kptr: STX *(P+8) = R6 (Pointer{T}) -> kptr_findings[(P,8)] = T.
        // Conflict on (P, 8) drops both.
        let slot_off: u32 = 8;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // Arena LDX path: r2 = *(u64*)(r1+8); r3 = *(u64*)(r2+0).
        // Kernel STX path: *(u64*)(r1+8) = r6.
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, slot_off as i16),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            stx(BPF_SIZE_DW, 1, 6, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: p_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: t_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "arena+kptr conflict on same slot must drop both: {map:?}"
        );
    }

    /// Two STX writes to the same `(struct, offset)` slot with
    /// different target struct ids collapse the kptr finding to
    /// `KptrEntry::Conflicting`. `finalize` skips conflicting
    /// entries, so the slot does not appear in the output map.
    ///
    /// Bare `map.is_empty()` cannot distinguish "Conflicting state
    /// was reached" from "the analyzer never recorded either STX".
    /// Both yield empty maps. The strengthened test runs THREE
    /// analyses to triangulate the production path:
    ///   (a) baseline single-STX of T1: must record (P, slot) -> T1.
    ///       Establishes that the STX path is functional and that
    ///       T1 is recoverable from R1.
    ///   (b) STX T1 then STX T2: must drop (collapse to Conflicting).
    ///       Same as the original test.
    ///   (c) STX T1, STX T2, STX T1 again: must STILL drop. Once a
    ///       slot transitions to Conflicting, every subsequent STX
    ///       (even of the original target) preserves Conflicting per
    ///       the `Some(_)` arm of the match in `handle_stx()` —
    ///       proves the slot did NOT revert to `Single(T1)` after
    ///       the third store. If the analyzer instead overwrote
    ///       Conflicting back to Single on a same-target restore,
    ///       (c) would emit (P, slot) -> T1 like (a) does.
    #[test]
    fn kptr_conflict_two_targets_drops() {
        // BTF: u64(1), T1(2, u64@0), T2(3, u64@0), P(4, u64@slot_off).
        // Seed R1=Pointer{T1}, R2=Pointer{T2}, R6=Pointer{P}.
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t1 = push_name(&mut strings, "T1");
        let n_t2 = push_name(&mut strings, "T2");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t1,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Struct {
                name_off: n_t2,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t1_id = 2;
        let t2_id = 3;
        let p_id = 4;
        let seeds = [
            InitialReg {
                reg: 1,
                struct_type_id: t1_id,
            },
            InitialReg {
                reg: 2,
                struct_type_id: t2_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ];

        // (a) Baseline: single STX of T1 records (P, slot) -> T1.
        // Without this anchor, the empty map from (b)/(c) below
        // could be explained by a non-functional STX path rather
        // than the Conflicting transition.
        let insns_single = vec![stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
        let map_single = analyze_casts(&insns_single, &btf, &seeds, &[], &[]);
        assert_eq!(
            map_single.len(),
            1,
            "(a) single STX must record exactly one finding: {map_single:?}"
        );
        assert_eq!(
            map_single.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t1_id,
                addr_space: AddrSpace::Kernel
            }),
            "(a) baseline records (P, slot) -> (T1, Kernel): {map_single:?}"
        );

        // (b) Two distinct targets — collapses to Conflicting and
        // finalize drops. Same shape as the historical test.
        let insns_conflict = vec![
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            stx(BPF_SIZE_DW, 6, 2, slot_off as i16),
            exit(),
        ];
        let map_conflict = analyze_casts(&insns_conflict, &btf, &seeds, &[], &[]);
        assert!(
            map_conflict.is_empty(),
            "(b) two distinct kptr targets on same slot must collapse to \
             Conflicting and drop: {map_conflict:?}"
        );

        // (c) Append a third STX of T1. If the slot is Conflicting,
        // the `Some(_)` arm preserves Conflicting (no revert to
        // Single). If the production code instead reset to
        // Single(T1) on a same-target restore, the map would
        // emit (P, slot) -> T1 like (a) does. Empty map confirms
        // Conflicting was reached and is sticky.
        let insns_three = vec![
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            stx(BPF_SIZE_DW, 6, 2, slot_off as i16),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map_three = analyze_casts(&insns_three, &btf, &seeds, &[], &[]);
        assert!(
            map_three.is_empty(),
            "(c) Conflicting state must be sticky across same-target \
             restore — third STX of T1 must not resurrect: {map_three:?}"
        );
    }

    // ----- OOB tests ----------------------------------------------

    /// A malformed `BpfInsn` with `dst >= 11` (out of the 0..=10
    /// valid register range) must NOT panic. The bounds check at
    /// the top of `step()` and `handle_*` rejects early. The
    /// analyzer treats the instruction as a no-op; output map is
    /// empty.
    #[test]
    fn oob_dst_reg_does_not_panic() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // LDX with dst=11 (invalid). Construct via BpfInsn::new
        // which masks to 4 bits (11 & 0x0f == 11). No panic; map empty.
        let bad = BpfInsn::new(BPF_CLASS_LDX | BPF_SIZE_DW | BPF_MODE_MEM, 11, 1, 8, 0);
        let insns = vec![bad, exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "OOB dst must not panic, map empty: {map:?}");
    }

    /// A malformed `BpfInsn` with `src == 15` (out of the 0..=10
    /// valid register range) must NOT panic. `BpfInsn::new` packs
    /// 15 into the 4-bit src field; `src_reg()` decodes back to 15.
    /// The bounds check rejects early; output map is empty.
    #[test]
    fn oob_src_reg_does_not_panic() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // LDX with src=15 (invalid). No panic; map empty.
        let bad = BpfInsn::new(BPF_CLASS_LDX | BPF_SIZE_DW | BPF_MODE_MEM, 2, 15, 8, 0);
        let insns = vec![bad, exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "OOB src must not panic, map empty: {map:?}");
    }

    // ----- Other tests --------------------------------------------

    /// Storing a `Pointer{P}` into a `u64` field of struct `P`
    /// itself is rejected: `parent == target` is almost always a
    /// structural error from ambiguous pointer aliasing in the
    /// analyzer, not a real kptr write. Production line
    /// `if parent_struct_id == target_struct_id { return; }`
    /// drops the finding; output map is empty.
    #[test]
    fn self_store_rejected() {
        // BTF: u64(1), P(2, u64@slot_off). No separate target type.
        let slot_off: u32 = 8;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_p = push_name(&mut strings, "P");
        let n_slot = push_name(&mut strings, "slot");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let p_id = 2;
        // R1 = Pointer{P}. STX *(R1 + slot_off) = R1. Self-store.
        let insns = vec![stx(BPF_SIZE_DW, 1, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: p_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "self-store must be rejected: {map:?}");
    }

    /// A FuncProto with a variadic sentinel parameter (`name_off=0
    /// AND type_id=0` per `Parameter::is_variadic`) must terminate
    /// the parameter scan: no parameter slot past the sentinel
    /// reseeds a register. With params `[T*, P*, variadic]` the
    /// non-variadic prefix seeds R1 = Pointer{T} and R2 = Pointer{P};
    /// the variadic sentinel terminates the scan so R3 stays Unknown
    /// even though a real BTF parameter slot follows. A subsequent
    /// STX through R3 must not record a kptr finding.
    #[test]
    fn variadic_param_breaks_seeding() {
        // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@slot_off1, u64@slot_off2),
        //      P*(5), FuncProto(6, params=[T*, P*, variadic, T*]).
        let slot_off1: u32 = 16;
        let slot_off2: u32 = 24;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot1 = push_name(&mut strings, "slot1");
        let n_slot2 = push_name(&mut strings, "slot2");
        let n_arg_t = push_name(&mut strings, "task");
        let n_arg_p = push_name(&mut strings, "parent");
        let n_arg_after = push_name(&mut strings, "after_variadic");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 }, // id 3: T*
            SynType::Struct {
                name_off: n_p,
                size: slot_off2 + 8,
                members: vec![
                    SynMember {
                        name_off: n_slot1,
                        type_id: 1,
                        byte_offset: slot_off1,
                    },
                    SynMember {
                        name_off: n_slot2,
                        type_id: 1,
                        byte_offset: slot_off2,
                    },
                ],
            },
            SynType::Ptr { type_id: 4 }, // id 5: P*
            // FuncProto with [T*, P*, variadic, T*]. The trailing
            // T* slot is BTF-reachable but unreachable in the BPF
            // calling convention because the variadic sentinel
            // terminates the scan; the analyzer must NOT seed R4
            // from it.
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![
                    SynParam {
                        name_off: n_arg_t,
                        type_id: 3,
                    },
                    SynParam {
                        name_off: n_arg_p,
                        type_id: 5,
                    },
                    SynParam {
                        name_off: 0,
                        type_id: 0,
                    },
                    SynParam {
                        name_off: n_arg_after,
                        type_id: 3,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let proto_id = 6;
        // FuncEntry seeds R1 = Pointer{T} (param 0), R2 = Pointer{P}
        // (param 1). R3 stays Unknown because the variadic sentinel
        // terminates the scan before param 3.
        // STX *(R2 + slot1) = R1 records (P, slot1) -> T.
        // STX *(R2 + slot2) = R3 must NOT record (R3 Unknown).
        let insns = vec![
            stx(BPF_SIZE_DW, 2, 1, slot_off1 as i16),
            stx(BPF_SIZE_DW, 2, 3, slot_off2 as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[],
            &[FuncEntry {
                insn_offset: 0,
                func_proto_id: proto_id,
            }],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off1)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "non-variadic params must seed R1 and R2: {map:?}"
        );
        assert!(
            !map.contains_key(&(p_id, slot_off2)),
            "variadic sentinel must terminate scan, R3 must stay Unknown: {map:?}"
        );
    }

    /// `FuncEntry` clears ALL registers (R0..R10) and the stack before
    /// seeding R1..R5 from the FuncProto. A typed pointer parked in
    /// any register by `InitialReg` is dropped at the entry PC —
    /// including callee-saved R6..R9 (the linear walk has no real
    /// caller, so preserving them would leak stale state).
    #[test]
    fn func_entry_clears_all_regs() {
        // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@slot_off),
        //      FuncProto(5, params=[T*]).
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let n_arg = push_name(&mut strings, "arg");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            // FuncProto(T*) -> void.
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![SynParam {
                    name_off: n_arg,
                    type_id: 3,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let proto_id = 5;
        // Seed R3 = Pointer{T} and R6 = Pointer{P} via InitialReg.
        // FuncEntry at PC 0 clears ALL registers (R0..R10), then
        // seeds R1 from param 0. Both R3 and R6 are now Unknown.
        // STX *(R6 + slot) = R3 must NOT record (both cleared).
        let insns = vec![stx(BPF_SIZE_DW, 6, 3, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 3,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[FuncEntry {
                insn_offset: 0,
                func_proto_id: proto_id,
            }],
            &[],
        );
        assert!(
            map.is_empty(),
            "FuncEntry pre-clear must drop R3 typed state: {map:?}"
        );
    }

    /// `BPF_PROBE_MEM` (`mode = 0x20`) is a post-verifier marker
    /// per linux `include/linux/filter.h` and never appears in
    /// pre-verification bytecode. Production treats any LDX with
    /// `mode != BPF_MODE_MEM` as Unknown; a subsequent deref
    /// through the resulting register records nothing.
    #[test]
    fn probe_mem_load_treated_as_unknown() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // PROBE_MEM mode = 0x20. code = LDX | DW | PROBE_MEM = 0x39.
        // dst=2, src=1, off=8 mimics the arena LDX shape but the
        // mode bits divert to the Unknown branch.
        const BPF_MODE_PROBE_MEM: u8 = 0x20;
        let probe_load = mk_insn(BPF_CLASS_LDX | BPF_SIZE_DW | BPF_MODE_PROBE_MEM, 2, 1, 8, 0);
        let insns = vec![probe_load, ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "BPF_PROBE_MEM load must mark dst Unknown: {map:?}"
        );
    }

    // ----- Finalize edge cases ------------------------------------

    /// `BPF_ADDR_SPACE_CAST` arena->kernel populates `arena_confirmed`
    /// even when the originating LDX never produces a downstream
    /// dereference, so `patterns[(T,8)]` carries an EMPTY access set.
    /// If the same `(T, off)` slot is also the parent of a STX-source
    /// kptr write, `finalize` must treat the slot as conflicting and
    /// drop BOTH observations: the kptr loop skips the conflicted key
    /// and the arena loop skips the empty-access entry independently.
    /// Without the `arena_confirmed`-side participation in the
    /// conflict set the kptr finding would emit even though the cast
    /// instruction proves the slot holds an arena address.
    #[test]
    fn finalize_arena_confirmed_conflicts_with_kptr() {
        // BTF: u64(1), T(2, u64@8), T*(3), Q(4, u64@0). T also acts
        // as the parent for the kptr STX (the slot at T+8). Q is the
        // distinct value type to keep the self-store rejection from
        // firing.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Struct {
                name_off: n_q,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let q_id = 4;
        // r2 = *(u64*)(r1 + 8)        ; r2 = LoadedU64Field{T, 8},
        //                               patterns[(T,8)] = {} (no deref)
        // r4 = (cast as(1)->as(0)) r2 ; arena_confirmed += (T, 8)
        // *(u64*)(r1 + 8) = r3        ; r1 still Pointer{T},
        //                               r3 = Pointer{Q} ->
        //                               kptr_findings[(T,8)] = Single(Q)
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 2, 1, 1);
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            cast,
            stx(BPF_SIZE_DW, 1, 3, 8),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 3,
                    struct_type_id: q_id,
                },
            ],
            &[],
            &[],
        );
        // Both observations must drop. Specifically: the kptr finding
        // for (T, 8) -> Q must NOT appear, and no arena entry can
        // appear either (the access set was empty anyway).
        assert!(
            !map.contains_key(&(t_id, 8)),
            "arena_confirmed + kptr conflict on (T, 8) must drop both: {map:?}"
        );
        assert!(map.is_empty(), "no other entries expected: {map:?}");
    }

    /// Loading a u64 field without dereferencing through it leaves
    /// `patterns[(T, off)]` populated but with an EMPTY access set.
    /// `finalize`'s arena loop short-circuits via
    /// `if accesses.is_empty() { continue }` and emits nothing. The
    /// slot stays absent from the output map even though the source
    /// register held a `LoadedU64Field` state at one point.
    #[test]
    fn finalize_empty_access_set_does_not_emit() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r2 = *(u64 *)(r1 + 8)  -- patterns[(T,8)] = {} (loaded only)
        // exit                    -- never dereferenced
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            !map.contains_key(&(t_id, 8)),
            "empty access set must not emit: {map:?}"
        );
        assert!(map.is_empty(), "no other entries expected: {map:?}");
    }

    /// The candidate-search intersection drops the source struct
    /// itself even when its layout matches every observed
    /// `(offset, size)` access. When the original candidate set is
    /// `{source, X}` (source plus exactly one foreign struct that also
    /// matches), `finalize` rejects the entire entry rather than
    /// emitting `X`: an `{source, X}` set means the true target could
    /// have been the source AND its access pattern happens to match
    /// X by coincidence. Picking X would be a false positive.
    #[test]
    fn finalize_source_in_candidates_with_others_emits_other() {
        // BTF: u64(1), T(2, u64@0 + u64@8 -- same shape T matches its
        // own access pattern), Q(3, u64@0 + u64@8). Loading T.f at
        // offset 0 then dereferencing through it at offsets 0 and 8
        // gives candidates {T, Q} -- both have u64s at those offsets.
        // Production must drop because the source T is in the set
        // alongside Q.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_a = push_name(&mut strings, "a");
        let n_b = push_name(&mut strings, "b");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![
                    SynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
            SynType::Struct {
                name_off: n_q,
                size: 16,
                members: vec![
                    SynMember {
                        name_off: n_a,
                        type_id: 1,
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_b,
                        type_id: 1,
                        byte_offset: 8,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        // Sequence: r2 = *(u64*)(r1 + 0); r3 = *(u64*)(r2 + 0);
        // r4 = *(u64*)(r2 + 8). Candidates for (offset=0, size=8) and
        // (offset=8, size=8) intersect to {T, Q}. Source T is removed
        // from candidates; Q remains as the sole non-source candidate
        // and is emitted.
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 0),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            ldx(BPF_SIZE_DW, 4, 2, 8),
            exit(),
        ];
        let q_id = 3;
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 0)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "source removed, sole non-source candidate Q emitted: {map:?}"
        );
    }

    /// When the only candidate matching the access pattern is the
    /// source struct itself, `finalize` removes it via
    /// `had_source = candidates.remove(source)` and the resulting set
    /// is empty -- nothing emits. This guards against self-typed
    /// casts (`source.f` -> `source*`) where a self-referential layout
    /// would silently win the intersection without disambiguating
    /// evidence.
    #[test]
    fn finalize_only_source_candidate_drops() {
        // BTF: u64(1), T(2, u64@8). T is the only struct in the BTF;
        // its layout matches the access pattern (offset=8, size=8).
        // After remove(source) the candidate set is empty -> skip emit.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_f = push_name(&mut strings, "f");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // T has u64@8 only; (offset=8, size=8) matches T.
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        // Sequence: r2 = *(u64*)(r1 + 8); r3 = *(u64*)(r2 + 8).
        // Pattern recorded: source=(T,8), access=(8,8). Layout maps
        // (8,8) -> {T}. After remove(source), candidates empty ->
        // skip emit.
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 8), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "candidate set containing only the source must drop: {map:?}"
        );
    }

    /// The candidate-search loop walks BTF ids `1..=max_id` where
    /// `max_id = max_seen_type_id + CANDIDATE_SEARCH_SLACK` capped at
    /// `MAX_BTF_ID_PROBE`. When the source struct id is small (e.g.
    /// 2), `max_seen_type_id` picks up the source id (via
    /// `note_type_id`) and the slack carries the search several
    /// thousand ids further. A target struct that lives WELL beyond
    /// the source id but well inside the slack window must still be
    /// found. This guards against a regression that would shrink the
    /// loop bound to `max_seen_type_id` itself, AND verifies that the
    /// `MAX_BTF_ID_PROBE` cap leaves room for the slack on small
    /// max_seen values.
    #[test]
    fn finalize_max_seen_type_id_slack_finds_distant_candidate() {
        // BTF: many filler Ptr types between T and Q so that Q's id
        // is far above max_seen_type_id (which the analyzer sets to
        // T's id when seeding R1). The slack (CANDIDATE_SEARCH_SLACK
        // = 65_536) more than covers any practical BTF id space, so
        // Q at id 203 must be found even though only T (id 2) is
        // touched during the forward pass. The MAX_BTF_ID_PROBE cap
        // (100_000) is far above id 203, so this also exercises the
        // .min() arm without truncating the search.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");
        // Build: id 1 = u64, id 2 = T (struct with u64@8), then 200
        // filler Ptr-to-u64 types (ids 3..=202), then id 203 = Q
        // (struct with u64@0). max_seen ends up at 2 (T), slack pushes
        // search through id 65_538, so id 203 is well within range.
        let mut types: Vec<SynType> = Vec::new();
        types.push(SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        });
        types.push(SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        });
        for _ in 0..200 {
            types.push(SynType::Ptr { type_id: 1 });
        }
        types.push(SynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        });
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let q_id = 203;
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "slack must carry search well past max_seen, capped within \
             MAX_BTF_ID_PROBE: {map:?}"
        );
    }

    // ----- BTF type edge cases ------------------------------------

    /// `struct_member_at` skips bitfield members (members with
    /// `bitfield_size > 0`) even when their byte offset matches the
    /// query. A LDX through a `Pointer{T}` register at the bitfield's
    /// byte offset must NOT seed a `LoadedU64Field` state, so no
    /// pattern accumulates and no cast emits.
    #[test]
    fn struct_member_at_skips_bitfield_at_target_offset() {
        // BTF: u64(1), T(kind_flag=1) with a u64 bitfield at byte
        // offset 8 (bitfield_size = 32 bits). The byte offset matches
        // the LDX target but the bitfield_size > 0 makes
        // struct_member_at return None.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_f = push_name(&mut strings, "f");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::StructBitfields {
                name_off: n_t,
                size: 16,
                members: vec![SynMemberBits {
                    name_off: n_f,
                    type_id: 1,
                    bit_offset: 8 * 8,      // byte offset 8
                    bitfield_size_bits: 32, // bitfield -> skip
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        // r2 = *(u64*)(r1 + 8): handle_ldx queries struct_member_at(T, 8).
        // The bitfield at 8 is skipped; struct_member_at returns None;
        // r2 becomes Unknown; no LoadedU64Field; no pattern recorded.
        // r3 = *(u64*)(r2 + 0) on Unknown source records nothing.
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "bitfield at target offset must not seed cast: {map:?}"
        );
    }

    /// `struct_member_at` skips members whose bit offset is not a
    /// multiple of 8 (`bit_off % 8 != 0`) -- bit-position members
    /// inside a kind_flag=1 struct that happen to lie between byte
    /// boundaries cannot serve as a u64 LDX source. Even though the
    /// byte derived from the bit position would round to the LDX
    /// target offset, the alignment guard rejects.
    #[test]
    fn struct_member_at_skips_non_byte_aligned_member() {
        // T (kind_flag=1) with a u64 member at bit_offset = 65 (NOT
        // a multiple of 8; integer-divided by 8 gives byte 8). The
        // alignment guard rejects regardless of bitfield_size_bits.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_f = push_name(&mut strings, "f");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::StructBitfields {
                name_off: n_t,
                size: 24,
                members: vec![SynMemberBits {
                    name_off: n_f,
                    type_id: 1,
                    bit_offset: 65,        // NOT a multiple of 8
                    bitfield_size_bits: 0, // not a bitfield
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        // r2 = *(u64*)(r1 + 8): struct_member_at scans T's members,
        // sees bit_offset 65 % 8 = 1, skips. r2 stays Unknown.
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "non-byte-aligned member must not seed cast: {map:?}"
        );
    }

    /// `member_size_bytes` returns `None` for terminal types whose
    /// size is not resolvable from BTF alone (Func, FuncProto, Void,
    /// Fwd, Var, Datasec). The `build_layout_index` loop must handle
    /// `None` by skipping the member rather than panicking. Stress
    /// the path by constructing a candidate struct whose members
    /// include one each of Fwd, Func, and Void; the matcher must not
    /// include those member positions in the (offset, size) layout
    /// map. Without this guard the matcher would either crash on the
    /// `expect`-style unwrap or emit a candidate the renderer cannot
    /// chase.
    #[test]
    fn member_size_bytes_unsupported_terminals_skipped() {
        // BTF:
        //   id 1: u64
        //   id 2: T   { u64 f @ 8 }       -- source struct
        //   id 3: Fwd struct (kind_flag=0)
        //   id 4: FuncProto returning void, no params
        //   id 5: Func -> id 4
        //   id 6: U  { fwd_ref @ 0; func_ref @ 8; void_ref @ 16; u64 v @ 24 }
        //
        //   The members typed as Fwd / Func / Void all return None
        //   from member_size_bytes, so layout_index for U skips them.
        //   The single u64 at offset 24 makes U a candidate at
        //   (offset=24, size=8) ONLY. The LDX pattern accesses
        //   offset=24 (size 8); intersection -> {U}.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_u = push_name(&mut strings, "U");
        let n_fwd_target = push_name(&mut strings, "fwd_struct");
        let n_func = push_name(&mut strings, "fn");
        let n_fwd_ref = push_name(&mut strings, "fwd_ref");
        let n_func_ref = push_name(&mut strings, "func_ref");
        let n_void_ref = push_name(&mut strings, "void_ref");
        let n_v = push_name(&mut strings, "v");
        let n_f = push_name(&mut strings, "f");
        let types = vec![
            // id 1: u64
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: source T { u64 f @ 8 }
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            // id 3: Fwd (struct).
            SynType::Fwd {
                name_off: n_fwd_target,
                kind_flag: 0,
            },
            // id 4: FuncProto -> void, no params.
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![],
            },
            // id 5: Func -> id 4.
            SynType::Func {
                name_off: n_func,
                type_id: 4,
                linkage: 1,
            },
            // id 6: U with members of unsupported sizes plus one
            // sized u64 member. Production must skip the unsupported
            // ones and include only (offset=24, size=8).
            SynType::Struct {
                name_off: n_u,
                size: 32,
                members: vec![
                    SynMember {
                        name_off: n_fwd_ref,
                        type_id: 3, // Fwd -> None size
                        byte_offset: 0,
                    },
                    SynMember {
                        name_off: n_func_ref,
                        type_id: 5, // Func -> None size
                        byte_offset: 8,
                    },
                    SynMember {
                        name_off: n_void_ref,
                        type_id: 0, // Void -> None size
                        byte_offset: 16,
                    },
                    SynMember {
                        name_off: n_v,
                        type_id: 1, // u64 -> Some(8)
                        byte_offset: 24,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let u_id = 6;
        // r2 = *(u64*)(r1 + 8); r3 = *(u64*)(r2 + 24). The pattern
        // (offset=24, size=8) must intersect to {U} only -- Fwd /
        // Func / Void members at offsets 0/8/16 are skipped during
        // layout indexing.
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            ldx(BPF_SIZE_DW, 3, 2, 24),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: u_id,
                addr_space: AddrSpace::Arena
            }),
            "unsupported terminals must be skipped without crashing: {map:?}"
        );
    }

    /// `build_layout_index` skips bitfield members (those with
    /// `bitfield_size > 0`). A struct whose only u64 sits as a
    /// bitfield does NOT register in the (offset, size) layout map,
    /// so it cannot be a candidate even when the access pattern
    /// "would" match its byte position. The matcher converges on the
    /// struct that has a NON-bitfield member at the queried position.
    #[test]
    fn build_layout_index_skips_bitfields_in_candidates() {
        // BTF: u64(1), T(2, u64@8) source. Q1(3, kind_flag=1) with a
        // u64 BITFIELD at byte 0 (size 32 bits) -- must NOT register
        // as a candidate. Q2(4) with a u64 NON-bitfield at byte 0 --
        // sole candidate. Pattern (offset=0, size=8) -> {Q2} only.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q1 = push_name(&mut strings, "Q1");
        let n_q2 = push_name(&mut strings, "Q2");
        let n_f = push_name(&mut strings, "f");
        let n_a = push_name(&mut strings, "a");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // T has u64@8 source field.
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            // Q1 (kind_flag=1) with bitfield u64 at byte 0
            // (bitfield_size = 32 bits). Production must skip during
            // layout indexing because bitfield_size > 0.
            SynType::StructBitfields {
                name_off: n_q1,
                size: 8,
                members: vec![SynMemberBits {
                    name_off: n_a,
                    type_id: 1,
                    bit_offset: 0,
                    bitfield_size_bits: 32,
                }],
            },
            // Q2 with a normal u64 at byte 0 -- included in layout.
            SynType::Struct {
                name_off: n_q2,
                size: 8,
                members: vec![SynMember {
                    name_off: n_a,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let q2_id = 4;
        // Sequence: r2 = *(u64*)(r1 + 8); r3 = *(u64*)(r2 + 0).
        // Pattern (0, 8): layout includes Q2 only (Q1's bitfield
        // skipped). Map records (T, 8) -> (Q2, Arena).
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q2_id,
                addr_space: AddrSpace::Arena
            }),
            "bitfield candidate must be skipped: {map:?}"
        );
    }

    /// `BTF_KIND_UNION` participates in `build_layout_index` and
    /// `struct_member_at` identically to `BTF_KIND_STRUCT` -- `btf-rs`
    /// aliases `Union = Struct`, and production matches both via
    /// `Type::Struct(s) | Type::Union(s)`. Verify a kptr STX through
    /// a parent typed as a union records correctly, AND a candidate
    /// search resolves to a union when its layout uniquely matches.
    #[test]
    fn union_works_like_struct_for_layout_and_member_lookup() {
        // BTF:
        //   id 1: u64
        //   id 2: T (struct, kptr target) { u64 x @ 0 }
        //   id 3: T*
        //   id 4: P (UNION) { u64 slot @ 16 }
        //   id 5: SourceU (struct) { u64 f @ 8 }
        //   id 6: TargetU (UNION) { u64 a @ 0 } -- candidate target
        //
        // Two checks in one test:
        //   (a) STX through a Pointer{P=union} into P.slot at offset
        //       16 records (P, 16) -> T (Kernel).
        //   (b) LDX through a SourceU producing LoadedU64Field then
        //       deref at offset 0 (size 8) finds TargetU (the only
        //       struct/union with u64@0 in this BTF after dropping
        //       SourceU which has u64@8 not u64@0).
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_su = push_name(&mut strings, "SourceU");
        let n_tu = push_name(&mut strings, "TargetU");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let n_f = push_name(&mut strings, "f");
        let n_a = push_name(&mut strings, "a");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // T has its u64 at offset 8 (NOT 0) so layout(0, 8) is
            // uniquely satisfied by TargetU below — production must
            // converge on TargetU only when intersecting candidates.
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            SynType::Ptr { type_id: 2 }, // id 3: T*
            // id 4: P as a UNION with a u64 slot at byte 16.
            SynType::Union {
                name_off: n_p,
                size: 24,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id 5: SourceU as a struct with u64 source field at
            // offset 8.
            SynType::Struct {
                name_off: n_su,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            // id 6: TargetU as a UNION with u64 a at offset 0. Sole
            // candidate for the (0, 8) access pattern (after dropping
            // the source struct SourceU which has u64@8 not u64@0).
            // Ensures the layout index walks Union the same as Struct.
            SynType::Union {
                name_off: n_tu,
                size: 8,
                members: vec![SynMember {
                    name_off: n_a,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let source_u_id = 5;
        let target_u_id = 6;
        // Block 1: kptr through union parent.
        //   r1 = Pointer{P=union}; r6 = Pointer{T}.
        //   *(u64*)(r1 + 16) = r6  -> kptr_findings[(P,16)] = T.
        // Block 2: arena LDX through union target.
        //   r2 = Pointer{SourceU}; r3 = *(u64*)(r2 + 8); r4 = *(u64*)(r3 + 0).
        //   Pattern (0, 8) -> {TargetU}; (SourceU, 8) -> (TargetU, Arena).
        let insns = vec![
            stx(BPF_SIZE_DW, 1, 6, 16),
            ldx(BPF_SIZE_DW, 3, 2, 8),
            ldx(BPF_SIZE_DW, 4, 3, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: p_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 2,
                    struct_type_id: source_u_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, 16)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "kptr through union parent must record: {map:?}"
        );
        assert_eq!(
            map.get(&(source_u_id, 8)),
            Some(&CastHit {
                target_type_id: target_u_id,
                addr_space: AddrSpace::Arena
            }),
            "union target must be a layout candidate: {map:?}"
        );
    }

    /// `build_layout_index`'s walk advances type ids `1..=max_id`
    /// using `consecutive_fail` to short-circuit pathological /
    /// synthesized BTFs. After 256 consecutive failed
    /// `resolve_type_by_id` calls the loop breaks. To exercise this
    /// path the test relies on the fact that `max_seen_type_id +
    /// CANDIDATE_SEARCH_SLACK` (= 65538 here) is far above the
    /// legitimate ids in the BTF, so the walk WOULD iterate ~65k
    /// failed lookups without the cap; the consecutive fail cap of
    /// 256 short-circuits early. Verify that:
    ///   - the matcher still finds the legitimate candidate (loop
    ///     visits valid ids before the cap kicks in),
    ///   - the matcher does not panic when many failed ids accumulate.
    #[test]
    fn build_layout_index_consecutive_fail_cap_short_circuits() {
        // BTF: u64(1), T(2, u64@8 source), Q(3, u64@0 unique target).
        // Type ids 4..=65538 would all fail -- production stops after
        // 256 consecutive fails. The legitimate candidate at id 3 is
        // found before the cap activates, so the cast still emits.
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        // Even though ids 4..=65538 (max_seen=2 + slack=65536) all
        // resolve to errors, the consecutive_fail cap (256) stops the
        // loop early without panic, and Q(3) is recorded normally.
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "valid candidate must be found before fail cap; sparse \
             BTF must not panic: {map:?}"
        );
    }

    /// Non-bitfield members (`bitfield_size_bits = 0`) inside a
    /// `kind_flag=1` struct ARE included in `build_layout_index`. The
    /// production guard
    /// `matches!(m.bitfield_size(), Some(s) if s > 0)` only matches
    /// when the size is strictly positive. With kind_flag=1 every
    /// member exposes `bitfield_size = Some(0)` for non-bitfield
    /// members; production must NOT skip them.
    #[test]
    fn kind_flag_struct_includes_non_bitfield_members() {
        // BTF:
        //   id 1: u64
        //   id 2: T (kind_flag=0) { u64 src @ 8 }   -- source struct
        //   id 3: Q (kind_flag=1) { u64 a @ 0,
        //                            bf u64 b @ 64 (bitfield 32) }
        // Q has a non-bitfield u64 at byte 0. Production must include
        // it in layout (kind_flag=1, but bitfield_size=Some(0) since
        // bitfield_size_bits=0). The bitfield member at byte 8 is
        // skipped (bitfield_size_bits=32 > 0). Pattern (0, 8) -> {Q}.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_src = push_name(&mut strings, "src");
        let n_a = push_name(&mut strings, "a");
        let n_b = push_name(&mut strings, "b");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_src,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            // Q (kind_flag=1) has a non-bitfield u64 at byte 0 AND a
            // bitfield u64 at byte 8 (32-bit field). The non-bitfield
            // member must remain in layout despite kind_flag=1.
            SynType::StructBitfields {
                name_off: n_q,
                size: 16,
                members: vec![
                    SynMemberBits {
                        name_off: n_a,
                        type_id: 1,
                        bit_offset: 0,
                        bitfield_size_bits: 0,
                    },
                    SynMemberBits {
                        name_off: n_b,
                        type_id: 1,
                        bit_offset: 64,
                        bitfield_size_bits: 32,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let q_id = 3;
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "non-bitfield member of kind_flag=1 struct must be a \
             layout candidate: {map:?}"
        );
    }

    // ----- Stack edge cases ---------------------------------------

    /// `handle_stx`'s r10 spill guard treats a store with `off >= 0`
    /// as out-of-spec for BPF (the stack grows DOWN; slots live at
    /// negative offsets). The guard removes any prior slot at that
    /// offset rather than saving state. Symmetrically, `handle_ldx`'s
    /// r10 reload guard rejects loads with `off >= 0` and produces
    /// Unknown. Verify the STX path's invalidation: a write with
    /// `off >= 0` through r10 must remove any previously saved slot
    /// state at that offset so a later DW reload returns Unknown.
    #[test]
    fn stack_off_non_negative_through_r10_invalidates() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // *(u64 *)(r10 + 0) = R1   ; off >= 0 spill: dropped, no
        //                            ; state saved at slot 0
        // R3 = *(u64 *)(r10 + 0)   ; off >= 0 reload: returns Unknown
        // *(u64 *)(R6 + slot_off) = R3
        //                          ; R3 Unknown -> no kptr finding
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, 0),
            ldx(BPF_SIZE_DW, 3, 10, 0),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "non-negative r10 store must not save state, reload \
             returns Unknown: {map:?}"
        );
    }

    /// `field_byte_offset` returns `None` for negative offsets when
    /// the base register is NOT r10 (stack-relative loads are handled
    /// separately via the r10 fast path). On a struct-pointer base,
    /// a negative `off` is undefined behavior -- kernel struct fields
    /// have non-negative byte offsets relative to the struct base.
    /// Production drops dst to Unknown via
    /// `field_byte_offset(off) -> None` and the LDX records nothing.
    #[test]
    fn negative_off_in_non_r10_context_drops() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r2 = *(u64 *)(r1 - 8): r1 is Pointer{T}, NOT r10. Negative
        // off goes to the Pointer arm of handle_ldx and produces
        // None in field_byte_offset, dropping dst to Unknown. No
        // pattern recorded.
        // r3 = *(u64 *)(r2 + 0): r2 Unknown -> no record.
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, -8),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "negative offset through Pointer{{T}} must drop, no \
             pattern recorded: {map:?}"
        );
    }

    /// Two consecutive STX-through-r10 spills with the SAME source
    /// register state (Pointer{T}) overwrite the slot with a value
    /// indistinguishable from the prior contents. Production stores
    /// the second spill via `self.stack_slots.insert(off, regs[src])`
    /// which replaces by key but does not collapse to Conflicting.
    /// A later reload restores the same Pointer{T}; a subsequent STX
    /// of the reloaded register into a parent slot must record the
    /// kptr finding as `Single(T)` -- NOT `Conflicting`.
    #[test]
    fn stack_spill_same_target_stays_single() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // *(u64 *)(r10 - 8) = R1   ; spill Pointer{T} (slot=Pointer{T})
        // *(u64 *)(r10 - 8) = R1   ; spill Pointer{T} again -- same
        //                            ; target type, slot replaced
        //                            ; with same value, NOT Conflicting
        // R3 = *(u64 *)(r10 - 8)   ; reload as Pointer{T}
        // *(u64 *)(R6 + slot_off) = R3
        //                          ; records (P, slot_off) -> T
        let insns = vec![
            stx(BPF_SIZE_DW, 10, 1, -8),
            stx(BPF_SIZE_DW, 10, 1, -8),
            ldx(BPF_SIZE_DW, 3, 10, -8),
            stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "spill of identical Pointer{{T}} must reload as Pointer{{T}}, \
             kptr stays Single: {map:?}"
        );
    }

    // ----- kfunc edge cases ---------------------------------------

    /// `handle_kfunc_call` walks the FuncProto's return type through
    /// `bpf_map::resolve_to_struct_id`. A return type that resolves
    /// to a non-struct pointer (e.g. `int *`, `void *`) yields `None`
    /// from the resolver, so R0 stays Unknown. A subsequent STX of
    /// R0 into a u64 slot must NOT record a kptr finding.
    #[test]
    fn kfunc_call_returning_int_ptr_leaves_r0_unknown() {
        let slot_off: u32 = 16;
        // BTF:
        //   id 1: u64
        //   id 2: P (struct with u64 slot @ slot_off) -- kptr parent
        //         seed type for the post-call STX
        //   id 3: int* (Ptr -> u64). Pointee peels to Type::Int, so
        //         resolve_to_struct_id returns None.
        //   id 4: FuncProto returning id 3 (int*)
        //   id 5: Func -> id 4
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_p = push_name(&mut strings, "P");
        let n_slot = push_name(&mut strings, "slot");
        let n_kfunc = push_name(&mut strings, "bpf_returns_int_ptr");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            SynType::Ptr { type_id: 1 }, // id 3: u64*
            SynType::FuncProto {
                return_type_id: 3,
                params: vec![],
            },
            SynType::Func {
                name_off: n_kfunc,
                type_id: 4,
                linkage: 1,
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let p_id = 2;
        let kfunc_id = 5;
        // call kfunc returning int*; *(P + slot) = R0. R0 must be
        // Unknown (the return type's pointee resolves to Int, not
        // Struct, so resolve_to_struct_id returns None).
        let insns = vec![
            kfunc_call(kfunc_id),
            stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: p_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "kfunc returning int* must leave R0 Unknown: {map:?}"
        );
    }

    /// `handle_kfunc_call` short-circuits when the FuncProto's
    /// `return_type_id == 0` (void return). R0 stays Unknown after
    /// the standard r0..r5 clobber. A subsequent STX of R0 into a
    /// u64 slot must NOT record a kptr finding.
    #[test]
    fn kfunc_call_void_return_leaves_r0_unknown() {
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_p = push_name(&mut strings, "P");
        let n_slot = push_name(&mut strings, "slot");
        let n_kfunc = push_name(&mut strings, "bpf_void_return");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            // id 3: FuncProto -> void (return_type_id = 0).
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![],
            },
            // id 4: Func -> id 3.
            SynType::Func {
                name_off: n_kfunc,
                type_id: 3,
                linkage: 1,
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let p_id = 2;
        let kfunc_id = 4;
        let insns = vec![
            kfunc_call(kfunc_id),
            stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: p_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "kfunc with void return must leave R0 Unknown: {map:?}"
        );
    }

    /// `handle_kfunc_call`'s `imm` may resolve directly to a
    /// `Type::FuncProto` (no `Type::Func` wrapper). Production peels
    /// `Type::Func -> Type::FuncProto` when needed but also accepts a
    /// FuncProto id directly. A kfunc call with `imm` == FuncProto id
    /// must seed R0 from the proto's return type just as it would
    /// from a Func wrapper.
    #[test]
    fn kfunc_call_with_funcproto_id_directly() {
        let slot_off: u32 = 16;
        // BTF:
        //   id 1: u64
        //   id 2: T (kptr target struct) { u64 x @ 0 }
        //   id 3: T*
        //   id 4: P (struct holding the kptr slot)
        //   id 5: FuncProto -> T*  (no Func wrapper)
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 }, // id 3: T*
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            // id 5: FuncProto returning T* -- pass id=5 directly to
            // kfunc_call's imm so the resolver hits the Type::FuncProto
            // arm, not Type::Func -> peel.
            SynType::FuncProto {
                return_type_id: 3,
                params: vec![],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let proto_id = 5;
        // kfunc call with imm = proto_id (id 5 = FuncProto). R0 must
        // be set to Pointer{T} via the FuncProto-direct path.
        let insns = vec![
            kfunc_call(proto_id),
            stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: p_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "kfunc with direct FuncProto id must seed R0 from return \
             type: {map:?}"
        );
    }

    // ----- Register protection tests ------------------------------
    //
    // R10 is the read-only frame pointer per the BPF ABI; the
    // analyzer enforces the invariant that `regs[10]` is always
    // Unknown by guarding MOV and LDX early. Out-of-range register
    // indices (11..=15) can appear in a malformed instruction stream
    // because BpfInsn packs each field into 4 bits; the bounds check
    // at the top of step() and each handle_* routine rejects without
    // panicking.

    /// MOV with `dst == r10` is rejected by the production guard in
    /// the ALU64-MOV-X arm (`if dst == BPF_REG_R10 { return; }`).
    /// Verifying r10's state directly is impossible because the
    /// stack-spill / reload path keys on the register index, not on
    /// `regs[10]`'s `RegState`. The probe instead routes through a
    /// second MOV: `MOV r3, r10` copies `regs[10]` into `regs[3]`,
    /// then a deref chain through r3 records a cast iff `regs[10]`
    /// was typed. With the rejection working, `regs[10]` stays
    /// Unknown, so r3 stays Unknown, and the deref chain produces no
    /// record.
    #[test]
    fn mov_to_r10_rejected_keeps_r10_unknown() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r2 = *(u64 *)(r1 + 8)   -- r2 = LoadedU64Field{T, 8}
        // r10 = r2                -- REJECTED, r10 stays Unknown
        // r3 = r10                -- r3 = regs[10] = Unknown
        // r4 = *(u64 *)(r3 + 0)   -- r3 Unknown, no record
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            mov_x(10, 2),
            mov_x(3, 10),
            ldx(BPF_SIZE_DW, 4, 3, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "MOV r10, r2 must be rejected so r10 stays Unknown: {map:?}"
        );
    }

    /// LDX with `dst == r10` is rejected by the production guard
    /// (`if dst == BPF_REG_R10 { return; }` in `handle_ldx`). The
    /// same routing trick as `mov_to_r10_rejected_keeps_r10_unknown`
    /// observes the rejection: a successful LDX into r10 would have
    /// seeded `regs[10] = LoadedU64Field`, and a follow-up
    /// `MOV r3, r10; LDX r4, [r3+0]` chain would record a cast.
    /// With the guard active, r10 stays Unknown and the chain
    /// produces nothing.
    #[test]
    fn ldx_into_r10_rejected_keeps_r10_unknown() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r10 = *(u64 *)(r1 + 8)  -- REJECTED, r10 stays Unknown
        // r3 = r10                -- r3 = regs[10] = Unknown
        // r4 = *(u64 *)(r3 + 0)   -- r3 Unknown, no record
        let insns = vec![
            ldx(BPF_SIZE_DW, 10, 1, 8),
            mov_x(3, 10),
            ldx(BPF_SIZE_DW, 4, 3, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "LDX r10, [r1+8] must be rejected so r10 stays Unknown: {map:?}"
        );
    }

    /// `BPF_STX | BPF_DW | BPF_MEM` with both `dst` and `src` out of
    /// the 0..=10 valid register range (encoded as 15) must NOT panic.
    /// `BpfInsn::new` masks each register field to 4 bits, so 15
    /// round-trips through `dst_reg()` / `src_reg()` as 15. The bounds
    /// check at the top of `step()` (and the redundant guard in
    /// `handle_stx`) reject before any array indexing.
    #[test]
    fn oob_stx_reg_does_not_panic() {
        let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let bad = BpfInsn::new(BPF_CLASS_STX | BPF_SIZE_DW | BPF_MODE_MEM, 15, 15, 0, 0);
        let insns = vec![bad, exit()];
        let map = analyze_casts(&insns, &btf, &[], &[], &[]);
        assert!(
            map.is_empty(),
            "OOB STX (dst=15, src=15) must not panic: {map:?}"
        );
    }

    /// `BPF_ALU64 | BPF_OP_MOV | BPF_SRC_X` with `dst == 15` (out of
    /// range) must NOT panic. The bounds check at the top of `step()`
    /// rejects before the MOV branch executes.
    #[test]
    fn oob_mov_reg_does_not_panic() {
        let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let bad = BpfInsn::new(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 15, 0, 0, 0);
        let insns = vec![bad, exit()];
        let map = analyze_casts(&insns, &btf, &[], &[], &[]);
        assert!(map.is_empty(), "OOB MOV (dst=15) must not panic: {map:?}");
    }

    /// `BPF_STX | BPF_DW | BPF_ATOMIC` with `dst` and `src` out of
    /// range (15) must NOT panic. The bounds check at the top of
    /// `step()` rejects before dispatch into `handle_atomic`; the
    /// redundant guard at the top of `handle_atomic` is a defense-in-
    /// depth backstop.
    #[test]
    fn oob_atomic_reg_does_not_panic() {
        let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let bad = BpfInsn::new(
            BPF_CLASS_STX | BPF_SIZE_DW | BPF_MODE_ATOMIC,
            15,
            15,
            0,
            BPF_FETCH | 0xe0,
        );
        let insns = vec![bad, exit()];
        let map = analyze_casts(&insns, &btf, &[], &[], &[]);
        assert!(
            map.is_empty(),
            "OOB ATOMIC (dst=15, src=15) must not panic: {map:?}"
        );
    }

    /// `MOV dst, src` where the source register is `Unknown`
    /// overwrites the destination's typed state with `Unknown`.
    /// Production unconditionally copies `regs[src]` into `regs[dst]`,
    /// so a previously-typed dst loses its `Pointer{T}` /
    /// `LoadedU64Field` state when an Unknown source is moved in. A
    /// subsequent deref through dst must NOT record a cast.
    #[test]
    fn mov_x_unknown_source_overwrites_typed_dst() {
        // Seed r2 with a load chain so it carries LoadedU64Field{T, 8}.
        // Then MOV r2 = r3, where r3 stays Unknown. r2 becomes Unknown.
        // The follow-up deref chain through r2 must produce no record.
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![
            // r2 = *(u64*)(r1+8)  -- r2 = LoadedU64Field{T, 8}
            ldx(BPF_SIZE_DW, 2, 1, 8),
            // r2 = r3            -- r3 Unknown -> r2 becomes Unknown
            mov_x(2, 3),
            // r4 = *(u64*)(r2+0) -- r2 Unknown, no record
            ldx(BPF_SIZE_DW, 4, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "MOV with Unknown source must overwrite typed dst: {map:?}"
        );
    }

    /// Self-copy `MOV r2, r2` preserves the register's state because
    /// production reads and writes `regs[2]` with no intermediate
    /// transformation. A typed register that self-copies continues
    /// to carry its typed state into subsequent operations.
    #[test]
    fn mov_x_self_copy_preserves_state() {
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![
            // r2 = *(u64*)(r1+8)  -- r2 = LoadedU64Field{T, 8}
            ldx(BPF_SIZE_DW, 2, 1, 8),
            // r2 = r2             -- self-copy, r2 stays LoadedU64Field
            mov_x(2, 2),
            // r3 = *(u64*)(r2+0)  -- records access, resolves to Q
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena
            }),
            "MOV self-copy must preserve LoadedU64Field state: {map:?}"
        );
    }

    /// 32-bit MOV (`BPF_CLASS_ALU | BPF_OP_MOV | BPF_SRC_X`) destroys
    /// typed-pointer state in the destination register because a
    /// 32-bit move truncates the upper 32 bits of any 64-bit pointer.
    /// Production routes 32-bit MOV to `set_reg(dst, Unknown)`
    /// regardless of source state. A subsequent deref through the
    /// destination must record nothing.
    #[test]
    fn mov32_destroys_typed_state() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // Build a 32-bit MOV: ALU (not ALU64) | MOV | SRC_X.
        let mov32 = mk_insn(BPF_CLASS_ALU | BPF_OP_MOV | BPF_SRC_X, 4, 2, 0, 0);
        let insns = vec![
            // r2 = *(u64*)(r1+8)  -- r2 = LoadedU64Field{T, 8}
            ldx(BPF_SIZE_DW, 2, 1, 8),
            // r4 = (u32) r2       -- 32-bit MOV truncates -> r4 = Unknown
            mov32,
            // r5 = *(u64*)(r4+0)  -- r4 Unknown, no record
            ldx(BPF_SIZE_DW, 5, 4, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "32-bit MOV must drop typed state: {map:?}");
    }

    // ----- ALU op tests -------------------------------------------
    //
    // All non-MOV ALU ops (ADD, SUB, AND, OR, LSH, etc.) and
    // immediate-source MOV (`BPF_OP_MOV | BPF_SRC_K`) destroy the
    // typed-pointer property of the destination register. Production
    // handles every such case via the catch-all in the ALU dispatch:
    // drop dst to Unknown.
    //
    // Tests verify destruction by setting up a Pointer{T} register
    // that would normally produce a kptr finding when stored into a
    // P struct's u64 slot, then applying the ALU op, then performing
    // the STX. With the destruction working, the STX records nothing.

    /// `BPF_ALU64 | BPF_ADD | BPF_SRC_X` with a typed pointer in dst
    /// destroys the pointer state. Pointer + integer is no longer a
    /// pointer to the same struct (it's a derived address), so the
    /// kptr finding must drop.
    #[test]
    fn alu64_add_x_destroys_typed_pointer() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // ALU64 ADD X: code = BPF_CLASS_ALU64 | BPF_ADD | BPF_SRC_X.
        // BPF_ADD = 0x00, so code = 0x07 | 0x00 | 0x08 = 0x0f.
        let add_x = mk_insn(
            BPF_CLASS_ALU64 | (bs::BPF_ADD as u8) | BPF_SRC_X,
            1,
            3,
            0,
            0,
        );
        // r1 starts Pointer{T}. ADD r1, r3 -> r1 Unknown.
        // STX *(r6+slot_off) = r1 -> no record.
        let insns = vec![add_x, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "ALU64 ADD X must destroy typed pointer: {map:?}"
        );
    }

    /// `BPF_ALU64 | BPF_SUB | BPF_SRC_X` destroys the typed pointer
    /// state of the destination register. Same shape as ADD: any
    /// arithmetic on a pointer produces an integer, not a typed
    /// pointer.
    #[test]
    fn alu64_sub_x_destroys_typed_pointer() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        let sub_x = mk_insn(
            BPF_CLASS_ALU64 | (bs::BPF_SUB as u8) | BPF_SRC_X,
            1,
            3,
            0,
            0,
        );
        let insns = vec![sub_x, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "ALU64 SUB X must destroy typed pointer: {map:?}"
        );
    }

    /// `BPF_ALU64 | BPF_AND | BPF_SRC_X` destroys the typed pointer
    /// state of the destination register. Bitwise AND on a pointer
    /// produces a masked integer, not a typed pointer.
    #[test]
    fn alu64_and_x_destroys_typed_pointer() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        let and_x = mk_insn(
            BPF_CLASS_ALU64 | (bs::BPF_AND as u8) | BPF_SRC_X,
            1,
            3,
            0,
            0,
        );
        let insns = vec![and_x, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "ALU64 AND X must destroy typed pointer: {map:?}"
        );
    }

    /// `BPF_ALU64 | BPF_ADD | BPF_SRC_K` (immediate ADD) destroys
    /// typed-pointer state in the destination register. Same code
    /// path as ADD with register source -- production drops dst to
    /// Unknown for any non-MOV-X-ALU64 op.
    #[test]
    fn alu64_add_k_destroys_typed_pointer() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // ALU64 ADD K: BPF_SRC_K is 0, so source field is unused.
        // imm carries the constant.
        let add_k = mk_insn(BPF_CLASS_ALU64 | (bs::BPF_ADD as u8), 1, 0, 0, 8);
        let insns = vec![add_k, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "ALU64 ADD K must destroy typed pointer: {map:?}"
        );
    }

    /// Immediate MOV (`mov_k`, `BPF_OP_MOV | BPF_SRC_K`) destroys the
    /// destination register's typed-pointer state. The catch-all in
    /// the ALU dispatch handles this case because the `BPF_OP_MOV +
    /// BPF_SRC_X` short-circuit only matches the register source
    /// variant; the immediate variant lands in the destruction branch.
    #[test]
    fn mov_k_destroys_typed_pointer() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // mov_k r1, 42 -> r1 Unknown.
        // STX *(r6+slot_off) = r1 -> no record.
        let insns = vec![
            mov_k(1, 42),
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(map.is_empty(), "mov_k must destroy typed pointer: {map:?}");
    }

    // ----- Additional BPF_ADDR_SPACE_CAST tests -------------------

    /// `BPF_ADDR_SPACE_CAST` with a reserved imm value (`imm == 2`,
    /// neither `1` nor `0x10000`) drops the destination register to
    /// Unknown. The verifier in `kernel/bpf/verifier.c check_alu_op`
    /// rejects programs that use any other imm for
    /// `BPF_ADDR_SPACE_CAST`, so seeing it in pre-verification
    /// bytecode is malformed; treating dst as Unknown is the
    /// conservative direction.
    #[test]
    fn addr_space_cast_unknown_imm_drops_dst() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r3 = *(u64*)(r1+8)             -- r3 = LoadedU64Field{T, 8}
        // r4 = (cast imm=2) r3            -- imm=2 is reserved, r4 Unknown
        // r5 = *(u64*)(r4+0)              -- no record
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 2);
        let insns = vec![
            ldx(BPF_SIZE_DW, 3, 1, 8),
            cast,
            ldx(BPF_SIZE_DW, 5, 4, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "BPF_ADDR_SPACE_CAST with reserved imm must drop dst: {map:?}"
        );
    }

    /// `BPF_ADDR_SPACE_CAST` arena -> kernel (`imm == 1`) on a
    /// `Pointer{T}` source (rather than a `LoadedU64Field`) propagates
    /// the typed pointer state into the destination register.
    /// Production unconditionally copies `regs[src]` into `regs[dst]`;
    /// the LoadedU64Field-only branch merely populates
    /// `arena_confirmed` as a side effect when the source matches
    /// that variant. Since Pointer{T} sources skip the
    /// `arena_confirmed` insertion, no false-positive arena evidence
    /// is recorded, and the typed pointer survives the cast for use
    /// as a kptr value in a subsequent STX.
    #[test]
    fn addr_space_cast_arena_imm1_on_pointer_propagates() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r4 = (cast imm=1) r3   -- r3 = Pointer{T}, r4 = Pointer{T}
        // STX *(r6+slot_off) = r4 -- records (P, slot_off) -> T kptr finding
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 1);
        let insns = vec![cast, stx(BPF_SIZE_DW, 6, 4, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 3,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel
            }),
            "ADDR_SPACE_CAST imm=1 on Pointer{{T}} must propagate state: {map:?}"
        );
    }

    /// `BPF_ADDR_SPACE_CAST` kernel -> arena (`imm == 0x10000`) on a
    /// `Pointer{T}` source drops the destination to Unknown, even
    /// though the source carries a typed kernel pointer. Production
    /// routes any imm other than `1` (including `0x10000`) through
    /// the else branch, which clears dst regardless of source state.
    /// A subsequent kptr STX through the destination must NOT record.
    ///
    /// Sibling test `addr_space_cast_kernel_to_arena_drops_dst`
    /// covers the `LoadedU64Field` source case; this test extends
    /// coverage to the `Pointer{T}` source case.
    #[test]
    fn addr_space_cast_kernel_arena_drops_pointer_source() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r4 = (cast imm=0x10000) r3   -- r3 = Pointer{T}, r4 = Unknown
        // STX *(r6+slot_off) = r4      -- r4 Unknown, no record
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 0x10000);
        let insns = vec![cast, stx(BPF_SIZE_DW, 6, 4, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 3,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "ADDR_SPACE_CAST imm=0x10000 must drop Pointer source: {map:?}"
        );
    }

    // ----- Misc tests ---------------------------------------------

    /// `BPF_LD | BPF_W | BPF_ABS` (`code == 0x20`) is the legacy
    /// packet-data load mode kept for socket filters; it loads from
    /// packet data into r0. Production sets r0 to Unknown for any LD
    /// mode that is not the LD_IMM64 two-slot form. A previously-
    /// typed r0 must lose its typed state, so a follow-up kptr STX
    /// through r0 produces no record.
    #[test]
    fn bpf_ld_abs_clears_r0() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_LD | BPF_W | BPF_ABS = 0x00 | 0x00 | 0x20 = 0x20.
        let ld_abs = mk_insn(BPF_CLASS_LD | BPF_SIZE_W | (bs::BPF_ABS as u8), 0, 0, 0, 0);
        // r0 starts Pointer{T}. After BPF_LD_ABS, r0 is Unknown.
        // STX *(r6+slot_off) = r0 -> no record.
        let insns = vec![ld_abs, stx(BPF_SIZE_DW, 6, 0, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 0,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "BPF_LD_ABS must clear r0 typed state: {map:?}"
        );
    }

    /// `BPF_LD | BPF_W | BPF_IND` (`code == 0x40`) is the indirect
    /// packet-data load mode for socket filters; it loads from
    /// `packet[src + imm]` into r0. Production treats it the same way
    /// as `BPF_LD_ABS`: r0 becomes Unknown. A previously-typed r0
    /// must lose its state.
    #[test]
    fn bpf_ld_ind_clears_r0() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_LD | BPF_W | BPF_IND = 0x00 | 0x00 | 0x40 = 0x40.
        let ld_ind = mk_insn(BPF_CLASS_LD | BPF_SIZE_W | (bs::BPF_IND as u8), 0, 0, 0, 0);
        let insns = vec![ld_ind, stx(BPF_SIZE_DW, 6, 0, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 0,
                    struct_type_id: t_id,
                },
                InitialReg {
                    reg: 6,
                    struct_type_id: p_id,
                },
            ],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "BPF_LD_IND must clear r0 typed state: {map:?}"
        );
    }

    /// A program consisting of a single `EXIT` instruction must not
    /// panic and must produce an empty `CastMap`. EXIT is in
    /// `BPF_CLASS_JMP` with `op == BPF_OP_EXIT`, which production
    /// explicitly leaves unmodified. The empty instruction-stream
    /// behavior is the baseline; this test guards against regressions
    /// where a "no recognizable ops" program produces phantom output.
    #[test]
    fn single_exit_does_not_panic() {
        let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![exit()];
        let map = analyze_casts(&insns, &btf, &[], &[], &[]);
        assert!(map.is_empty(), "single EXIT must yield empty map: {map:?}");
    }

    /// A program of only jump / branch instructions (no LDX, STX, MOV,
    /// or call) carries no data flow that the analyzer could track.
    /// Production processes each insn through `step()` but the JMP
    /// arm only mutates state on CALL -- branches and EXIT are no-ops
    /// for state. The forward walk completes without panicking; the
    /// output map is empty even though every PC is processed.
    #[test]
    fn jumps_only_program_does_not_panic() {
        let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_JEQ_K = 0x10. Construct a sequence of conditional and
        // unconditional jumps that branch within bounds.
        // pc 0: if r1 == 0 goto +1 (target = pc 2)
        // pc 1: ja +1               (target = pc 3)
        // pc 2: ja -2               (target = pc 1)
        // pc 3: exit
        let jeq = mk_insn(BPF_CLASS_JMP | 0x10, 1, 0, 1, 0);
        let ja_plus = mk_insn(BPF_CLASS_JMP, 0, 0, 1, 0);
        let ja_minus = mk_insn(BPF_CLASS_JMP, 0, 0, -2, 0);
        let insns = vec![jeq, ja_plus, ja_minus, exit()];
        let map = analyze_casts(&insns, &btf, &[], &[], &[]);
        assert!(
            map.is_empty(),
            "all-jumps program must yield empty map: {map:?}"
        );
    }

    // ----- BSS / Datasec kptr detection ---------------------------
    //
    // These tests exercise the `BPF_PSEUDO_MAP_VALUE` path in the
    // `BPF_LD_IMM64` arm. The tests build a synthetic BTF with a
    // `BTF_KIND_DATASEC` over a `BTF_KIND_VAR` whose underlying
    // type is a plain u64 — the BSS layout libbpf generates for
    // `__u64 my_kptr;`. The `DatasecPointer` annotation passed to
    // `analyze_casts` mirrors what the host-side cast loader
    // emits after walking `.rel.text` against the program's
    // datasec sections.

    /// Helper: emit `BPF_LD_IMM64 dst, imm` as the two-instruction
    /// pseudo. The second slot's `code` is 0 per linux uapi
    /// `bpf.h`. The analyzer's `skip_next` flag swallows the
    /// second slot.
    fn ld_imm64(dst: u8, imm: i32) -> [BpfInsn; 2] {
        let lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, dst, 0, 0, imm);
        let hi = mk_insn(0, 0, 0, 0, 0);
        [lo, hi]
    }

    /// Build a synthetic BTF that declares a `BTF_KIND_DATASEC`
    /// (`.bss`) containing a single u64 global variable
    /// (`my_kptr`) at offset 0. Returns `(blob, datasec_id,
    /// kptr_target_id, var_byte_offset, kfunc_btf_id)` where
    /// `kptr_target_id` is a separate struct
    /// (`task_struct`-stand-in) that the STX path stores INTO
    /// the u64 slot.
    ///
    /// Layout (BTF type ids assigned in order):
    /// - id 1: int u64 (size=8, bits=64)
    /// - id 2: struct task_struct { u64 x @ 0 }   -- kptr target
    /// - id 3: T*
    /// - id 4: BTF_KIND_VAR(name="my_kptr", type=1, linkage=GLOBAL)
    /// - id 5: BTF_KIND_DATASEC(name=".bss", size=8, entries=[
    ///   {type=4, offset=0, size=8}])
    /// - id 6: FuncProto returning T*
    /// - id 7: Func("bpf_task_acquire") -> id 6
    fn btf_bss_with_kptr() -> (Vec<u8>, u32, u32, u32, u32) {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "task_struct");
        let n_x = push_name(&mut strings, "x");
        let n_kptr = push_name(&mut strings, "my_kptr");
        let n_bss = push_name(&mut strings, ".bss");
        let n_kfunc = push_name(&mut strings, "bpf_task_acquire");
        let types = vec![
            // id 1: u64
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: struct task_struct { u64 x @ 0 }
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            // id 3: T*
            SynType::Ptr { type_id: 2 },
            // id 4: Var("my_kptr", type=u64=1, linkage=GLOBAL)
            SynType::Var {
                name_off: n_kptr,
                type_id: 1,
                linkage: 1,
            },
            // id 5: Datasec(".bss") containing my_kptr at offset 0
            SynType::Datasec {
                name_off: n_bss,
                size: 8,
                entries: vec![SynVarSecinfo {
                    type_id: 4,
                    offset: 0,
                    size: 8,
                }],
            },
            // id 6: FuncProto -> T*
            SynType::FuncProto {
                return_type_id: 3,
                params: vec![],
            },
            // id 7: Func bpf_task_acquire (linkage = global)
            SynType::Func {
                name_off: n_kfunc,
                type_id: 6,
                linkage: 1,
            },
        ];
        let blob = build_btf(&types, &strings);
        (blob, 5, 2, 0, 7)
    }

    /// `BPF_LD_IMM64` with a `DatasecPointer` annotation must type
    /// the destination register as a typed pointer into the
    /// datasec. The follow-up STX through that register records a
    /// kptr finding keyed on `(datasec_id, var_byte_offset)`.
    ///
    /// Sequence (mirrors clang's `my_kptr = bpf_task_acquire(...)`
    /// codegen):
    ///   call kfunc bpf_task_acquire   ; r0 = T*
    ///   r1 = LD_IMM64(.bss, 0)        ; r1 = DatasecPointer{bss, 0}
    ///   *(u64 *)(r1 + 0) = r0         ; STX r0 into .bss[my_kptr]
    ///
    /// Expected: CastMap entry
    /// `(datasec_id, 0) -> (T, AddrSpace::Kernel)`.
    #[test]
    fn bss_kptr_records_kernel_cast() {
        let (blob, datasec_id, t_id, var_off, kfunc_id) = btf_bss_with_kptr();
        let btf = Btf::from_bytes(&blob).unwrap();
        let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
        let stx_kptr = stx(BPF_SIZE_DW, 1, 0, 0);
        let insns = vec![kfunc_call(kfunc_id), ld_lo, ld_hi, stx_kptr, exit()];
        // PC numbering: 0=call, 1=ld_lo, 2=ld_hi (skipped via
        // skip_next), 3=stx, 4=exit. The DatasecPointer marks PC=1
        // (the BPF_LD_IMM64 lo slot) as targeting the .bss
        // datasec at the my_kptr offset.
        let datasec_pointers = vec![DatasecPointer {
            insn_offset: 1,
            datasec_type_id: datasec_id,
            base_offset: var_off,
        }];
        let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers);
        assert_eq!(
            map.get(&(datasec_id, var_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "kfunc-returned T* stored into .bss[my_kptr] must record \
             (datasec_id, 0) -> (T, Kernel): {map:?}"
        );
    }

    /// `BPF_LD_IMM64` WITHOUT a `DatasecPointer` annotation leaves
    /// the destination register as `Unknown`, so a follow-up STX
    /// through it cannot record a kptr finding. This guards
    /// against a regression where the analyzer accidentally types
    /// the LD_IMM64 destination as the global variable's
    /// underlying integer type just from the BTF.
    #[test]
    fn ld_imm64_without_annotation_no_record() {
        let (blob, _datasec_id, _t_id, var_off, kfunc_id) = btf_bss_with_kptr();
        let btf = Btf::from_bytes(&blob).unwrap();
        let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
        let stx_kptr = stx(BPF_SIZE_DW, 1, 0, 0);
        let insns = vec![kfunc_call(kfunc_id), ld_lo, ld_hi, stx_kptr, exit()];
        // Empty datasec_pointers — analyzer has no way to recover
        // the parent datasec id, so the LD_IMM64 destination
        // stays Unknown.
        let map = analyze_casts(&insns, &btf, &[], &[], &[]);
        assert!(
            map.is_empty(),
            "LD_IMM64 without DatasecPointer annotation must not record \
             a kptr finding: {map:?}"
        );
    }

    /// `BPF_LD_IMM64` with a `DatasecPointer` annotation but the
    /// follow-up STX uses an untyped value register (literal
    /// constant via mov_k) records nothing. The kptr path
    /// requires both base AND value registers to be typed.
    #[test]
    fn bss_stx_with_untyped_value_no_record() {
        let (blob, datasec_id, _t_id, var_off, _kfunc_id) = btf_bss_with_kptr();
        let btf = Btf::from_bytes(&blob).unwrap();
        let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
        // r0 = literal 0 (mov_k clobbers any prior typed state)
        // *(u64 *)(r1 + 0) = r0      ; r0 Unknown -> no record
        let mov_zero = mov_k(0, 0);
        let stx_kptr = stx(BPF_SIZE_DW, 1, 0, 0);
        let insns = vec![ld_lo, ld_hi, mov_zero, stx_kptr, exit()];
        let datasec_pointers = vec![DatasecPointer {
            insn_offset: 0,
            datasec_type_id: datasec_id,
            base_offset: var_off,
        }];
        let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers);
        assert!(
            map.is_empty(),
            "STX with untyped value register must not record kptr: {map:?}"
        );
    }

    /// Multi-variable BSS layout: a single datasec contains TWO
    /// u64 globals at distinct offsets. The analyzer must key
    /// each kptr finding on the right `(datasec_id, var_offset)`
    /// pair without conflating them.
    #[test]
    fn bss_multi_variable_layout() {
        // BTF: u64(1), T(2, u64@0), T*(3), Var "kptr_a"(4),
        // Var "kptr_b"(5), Datasec(6, [(4,0,8), (5,16,8)]),
        // FuncProto(7), Func(8).
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "task_struct");
        let n_x = push_name(&mut strings, "x");
        let n_a = push_name(&mut strings, "kptr_a");
        let n_b = push_name(&mut strings, "kptr_b");
        let n_bss = push_name(&mut strings, ".bss");
        let n_kfunc = push_name(&mut strings, "bpf_task_acquire");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Var {
                name_off: n_a,
                type_id: 1,
                linkage: 1,
            },
            SynType::Var {
                name_off: n_b,
                type_id: 1,
                linkage: 1,
            },
            SynType::Datasec {
                name_off: n_bss,
                size: 24,
                entries: vec![
                    SynVarSecinfo {
                        type_id: 4,
                        offset: 0,
                        size: 8,
                    },
                    SynVarSecinfo {
                        type_id: 5,
                        offset: 16,
                        size: 8,
                    },
                ],
            },
            SynType::FuncProto {
                return_type_id: 3,
                params: vec![],
            },
            SynType::Func {
                name_off: n_kfunc,
                type_id: 7,
                linkage: 1,
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let datasec_id = 6;
        let t_id = 2;
        let kfunc_id = 8;
        let [ld_a_lo, ld_a_hi] = ld_imm64(1, 0);
        let [ld_b_lo, ld_b_hi] = ld_imm64(2, 16);
        let insns = vec![
            kfunc_call(kfunc_id),
            ld_a_lo,
            ld_a_hi,
            stx(BPF_SIZE_DW, 1, 0, 0),
            kfunc_call(kfunc_id),
            ld_b_lo,
            ld_b_hi,
            stx(BPF_SIZE_DW, 2, 0, 0),
            exit(),
        ];
        // PC numbering: 0=call, 1=ld_a_lo, 2=ld_a_hi, 3=stx_a,
        // 4=call, 5=ld_b_lo, 6=ld_b_hi, 7=stx_b, 8=exit.
        let datasec_pointers = vec![
            DatasecPointer {
                insn_offset: 1,
                datasec_type_id: datasec_id,
                base_offset: 0,
            },
            DatasecPointer {
                insn_offset: 5,
                datasec_type_id: datasec_id,
                base_offset: 16,
            },
        ];
        let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers);
        assert_eq!(
            map.get(&(datasec_id, 0)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "kptr_a at offset 0: {map:?}"
        );
        assert_eq!(
            map.get(&(datasec_id, 16)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "kptr_b at offset 16: {map:?}"
        );
    }

    /// `struct_member_at` on a Datasec parent finds the variable
    /// whose byte range contains the queried offset. A query
    /// inside a multi-byte variable's range returns the
    /// variable's start offset, NOT the queried offset, in
    /// `MemberAt::Datasec::var_byte_offset`. A query that lands
    /// outside any variable's range returns None.
    #[test]
    fn struct_member_at_datasec_resolves_variables() {
        let (blob, datasec_id, _t_id, _var_off, _kfunc_id) = btf_bss_with_kptr();
        let btf = Btf::from_bytes(&blob).unwrap();
        // Exact-offset hit: my_kptr starts at byte 0.
        let m0 = struct_member_at(&btf, datasec_id, 0).expect("byte 0 must hit my_kptr");
        match m0 {
            MemberAt::Datasec {
                var_byte_offset, ..
            } => assert_eq!(var_byte_offset, 0),
            MemberAt::Struct { .. } => panic!("Datasec parent must yield Datasec match"),
        }
        // Mid-variable hit: byte 4 lands inside my_kptr's [0, 8)
        // range; should return the variable's start (0).
        let m4 = struct_member_at(&btf, datasec_id, 4).expect("byte 4 must hit my_kptr range");
        match m4 {
            MemberAt::Datasec {
                var_byte_offset, ..
            } => assert_eq!(var_byte_offset, 0),
            MemberAt::Struct { .. } => panic!("Datasec parent must yield Datasec match"),
        }
        // Out-of-range hit: byte 100 is past the section.
        assert!(
            struct_member_at(&btf, datasec_id, 100).is_none(),
            "byte 100 outside section must return None"
        );
    }

    /// End-to-end: a BSS u64 stores a kfunc-returned pointer
    /// (mirrors `__u64 my_kptr; my_kptr = bpf_task_acquire(...)`
    /// at the analyzer level). Produces exactly one CastMap entry
    /// keyed on `(datasec_id, 0)` -> `(task_struct, Kernel)`.
    #[test]
    fn end_to_end_bss_global_stores_kfunc_pointer() {
        let (blob, datasec_id, t_id, var_off, kfunc_id) = btf_bss_with_kptr();
        let btf = Btf::from_bytes(&blob).unwrap();
        let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
        let insns = vec![
            kfunc_call(kfunc_id),
            ld_lo,
            ld_hi,
            stx(BPF_SIZE_DW, 1, 0, 0),
            exit(),
        ];
        let datasec_pointers = vec![DatasecPointer {
            insn_offset: 1,
            datasec_type_id: datasec_id,
            base_offset: var_off,
        }];
        let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers);
        assert_eq!(map.len(), 1, "exactly one finding expected: {map:?}");
        assert_eq!(
            map.get(&(datasec_id, var_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
        );
    }

    // ----- Edge case tests: kfunc imm=0 ---------------------------

    /// `handle_kfunc_call` short-circuits on `imm <= 0`. Typically
    /// `imm = -1` for an unrelocated kfunc placeholder; `imm = 0`
    /// also hits the short-circuit. R0 stays Unknown after the
    /// standard R0..R5 clobber.
    #[test]
    fn kfunc_call_imm_zero_leaves_r0_unknown() {
        let slot_off: u32 = 16;
        let (blob, _t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![
            kfunc_call(0),
            stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: p_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "kfunc_call imm=0 must leave R0 Unknown: {map:?}"
        );
    }

    // ----- Edge case tests: jumps ---------------------------------

    /// `BPF_JMP32 | BPF_JA` (gotol, op=0x00) uses `insn.imm` as the
    /// 32-bit jump offset per `jump_targets`. The target PC is reset.
    #[test]
    fn jmp32_gotol_resets_state_at_target() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // pc 0: r2 = T.f (LoadedU64Field).
        // pc 1: gotol +1 (JMP32|JA, imm=1). Target = pc 3.
        // pc 2: exit (skipped).
        // pc 3: r3 = *(u64 *)(r2 + 0) — state reset, no record.
        let gotol = mk_insn(BPF_CLASS_JMP32, 0, 0, 0, 1);
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            gotol,
            exit(),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(map.is_empty(), "JMP32|JA target must reset state: {map:?}");
    }

    /// Out-of-range jump targets (negative resolved address, or past
    /// `insns.len()`) are silently dropped. State survives.
    #[test]
    fn out_of_range_jump_targets_dropped() {
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let jeq_neg = mk_insn(BPF_CLASS_JMP | 0x10, 2, 0, -100, 0);
        let jeq_pos = mk_insn(BPF_CLASS_JMP | 0x10, 2, 0, 100, 0);
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            jeq_neg,
            jeq_pos,
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "out-of-range jumps must drop, state survives: {map:?}"
        );
    }

    /// All conditional jump opcodes register their targets per
    /// `jump_targets`. JEQ, JGT, JGE, JSET, JNE, JSGT, JSGE, JLT,
    /// JLE, JSLT, JSLE — each one's target PC must reset state.
    #[test]
    fn all_conditional_jumps_register_targets() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let ops: [u8; 11] = [
            0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0xa0, 0xb0, 0xc0, 0xd0,
        ];
        for op in ops {
            let cond = mk_insn(BPF_CLASS_JMP | op, 2, 0, 1, 0);
            let insns = vec![
                ldx(BPF_SIZE_DW, 2, 1, 8),
                cond,
                exit(),
                ldx(BPF_SIZE_DW, 3, 2, 0),
                exit(),
            ];
            let map = analyze_casts(
                &insns,
                &btf,
                &[InitialReg {
                    reg: 1,
                    struct_type_id: t_id,
                }],
                &[],
                &[],
            );
            assert!(
                map.is_empty(),
                "JMP op 0x{op:02x} target must reset state: {map:?}"
            );
        }
    }

    // ----- Edge case tests: FuncEntry -----------------------------

    /// Multiple `FuncEntry` entries at the same PC are processed in
    /// order — last one wins. Each entry's `seed_from_func_proto`
    /// clears all registers before seeding.
    /// Two FuncProtos at PC 0:
    ///   A: ([T*, P*]) — seeds R1=T*, R2=P*.
    ///   B: ([P*, T*]) — seeds R1=P*, R2=T*.
    /// With B processed second, R1=P* and R2=T*. Records (P, slot) -> T.
    #[test]
    fn func_entry_multiple_at_same_pc_last_wins() {
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let n_arg_t = push_name(&mut strings, "arg_t");
        let n_arg_p = push_name(&mut strings, "arg_p");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            SynType::Ptr { type_id: 4 },
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![
                    SynParam {
                        name_off: n_arg_t,
                        type_id: 3,
                    },
                    SynParam {
                        name_off: n_arg_p,
                        type_id: 5,
                    },
                ],
            },
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![
                    SynParam {
                        name_off: n_arg_p,
                        type_id: 5,
                    },
                    SynParam {
                        name_off: n_arg_t,
                        type_id: 3,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let proto_a = 6;
        let proto_b = 7;
        let insns = vec![stx(BPF_SIZE_DW, 1, 2, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[],
            &[
                FuncEntry {
                    insn_offset: 0,
                    func_proto_id: proto_a,
                },
                FuncEntry {
                    insn_offset: 0,
                    func_proto_id: proto_b,
                },
            ],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "later FuncEntry at same PC must win: {map:?}"
        );
    }

    /// `FuncEntry` with `insn_offset` past `insns.len()` is silently
    /// skipped — the loop never finds a matching PC.
    #[test]
    fn func_entry_past_insns_len_no_op() {
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[FuncEntry {
                insn_offset: 999,
                func_proto_id: 1,
            }],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "FuncEntry past insns.len() must not affect run: {map:?}"
        );
    }

    /// `FuncEntry` at PC 0 with no params clears all registers, then
    /// iterates an empty param list (no seeding). InitialReg state
    /// is wiped.
    #[test]
    fn func_entry_pc0_no_params_clears_initial_regs() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            SynType::Struct {
                name_off: n_q,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let proto_id = 4;
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[FuncEntry {
                insn_offset: 0,
                func_proto_id: proto_id,
            }],
            &[],
        );
        assert!(
            map.is_empty(),
            "FuncEntry with empty params must clear all regs: {map:?}"
        );
    }

    /// `FuncEntry` with `func_proto_id == 0` (Void) hits the
    /// `_ => return` arm — but only AFTER all registers are cleared.
    #[test]
    fn func_entry_proto_id_zero_clears_regs_no_seed() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[FuncEntry {
                insn_offset: 0,
                func_proto_id: 0,
            }],
            &[],
        );
        assert!(
            map.is_empty(),
            "FuncEntry with proto_id=0 must clear regs and not seed: {map:?}"
        );
    }

    /// `FuncEntry` at PC > 0 reseeds at the matching PC mid-stream.
    #[test]
    fn func_entry_pc_gt_0_reseeds_mid_stream() {
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
        let n_arg_t = push_name(&mut strings, "arg_t");
        let n_arg_p = push_name(&mut strings, "arg_p");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
            SynType::Ptr { type_id: 2 },
            SynType::Struct {
                name_off: n_p,
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            SynType::Ptr { type_id: 4 },
            SynType::FuncProto {
                return_type_id: 0,
                params: vec![
                    SynParam {
                        name_off: n_arg_t,
                        type_id: 3,
                    },
                    SynParam {
                        name_off: n_arg_p,
                        type_id: 5,
                    },
                ],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id = 2;
        let p_id = 4;
        let proto_id = 6;
        // pc 0: exit. pc 1: STX *(R2 + slot_off) = R1.
        // FuncEntry at PC 1 reseeds R1=T*, R2=P*. Records (P, slot) -> T.
        let insns = vec![exit(), stx(BPF_SIZE_DW, 2, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[],
            &[FuncEntry {
                insn_offset: 1,
                func_proto_id: proto_id,
            }],
            &[],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "FuncEntry at PC>0 must reseed: {map:?}"
        );
    }

    // ----- Misc edge case tests -----------------------------------

    /// The second slot of `BPF_LD_IMM64` is skipped per `skip_next`.
    /// Even non-zero content must not be interpreted as instruction.
    #[test]
    fn ld_imm64_second_slot_with_non_zero_content_skipped() {
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // pc 0: LD_IMM64 first slot (with non-zero imm = 42).
        // pc 1: second slot — emit a fake "instruction" that LOOKS
        //       like an ALU64|MOV|X. It must be skipped.
        // pc 2: r2 = T.f
        // pc 3: r3 = *r2
        let ld_imm64_lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, 6, 0, 0, 42);
        let fake_mov = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 0, 0);
        let insns = vec![
            ld_imm64_lo,
            fake_mov,
            ldx(BPF_SIZE_DW, 2, 1, 8),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "non-zero LD_IMM64 second slot must skip: {map:?}"
        );
    }

    /// `seed` iterates `initial_regs` in order; later seeds for the
    /// same register overwrite earlier ones (last wins).
    #[test]
    fn initial_reg_duplicate_seeds_last_wins() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_s1 = push_name(&mut strings, "S1");
        let n_s2 = push_name(&mut strings, "S2");
        let n_q = push_name(&mut strings, "Q");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");
        let types = vec![
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            // id 2: S1 { u64 f @ 8 }, size 16
            SynType::Struct {
                name_off: n_s1,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 8,
                }],
            },
            // id 3: S2 { u64 f @ 16 }, size 24
            SynType::Struct {
                name_off: n_s2,
                size: 24,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 1,
                    byte_offset: 16,
                }],
            },
            // id 4: Q { u64 x @ 0 }, size 8
            SynType::Struct {
                name_off: n_q,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 0,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let s1_id = 2;
        let s2_id = 3;
        let q_id = 4;
        // Seed R1 first as S1, then as S2 — last wins, R1 = S2.
        // Sequence: r2 = *(u64*)(r1+16) = S2.f, then r3 = *r2 at 0.
        // Records (S2, 16) -> Q. If first seed had won, S1 has no
        // field at offset 16, so no record.
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 16),
            ldx(BPF_SIZE_DW, 3, 2, 0),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[
                InitialReg {
                    reg: 1,
                    struct_type_id: s1_id,
                },
                InitialReg {
                    reg: 1,
                    struct_type_id: s2_id,
                },
            ],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(s2_id, 16)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "duplicate InitialReg seed must use last value: {map:?}"
        );
        assert!(
            !map.contains_key(&(s1_id, 16)),
            "first InitialReg seed must NOT take effect: {map:?}"
        );
    }

    /// `InitialReg` with `struct_type_id == 0` is silently dropped.
    #[test]
    fn initial_reg_struct_type_id_zero_dropped() {
        let (blob, _t, _q) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: 0,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "InitialReg with struct_type_id=0 must be dropped: {map:?}"
        );
    }

    // ----- Stress / boundary tests -------------------------------
    //
    // These tests target performance regressions (quadratic blowups
    // over the BTF id walk, the patterns set, the layout index) and
    // boundary panics (OOB indexing on max-stack, depth-limit chains
    // through `peel_modifiers`, all-LD_IMM64 streams driving
    // `skip_next` to the program end). Assertions verify exact
    // `CastMap` contents — counting only would mask both spurious
    // entries and missed entries.

    /// 10,000-instruction program: stuffed with `r0 = 0` no-ops with
    /// a single arena cast pattern buried near the middle. Verifies
    /// the analyzer's forward-pass cost remains linear in instruction
    /// count and that its register tracking does not lose the typed
    /// state across thousands of unrelated instructions. Single-slot
    /// `r0 = 0` only clobbers `r0`, so the seeded `r1 = T*` and the
    /// loaded `r2 = LoadedU64Field{T, 8}` survive across the no-op
    /// padding and the cast resolves uniquely to `Q`. Real `BPF_JA +0`
    /// would add every PC to the jump-target set and reset register
    /// state at every step.
    #[test]
    fn large_program_buried_cast_recorded() {
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let mut insns: Vec<BpfInsn> = Vec::with_capacity(10_001);
        for _ in 0..4_999 {
            insns.push(mov_k(0, 0));
        }
        insns.push(ldx(BPF_SIZE_DW, 2, 1, 8));
        insns.push(ldx(BPF_SIZE_DW, 3, 2, 0));
        for _ in 0..4_998 {
            insns.push(mov_k(0, 0));
        }
        insns.push(exit());
        assert_eq!(insns.len(), 10_000);
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.len(),
            1,
            "exactly one cast in 10k-insn program: {map:?}"
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "buried cast must resolve: {map:?}"
        );
    }

    /// 100 distinct `FuncEntry` records at consecutive PCs, each
    /// pointing at a distinct `FuncProto(T_i*, P*) -> void`. After
    /// each entry's reseeding, the single instruction at that PC is
    /// `STX *(R2 + slot_off_i) = R1`, recording `(P, slot_off_i) ->
    /// (T_i, Kernel)`. Verifies that the analyzer applies every
    /// `FuncEntry` (no off-by-one, no early-exit on the entry list
    /// scan) and that 100 distinct kptr findings land in the output.
    /// Sized below the `i16` byte-offset bound (100 * 8 = 800).
    #[test]
    fn many_func_entries_each_seeds() {
        const N: usize = 100;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_p = push_name(&mut strings, "P");
        let n_arg_t = push_name(&mut strings, "task");
        let n_arg_p = push_name(&mut strings, "parent");
        let mut t_name_offs = Vec::with_capacity(N);
        for i in 0..N {
            t_name_offs.push(push_name(&mut strings, &format!("T{i}")));
        }
        let mut slot_name_offs = Vec::with_capacity(N);
        for i in 0..N {
            slot_name_offs.push(push_name(&mut strings, &format!("slot{i}")));
        }
        // Type id layout (1-indexed; id 0 is Void):
        //   1: u64; 2..=N+1: T_i (each with u64@0);
        //   N+2..=2N+1: T_i*; 2N+2: P (N u64 fields); 2N+3: P*;
        //   2N+4..=3N+3: N FuncProtos (T_i*, P*) -> void.
        let mut types: Vec<SynType> = Vec::new();
        types.push(SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        });
        for &name_off in t_name_offs.iter().take(N) {
            types.push(SynType::Struct {
                name_off,
                size: 8,
                members: vec![SynMember {
                    name_off: 0,
                    type_id: 1,
                    byte_offset: 0,
                }],
            });
        }
        for i in 0..N {
            types.push(SynType::Ptr {
                type_id: (2 + i) as u32,
            });
        }
        let p_size: u32 = 8 * (N as u32);
        let p_members: Vec<SynMember> = (0..N)
            .map(|i| SynMember {
                name_off: slot_name_offs[i],
                type_id: 1,
                byte_offset: 8 * i as u32,
            })
            .collect();
        types.push(SynType::Struct {
            name_off: n_p,
            size: p_size,
            members: p_members,
        });
        let p_id: u32 = 2 + 2 * N as u32;
        types.push(SynType::Ptr { type_id: p_id });
        let p_ptr_id: u32 = 2 * N as u32 + 3;
        for i in 0..N {
            types.push(SynType::FuncProto {
                return_type_id: 0,
                params: vec![
                    SynParam {
                        name_off: n_arg_t,
                        type_id: (N as u32 + 2 + i as u32),
                    },
                    SynParam {
                        name_off: n_arg_p,
                        type_id: p_ptr_id,
                    },
                ],
            });
        }
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let mut insns: Vec<BpfInsn> = Vec::with_capacity(N + 1);
        let mut func_entries: Vec<FuncEntry> = Vec::with_capacity(N);
        for i in 0..N {
            insns.push(stx(BPF_SIZE_DW, 2, 1, (8 * i) as i16));
            let proto_id: u32 = 2 * N as u32 + 4 + i as u32;
            func_entries.push(FuncEntry {
                insn_offset: i,
                func_proto_id: proto_id,
            });
        }
        insns.push(exit());
        let map = analyze_casts(&insns, &btf, &[], &func_entries, &[]);
        assert_eq!(map.len(), N, "expected {N} kptr findings: {map:?}");
        for i in 0..N {
            let t_id = (2 + i) as u32;
            assert_eq!(
                map.get(&(p_id, 8 * i as u32)),
                Some(&CastHit {
                    target_type_id: t_id,
                    addr_space: AddrSpace::Kernel,
                }),
                "FuncEntry #{i} at PC {i} must record (P, {}) -> T{i}: {map:?}",
                8 * i as u32
            );
        }
    }

    /// 500 distinct struct types in the BTF; only one matches the
    /// observed access pattern. Verifies that the matcher's
    /// intersection over `build_layout_index` correctly narrows the
    /// candidate set when nearly every other type matches a
    /// disambiguating-but-not-target shape.
    ///
    /// Layout: `Qtarget` has `(u64@40, u32@80)`. The other 499 each
    /// carry only a single `u64@0` — they match neither `(40, 8)` nor
    /// `(80, 4)`, so the intersection collapses to `Qtarget`. Source
    /// `T` has a single u64@8; T's u64@0 is absent, avoiding the
    /// "had_source && others remain" ambiguity drop.
    #[test]
    fn many_struct_types_unique_match_resolves() {
        const N_FILLER: usize = 499;
        let mut strings: Vec<u8> = vec![0];
        let n_u32 = push_name(&mut strings, "u32");
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_qtarget = push_name(&mut strings, "Qtarget");
        let n_filler_a = push_name(&mut strings, "a");
        let n_filler_b = push_name(&mut strings, "b");
        let n_f = push_name(&mut strings, "f");
        let mut filler_name_offs = Vec::with_capacity(N_FILLER);
        for i in 0..N_FILLER {
            filler_name_offs.push(push_name(&mut strings, &format!("Q{i}")));
        }
        let mut types: Vec<SynType> = vec![
            SynType::Int {
                name_off: n_u32,
                size: 4,
                encoding: 0,
                offset: 0,
                bits: 32,
            },
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 16,
                members: vec![SynMember {
                    name_off: n_f,
                    type_id: 2,
                    byte_offset: 8,
                }],
            },
            SynType::Struct {
                name_off: n_qtarget,
                size: 84,
                members: vec![
                    SynMember {
                        name_off: n_filler_a,
                        type_id: 2,
                        byte_offset: 40,
                    },
                    SynMember {
                        name_off: n_filler_b,
                        type_id: 1,
                        byte_offset: 80,
                    },
                ],
            },
        ];
        for &name_off in filler_name_offs.iter().take(N_FILLER) {
            types.push(SynType::Struct {
                name_off,
                size: 8,
                members: vec![SynMember {
                    name_off: n_filler_a,
                    type_id: 2,
                    byte_offset: 0,
                }],
            });
        }
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id: u32 = 3;
        let qtarget_id: u32 = 4;
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            ldx(BPF_SIZE_DW, 3, 2, 40),
            ldx(BPF_SIZE_W, 4, 2, 80),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.len(),
            1,
            "single unique cast across 500 candidates: {map:?}"
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: qtarget_id,
                addr_space: AddrSpace::Arena,
            }),
            "unique match must resolve to Qtarget: {map:?}"
        );
    }

    /// 30-level chain of cycling `Typedef -> Const -> Volatile`
    /// modifiers wrapping a `u64` `Int`. `peel_modifiers` walks 30
    /// peel iterations (well below the `MAX_MODIFIER_DEPTH = 32`
    /// cap) before resolving the underlying type. The struct member
    /// at `T.f` carries this deep chain as its declared type; the
    /// analyzer's cast path must still recognize the field as a
    /// plain `u64` and seed `LoadedU64Field` on the LDX. The
    /// follow-up deref then records `(T, 8) -> (Q, Arena)`.
    #[test]
    fn deep_modifier_chain_resolves_to_u64() {
        const CHAIN_LEN: usize = 30;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q = push_name(&mut strings, "Q");
        let n_f = push_name(&mut strings, "f");
        let n_x = push_name(&mut strings, "x");
        let n_typedef = push_name(&mut strings, "alias_t");
        let mut types: Vec<SynType> = Vec::new();
        types.push(SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        });
        for i in 0..CHAIN_LEN {
            let inner_id = 1 + i as u32;
            let kind = i % 3;
            let chain_node = match kind {
                0 => SynType::Typedef {
                    name_off: n_typedef,
                    type_id: inner_id,
                },
                1 => SynType::Const { type_id: inner_id },
                _ => SynType::Volatile { type_id: inner_id },
            };
            types.push(chain_node);
        }
        let chain_head_id: u32 = (CHAIN_LEN as u32) + 1;
        types.push(SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: chain_head_id,
                byte_offset: 8,
            }],
        });
        types.push(SynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        });
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id: u32 = chain_head_id + 1;
        let q_id: u32 = chain_head_id + 2;
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "30-level modifier chain must peel to u64 and seed cast: {map:?}"
        );
    }

    /// All 64 stack slots filled with distinct typed pointers (each
    /// produced by a kfunc-call return), then all reloaded and stored
    /// into 64 distinct `(P, slot_off)` slots — yielding 64 kernel
    /// kptr findings. Verifies that `stack_slots` (a `BTreeMap`)
    /// handles a fully-loaded BPF stack frame (512 bytes at 8 bytes/slot)
    /// with no slot lost or aliased on reload. Each kfunc returns a
    /// different `T_i*` so the assertion validates that saved register
    /// state per slot is preserved independently.
    #[test]
    fn maximum_stack_slots_all_recorded() {
        const N: usize = 64;
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let n_p = push_name(&mut strings, "P");
        let mut t_names = Vec::with_capacity(N);
        let mut slot_names = Vec::with_capacity(N);
        let mut kfunc_names = Vec::with_capacity(N);
        for i in 0..N {
            t_names.push(push_name(&mut strings, &format!("T{i}")));
            slot_names.push(push_name(&mut strings, &format!("slot{i}")));
            kfunc_names.push(push_name(&mut strings, &format!("kfunc_acquire_{i}")));
        }
        // Type id layout:
        //   1: u64; 2..=N+1: T_i; N+2..=2N+1: T_i*;
        //   2N+2: P (N u64 fields); 2N+3..=3N+2: FuncProtos returning T_i*;
        //   3N+3..=4N+2: Func entries.
        let mut types: Vec<SynType> = Vec::new();
        types.push(SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        });
        for &name_off in t_names.iter().take(N) {
            types.push(SynType::Struct {
                name_off,
                size: 8,
                members: vec![SynMember {
                    name_off: 0,
                    type_id: 1,
                    byte_offset: 0,
                }],
            });
        }
        for i in 0..N {
            types.push(SynType::Ptr {
                type_id: (2 + i) as u32,
            });
        }
        let p_members: Vec<SynMember> = (0..N)
            .map(|i| SynMember {
                name_off: slot_names[i],
                type_id: 1,
                byte_offset: 8 * i as u32,
            })
            .collect();
        types.push(SynType::Struct {
            name_off: n_p,
            size: 8 * N as u32,
            members: p_members,
        });
        for i in 0..N {
            types.push(SynType::FuncProto {
                return_type_id: (N as u32 + 2 + i as u32),
                params: vec![],
            });
        }
        for (i, &name_off) in kfunc_names.iter().enumerate().take(N) {
            types.push(SynType::Func {
                name_off,
                type_id: (2 * N as u32 + 3 + i as u32),
                linkage: 1,
            });
        }
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let p_id: u32 = 2 * N as u32 + 2;
        // r6 is seeded with Pointer{P} via InitialReg and is callee-
        // saved across kfunc CALL (per BPF ABI, R6..R9 are not clobbered).
        let mut insns: Vec<BpfInsn> = Vec::with_capacity(4 * N + 1);
        for i in 0..N {
            let func_id: u32 = 3 * N as u32 + 3 + i as u32;
            insns.push(kfunc_call(func_id));
            insns.push(stx(BPF_SIZE_DW, 10, 0, -((i as i16 + 1) * 8)));
        }
        for i in 0..N {
            insns.push(ldx(BPF_SIZE_DW, 3, 10, -((i as i16 + 1) * 8)));
            insns.push(stx(BPF_SIZE_DW, 6, 3, (8 * i) as i16));
        }
        insns.push(exit());
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: p_id,
            }],
            &[],
            &[],
        );
        assert_eq!(map.len(), N, "expected {N} kptr findings: {map:?}");
        for i in 0..N {
            let t_id: u32 = (2 + i) as u32;
            assert_eq!(
                map.get(&(p_id, 8 * i as u32)),
                Some(&CastHit {
                    target_type_id: t_id,
                    addr_space: AddrSpace::Kernel,
                }),
                "stack slot {i} (off={}) must record T{i}: {map:?}",
                -((i as i16 + 1) * 8)
            );
        }
    }

    /// Source struct `T` with 100 `u64` members; cast pattern triggers
    /// at `f50` (offset 400) and `f99` (offset 792). Each load enters
    /// `LoadedU64Field`, then a follow-up deref records a unique-shape
    /// access against a single matching candidate (`Q50` for `f50`,
    /// `Q99` for `f99`). The dereference offsets/sizes are chosen so
    /// `T`'s u64-at-multiple-of-8 layout matches NEITHER pattern,
    /// avoiding the ambiguity drop. Verifies the matcher scales when
    /// the source struct has many fields.
    #[test]
    fn many_field_struct_records_two_distinct_casts() {
        const N: u32 = 100;
        let mut strings: Vec<u8> = vec![0];
        let n_u8 = push_name(&mut strings, "u8");
        let n_u32 = push_name(&mut strings, "u32");
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_q50 = push_name(&mut strings, "Q50");
        let n_q99 = push_name(&mut strings, "Q99");
        let n_x = push_name(&mut strings, "x");
        let mut t_field_names = Vec::with_capacity(N as usize);
        for i in 0..N {
            t_field_names.push(push_name(&mut strings, &format!("f{i}")));
        }
        let t_members: Vec<SynMember> = (0..N)
            .map(|i| SynMember {
                name_off: t_field_names[i as usize],
                type_id: 3,
                byte_offset: 8 * i,
            })
            .collect();
        // Type ids:
        //   1: u8, 2: u32, 3: u64;
        //   4: T (100 u64 fields at 0, 8, ..., 792);
        //   5: Q50 (single u32@4 — pattern (4, 4) matches only this);
        //   6: Q99 (single u8@5 — pattern (5, 1) matches only this).
        let types = vec![
            SynType::Int {
                name_off: n_u8,
                size: 1,
                encoding: 0,
                offset: 0,
                bits: 8,
            },
            SynType::Int {
                name_off: n_u32,
                size: 4,
                encoding: 0,
                offset: 0,
                bits: 32,
            },
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8 * N,
                members: t_members,
            },
            SynType::Struct {
                name_off: n_q50,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 2,
                    byte_offset: 4,
                }],
            },
            SynType::Struct {
                name_off: n_q99,
                size: 8,
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: 5,
                }],
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id: u32 = 4;
        let q50_id: u32 = 5;
        let q99_id: u32 = 6;
        // f50 at offset 400; f99 at offset 792.
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 400),
            ldx(BPF_SIZE_W, 3, 2, 4),
            ldx(BPF_SIZE_DW, 2, 1, 792),
            ldx(BPF_SIZE_B, 4, 2, 5),
            exit(),
        ];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(map.len(), 2, "two distinct casts expected: {map:?}");
        assert_eq!(
            map.get(&(t_id, 400)),
            Some(&CastHit {
                target_type_id: q50_id,
                addr_space: AddrSpace::Arena,
            }),
            "f50 at offset 400: {map:?}"
        );
        assert_eq!(
            map.get(&(t_id, 792)),
            Some(&CastHit {
                target_type_id: q99_id,
                addr_space: AddrSpace::Arena,
            }),
            "f99 at offset 792: {map:?}"
        );
    }

    /// 20 distinct `(source_struct, field_offset)` cast patterns in a
    /// single program. Source struct `T` has 20 `u64` fields at
    /// offsets `0, 8, ..., 152`. Each `T.f_i` is loaded then
    /// dereferenced at offset `(i+1)` size 1, matching exactly one of
    /// 20 distinct `Q_i` target structs (each with a single `u8` at
    /// the matching offset). Verifies that the analyzer's `patterns`
    /// map and the matcher's per-pattern intersection scale to many
    /// distinct cast emissions in one walk.
    #[test]
    fn many_cast_patterns_in_one_program() {
        const N: u32 = 20;
        let mut strings: Vec<u8> = vec![0];
        let n_u8 = push_name(&mut strings, "u8");
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_x = push_name(&mut strings, "x");
        let mut t_field_names = Vec::with_capacity(N as usize);
        let mut q_names = Vec::with_capacity(N as usize);
        for i in 0..N {
            t_field_names.push(push_name(&mut strings, &format!("f{i}")));
            q_names.push(push_name(&mut strings, &format!("Q{i}")));
        }
        // Type ids:
        //   1: u8, 2: u64; 3: T (N u64 fields);
        //   4..=3+N: Q_i, each with single u8@(i+1).
        let t_members: Vec<SynMember> = (0..N)
            .map(|i| SynMember {
                name_off: t_field_names[i as usize],
                type_id: 2,
                byte_offset: 8 * i,
            })
            .collect();
        let mut types: Vec<SynType> = vec![
            SynType::Int {
                name_off: n_u8,
                size: 1,
                encoding: 0,
                offset: 0,
                bits: 8,
            },
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Struct {
                name_off: n_t,
                size: 8 * N,
                members: t_members,
            },
        ];
        for i in 0..N {
            types.push(SynType::Struct {
                name_off: q_names[i as usize],
                size: i + 2, // u8@(i+1) requires size >= i+2
                members: vec![SynMember {
                    name_off: n_x,
                    type_id: 1,
                    byte_offset: i + 1,
                }],
            });
        }
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let t_id: u32 = 3;
        let mut insns: Vec<BpfInsn> = Vec::with_capacity(2 * N as usize + 1);
        for i in 0..N {
            insns.push(ldx(BPF_SIZE_DW, 2, 1, (8 * i) as i16));
            insns.push(ldx(BPF_SIZE_B, 3, 2, (i + 1) as i16));
        }
        insns.push(exit());
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert_eq!(map.len(), N as usize, "expected {N} cast patterns: {map:?}");
        for i in 0..N {
            let q_id: u32 = 4 + i;
            assert_eq!(
                map.get(&(t_id, 8 * i)),
                Some(&CastHit {
                    target_type_id: q_id,
                    addr_space: AddrSpace::Arena,
                }),
                "pattern #{i} at (T, {}) must resolve to Q{i}: {map:?}",
                8 * i
            );
        }
    }

    /// BTF with no struct types at all (only a single `u64` Int).
    /// `build_layout_index` walks the id space without finding any
    /// struct/union; `finalize` emits no findings. Verifies the
    /// analyzer does not panic on a degenerate BTF that contains no
    /// aggregate types, and that the empty layout index correctly
    /// produces an empty `CastMap` even when the instruction stream
    /// contains LDX patterns. The seed of `r1 = struct_type_id 1` is
    /// silently dropped by `resolve_to_struct_id` (id 1 is `u64`).
    #[test]
    fn empty_btf_no_panic() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_name(&mut strings, "u64");
        let types = vec![SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        }];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: 1,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "no struct types in BTF must produce empty CastMap: {map:?}"
        );
    }

    /// BTF containing only `Int` types and no structs. The seed
    /// targets a non-struct id, so `resolve_to_struct_id` returns
    /// `None` and the seed is dropped. The instruction stream's LDX
    /// cannot type any register, the `patterns` map stays empty, and
    /// `build_layout_index` finds no struct/union to index. Verifies
    /// the analyzer handles a scalar-only BTF without panic.
    #[test]
    fn btf_only_ints_no_panic() {
        let mut strings: Vec<u8> = vec![0];
        let n_u8 = push_name(&mut strings, "u8");
        let n_u16 = push_name(&mut strings, "u16");
        let n_u32 = push_name(&mut strings, "u32");
        let n_u64 = push_name(&mut strings, "u64");
        let n_s32 = push_name(&mut strings, "s32");
        // BTF int encoding bit `BTF_INT_SIGNED` per linux uapi `btf.h`.
        const BTF_INT_SIGNED: u32 = 1;
        let types = vec![
            SynType::Int {
                name_off: n_u8,
                size: 1,
                encoding: 0,
                offset: 0,
                bits: 8,
            },
            SynType::Int {
                name_off: n_u16,
                size: 2,
                encoding: 0,
                offset: 0,
                bits: 16,
            },
            SynType::Int {
                name_off: n_u32,
                size: 4,
                encoding: 0,
                offset: 0,
                bits: 32,
            },
            SynType::Int {
                name_off: n_u64,
                size: 8,
                encoding: 0,
                offset: 0,
                bits: 64,
            },
            SynType::Int {
                name_off: n_s32,
                size: 4,
                encoding: BTF_INT_SIGNED,
                offset: 0,
                bits: 32,
            },
        ];
        let blob = build_btf(&types, &strings);
        let btf = Btf::from_bytes(&blob).unwrap();
        // Seed targets the u64 id — `resolve_to_struct_id` walks
        // Int as terminal and returns None. The seed silently drops.
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: 4,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "Int-only BTF must produce empty CastMap: {map:?}"
        );
    }

    /// Long stream of `BPF_LD_IMM64` two-slot instructions back-to-
    /// back, terminated by `exit`. Every `lo` slot sets `skip_next`,
    /// and every `hi` slot is the upper-immediate placeholder that
    /// the analyzer must not interpret. Verifies the
    /// `skip_next`-driven decode path does not run off the end of the
    /// slice or misaccount its position when the program is densely
    /// packed with two-slot ops. Also exercises the same pattern in
    /// `jump_targets`'s pre-pass — both must agree on which slots are
    /// second-half placeholders.
    #[test]
    fn only_ld_imm64_no_oob() {
        const N_PAIRS: usize = 50;
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let mut insns: Vec<BpfInsn> = Vec::with_capacity(2 * N_PAIRS + 1);
        let lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, 2, 0, 0, 0);
        let hi = mk_insn(0, 0, 0, 0, 0);
        for _ in 0..N_PAIRS {
            insns.push(lo);
            insns.push(hi);
        }
        insns.push(exit());
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
            &[],
        );
        assert!(
            map.is_empty(),
            "all-LD_IMM64 stream must produce no findings, no OOB panic: {map:?}"
        );
    }
}
