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
//! Function entry seeding (via [`FuncEntry`]) reseeds R1..Rn with the
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
//! - [`CastMap`]: ordered map from `(source_btf_type_id,
//!   field_byte_offset)` to `(target_btf_type_id, AddrSpace)`.
//! - [`InitialReg`]: caller-supplied seed register state for entry
//!   parameters / known typed values returned from helpers.
//! - [`FuncEntry`]: function-entry PC + BTF FuncProto id for
//!   automatic R1..Rn seeding from the proto's parameters.
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
    /// to read; the field is `pub` only for raw construction.
    pub regs: u8,
    /// Signed 16-bit offset (PC-relative for jumps, byte offset for
    /// mem ops, atomic-op subselect for `BPF_MODE_ATOMIC`).
    pub off: i16,
    /// Signed 32-bit immediate (constant operand, or — for
    /// `BPF_PSEUDO_KFUNC_CALL` — the BTF id of the kfunc).
    pub imm: i32,
}

#[allow(dead_code)]
impl BpfInsn {
    /// Construct an instruction with explicit fields. `dst` and `src`
    /// are 0..=15 register indices (the analyzer rejects 11..=15 at
    /// decode time per `step()`).
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
}

/// Caller-supplied initial state for one BPF register.
///
/// Used to seed entry-parameter typing or the typed return value of
/// a kfunc. Empty seed lists yield no findings — the analysis only
/// produces output along chains rooted in registers it knows are
/// typed pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialReg {
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
/// R1..Rn from the FuncProto's parameter list:
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
pub struct FuncEntry {
    /// Instruction index of the function's first instruction.
    pub insn_offset: usize,
    /// BTF id of the function's prototype (`BTF_KIND_FUNC_PROTO`).
    pub func_proto_id: u32,
}

/// Address space of a recovered cast target.
///
/// Distinguishes the two detection paths: arena pointers carry an
/// arena virtual address; kernel kptrs carry a kernel virtual
/// address (slab / vmalloc / per-cpu). Both share the same
/// `(source, offset) -> target` shape.
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

/// Output of [`analyze_casts`].
///
/// Maps `(source_btf_type_id, field_byte_offset)` to the recovered
/// target struct's BTF type id paired with its [`AddrSpace`]. The
/// map is `BTreeMap` so iteration order is deterministic, which
/// makes test assertions stable without a sort step at every
/// assertion site.
pub type CastMap = BTreeMap<(u32, u32), (u32, AddrSpace)>;

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
/// instruction at that PC executes.
///
/// `initial_regs` and `func_entries` compose: seeds apply once at
/// PC 0, function-entry reseeding applies at every matching
/// `insn_offset`. Reseeding clears ALL registers (R0..R10) and
/// drops every stack slot (subprog entry semantics: the callee's
/// frame is fresh, and stale R6..R9 from linearly-preceding
/// unrelated functions must not leak). R1..R5 are then re-seeded
/// from the FuncProto's parameter types where they resolve to
/// struct pointers.
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
) -> CastMap {
    let mut analyzer = Analyzer::new(btf);
    analyzer.seed(initial_regs);
    let targets = jump_targets(insns);
    analyzer.run(insns, &targets, func_entries);
    analyzer.finalize()
}

struct Analyzer<'a> {
    btf: &'a Btf,
    regs: [RegState; 11],
    /// Per `(source_struct, field_offset)` set of `(target_offset,
    /// target_size)` accesses observed via the arena LDX path.
    patterns: BTreeMap<(u32, u32), BTreeSet<Access>>,
    /// Direct kptr findings keyed by `(parent_struct_id,
    /// field_offset)`. Populated by the STX path when a
    /// `Pointer{T}` source is stored into a `u64` field. The value
    /// is the inner struct id `T`. Conflicting writes (same slot,
    /// different `T`) collapse to a sentinel that finalize() drops
    /// — ambiguity is a false negative, never a false positive.
    kptr_findings: BTreeMap<(u32, u32), KptrEntry>,
    /// Stack-slot map keyed by frame-pointer-relative byte offset
    /// (always negative). STX through r10 saves the source register
    /// state; LDX through r10 restores it. Cleared at every
    /// jump-target PC alongside the register file.
    stack_slots: BTreeMap<i16, RegState>,
    /// Fields confirmed as arena pointers by a `BPF_ADDR_SPACE_CAST`
    /// instruction (code=0xBF, off=1, imm=1). Keyed by
    /// `(source_struct_id, field_byte_offset)`. When present,
    /// `finalize()` emits an `AddrSpace::Arena` entry even if the
    /// access-pattern intersection didn't uniquely resolve a target
    /// — the cast instruction is authoritative evidence that the
    /// field holds an arena pointer.
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
        // Pre-clear ALL registers unconditionally. Even if proto resolution
        // fails below, stale typed-pointer state from a prior function
        // must not survive past this entry — the new function's R1..R5
        // are defined by the FuncProto, not by whatever happened to be
        // in those registers when the previous function exited.
        // Clear ALL registers. The linear forward walk concatenates
        // subprogram instruction streams; a function entry reached
        // by fall-through from a prior EXIT inherits stale R6..R9
        // state from an unrelated function. BPF ABI says R6..R9 are
        // callee-saved (inherited from the CALLER), but in the linear
        // walk there is no real caller — the prior function is
        // unrelated. Preserving R6..R9 would let stale typed pointers
        // leak across function boundaries, risking false positives.
        // The cost: legitimate call-inherited R6..R9 typing is lost
        // (false negative), which is the safe direction.
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
        // Cap at R5 — BPF ABI passes args 1..5 in registers. R0 is
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
    ) {
        // BPF_LD_IMM64 is a two-insn pseudo-instruction. The decoder
        // skips its second slot via this flag.
        let mut skip_next = false;

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
            for fe in func_entries {
                if fe.insn_offset == pc {
                    self.seed_from_func_proto(fe.func_proto_id);
                }
            }

            self.step(*insn, &mut skip_next);
        }
    }

    fn step(&mut self, insn: BpfInsn, skip_next: &mut bool) {
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
                // first slot's `imm` may be a map fd / btf id /
                // pseudo-value; the destination receives a 64-bit
                // immediate, never a typed kernel pointer the
                // renderer needs to chase. Mark dst Unknown and
                // tell the caller to skip slot 2.
                if insn.code == (BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM) {
                    self.set_reg(dst, RegState::Unknown);
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

        match self.regs[src] {
            RegState::Pointer { struct_type_id } => {
                // r_dst = *(size *) (r_src_struct_pointer + off).
                // Look up the field at that byte offset in the
                // source struct via BTF.
                let field_off = match field_byte_offset(off) {
                    Some(o) => o,
                    None => {
                        self.set_reg(dst, RegState::Unknown);
                        return;
                    }
                };

                if let Some(member) = struct_member_at(self.btf, struct_type_id, field_off) {
                    let member_type_id = match member.get_type_id() {
                        Ok(id) => id,
                        Err(_) => {
                            self.set_reg(dst, RegState::Unknown);
                            return;
                        }
                    };
                    let resolved = super::btf_render::peel_modifiers(self.btf, member_type_id);
                    match (size_bytes, resolved) {
                        // Ptr field directly -- BTF already typed.
                        // Mark dst as a typed pointer so chained
                        // accesses can be tracked; the renderer
                        // already handles this case so no cast
                        // needs to be recorded.
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
                        // Plain u64 field -- THIS is the cast
                        // target. Record source for the matching
                        // pass and remember dst as "loaded u64".
                        (Some(8), Some(Type::Int(int))) => {
                            if int.size() == 8
                                && !int.is_signed()
                                && !int.is_bool()
                                && !int.is_char()
                            {
                                self.set_reg(
                                    dst,
                                    RegState::LoadedU64Field {
                                        source_struct_id: struct_type_id,
                                        field_offset: field_off,
                                    },
                                );
                                self.note_type_id(struct_type_id);
                                self.patterns
                                    .entry((struct_type_id, field_off))
                                    .or_default();
                            } else {
                                self.set_reg(dst, RegState::Unknown);
                            }
                        }
                        _ => {
                            // Other field shapes (sub-u64 ints,
                            // structs, unions, enums, arrays,
                            // floats, FuncProto) cannot become a
                            // pointer by load alone. Drop dst.
                            self.set_reg(dst, RegState::Unknown);
                        }
                    }
                } else {
                    self.set_reg(dst, RegState::Unknown);
                }
            }
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
        }
    }

    /// `STX [r_dst_base + off] = r_src_value`.
    ///
    /// Two roles:
    /// 1. Stack spill — when the destination base is r10, save the
    ///    source register's full RegState into `stack_slots[off]`.
    ///    Sub-DW stores invalidate the slot (truncating writes
    ///    cannot preserve a 64-bit pointer); BPF_DW writes through
    ///    r10 with non-negative `off` are out-of-spec, drop them.
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
        let RegState::Pointer {
            struct_type_id: parent_struct_id,
        } = self.regs[dst]
        else {
            return;
        };
        let RegState::Pointer {
            struct_type_id: target_struct_id,
        } = self.regs[src]
        else {
            return;
        };
        let Some(field_off) = field_byte_offset(off as i32) else {
            return;
        };
        // BTF gate: the destination field at this offset must be a
        // plain `u64`. A typed Ptr field is the BTF-already-typed
        // case the renderer handles natively; recording a kptr
        // there would duplicate work. A non-u64 field (sub-u64
        // int, struct, array) is not a pointer slot at all — the
        // store is undefined behavior we drop conservatively.
        let Some(member) = struct_member_at(self.btf, parent_struct_id, field_off) else {
            return;
        };
        let Ok(member_type_id) = member.get_type_id() else {
            return;
        };
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
        // positive bar high.
        if parent_struct_id == target_struct_id {
            return;
        }
        self.note_type_id(parent_struct_id);
        self.note_type_id(target_struct_id);
        let key = (parent_struct_id, field_off);
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
    /// linux uapi `bpf.h`). Two of them write a register with the
    /// PRIOR memory value:
    /// - `BPF_XCHG = 0xe0 | BPF_FETCH`: `src_reg = atomic_xchg(...)`.
    ///   `src_reg` is overwritten with the old memory value, which
    ///   the analyzer cannot type from the source register's prior
    ///   RegState — drop it to Unknown.
    /// - `BPF_CMPXCHG = 0xf0 | BPF_FETCH`: `r0 = atomic_cmpxchg(...)`.
    ///   R0 is overwritten with the old memory value — drop R0.
    ///
    /// Other atomic ops (BPF_ADD/AND/OR/XOR with optional BPF_FETCH)
    /// either leave registers untouched (no FETCH) or write src_reg
    /// with the prior arithmetic value (FETCH variants). The
    /// arithmetic-fetch variants also cannot produce a typed
    /// pointer the analyzer can model — clobber src on any FETCH
    /// imm that is not the two named above.
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

        // Stack-slot invalidation: an atomic store through r10 (any
        // imm encoding) overwrites the slot's prior content. Drop
        // before per-register logic so all atomic flavours invalidate
        // — XCHG, CMPXCHG, LOAD_ACQ-via-r10 (rare but possible),
        // STORE_REL through r10, and arithmetic atomics.
        if dst == BPF_REG_R10 {
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
        if has_fetch {
            self.set_reg(src, RegState::Unknown);
            return;
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
            // observed (offset, size).
            let mut iter = accesses.iter();
            let first = iter.next().expect("non-empty checked above");
            let mut candidates: BTreeSet<u32> = layout
                .get(&(first.offset, first.size))
                .cloned()
                .unwrap_or_default();
            for acc in iter {
                let next: BTreeSet<u32> = layout
                    .get(&(acc.offset, acc.size))
                    .cloned()
                    .unwrap_or_default();
                candidates = candidates.intersection(&next).copied().collect();
                if candidates.is_empty() {
                    break;
                }
            }

            // Drop the source struct from candidates — a self-typed
            // cast (source.f → source*) matches tautologically and
            // carries no disambiguating evidence. Only emit when a
            // single NON-source candidate remains AND the original
            // set didn't include the source alongside it (ambiguous
            // {source, X} sets drop entirely to avoid the false
            // positive where true target was source but X gets
            // emitted). This is stricter than "remove source, emit
            // remainder" — it prevents the asymmetric false positive
            // the kptr STX path's self-rejection (line ~1008) avoids
            // on its side.
            let had_source = candidates.remove(source);
            if had_source && !candidates.is_empty() {
                // Original set was {source, ...others} — ambiguous.
                // Drop rather than guess.
                continue;
            }

            if candidates.len() == 1 {
                let target = *candidates.iter().next().unwrap();
                out.insert((*source, *field_off), (target, AddrSpace::Arena));
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
            out.insert(key, (target, AddrSpace::Kernel));
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

/// Find the member of a struct/union at the given byte offset, if
/// one exists at exactly that offset and is not a bitfield. Returns
/// the matching `btf_rs::Member`.
fn struct_member_at(btf: &Btf, struct_type_id: u32, byte_offset: u32) -> Option<btf_rs::Member> {
    let t = btf.resolve_type_by_id(struct_type_id).ok()?;
    let s = match t {
        Type::Struct(s) | Type::Union(s) => s,
        _ => return None,
    };
    for m in &s.members {
        if matches!(m.bitfield_size(), Some(s) if s > 0) {
            continue;
        }
        let bit_off = m.bit_offset();
        if bit_off % 8 != 0 {
            continue;
        }
        if bit_off / 8 == byte_offset {
            return Some(m.clone());
        }
    }
    None
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
const BPF_PSEUDO_KFUNC_CALL: u8 = bs::BPF_PSEUDO_KFUNC_CALL as u8;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
    const BTF_KIND_FUNC: u32 = 12;
    const BTF_KIND_FUNC_PROTO: u32 = 13;

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

    /// One synthetic BTF type. Tests build a Vec<SynType>; the
    /// writer assigns ids starting at 1 (id 0 is Void) and emits
    /// the type section in order.
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        let map = analyze_casts(&[], &btf, &[], &[]);
        assert!(map.is_empty());
    }

    #[test]
    fn no_initial_seed_yields_empty_map() {
        // Without seeding any register as a struct pointer, the
        // analyzer cannot identify the source type of an LDX.
        let (blob, _t, _q) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
        let map = analyze_casts(&insns, &btf, &[], &[]);
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
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&(q_id, AddrSpace::Arena)),
            "got: {map:?}"
        );
    }

    #[test]
    fn ambiguous_targets_drop_silently() {
        // Build BTF with two structs having a u64 at offset 0
        // (both Q1 and Q2 match the access pattern). Cast must NOT
        // be recorded because false positives are unacceptable.
        let mut strings: Vec<u8> = vec![0];
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&(q1_id, AddrSpace::Arena)),
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&(q1_id, AddrSpace::Arena)),
            "f1: {map:?}"
        );
        assert_eq!(
            map.get(&(t_id, 16)),
            Some(&(q2_id, AddrSpace::Arena)),
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert!(
            map.is_empty(),
            "typed Ptr field must not be recorded as cast: {map:?}"
        );
    }

    #[test]
    fn null_check_fall_through_preserves_state() {
        // if r2 == 0 goto SKIP; deref r2; SKIP: exit.
        // The deref happens at the FALL-THROUGH after the
        // conditional jump, so the state survives. The analyzer
        // should still record the cast.
        let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // pc 0: r2 = T.f
        // pc 1: if r2 == 0 goto +1 (jump to pc=3, skip the deref)
        // pc 2: r3 = *r2  (fall-through; r2 still LoadedU64Field)
        // pc 3: exit.
        let jeq = mk_insn(BPF_CLASS_JMP | 0x10, 2, 0, 1, 0); // BPF_JEQ_K = 0x10
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            jeq,
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
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&(q_id, AddrSpace::Arena)),
            "fall-through deref must record: {map:?}"
        );
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
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&(q_id, AddrSpace::Arena)),
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
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&(q_id, AddrSpace::Arena)),
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
        );
        assert!(map.is_empty(), "r10 seed must be ignored: {map:?}");
    }

    #[test]
    fn nonu64_field_at_source_offset_not_tracked() {
        // T has a u32 at offset 8 (not u64). Loading from there
        // and treating as a pointer is meaningless — the analyzer
        // must not seed LoadedU64Field.
        let mut strings: Vec<u8> = vec![0];
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&(t_id, AddrSpace::Kernel)),
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
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&(t_id, AddrSpace::Kernel)),
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&(t_id, AddrSpace::Kernel)),
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert_eq!(
            map.get(&(m_id, 0)),
            Some(&(a_id, AddrSpace::Arena)),
            "arena cast missing: {map:?}"
        );
        assert_eq!(
            map.get(&(m_id, 16)),
            Some(&(t_id, AddrSpace::Kernel)),
            "kernel kptr missing: {map:?}"
        );
    }

    #[test]
    fn func_entry_seeding_from_btf() {
        // FuncProto with two parameters: param 0 = T* (typed
        // source), param 1 = P* (parent base). FuncEntry seeds
        // R1 = Pointer{T} and R2 = Pointer{P}. The single
        // instruction stores R1 into R2's u64 slot at slot_off,
        // recording (P, slot_off) -> (T, Kernel). Both base and
        // value registers come from the FuncProto seeding because
        // the entry-PC reseed clears all registers before
        // populating R1..R5 — InitialReg state set before the run
        // does not survive into a function entry at PC 0.
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
        let n_u64 = push_name(&mut strings, "u64");
        let n_t = push_name(&mut strings, "T");
        let n_p = push_name(&mut strings, "P");
        let n_x = push_name(&mut strings, "x");
        let n_slot = push_name(&mut strings, "slot");
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
                size: slot_off + 8,
                members: vec![SynMember {
                    name_off: n_slot,
                    type_id: 1,
                    byte_offset: slot_off,
                }],
            },
            SynType::Ptr { type_id: 4 }, // id 5: P*
            // id 6: FuncProto(T*, P*) -> void
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
        // STX *(R2 + slot_off) = R1: R2 holds Pointer{P} (from
        // FuncProto param 1), R1 holds Pointer{T} (from FuncProto
        // param 0). Records the kptr finding.
        let insns = vec![stx(BPF_SIZE_DW, 2, 1, slot_off as i16), exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[],
            &[FuncEntry {
                insn_offset: 0,
                func_proto_id: proto_id,
            }],
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&(t_id, AddrSpace::Kernel)),
            "FuncEntry param seeding must populate R1 and R2: {map:?}"
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
    #[test]
    fn addr_space_cast_arena_alone_does_not_emit() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r3 = *(u64 *)(r1 + 8)         ; r3 = LoadedU64Field{T, 8}
        // r4 = (cast as(1) -> as(0)) r3 ; arena_confirmed += (T, 8)
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 1);
        let insns = vec![ldx(BPF_SIZE_DW, 3, 1, 8), cast, exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 1,
                struct_type_id: t_id,
            }],
            &[],
        );
        // arena_confirmed without shape inference does NOT emit a
        // standalone entry (target_type_id=0 produces unusable
        // renders). It participates in conflict detection only.
        // Without a subsequent deref pattern that uniquely resolves
        // the target, the map stays empty.
        assert!(
            map.is_empty(),
            "arena_confirmed alone (no deref pattern) must not emit: {map:?}"
        );
    }

    /// `BPF_ADDR_SPACE_CAST` kernel -> arena (`imm == 0x10000`) drops
    /// the destination register state. Per kernel `verifier.c
    /// check_alu_op` the result is a 32-bit arena address, not a
    /// kernel pointer the analyzer can track. A subsequent LDX
    /// through the cast result must NOT record any access pattern,
    /// so no entry appears in the output map.
    #[test]
    fn addr_space_cast_kernel_to_arena_drops_dst() {
        let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
        let btf = Btf::from_bytes(&blob).unwrap();
        // r3 = *(u64 *)(r1 + 8)            ; r3 = LoadedU64Field{T, 8}
        // r4 = (cast as(0) -> as(1)) r3    ; r4 = Unknown
        // r5 = *(u64 *)(r4 + 0)            ; r4 Unknown -> no record
        let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 0x10000);
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
        );
        assert!(
            map.is_empty(),
            "kernel -> arena cast must drop dst, no record: {map:?}"
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
    /// `u64` slot still records the kptr finding.
    #[test]
    fn atomic_non_fetch_preserves_regs() {
        let slot_off: u32 = 16;
        let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
        let btf = Btf::from_bytes(&blob).unwrap();
        // BPF_ADD without BPF_FETCH: imm = 0x00. R1 stays
        // Pointer{T}; STX R1 into P.slot records normally.
        let insns = vec![
            atomic_stx(2, 1, 0, 0x00),
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
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&(t_id, AddrSpace::Kernel)),
            "non-fetch ATOMIC must preserve src register: {map:?}"
        );
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
        );
        assert!(
            map.is_empty(),
            "ATOMIC on stack slot must invalidate, reload yields Unknown: {map:?}"
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&(t2_id, AddrSpace::Kernel)),
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
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&(t_id, AddrSpace::Kernel)),
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
        );
        assert!(
            map.is_empty(),
            "sub-DW store must invalidate slot, reload Unknown: {map:?}"
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
    #[test]
    fn kptr_conflict_two_targets_drops() {
        // BTF: u64(1), T1(2, u64@0), T2(3, u64@0), P(4, u64@slot_off).
        // Seed R1=Pointer{T1}, R2=Pointer{T2}, R6=Pointer{P}.
        // STX R1 into P.slot, then STX R2 into P.slot. Conflict.
        let slot_off: u32 = 16;
        let mut strings: Vec<u8> = vec![0];
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        let insns = vec![
            stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
            stx(BPF_SIZE_DW, 6, 2, slot_off as i16),
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
        );
        assert!(
            map.is_empty(),
            "two distinct kptr targets on same slot must collapse to Conflicting and drop: {map:?}"
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert_eq!(
            map.get(&(p_id, slot_off1)),
            Some(&(t_id, AddrSpace::Kernel)),
            "non-variadic params must seed R1 and R2: {map:?}"
        );
        assert!(
            map.get(&(p_id, slot_off2)).is_none(),
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
        let push_name = |s: &mut Vec<u8>, name: &str| {
            let off = s.len() as u32;
            s.extend_from_slice(name.as_bytes());
            s.push(0);
            off
        };
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
        );
        assert!(
            map.is_empty(),
            "BPF_PROBE_MEM load must mark dst Unknown: {map:?}"
        );
    }
}
