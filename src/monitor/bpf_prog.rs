//! Host-side BPF program enumeration via guest physical memory.
//!
//! Walks the kernel's `prog_idr` xarray from the host to discover
//! loaded BPF programs and read verifier stats from `bpf_prog_aux`.
//! No guest cooperation is needed — all reads go through the guest
//! physical memory mapping.

use super::btf_offsets::BpfProgOffsets;
use super::idr::{translate_any_kva, xa_load};
use super::reader::GuestMem;
use super::symbols::text_kva_to_pa;

/// BPF_PROG_TYPE_STRUCT_OPS from include/uapi/linux/bpf.h.
const BPF_PROG_TYPE_STRUCT_OPS: u32 = 27;

/// BPF_OBJ_NAME_LEN from include/linux/bpf.h.
const BPF_OBJ_NAME_LEN: usize = 16;

/// Per-program BPF verifier statistics collected from the host.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgVerifierStats {
    /// Program name as registered with the kernel.
    pub name: String,
    /// Instructions accepted by the verifier (from
    /// `bpf_prog_aux->verified_insns`).
    pub verified_insns: u32,
}

/// Enumerate struct_ops BPF programs from the kernel's `prog_idr`.
///
/// Reads `prog_idr` from guest memory, walks the xarray, and for
/// each `bpf_prog` with `type == BPF_PROG_TYPE_STRUCT_OPS`, reads
/// `aux->verified_insns` and `aux->name`.
pub(crate) fn find_struct_ops_progs(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    l5: bool,
) -> Vec<ProgVerifierStats> {
    let idr_pa = text_kva_to_pa(prog_idr_kva);

    let xa_head = mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    let idr_next = mem.read_u32(idr_pa, offsets.idr_next);

    let mut progs = Vec::new();

    for id in 0..idr_next {
        let Some(entry) = xa_load(
            mem,
            page_offset,
            xa_head,
            id as u64,
            offsets.xa_node_slots,
            offsets.xa_node_shift,
        ) else {
            continue;
        };
        if entry == 0 {
            continue;
        }

        // bpf_prog is SLAB-allocated or vmalloc'd.
        let Some(prog_pa) = translate_any_kva(mem, cr3_pa, page_offset, entry, l5) else {
            continue;
        };

        let prog_type = mem.read_u32(prog_pa, offsets.prog_type);
        if prog_type != BPF_PROG_TYPE_STRUCT_OPS {
            continue;
        }

        let aux_kva = mem.read_u64(prog_pa, offsets.prog_aux);
        if aux_kva == 0 {
            continue;
        }

        // bpf_prog_aux is kmalloc'd (SLAB, direct mapping).
        let Some(aux_pa) = translate_any_kva(mem, cr3_pa, page_offset, aux_kva, l5) else {
            continue;
        };

        let verified_insns = mem.read_u32(aux_pa, offsets.aux_verified_insns);

        let mut name_buf = [0u8; BPF_OBJ_NAME_LEN];
        mem.read_bytes(aux_pa + offsets.aux_name as u64, &mut name_buf);
        let name_len = name_buf
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(BPF_OBJ_NAME_LEN);
        let name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();

        progs.push(ProgVerifierStats {
            name,
            verified_insns,
        });
    }

    progs
}

/// Per-program runtime stats summed across all CPUs.
///
/// Mirrors the kernel's `struct bpf_prog_stats` (include/linux/filter.h):
/// `cnt` (invocations), `nsecs` (cumulative runtime), `misses` (recursion
/// re-entries skipped via `bpf_prog_inc_misses_counter`,
/// kernel/bpf/syscall.c). All three counters are u64 monotonics summed
/// across the program's per-CPU `bpf_prog_stats` slots.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProgRuntimeStats {
    /// Program name as registered with the kernel.
    pub name: String,
    /// Total invocation count across all CPUs.
    pub cnt: u64,
    /// Total CPU time in nanoseconds across all CPUs.
    pub nsecs: u64,
    /// Total recursion misses across all CPUs. A miss is a re-entry
    /// attempt blocked by the program's per-CPU recursion guard.
    pub misses: u64,
}

impl std::fmt::Display for ProgRuntimeStats {
    /// One-line summary used by [`super::dump::FailureDumpReport`]'s
    /// human-readable rendering: name + the three counter sums.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: cnt={} nsecs={} misses={}",
            self.name, self.cnt, self.nsecs, self.misses
        )
    }
}

/// Cached per-program info for repeated stats reads in the monitor loop.
/// Pre-resolved at startup to avoid IDR walks each cycle.
#[derive(Debug, Clone)]
pub struct CachedProgInfo {
    /// Program name from bpf_prog_aux.
    pub name: String,
    /// Per-CPU `bpf_prog_stats` KVA (the __percpu base pointer).
    pub stats_percpu_kva: u64,
}

/// Enumerate struct_ops programs and cache their stats pointers.
///
/// Walks `prog_idr` once. For each struct_ops program reads
/// `bpf_prog->stats` (percpu pointer) and `bpf_prog_aux->name` via
/// the aux pointer on `bpf_prog`. Returns cached info for use by
/// `read_prog_runtime_stats` in the monitor loop.
pub(crate) fn discover_struct_ops_stats(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    l5: bool,
) -> Vec<CachedProgInfo> {
    let idr_pa = text_kva_to_pa(prog_idr_kva);

    let xa_head = mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    let idr_next = mem.read_u32(idr_pa, offsets.idr_next);

    let mut cached = Vec::new();

    for id in 0..idr_next {
        let Some(entry) = xa_load(
            mem,
            page_offset,
            xa_head,
            id as u64,
            offsets.xa_node_slots,
            offsets.xa_node_shift,
        ) else {
            continue;
        };
        if entry == 0 {
            continue;
        }

        let Some(prog_pa) = translate_any_kva(mem, cr3_pa, page_offset, entry, l5) else {
            continue;
        };

        let prog_type = mem.read_u32(prog_pa, offsets.prog_type);
        if prog_type != BPF_PROG_TYPE_STRUCT_OPS {
            continue;
        }

        let aux_kva = mem.read_u64(prog_pa, offsets.prog_aux);
        if aux_kva == 0 {
            continue;
        }

        let Some(aux_pa) = translate_any_kva(mem, cr3_pa, page_offset, aux_kva, l5) else {
            continue;
        };

        let mut name_buf = [0u8; BPF_OBJ_NAME_LEN];
        mem.read_bytes(aux_pa + offsets.aux_name as u64, &mut name_buf);
        let name_len = name_buf
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(BPF_OBJ_NAME_LEN);
        let name = String::from_utf8_lossy(&name_buf[..name_len]).to_string();

        let stats_percpu_kva = mem.read_u64(prog_pa, offsets.prog_stats);
        if stats_percpu_kva == 0 {
            continue;
        }

        cached.push(CachedProgInfo {
            name,
            stats_percpu_kva,
        });
    }

    cached
}

/// Read per-CPU runtime stats for a set of cached programs.
///
/// For each program, reads `cnt`, `nsecs`, and `misses` from each
/// CPU's `bpf_prog_stats` and sums across CPUs. Uses pre-resolved
/// `__per_cpu_offset` array for address resolution.
///
/// Address translation uses [`translate_any_kva`], which tries the
/// direct mapping first and falls through to a page-table walk when
/// the percpu KVA lives in vmalloc'd memory (large dynamic per-CPU
/// allocations served by `pcpu_get_vm_areas`). The pre-fix path
/// assumed `bpf_prog_stats` always lived in the direct mapping —
/// which is true for static percpu and small kmalloc'd percpu, but
/// not for vmalloc-backed percpu — and silently dropped per-CPU
/// readings whose PA fell outside the direct mapping.
pub(crate) fn read_prog_runtime_stats(
    mem: &GuestMem,
    cached: &[CachedProgInfo],
    per_cpu_offsets: &[u64],
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    offsets: &BpfProgOffsets,
) -> Vec<ProgRuntimeStats> {
    cached
        .iter()
        .map(|prog| {
            let mut cnt: u64 = 0;
            let mut nsecs: u64 = 0;
            let mut misses: u64 = 0;
            for &cpu_off in per_cpu_offsets {
                let stats_kva = prog.stats_percpu_kva.wrapping_add(cpu_off);
                if let Some(stats_pa) = translate_any_kva(mem, cr3_pa, page_offset, stats_kva, l5)
                    && stats_pa < mem.size()
                {
                    // saturating_add: per-CPU `bpf_prog_stats.{cnt,
                    // nsecs, misses}` are kernel-side u64 counters
                    // that monotonically increase on every program
                    // execution. Summing N CPUs' values can in
                    // principle overflow on a long-running guest with
                    // a hot BPF program; observed in nextest runs
                    // where uninitialized / scrambled per-CPU pages
                    // yield near-u64::MAX values. Saturating to
                    // u64::MAX is the right semantics — the consumer
                    // (`ProgRuntimeStats` viewer) never produces
                    // signed deltas off this so a saturated sum still
                    // sorts correctly, and it prevents an `attempt to
                    // add with overflow` panic in the monitor thread
                    // that would tear the whole VM down.
                    cnt = cnt.saturating_add(mem.read_u64(stats_pa, offsets.stats_cnt));
                    nsecs = nsecs.saturating_add(mem.read_u64(stats_pa, offsets.stats_nsecs));
                    misses = misses.saturating_add(mem.read_u64(stats_pa, offsets.stats_misses));
                }
            }
            ProgRuntimeStats {
                name: prog.name.clone(),
                cnt,
                nsecs,
                misses,
            }
        })
        .collect()
}

/// Host-side BPF program accessor for a running guest VM.
pub struct BpfProgAccessor<'a> {
    kernel: &'a super::guest::GuestKernel<'a>,
    prog_idr_kva: u64,
    /// Borrowed from the caller. Mirrors the `BpfMapAccessor` pattern:
    /// `BpfProgOffsets` is a ~160-byte POD built once from the
    /// vmlinux BTF, and every hot-path method reads it by reference,
    /// so owning it in the accessor would charge a clone that serves
    /// no purpose.
    offsets: &'a BpfProgOffsets,
}

impl<'a> BpfProgAccessor<'a> {
    /// Create from an existing [`GuestKernel`](super::guest::GuestKernel)
    /// and a caller-owned [`BpfProgOffsets`]. The accessor borrows both
    /// for its lifetime — build `offsets` once via
    /// [`BpfProgOffsets::from_vmlinux`] and reuse across calls.
    pub fn from_guest_kernel(
        kernel: &'a super::guest::GuestKernel<'a>,
        offsets: &'a BpfProgOffsets,
    ) -> anyhow::Result<Self> {
        let prog_idr_kva = kernel
            .symbol_kva("prog_idr")
            .ok_or_else(|| anyhow::anyhow!("prog_idr symbol not found in vmlinux"))?;

        Ok(Self {
            kernel,
            prog_idr_kva,
            offsets,
        })
    }

    /// Enumerate struct_ops BPF programs and collect verifier stats.
    pub fn struct_ops_progs(&self) -> Vec<ProgVerifierStats> {
        find_struct_ops_progs(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.prog_idr_kva,
            self.offsets,
            self.kernel.l5(),
        )
    }

    /// Snapshot per-program runtime stats (`cnt`, `nsecs`, `misses`)
    /// summed across all CPUs.
    ///
    /// One-shot helper for dump-time capture: walks `prog_idr` to
    /// resolve every struct_ops program's per-CPU `bpf_prog_stats`
    /// pointer, then reads each CPU slot via the supplied
    /// `per_cpu_offsets` array (typically obtained from
    /// [`super::symbols::read_per_cpu_offsets`]). Returns an empty
    /// vector when the kernel exposes no struct_ops programs.
    ///
    /// Mirrors the kernel-side per-CPU accumulation: `cnt` is
    /// bumped via `u64_stats_inc` and `nsecs` is bumped via
    /// `u64_stats_add(&stats->nsecs, duration)` inside
    /// `__bpf_prog_run` (include/linux/filter.h), invoked through
    /// the JIT-emitted entry path on every program invocation.
    /// `misses` is bumped by `bpf_prog_inc_misses_counter`
    /// (defined in `kernel/bpf/syscall.c`) called from
    /// `kernel/bpf/trampoline.c::__bpf_prog_enter_recur` when a
    /// program re-enters and the recursion guard rejects it.
    pub fn runtime_stats(&self, per_cpu_offsets: &[u64]) -> Vec<ProgRuntimeStats> {
        let cached = discover_struct_ops_stats(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.prog_idr_kva,
            self.offsets,
            self.kernel.l5(),
        );
        if cached.is_empty() {
            return Vec::new();
        }
        read_prog_runtime_stats(
            self.kernel.mem(),
            &cached,
            per_cpu_offsets,
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.kernel.l5(),
            self.offsets,
        )
    }
}

/// Owns a [`super::guest::GuestKernel`] and a [`BpfProgOffsets`],
/// providing BPF program access through a borrowed
/// [`BpfProgAccessor`].
///
/// Mirrors [`super::bpf_map::GuestMemAccessorOwned`] for the
/// program-side surface: callers that don't already hold a
/// `GuestKernel` + `BpfProgOffsets` pair (e.g. the freeze
/// coordinator) construct one of these once at start, retain it
/// across the run, and borrow [`Self::as_accessor`] for each
/// read. Owning the offsets here keeps the BTF parse to once per
/// VM run rather than once per dump.
pub struct BpfProgAccessorOwned<'a> {
    kernel: super::guest::GuestKernel<'a>,
    offsets: BpfProgOffsets,
}

impl<'a> BpfProgAccessorOwned<'a> {
    /// One-shot constructor: builds a [`super::guest::GuestKernel`]
    /// from `vmlinux`, parses BTF to resolve the BPF-program-related
    /// struct offsets, and verifies the `prog_idr` symbol exists.
    /// The resulting handle owns both the `GuestKernel` and the
    /// `BpfProgOffsets`.
    ///
    /// Errors when the vmlinux ELF / BTF parse fails, when the
    /// `GuestKernel` handshake fails (still-booting guest), or
    /// when `prog_idr` is missing from the symbol table.
    pub fn new(mem: &'a super::reader::GuestMem, vmlinux: &std::path::Path) -> anyhow::Result<Self> {
        let kernel = super::guest::GuestKernel::new(mem, vmlinux)?;
        let offsets = BpfProgOffsets::from_vmlinux(vmlinux)?;
        // Validate prog_idr resolves so a borrowed `as_accessor`
        // can't fail later — same pre-flight pattern as
        // `GuestMemAccessorOwned::new`.
        if kernel.symbol_kva("prog_idr").is_none() {
            return Err(anyhow::anyhow!(
                "prog_idr symbol not found in vmlinux"
            ));
        }
        Ok(Self { kernel, offsets })
    }

    /// Borrow as a [`BpfProgAccessor`] for program operations.
    ///
    /// The returned accessor borrows `self.offsets`; no clone on
    /// the hot path. Errors when `prog_idr` cannot be resolved
    /// (kept for surface symmetry with
    /// [`BpfProgAccessor::from_guest_kernel`]; in practice `new`
    /// already validated the symbol so this is infallible).
    pub fn as_accessor(&self) -> anyhow::Result<BpfProgAccessor<'_>> {
        BpfProgAccessor::from_guest_kernel(&self.kernel, &self.offsets)
    }

    /// Access the underlying [`super::guest::GuestKernel`] for
    /// callers that need symbol resolution / page-walk primitives
    /// outside the prog-discovery surface (e.g. resolving
    /// `__per_cpu_offset` for `runtime_stats`).
    #[allow(dead_code)]
    pub fn guest_kernel(&self) -> &super::guest::GuestKernel<'a> {
        &self.kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prog_verifier_stats_serde_roundtrip() {
        let info = ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 42000,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "dispatch");
        assert_eq!(loaded.verified_insns, 42000);
    }

    #[test]
    fn prog_verifier_stats_vec_serde_roundtrip() {
        let stats = vec![
            ProgVerifierStats {
                name: "dispatch".to_string(),
                verified_insns: 100000,
            },
            ProgVerifierStats {
                name: "enqueue".to_string(),
                verified_insns: 50000,
            },
        ];
        let json = serde_json::to_vec(&stats).unwrap();
        let loaded: Vec<ProgVerifierStats> = serde_json::from_slice(&json).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "dispatch");
        assert_eq!(loaded[0].verified_insns, 100000);
        assert_eq!(loaded[1].name, "enqueue");
        assert_eq!(loaded[1].verified_insns, 50000);
    }

    #[test]
    fn prog_verifier_stats_empty_name() {
        let info = ProgVerifierStats {
            name: String::new(),
            verified_insns: 0,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "");
        assert_eq!(loaded.verified_insns, 0);
    }

    #[test]
    fn prog_verifier_stats_max_values() {
        let info = ProgVerifierStats {
            name: "x".repeat(16),
            verified_insns: u32::MAX,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.verified_insns, u32::MAX);
        assert_eq!(loaded.name.len(), 16);
    }

    #[test]
    fn prog_runtime_stats_serde_roundtrip() {
        let info = ProgRuntimeStats {
            name: "ktstr_dispatch".to_string(),
            cnt: 12345,
            nsecs: 9_876_543,
            misses: 7,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgRuntimeStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "ktstr_dispatch");
        assert_eq!(loaded.cnt, 12345);
        assert_eq!(loaded.nsecs, 9_876_543);
        assert_eq!(loaded.misses, 7);
    }

    /// All three counters use `saturating_add` in
    /// [`read_prog_runtime_stats`] when summing per-CPU slots, so a
    /// long-running guest with a hot BPF program (or scrambled
    /// per-CPU pages from an unmapped slot) can produce a `u64::MAX`
    /// sum instead of wrapping. Pinning the wire shape here proves
    /// the serde codec preserves the saturated value end-to-end —
    /// any future migration that swaps the field type would surface
    /// here before bleeding into the failure-dump consumers.
    #[test]
    fn prog_runtime_stats_max_u64_saturation_roundtrip() {
        let info = ProgRuntimeStats {
            name: "saturated".to_string(),
            cnt: u64::MAX,
            nsecs: u64::MAX,
            misses: u64::MAX,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgRuntimeStats = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.cnt, u64::MAX);
        assert_eq!(loaded.nsecs, u64::MAX);
        assert_eq!(loaded.misses, u64::MAX);
    }

    #[test]
    fn prog_runtime_stats_default_zero() {
        let info = ProgRuntimeStats::default();
        assert_eq!(info.name, "");
        assert_eq!(info.cnt, 0);
        assert_eq!(info.nsecs, 0);
        assert_eq!(info.misses, 0);
    }

    /// The Display impl is the entry point used by
    /// [`super::dump::FailureDumpReport`]'s human-readable rendering;
    /// pin the format so a downstream change to the impl is caught
    /// before the failure-dump output silently changes shape.
    #[test]
    fn prog_runtime_stats_display_format() {
        let info = ProgRuntimeStats {
            name: "ktstr_enqueue".to_string(),
            cnt: 100,
            nsecs: 200,
            misses: 3,
        };
        assert_eq!(
            format!("{info}"),
            "ktstr_enqueue: cnt=100 nsecs=200 misses=3"
        );
    }
}
