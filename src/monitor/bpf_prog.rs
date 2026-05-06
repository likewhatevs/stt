//! Host-side BPF program enumeration via guest physical memory.
//!
//! Walks the kernel's `prog_idr` xarray from the host to discover
//! loaded BPF programs and read verifier stats from `bpf_prog_aux`.
//! No guest cooperation is needed — all reads go through the guest
//! physical memory mapping.

use super::btf_offsets::BpfProgOffsets;
use super::idr::{translate_any_kva, xa_load};
use super::reader::{GuestMem, WalkContext};
use super::symbols::text_kva_to_pa_with_base;

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
/// `aux->verified_insns` and `aux->name`. `start_kernel_map` is the
/// runtime kernel image base used to translate `prog_idr_kva` to a
/// guest physical address.
pub(crate) fn find_struct_ops_progs(
    mem: &GuestMem,
    walk: WalkContext,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    start_kernel_map: u64,
) -> Vec<ProgVerifierStats> {
    let idr_pa = text_kva_to_pa_with_base(prog_idr_kva, start_kernel_map);

    let xa_head = mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    let idr_next = mem.read_u32(idr_pa, offsets.idr_next);

    let mut progs = Vec::new();

    for id in 0..idr_next {
        let Some(entry) = xa_load(
            mem,
            walk.page_offset,
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
        let Some(prog_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            entry,
            walk.l5,
            walk.tcr_el1,
        ) else {
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
        let Some(aux_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            aux_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
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

impl ProgRuntimeStats {
    /// Mean nanoseconds per invocation: `nsecs / cnt`. Returns
    /// `0.0` when `cnt == 0` (program never ran or counter not
    /// running) so the result never propagates `NaN` / `Infinity`
    /// into downstream `finite_or_zero` filters. Method-only access
    /// (no stored shadow) — recomputed every call from the raw
    /// fields, matching the [`super::super::assert::CgroupStats::wake_latency_tail_ratio`]
    /// derived-ratio convention.
    ///
    /// Unitless-from-bpftop's perspective: bpftop-style triage
    /// reads "ns/call" as the primary cost-per-invocation metric;
    /// surfacing it here lets a failure-dump consumer compare two
    /// programs' per-call cost without dividing the wire counters
    /// manually.
    pub fn ns_per_call(&self) -> f64 {
        if self.cnt > 0 {
            self.nsecs as f64 / self.cnt as f64
        } else {
            0.0
        }
    }

    /// Fraction of invocation attempts blocked by the per-CPU
    /// recursion guard: `misses / (cnt + misses)`. Returns `0.0`
    /// when both counters are zero (no signal); never produces
    /// `NaN` / `Infinity` even on a saturated `cnt + misses`
    /// overflow because `saturating_add` floors at `u64::MAX` and
    /// the resulting denominator is non-zero.
    ///
    /// A non-trivial miss rate signals lock contention or a
    /// misconfigured recursion guard — bpftop-style triage flags
    /// any program with `miss_rate > 0.01` as a hot recursion
    /// path. Method-only access (no stored shadow); the wire
    /// format carries `cnt` and `misses` separately so consumers
    /// who want the raw counts can recover them.
    pub fn miss_rate(&self) -> f64 {
        let total = self.cnt.saturating_add(self.misses);
        if total > 0 {
            self.misses as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for ProgRuntimeStats {
    /// One-line summary used by [`super::dump::FailureDumpReport`]'s
    /// human-readable rendering: name + the three counter sums plus
    /// the bpftop-style derived metrics (ns/call, miss-rate fraction).
    /// Derived metrics elide when their guards fire (cnt==0 or
    /// cnt+misses==0) so a program that never ran renders without
    /// misleading "0.000 ns/call" noise.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: cnt={} nsecs={} misses={}",
            self.name, self.cnt, self.nsecs, self.misses
        )?;
        if self.cnt > 0 {
            // Three decimals on ns/call: bpftop uses two; we add
            // one for sub-microsecond precision since scheduler
            // BPF ops typically run in tens of nanoseconds.
            write!(f, " ns/call={:.3}", self.ns_per_call())?;
        }
        if self.cnt.saturating_add(self.misses) > 0 && self.misses > 0 {
            // Render miss_rate only when there were actual misses
            // — `0.000` would just be noise on healthy programs.
            // Four decimals: a 0.0001 (= 1 in 10K) miss rate is
            // already actionable for a hot scheduler op.
            write!(f, " miss_rate={:.4}", self.miss_rate())?;
        }
        Ok(())
    }
}

/// Walk `prog_idr` and produce per-program runtime stats in a single
/// IDR pass.
///
/// Folds the previous discover-then-read split into one visitor: for
/// each struct_ops program reached via `xa_load`, read
/// `bpf_prog->stats` (per-CPU base) and `bpf_prog_aux->name` and then
/// sum `cnt`/`nsecs`/`misses` across `per_cpu_offsets`. Halves the
/// per-prog kernel-memory reads relative to the prior split (one
/// `prog_idr` walk and one `bpf_prog`/`aux` translate per program
/// instead of two of each).
///
/// `cnt`/`nsecs`/`misses` are u64 monotonic counters per the kernel's
/// `struct bpf_prog_stats` (include/linux/filter.h) — see
/// [`ProgRuntimeStats`] for provenance and the saturation contract.
/// Address translation uses [`translate_any_kva`] so per-CPU pages
/// served from vmalloc'd memory (`pcpu_get_vm_areas`) translate
/// correctly alongside direct-mapping percpu allocations.
pub(crate) fn walk_struct_ops_runtime_stats(
    mem: &GuestMem,
    walk: WalkContext,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    per_cpu_offsets: &[u64],
    start_kernel_map: u64,
) -> Vec<ProgRuntimeStats> {
    let idr_pa = text_kva_to_pa_with_base(prog_idr_kva, start_kernel_map);

    let xa_head = mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    let idr_next = mem.read_u32(idr_pa, offsets.idr_next);

    let mut stats_out = Vec::new();

    for id in 0..idr_next {
        let Some(entry) = xa_load(
            mem,
            walk.page_offset,
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

        let Some(prog_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            entry,
            walk.l5,
            walk.tcr_el1,
        ) else {
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
        let Some(aux_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            aux_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
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

        // Per-CPU sum. saturating_add prevents the
        // `attempt to add with overflow` panic that's been
        // observed when uninitialized / scrambled per-CPU pages
        // yield near-u64::MAX values; see `ProgRuntimeStats`.
        let mut cnt: u64 = 0;
        let mut nsecs: u64 = 0;
        let mut misses: u64 = 0;
        for (cpu_index, &cpu_off) in per_cpu_offsets.iter().enumerate() {
            // Out-of-range CPU detection: kernel `setup_per_cpu_areas`
            // only writes `__per_cpu_offset[cpu]` for CPUs in
            // `for_each_possible_cpu`, leaving slots beyond
            // `nr_cpu_ids` at the BSS-initialized 0. Real SMP
            // kernels assign each possible CPU a strictly-positive
            // offset for `cpu > 0`; only the BSP (cpu_index == 0)
            // can legitimately observe a zero offset. Skip
            // `cpu_off == 0 && cpu_index > 0` to avoid double-
            // counting CPU 0's stats for every BSS-zero tail slot.
            // Mirrors the guard in
            // [`super::bpf_map::read_percpu_array_value`].
            if cpu_off == 0 && cpu_index > 0 {
                continue;
            }
            let stats_kva = stats_percpu_kva.wrapping_add(cpu_off);
            if let Some(stats_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                stats_kva,
                walk.l5,
                walk.tcr_el1,
            ) && stats_pa < mem.size()
            {
                // Batch the three u64 stat reads into one bulk
                // `read_bytes` covering the contiguous span from
                // `min(cnt, nsecs, misses)` to `max(...) + 8`. The
                // kernel's `struct bpf_prog_stats` packs `cnt`,
                // `nsecs`, and `misses` as adjacent u64_stats_t
                // (8 bytes each) and the BTF resolver accepts only
                // layouts where the three fields land in 24
                // contiguous bytes. The bulk read pays one bounds
                // check + region resolve instead of three per CPU,
                // and parses the values from the local buffer
                // without further volatile loads.
                let lo = offsets
                    .stats_cnt
                    .min(offsets.stats_nsecs)
                    .min(offsets.stats_misses);
                let hi = offsets
                    .stats_cnt
                    .max(offsets.stats_nsecs)
                    .max(offsets.stats_misses)
                    + 8;
                let span = hi - lo;
                if span <= 64 {
                    let mut buf = [0u8; 64];
                    let n = mem.read_bytes(stats_pa + lo as u64, &mut buf[..span]);
                    if n == span {
                        let parse = |off: usize| -> u64 {
                            let i = off - lo;
                            u64::from_ne_bytes(buf[i..i + 8].try_into().unwrap())
                        };
                        cnt = cnt.saturating_add(parse(offsets.stats_cnt));
                        nsecs = nsecs.saturating_add(parse(offsets.stats_nsecs));
                        misses = misses.saturating_add(parse(offsets.stats_misses));
                    } else {
                        // Partial copy (page straddle / end-of-DRAM)
                        // — fall back to scalar reads to retain the
                        // original semantics.
                        cnt = cnt.saturating_add(mem.read_u64(stats_pa, offsets.stats_cnt));
                        nsecs = nsecs.saturating_add(mem.read_u64(stats_pa, offsets.stats_nsecs));
                        misses = misses.saturating_add(mem.read_u64(stats_pa, offsets.stats_misses));
                    }
                } else {
                    // Span exceeds the inline buffer. Should be
                    // unreachable for the production
                    // `bpf_prog_stats` layout (24 bytes), but
                    // tolerate exotic layouts via the scalar path
                    // rather than panicking.
                    cnt = cnt.saturating_add(mem.read_u64(stats_pa, offsets.stats_cnt));
                    nsecs = nsecs.saturating_add(mem.read_u64(stats_pa, offsets.stats_nsecs));
                    misses = misses.saturating_add(mem.read_u64(stats_pa, offsets.stats_misses));
                }
            }
        }

        stats_out.push(ProgRuntimeStats {
            name,
            cnt,
            nsecs,
            misses,
        });
    }

    stats_out
}

/// Read-only abstraction over BPF program enumeration and per-program
/// stats reads across data sources. Mirror of
/// [`super::bpf_map::BpfMapAccessor`] for the program side.
///
/// Currently one implementation: [`GuestMemProgAccessor`] (PTE-walks a
/// frozen guest's `prog_idr`). The planned live-host backend
/// will walk loaded programs via `BPF_PROG_GET_NEXT_ID` /
/// `BPF_OBJ_GET_INFO_BY_FD` and produce the same
/// `Vec<ProgVerifierStats>` / `Vec<ProgRuntimeStats>` shapes, so the
/// failure-dump renderer stays data-source-agnostic.
pub trait BpfProgAccessor {
    /// Enumerate struct_ops BPF programs and collect verifier stats.
    fn struct_ops_progs(&self) -> Vec<ProgVerifierStats>;

    /// Snapshot per-program runtime stats (`cnt`, `nsecs`, `misses`)
    /// for every struct_ops BPF program, summed across all CPUs.
    ///
    /// `per_cpu_offsets` is the kernel's `__per_cpu_offset[]` array,
    /// typically obtained via [`super::symbols::read_per_cpu_offsets`].
    /// The live-host backend will ignore this argument (the kernel
    /// provides per-CPU sums via `BPF_OBJ_GET_INFO_BY_FD`).
    fn struct_ops_runtime_stats(&self, per_cpu_offsets: &[u64]) -> Vec<ProgRuntimeStats>;
}

/// Host-side BPF program accessor backed by direct guest physical-memory
/// reads. PTE-walks a frozen guest's `prog_idr` to enumerate loaded
/// programs and reads `bpf_prog_stats` per-CPU slots inline.
pub struct GuestMemProgAccessor<'a> {
    kernel: &'a super::guest::GuestKernel<'a>,
    prog_idr_kva: u64,
    /// Borrowed from the caller. Mirrors the
    /// [`super::bpf_map::GuestMemMapAccessor`] pattern:
    /// `BpfProgOffsets` is a ~160-byte POD built once from the
    /// vmlinux BTF, and every hot-path method reads it by reference,
    /// so owning it in the accessor would charge a clone that serves
    /// no purpose.
    offsets: &'a BpfProgOffsets,
}

impl<'a> GuestMemProgAccessor<'a> {
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
}

impl BpfProgAccessor for GuestMemProgAccessor<'_> {
    fn struct_ops_progs(&self) -> Vec<ProgVerifierStats> {
        find_struct_ops_progs(
            self.kernel.mem(),
            self.kernel.walk_context(),
            self.prog_idr_kva,
            self.offsets,
            self.kernel.start_kernel_map(),
        )
    }

    /// Mirrors the kernel-side per-CPU accumulation: `cnt` is
    /// bumped via `u64_stats_inc` and `nsecs` is bumped via
    /// `u64_stats_add(&stats->nsecs, duration)` inside
    /// `__bpf_prog_run` (include/linux/filter.h), invoked through
    /// the JIT-emitted entry path on every program invocation.
    /// `misses` is bumped by `bpf_prog_inc_misses_counter`
    /// (defined in `kernel/bpf/syscall.c`) called from
    /// `kernel/bpf/trampoline.c::__bpf_prog_enter_recur` when a
    /// program re-enters and the recursion guard rejects it.
    fn struct_ops_runtime_stats(&self, per_cpu_offsets: &[u64]) -> Vec<ProgRuntimeStats> {
        walk_struct_ops_runtime_stats(
            self.kernel.mem(),
            self.kernel.walk_context(),
            self.prog_idr_kva,
            self.offsets,
            per_cpu_offsets,
            self.kernel.start_kernel_map(),
        )
    }
}

/// Owns a [`super::guest::GuestKernel`] and a [`BpfProgOffsets`],
/// providing BPF program access through a borrowed
/// [`GuestMemProgAccessor`].
///
/// Mirrors [`super::bpf_map::GuestMemMapAccessorOwned`] for the
/// program-side surface: callers that don't already hold a
/// `GuestKernel` + `BpfProgOffsets` pair (e.g. the freeze
/// coordinator) construct one of these once at start, retain it
/// across the run, and borrow [`Self::as_accessor`] for each
/// read. Owning the offsets here keeps the BTF parse to once per
/// VM run rather than once per dump.
pub struct GuestMemProgAccessorOwned<'a> {
    kernel: super::guest::GuestKernel<'a>,
    prog_idr_kva: u64,
    offsets: BpfProgOffsets,
}

impl<'a> GuestMemProgAccessorOwned<'a> {
    /// One-shot constructor: builds a [`super::guest::GuestKernel`]
    /// from `vmlinux`, parses BTF to resolve the BPF-program-related
    /// struct offsets, and resolves the `prog_idr` symbol KVA. The
    /// resulting handle owns both the `GuestKernel` and the
    /// `BpfProgOffsets`, with `prog_idr_kva` cached so
    /// [`Self::as_accessor`] is infallible.
    ///
    /// Errors when the vmlinux ELF / BTF parse fails, when the
    /// `GuestKernel` handshake fails (still-booting guest), or
    /// when `prog_idr` is missing from the symbol table.
    pub fn new(
        mem: &'a super::reader::GuestMem,
        vmlinux: &std::path::Path,
        tcr_el1: u64,
    ) -> anyhow::Result<Self> {
        let kernel = super::guest::GuestKernel::new(mem, vmlinux, tcr_el1)?;
        let offsets = BpfProgOffsets::from_vmlinux(vmlinux)?;
        let prog_idr_kva = kernel
            .symbol_kva("prog_idr")
            .ok_or_else(|| anyhow::anyhow!("prog_idr symbol not found in vmlinux"))?;
        Ok(Self {
            kernel,
            prog_idr_kva,
            offsets,
        })
    }

    /// Borrow as a [`GuestMemProgAccessor`] for program operations.
    ///
    /// Infallible — `new` already resolved `prog_idr_kva` and the
    /// borrow returns the cached KVA directly. Mirrors
    /// [`super::bpf_map::GuestMemMapAccessorOwned::as_accessor`].
    pub fn as_accessor(&self) -> GuestMemProgAccessor<'_> {
        GuestMemProgAccessor {
            kernel: &self.kernel,
            prog_idr_kva: self.prog_idr_kva,
            offsets: &self.offsets,
        }
    }

    /// Access the underlying [`super::guest::GuestKernel`] for
    /// callers that need symbol resolution / page-walk primitives
    /// outside the prog-discovery surface (e.g. resolving
    /// `__per_cpu_offset` for `struct_ops_runtime_stats`).
    #[allow(dead_code)]
    pub fn guest_kernel(&self) -> &super::guest::GuestKernel<'a> {
        &self.kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::START_KERNEL_MAP;

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
    ///
    /// Two derived metrics surface on the line when their guards
    /// pass: `ns/call` whenever `cnt > 0`, and `miss_rate`
    /// whenever there are any misses. A program that never ran
    /// (cnt=0) elides both — `prog_runtime_stats_display_zero_counters_elides_derived`
    /// covers that branch.
    #[test]
    fn prog_runtime_stats_display_format() {
        let info = ProgRuntimeStats {
            name: "ktstr_enqueue".to_string(),
            cnt: 100,
            nsecs: 200,
            misses: 3,
        };
        // cnt=100, nsecs=200 → ns/call = 2.000.
        // misses=3, cnt+misses=103 → miss_rate = 3/103 ≈ 0.0291.
        assert_eq!(
            format!("{info}"),
            "ktstr_enqueue: cnt=100 nsecs=200 misses=3 ns/call=2.000 miss_rate=0.0291",
        );
    }

    /// A program that never ran (cnt=0) renders only the four
    /// raw counters — both derived metrics are guarded out.
    /// Pin the elision so a regression that strips the guard and
    /// emits "ns/call=0.000 miss_rate=0.0000" surfaces here.
    #[test]
    fn prog_runtime_stats_display_zero_counters_elides_derived() {
        let info = ProgRuntimeStats {
            name: "never_ran".to_string(),
            cnt: 0,
            nsecs: 0,
            misses: 0,
        };
        let s = format!("{info}");
        assert_eq!(s, "never_ran: cnt=0 nsecs=0 misses=0");
        assert!(!s.contains("ns/call"), "ns/call must elide when cnt=0: {s}");
        assert!(
            !s.contains("miss_rate"),
            "miss_rate must elide when total=0: {s}"
        );
    }

    /// Healthy program with no recursion misses — `ns/call`
    /// surfaces but `miss_rate` elides (since misses=0).
    /// A regression that flipped the gate and rendered a
    /// "miss_rate=0.0000" line on every healthy program would
    /// trip here.
    #[test]
    fn prog_runtime_stats_display_no_misses_elides_miss_rate() {
        let info = ProgRuntimeStats {
            name: "healthy".to_string(),
            cnt: 1000,
            nsecs: 50_000,
            misses: 0,
        };
        let s = format!("{info}");
        assert!(s.contains("ns/call=50.000"), "ns/call must render: {s}");
        assert!(
            !s.contains("miss_rate"),
            "miss_rate must elide when misses=0: {s}",
        );
    }

    /// `ns_per_call` derived accessor: pin happy-path math + zero-
    /// divisor guard. Mirrors the `CgroupStats::wake_latency_tail_ratio`
    /// test pattern from assert.rs.
    #[test]
    fn prog_runtime_stats_ns_per_call_derived() {
        // Happy path: 1000 cnt + 50000 nsecs = 50 ns/call.
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 1000,
            nsecs: 50_000,
            misses: 0,
        };
        assert_eq!(info.ns_per_call(), 50.0);
        assert!(info.ns_per_call().is_finite());

        // Zero divisor: cnt=0 → 0.0 (not NaN).
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 0,
            nsecs: 999_999,
            misses: 0,
        };
        assert_eq!(info.ns_per_call(), 0.0);
        assert!(info.ns_per_call().is_finite());
    }

    /// `miss_rate` derived accessor: pin happy-path math + zero-
    /// divisor guard + saturating_add edge.
    #[test]
    fn prog_runtime_stats_miss_rate_derived() {
        // Happy path: 9 misses / (1 cnt + 9 misses) = 0.9.
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 1,
            nsecs: 0,
            misses: 9,
        };
        assert!((info.miss_rate() - 0.9).abs() < 1e-12);
        assert!(info.miss_rate().is_finite());

        // Zero divisor: both counters zero → 0.0 (not NaN).
        let info = ProgRuntimeStats::default();
        assert_eq!(info.miss_rate(), 0.0);
        assert!(info.miss_rate().is_finite());

        // Saturating-add edge: cnt at u64::MAX, misses also non-
        // trivial — `saturating_add` floors at u64::MAX, so the
        // denominator stays non-zero and the rate is finite.
        let info = ProgRuntimeStats {
            name: "saturated".to_string(),
            cnt: u64::MAX,
            nsecs: 0,
            misses: 1000,
        };
        assert!(info.miss_rate().is_finite());
        // Result is essentially 0 (1000 / u64::MAX) but the
        // important contract is finiteness — a regression that
        // overflowed and produced inf/NaN trips here.
        assert!(info.miss_rate() >= 0.0);
    }

    /// Wire format must NOT carry the derived ratios — they are
    /// method-only and recomputed on read. Pin so a regression
    /// that re-introduces a stored shadow trips here.
    #[test]
    fn prog_runtime_stats_wire_format_omits_derived_keys() {
        let info = ProgRuntimeStats {
            name: "x".to_string(),
            cnt: 100,
            nsecs: 200,
            misses: 3,
        };
        let json = serde_json::to_value(&info).unwrap();
        let map = match json {
            serde_json::Value::Object(m) => m,
            other => panic!("expected object, got {other:?}"),
        };
        assert!(
            !map.contains_key("ns_per_call"),
            "derived methods must NOT appear as wire fields: {map:#?}",
        );
        assert!(
            !map.contains_key("miss_rate"),
            "derived methods must NOT appear as wire fields: {map:#?}",
        );
        // Cross-check: methods still compute correctly.
        assert_eq!(info.ns_per_call(), 2.0);
        assert!((info.miss_rate() - 3.0_f64 / 103.0).abs() < 1e-12);
    }

    /// Build a minimal `BpfProgOffsets` keyed for the synthetic
    /// chain test below. The exact field offsets are arbitrary —
    /// they only need to be consistent with how the test buffer
    /// is laid out — but `stats_cnt`/`stats_nsecs`/`stats_misses`
    /// MUST sit within a 24-byte window so the bulk-read path
    /// fires (`span <= 64`). Drift in these three offsets would
    /// silently switch the walker to the scalar fallback and
    /// the bulk-read assertion below would still pass for the
    /// wrong reason.
    fn synthetic_prog_offsets() -> BpfProgOffsets {
        BpfProgOffsets {
            prog_type: 0,
            prog_aux: 8,
            aux_verified_insns: 0,
            aux_name: 8,
            xa_node_slots: 16,
            xa_node_shift: 0,
            idr_xa_head: 0,
            idr_next: 8,
            prog_stats: 16,
            stats_cnt: 0,
            stats_nsecs: 8,
            stats_misses: 16,
        }
    }

    /// End-to-end chain test for the bulk 24-byte
    /// `bpf_prog_stats` read inside
    /// [`walk_struct_ops_runtime_stats`]. The walker reads `cnt`,
    /// `nsecs`, and `misses` (three adjacent u64s in the kernel
    /// `struct bpf_prog_stats`) via one `read_bytes` over the
    /// `[lo, hi)` span and parses each value from the local
    /// buffer. This test pins the contract by:
    ///
    /// 1. Laying out a synthetic IDR + bpf_prog + bpf_prog_aux
    ///    + per-CPU stats slot in a flat buffer, using the
    ///    direct-mapping `kva = page_offset + pa` shortcut so
    ///    `translate_any_kva` resolves through the direct path
    ///    without building a page table.
    /// 2. Writing known u64 values at the three stats offsets.
    /// 3. Running the walker end-to-end and asserting the parsed
    ///    `cnt`/`nsecs`/`misses` match the bytes the bulk read
    ///    consumed.
    ///
    /// A regression that swapped two offsets in the parse closure
    /// (e.g. `parse(stats_nsecs)` returning `cnt`) would surface
    /// here as a value mismatch, NOT as a silent count-1 sum
    /// drift that handler-level tests miss.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn walk_struct_ops_runtime_stats_bulk_24byte_read_parses_three_offsets() {
        use crate::monitor::reader::{GuestMem, WalkContext};

        // Layout (all PAs offset by `page_offset` to form KVAs in
        // the direct-mapping range, except `prog_idr_kva` which
        // sits in the kernel-text range and translates via
        // `text_kva_to_pa_with_base`):
        //
        //   0x0000  prog_idr (xa_head + idr_next)
        //   0x1000  bpf_prog (prog_type, prog_aux, prog_stats)
        //   0x2000  bpf_prog_aux (verified_insns, name)
        //   0x3000  per-CPU bpf_prog_stats (cnt, nsecs, misses)
        let total: usize = 0x4000;
        let mut buf = vec![0u8; total];

        let page_offset: u64 = 0xFFFF_8880_0000_0000;
        let pa_to_kva = |pa: u64| -> u64 { page_offset.wrapping_add(pa) };

        let idr_pa: u64 = 0x0000;
        let prog_pa: u64 = 0x1000;
        let aux_pa: u64 = 0x2000;
        let stats_pa: u64 = 0x3000;

        // Single-entry xarray: `xa_head` IS the prog KVA with
        // bit 1 clear (leaf marker). `pa_to_kva(prog_pa)` has
        // bit 1 clear because prog_pa is 4 KiB-aligned.
        let prog_kva = pa_to_kva(prog_pa);
        assert_eq!(prog_kva & 2, 0, "prog_kva must be a leaf entry");

        let offsets = synthetic_prog_offsets();
        // Sanity: the bulk-read fast path requires
        // `span = hi - lo <= 64`. With offsets {0, 8, 16}:
        // lo = 0, hi = 16 + 8 = 24, span = 24. Pinning here so
        // a future offset change that pushed `span > 64`
        // (forcing the scalar fallback) trips the assert
        // before the test runs.
        let lo = offsets
            .stats_cnt
            .min(offsets.stats_nsecs)
            .min(offsets.stats_misses);
        let hi = offsets
            .stats_cnt
            .max(offsets.stats_nsecs)
            .max(offsets.stats_misses)
            + 8;
        assert!(
            hi - lo <= 64,
            "test premise: stats span must be small enough for the bulk path"
        );

        let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
            let off = pa as usize;
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        };
        let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
            let off = pa as usize;
            buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
        };

        // IDR: xa_head = prog_kva, idr_next = 1.
        write_u64(&mut buf, idr_pa + offsets.idr_xa_head as u64, prog_kva);
        write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 1);

        // bpf_prog: type = STRUCT_OPS, aux = aux_kva, stats = stats_kva.
        write_u32(&mut buf, prog_pa + offsets.prog_type as u64, BPF_PROG_TYPE_STRUCT_OPS);
        write_u64(&mut buf, prog_pa + offsets.prog_aux as u64, pa_to_kva(aux_pa));
        write_u64(&mut buf, prog_pa + offsets.prog_stats as u64, pa_to_kva(stats_pa));

        // bpf_prog_aux: verified_insns + name. Name must NUL-
        // terminate within BPF_OBJ_NAME_LEN so the walker's
        // `position(|&b| b == 0)` finds the end.
        write_u32(&mut buf, aux_pa + offsets.aux_verified_insns as u64, 12_345);
        let name = b"bulk_test";
        let name_pa = (aux_pa + offsets.aux_name as u64) as usize;
        buf[name_pa..name_pa + name.len()].copy_from_slice(name);

        // Stats: write the three u64 counters at the synthetic
        // offsets. These are the bytes the bulk read MUST surface
        // through the parse closure.
        let known_cnt: u64 = 0x1111_1111_1111_1111;
        let known_nsecs: u64 = 0x2222_2222_2222_2222;
        let known_misses: u64 = 0x3333_3333_3333_3333;
        write_u64(&mut buf, stats_pa + offsets.stats_cnt as u64, known_cnt);
        write_u64(&mut buf, stats_pa + offsets.stats_nsecs as u64, known_nsecs);
        write_u64(&mut buf, stats_pa + offsets.stats_misses as u64, known_misses);

        // SAFETY: buf is a live local Vec<u8> whose backing storage
        // outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let walk = WalkContext {
            cr3_pa: 0,
            page_offset,
            l5: false,
            tcr_el1: 0,
        };
        // One CPU. `cpu_off == 0` is allowed at `cpu_index == 0`
        // (BSP). `stats_kva + 0 = stats_kva`, which translates
        // through the direct mapping to `stats_pa`.
        let per_cpu_offsets = vec![0u64];

        let prog_idr_kva = idr_pa + START_KERNEL_MAP;
        let stats = walk_struct_ops_runtime_stats(
            &mem,
            walk,
            prog_idr_kva,
            &offsets,
            &per_cpu_offsets,
            START_KERNEL_MAP,
        );

        assert_eq!(stats.len(), 1, "single STRUCT_OPS prog must surface");
        assert_eq!(stats[0].name, "bulk_test");
        assert_eq!(
            stats[0].cnt, known_cnt,
            "bulk read must parse cnt at offsets.stats_cnt within the 24-byte window",
        );
        assert_eq!(
            stats[0].nsecs, known_nsecs,
            "bulk read must parse nsecs at offsets.stats_nsecs within the 24-byte window",
        );
        assert_eq!(
            stats[0].misses, known_misses,
            "bulk read must parse misses at offsets.stats_misses within the 24-byte window",
        );
    }

    /// Format chain integration: the `ProgRuntimeStats` Display
    /// output must appear verbatim inside `FailureDumpReport`'s
    /// Display output. Pins the chain
    /// `ProgRuntimeStats::fmt` (bpf_prog.rs) →
    /// `FailureDumpReport::fmt::std::fmt::Display::fmt(stats, f)`
    /// (dump/display.rs `prog_runtime_stats:` arm).
    ///
    /// The standalone `prog_runtime_stats_display_format` test pins
    /// the inner Display in isolation; the dump-side
    /// `report_display_renders_prog_runtime_stats` test pins the
    /// outer section header. Neither catches a regression that
    /// SUBSTITUTED the inner Display call (e.g. introducing a
    /// custom rendering branch in the outer formatter that bypasses
    /// `ProgRuntimeStats::fmt`). This test catches that drift by
    /// asserting BOTH layers render identically and the inner
    /// string appears as a substring of the outer — a substitution
    /// would break either equality.
    #[test]
    fn prog_runtime_stats_format_chain_inner_appears_in_outer() {
        use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
        let info = ProgRuntimeStats {
            name: "chain_test".to_string(),
            cnt: 7,
            nsecs: 42,
            misses: 1,
        };
        let inner = format!("{info}");
        // Direct Display on ProgRuntimeStats: pinned shape includes
        // the bpftop-style derived metrics. cnt=7 nsecs=42 →
        // ns/call=6.000; misses=1 → miss_rate=1/8=0.1250.
        assert_eq!(
            inner,
            "chain_test: cnt=7 nsecs=42 misses=1 ns/call=6.000 miss_rate=0.1250",
        );

        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            prog_runtime_stats: vec![info],
            ..Default::default()
        };
        let outer = format!("{report}");
        // The outer's `prog_runtime_stats:` section calls
        // `std::fmt::Display::fmt(stats, f)` on each entry; that
        // call dispatches through THIS module's Display impl. If a
        // future regression replaced the dispatch with a custom
        // formatter, the inner string would no longer appear in
        // the outer output — surfacing as substring failure.
        assert!(
            outer.contains(&inner),
            "FailureDumpReport's Display chain must dispatch through \
             ProgRuntimeStats::fmt — inner {inner:?} must appear \
             verbatim inside outer:\n{outer}",
        );
        // Sanity: the outer also wraps with the expected section
        // header, so the substring match is finding the chain
        // through the correct arm of FailureDumpReport's fmt and
        // not (e.g.) a coincidence in the schema marker.
        assert!(
            outer.contains("prog_runtime_stats:"),
            "outer Display must carry the prog_runtime_stats section \
             header; without it the chain test could pass even when the \
             inner string matched a different format arm:\n{outer}",
        );
    }
}
