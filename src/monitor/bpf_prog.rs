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
    cr3_pa: u64,
    page_offset: u64,
    prog_idr_kva: u64,
    offsets: &BpfProgOffsets,
    l5: bool,
    per_cpu_offsets: &[u64],
) -> Vec<ProgRuntimeStats> {
    let idr_pa = text_kva_to_pa(prog_idr_kva);

    let xa_head = mem.read_u64(idr_pa, offsets.idr_xa_head);
    if xa_head == 0 {
        return Vec::new();
    }
    let idr_next = mem.read_u32(idr_pa, offsets.idr_next);

    let mut stats_out = Vec::new();

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

        // Per-CPU sum. saturating_add prevents the
        // `attempt to add with overflow` panic that's been
        // observed when uninitialized / scrambled per-CPU pages
        // yield near-u64::MAX values; see `ProgRuntimeStats`.
        let mut cnt: u64 = 0;
        let mut nsecs: u64 = 0;
        let mut misses: u64 = 0;
        for &cpu_off in per_cpu_offsets {
            let stats_kva = stats_percpu_kva.wrapping_add(cpu_off);
            if let Some(stats_pa) = translate_any_kva(mem, cr3_pa, page_offset, stats_kva, l5)
                && stats_pa < mem.size()
            {
                cnt = cnt.saturating_add(mem.read_u64(stats_pa, offsets.stats_cnt));
                nsecs = nsecs.saturating_add(mem.read_u64(stats_pa, offsets.stats_nsecs));
                misses = misses.saturating_add(mem.read_u64(stats_pa, offsets.stats_misses));
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
/// frozen guest's `prog_idr`). The planned live-host backend (#9)
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
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.prog_idr_kva,
            self.offsets,
            self.kernel.l5(),
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
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.prog_idr_kva,
            self.offsets,
            self.kernel.l5(),
            per_cpu_offsets,
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
    pub fn new(mem: &'a super::reader::GuestMem, vmlinux: &std::path::Path) -> anyhow::Result<Self> {
        let kernel = super::guest::GuestKernel::new(mem, vmlinux)?;
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
        assert!(!s.contains("miss_rate"), "miss_rate must elide when total=0: {s}");
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

    /// Format chain integration: the `ProgRuntimeStats` Display
    /// output must appear verbatim inside `FailureDumpReport`'s
    /// Display output. Pins the chain
    /// `ProgRuntimeStats::fmt` (bpf_prog.rs) →
    /// `FailureDumpReport::fmt::std::fmt::Display::fmt(stats, f)`
    /// (dump.rs `prog_runtime_stats:` arm).
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
        use super::dump::{FailureDumpReport, SCHEMA_SINGLE};
        let info = ProgRuntimeStats {
            name: "chain_test".to_string(),
            cnt: 7,
            nsecs: 42,
            misses: 1,
        };
        let inner = format!("{info}");
        // Direct Display on ProgRuntimeStats: pinned shape includes
        // the bpftop-style derived metrics (#22). cnt=7 nsecs=42 →
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
