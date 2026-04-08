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
pub struct ProgVerifierInfo {
    pub name: String,
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
) -> Vec<ProgVerifierInfo> {
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

        progs.push(ProgVerifierInfo {
            name,
            verified_insns,
        });
    }

    progs
}

/// Per-program runtime stats summed across all CPUs.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProgRuntimeStats {
    pub name: String,
    /// Total invocation count across all CPUs.
    pub cnt: u64,
    /// Total CPU time in nanoseconds across all CPUs.
    pub nsecs: u64,
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
/// Walks `prog_idr` once, reads `bpf_prog->stats` (percpu pointer)
/// for each struct_ops program. Returns cached info for use by
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
/// For each program, reads `cnt` and `nsecs` from each CPU's
/// `bpf_prog_stats` and sums across CPUs. Uses pre-resolved
/// `__per_cpu_offset` array for address resolution.
pub(crate) fn read_prog_runtime_stats(
    mem: &GuestMem,
    cached: &[CachedProgInfo],
    per_cpu_offsets: &[u64],
    page_offset: u64,
    offsets: &BpfProgOffsets,
) -> Vec<ProgRuntimeStats> {
    cached
        .iter()
        .map(|prog| {
            let mut cnt: u64 = 0;
            let mut nsecs: u64 = 0;
            for &cpu_off in per_cpu_offsets {
                let stats_kva = prog.stats_percpu_kva.wrapping_add(cpu_off);
                let stats_pa = super::symbols::kva_to_pa(stats_kva, page_offset);
                if stats_pa < mem.size() {
                    cnt += mem.read_u64(stats_pa, offsets.stats_cnt);
                    nsecs += mem.read_u64(stats_pa, offsets.stats_nsecs);
                }
            }
            ProgRuntimeStats {
                name: prog.name.clone(),
                cnt,
                nsecs,
            }
        })
        .collect()
}

/// Host-side BPF program accessor for a running guest VM.
pub struct BpfProgAccessor<'a> {
    kernel: &'a super::guest::GuestKernel<'a>,
    prog_idr_kva: u64,
    offsets: BpfProgOffsets,
}

impl<'a> BpfProgAccessor<'a> {
    /// Create from an existing [`GuestKernel`] and vmlinux path.
    pub fn from_guest_kernel(
        kernel: &'a super::guest::GuestKernel<'a>,
        vmlinux: &std::path::Path,
    ) -> anyhow::Result<Self> {
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

    /// Enumerate struct_ops BPF programs and collect verifier stats.
    pub fn struct_ops_progs(&self) -> Vec<ProgVerifierInfo> {
        find_struct_ops_progs(
            self.kernel.mem(),
            self.kernel.cr3_pa(),
            self.kernel.page_offset(),
            self.prog_idr_kva,
            &self.offsets,
            self.kernel.l5(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prog_verifier_info_serde_roundtrip() {
        let info = ProgVerifierInfo {
            name: "dispatch".to_string(),
            verified_insns: 42000,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "dispatch");
        assert_eq!(loaded.verified_insns, 42000);
    }

    #[test]
    fn prog_verifier_info_vec_serde_roundtrip() {
        let stats = vec![
            ProgVerifierInfo {
                name: "dispatch".to_string(),
                verified_insns: 100000,
            },
            ProgVerifierInfo {
                name: "enqueue".to_string(),
                verified_insns: 50000,
            },
        ];
        let json = serde_json::to_vec(&stats).unwrap();
        let loaded: Vec<ProgVerifierInfo> = serde_json::from_slice(&json).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "dispatch");
        assert_eq!(loaded[0].verified_insns, 100000);
        assert_eq!(loaded[1].name, "enqueue");
        assert_eq!(loaded[1].verified_insns, 50000);
    }

    #[test]
    fn prog_verifier_info_empty_name() {
        let info = ProgVerifierInfo {
            name: String::new(),
            verified_insns: 0,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, "");
        assert_eq!(loaded.verified_insns, 0);
    }

    #[test]
    fn prog_verifier_info_max_values() {
        let info = ProgVerifierInfo {
            name: "x".repeat(16),
            verified_insns: u32::MAX,
        };
        let json = serde_json::to_string(&info).unwrap();
        let loaded: ProgVerifierInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.verified_insns, u32::MAX);
        assert_eq!(loaded.name.len(), 16);
    }
}
