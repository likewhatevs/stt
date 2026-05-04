//! BTF offsets and stable enum indices for per-node NUMA-event
//! counter capture.
//!
//! The walker resolves the kernel symbol `node_data` (an array of
//! `pglist_data *` indexed by node id), dereferences each entry to
//! reach the `pglist_data` for that node, walks
//! `pglist_data.node_zones[]` (an inline array of `struct zone`), and
//! reads `zone.vm_numa_event[NR_VM_NUMA_EVENT_ITEMS]` (an array of
//! `atomic_long_t`) to produce per-node counters.
//!
//! Items are `#[allow(dead_code)]` because the live walker that
//! consumes the offsets/indices is pending — the wire shape and
//! BTF resolver are landed but no caller resolves the offsets yet.

#![allow(dead_code)]

use anyhow::Result;
use btf_rs::Btf;

use super::{find_struct, member_byte_offset};

/// Stable indices into `zone.vm_numa_event[NR_VM_NUMA_EVENT_ITEMS]`
/// from `enum numa_stat_item` (`include/linux/mmzone.h`). The kernel
/// pins this order; external readers (`/sys/devices/system/node/nodeN/numastat`,
/// `/proc/zoneinfo`) depend on it. Hard-coded here for the same
/// reason the [`super::CPUTIME_USER`] family is hard-coded — BTF only
/// encodes the array length, not the enum-to-position mapping, so a
/// BTF-driven read would require resolving the enum separately
/// (which is a UAPI break, not a layout drift this code can adapt
/// to).
pub const NUMA_HIT: usize = 0;
/// Pages allocated on the requested non-local node when the local
/// node was full. See [`NUMA_HIT`].
pub const NUMA_MISS: usize = 1;
/// Pages allocated on this node by a process whose policy targeted
/// a different node. See [`NUMA_HIT`].
pub const NUMA_FOREIGN: usize = 2;
/// Allocations from an interleave policy that hit this node.
/// See [`NUMA_HIT`].
pub const NUMA_INTERLEAVE_HIT: usize = 3;
/// Allocations on this node by a process running on this node.
/// See [`NUMA_HIT`].
pub const NUMA_LOCAL: usize = 4;
/// Allocations on this node by a process running on a different
/// node. See [`NUMA_HIT`].
pub const NUMA_OTHER: usize = 5;

/// Number of `numa_stat_item` slots per `zone.vm_numa_event[]`.
/// Mirrors `NR_VM_NUMA_EVENT_ITEMS` in `include/linux/mmzone.h`
/// (= 6 in current mainline). The pin-via-constant is intentional:
/// adding a slot to the kernel enum is a UAPI break that warrants a
/// host-side update of this constant, not a BTF-driven autodiscover.
pub const NR_VM_NUMA_EVENT_ITEMS: usize = 6;

/// Names of every NUMA event slot, indexed by [`NUMA_HIT`] etc.
/// Surfaced in failure-dump JSON so a downstream consumer reading
/// `vm_numa_event[i]` knows which counter each slot represents
/// without chasing the kernel header. Mirrors [`super::SOFTIRQ_NAMES`]'s
/// rationale.
pub const NUMA_EVENT_NAMES: [&str; NR_VM_NUMA_EVENT_ITEMS] = [
    "NUMA_HIT",
    "NUMA_MISS",
    "NUMA_FOREIGN",
    "NUMA_INTERLEAVE_HIT",
    "NUMA_LOCAL",
    "NUMA_OTHER",
];

/// Byte offsets used to read per-node NUMA-event counters from
/// guest memory.
///
/// The walk path is:
///   1. Resolve kernel symbol `node_data` — an array of
///      `pglist_data *` indexed by node id (declared in
///      `arch/x86/mm/numa.c::node_data[]` on x86 / `arch/arm64/mm/numa.c`
///      on arm64).
///   2. For each node, dereference `node_data[node]` to reach the
///      `pglist_data` for that node.
///   3. Walk `pglist_data.node_zones[MAX_NR_ZONES]` (an inline array
///      of `struct zone`).
///   4. For each zone, read `zone.vm_numa_event[]` (an array of
///      `atomic_long_t`) and sum across zones to produce per-node
///      counters.
///
/// `pglist_data_node_zones` and `zone_vm_numa_event` are the two
/// offsets the walker needs after the `node_data` symbol is
/// resolved; `zone_size` lets the walker stride to
/// `node_zones[zone_idx]`. `MAX_NR_ZONES` is hard-coded to 5
/// (matching mainline x86_64 and arm64: ZONE_DMA, ZONE_DMA32,
/// ZONE_NORMAL, ZONE_MOVABLE, ZONE_DEVICE) — a kernel without
/// CONFIG_ZONE_DEVICE drops the trailing slot but still reports
/// the others, so iterating up to 5 is safe (indices past the
/// kernel's actual count read all-zero).
///
/// Resolution returns `Err` when `pglist_data` or `zone` are
/// missing from BTF — universal types whose absence indicates a
/// stripped vmlinux. `vm_numa_event` is gated on
/// `CONFIG_NUMA + CONFIG_VM_EVENT_COUNTERS` (the latter defaults
/// to y on every modern kernel); when missing the resolver returns
/// Err so the caller skips the capture.
#[derive(Debug, Clone, Copy)]
pub struct NumaStatsOffsets {
    /// Offset of `node_zones[]` within `struct pglist_data`.
    pub pglist_data_node_zones: usize,
    /// Offset of `vm_numa_event[]` within `struct zone`. Read as
    /// `NR_VM_NUMA_EVENT_ITEMS` consecutive `atomic_long_t`
    /// (8 bytes each on 64-bit).
    pub zone_vm_numa_event: usize,
    /// Size of `struct zone` in bytes. Used to stride
    /// `node_zones[zone_idx]`.
    pub zone_size: usize,
}

impl NumaStatsOffsets {
    /// Resolve NUMA-event offsets from a pre-loaded BTF object.
    /// Returns Err when any required type/field is missing
    /// (stripped vmlinux, kernel without `CONFIG_NUMA` or without
    /// `CONFIG_VM_EVENT_COUNTERS`).
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (pglist_data, _) = find_struct(btf, "pglist_data")?;
        let pglist_data_node_zones = member_byte_offset(btf, &pglist_data, "node_zones")?;

        let (zone, _) = find_struct(btf, "zone")?;
        let zone_vm_numa_event = member_byte_offset(btf, &zone, "vm_numa_event")?;
        let zone_size = zone.size();

        Ok(Self {
            pglist_data_node_zones,
            zone_vm_numa_event,
            zone_size,
        })
    }
}
