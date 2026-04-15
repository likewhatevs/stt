//! Host-side KVM binary stats reader via KVM_GET_STATS_FD.
//!
//! The ioctl returns an fd whose contents are a self-describing binary
//! format: header -> descriptors -> data. Descriptors are static
//! (parsed once at open time). Data values are live u64s re-read via
//! pread at the header's data_offset.
//!
//! Stats fds have `noop_llseek` but `FMODE_PREAD`. The initial
//! sequential read works (position starts at 0); subsequent reads
//! use pread.

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use vmm_sys_util::ioctl_io_nr;

ioctl_io_nr!(KVM_GET_STATS_FD, kvm_bindings::KVMIO, 0xce);

/// Size of `kvm_stats_desc` fixed fields before the variable-length name.
const DESC_FIXED_SIZE: usize = 16;

/// Issue KVM_GET_STATS_FD on an fd (VmFd or VcpuFd).
fn get_stats_fd<F: AsRawFd>(fd: &F) -> Option<OwnedFd> {
    let ret = unsafe { libc::ioctl(fd.as_raw_fd(), KVM_GET_STATS_FD()) };
    if ret < 0 {
        None
    } else {
        Some(unsafe { OwnedFd::from_raw_fd(ret) })
    }
}

/// Parsed descriptor for a single stat.
#[derive(Debug, Clone)]
struct StatDesc {
    name: String,
    /// Number of u64 values (1 for scalars).
    size: usize,
    /// Byte offset from the start of the data section.
    offset: usize,
}

/// Metadata parsed from the stats fd. Static for the fd's lifetime.
#[derive(Debug)]
struct StatsMeta {
    /// File offset where data values begin.
    data_offset: usize,
    /// Byte size of the data section.
    data_size: usize,
    /// Parsed descriptors.
    descs: Vec<StatDesc>,
}

/// Read the entire stats fd content via sequential read. Must be
/// called once immediately after opening (position starts at 0;
/// noop_llseek prevents resetting).
fn read_initial(fd: RawFd) -> Option<Vec<u8>> {
    let mut buf = Vec::with_capacity(8192);
    let mut chunk = [0u8; 4096];
    loop {
        let n = unsafe { libc::read(fd, chunk.as_mut_ptr() as *mut _, chunk.len()) };
        if n < 0 {
            return None;
        }
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n as usize]);
    }
    if buf.len() < 24 {
        return None;
    }
    Some(buf)
}

/// Parse header + descriptors from the initial read buffer.
fn parse_meta_from_buf(buf: &[u8]) -> Option<StatsMeta> {
    let name_size = u32::from_ne_bytes(buf[4..8].try_into().ok()?) as usize;
    let num_desc = u32::from_ne_bytes(buf[8..12].try_into().ok()?) as usize;
    let desc_offset = u32::from_ne_bytes(buf[16..20].try_into().ok()?) as usize;
    let data_offset = u32::from_ne_bytes(buf[20..24].try_into().ok()?) as usize;

    let desc_stride = DESC_FIXED_SIZE + name_size;

    let mut descs = Vec::with_capacity(num_desc);
    for i in 0..num_desc {
        let d_start = desc_offset + i * desc_stride;
        if d_start + desc_stride > buf.len() {
            break;
        }
        let d = &buf[d_start..d_start + desc_stride];

        let size = u16::from_ne_bytes(d[6..8].try_into().ok()?) as usize;
        let offset = u32::from_ne_bytes(d[8..12].try_into().ok()?) as usize;

        let name_bytes = &d[DESC_FIXED_SIZE..DESC_FIXED_SIZE + name_size];
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_size);
        let name = String::from_utf8_lossy(&name_bytes[..name_end]).to_string();

        descs.push(StatDesc { name, size, offset });
    }

    let data_size = buf.len().saturating_sub(data_offset);

    Some(StatsMeta {
        data_offset,
        data_size,
        descs,
    })
}

/// Read live data section via pread.
fn pread_data(fd: RawFd, meta: &StatsMeta) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; meta.data_size];
    let n = unsafe {
        libc::pread(
            fd,
            buf.as_mut_ptr() as *mut _,
            meta.data_size,
            meta.data_offset as libc::off_t,
        )
    };
    if n < 0 {
        return None;
    }
    buf.truncate(n as usize);
    Some(buf)
}

/// Extract scalar stat values from a data buffer.
/// Descriptor offsets are byte offsets into the data section.
fn extract_stats(meta: &StatsMeta, data: &[u8]) -> HashMap<String, u64> {
    let mut map = HashMap::with_capacity(meta.descs.len());
    for desc in &meta.descs {
        if desc.size != 1 {
            continue;
        }
        let off = desc.offset;
        if off + 8 > data.len() {
            continue;
        }
        let val = u64::from_ne_bytes(data[off..off + 8].try_into().unwrap());
        map.insert(desc.name.clone(), val);
    }
    map
}

/// Per-vCPU stats reader. Holds the stats fd and pre-parsed metadata.
/// Opened before vCPUs move to threads; the stats fd holds a kernel
/// reference independent of VcpuFd ownership.
struct VcpuStatsReader {
    fd: OwnedFd,
    meta: StatsMeta,
}

impl VcpuStatsReader {
    /// Open a stats fd and parse metadata from the initial read.
    fn open<F: AsRawFd>(vcpu: &F) -> Option<Self> {
        let fd = get_stats_fd(vcpu)?;
        let buf = read_initial(fd.as_raw_fd())?;
        let meta = parse_meta_from_buf(&buf)?;
        Some(VcpuStatsReader { fd, meta })
    }

    /// Read current absolute stat values via pread.
    fn read_snapshot(&self) -> HashMap<String, u64> {
        let data = match pread_data(self.fd.as_raw_fd(), &self.meta) {
            Some(d) => d,
            None => return HashMap::new(),
        };
        extract_stats(&self.meta, &data)
    }
}

/// Holds pre-opened stats readers for all vCPUs. Opened before vCPUs
/// move to threads; read after VM exit to capture cumulative totals.
pub(crate) struct StatsContext {
    readers: Vec<VcpuStatsReader>,
}

impl StatsContext {
    /// Read absolute cumulative stats from each vCPU after VM exit.
    pub(crate) fn read_stats(&self) -> crate::vmm::KvmStatsTotals {
        let per_vcpu = self.readers.iter().map(|r| r.read_snapshot()).collect();
        crate::vmm::KvmStatsTotals { per_vcpu }
    }
}

/// Open stats fds for all vCPUs. Called before vCPUs move to threads.
/// Returns None if KVM_GET_STATS_FD is not supported.
pub(crate) fn open_stats_context(vcpus: &[kvm_ioctls::VcpuFd]) -> Option<StatsContext> {
    let mut readers = Vec::with_capacity(vcpus.len());
    for vcpu in vcpus {
        readers.push(VcpuStatsReader::open(vcpu)?);
    }
    Some(StatsContext { readers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::topology::Topology;
    use crate::vmm::x86_64::kvm::KtstrKvm;

    #[test]
    fn stats_fd_returns_some_on_modern_kernel() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        if let Some(fd) = get_stats_fd(&vm.vm_fd) {
            assert!(fd.as_raw_fd() >= 0);
        }
    }

    #[test]
    fn vcpu_stats_fd_returns_some() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        if let Some(fd) = get_stats_fd(&vm.vcpus[0]) {
            assert!(fd.as_raw_fd() >= 0);
        }
    }

    #[test]
    fn parse_meta_from_vcpu() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        if let Some(fd) = get_stats_fd(&vm.vcpus[0]) {
            let buf = read_initial(fd.as_raw_fd()).unwrap();
            let meta = parse_meta_from_buf(&buf).unwrap();
            assert!(!meta.descs.is_empty());
            assert!(meta.descs.iter().any(|d| d.name == "exits"));
        }
    }

    #[test]
    fn pread_data_after_initial_read() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        if let Some(fd) = get_stats_fd(&vm.vcpus[0]) {
            let buf = read_initial(fd.as_raw_fd()).unwrap();
            let meta = parse_meta_from_buf(&buf).unwrap();
            let data = pread_data(fd.as_raw_fd(), &meta);
            assert!(data.is_some(), "pread should work after initial read");
            assert!(!data.unwrap().is_empty());
        }
    }

    #[test]
    fn vcpu_reader_opens_and_reads() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        if let Some(reader) = VcpuStatsReader::open(&vm.vcpus[0]) {
            let snap = reader.read_snapshot();
            assert!(snap.contains_key("exits"), "should contain 'exits'");
        }
    }

    #[test]
    fn stats_context_opens_and_reads() {
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
        };
        let vm = KtstrKvm::new(topo, 64, false).unwrap();
        if let Some(ctx) = open_stats_context(&vm.vcpus) {
            assert_eq!(ctx.readers.len(), 2);
            let stats = ctx.read_stats();
            assert_eq!(stats.per_vcpu.len(), 2);
        }
    }

    #[test]
    fn interesting_stats_are_trust_relevant() {
        use crate::vmm::KVM_INTERESTING_STATS;
        assert!(KVM_INTERESTING_STATS.contains(&"exits"));
        assert!(KVM_INTERESTING_STATS.contains(&"halt_exits"));
        assert!(KVM_INTERESTING_STATS.contains(&"halt_successful_poll"));
        assert!(KVM_INTERESTING_STATS.contains(&"halt_attempted_poll"));
        assert!(KVM_INTERESTING_STATS.contains(&"halt_wait_ns"));
        assert!(KVM_INTERESTING_STATS.contains(&"signal_exits"));
        assert!(KVM_INTERESTING_STATS.contains(&"hypercalls"));
        assert!(KVM_INTERESTING_STATS.contains(&"preemption_reported"));
    }

    #[test]
    fn kvm_stats_totals_serde_roundtrip() {
        use crate::vmm::KvmStatsTotals;
        let mut totals = KvmStatsTotals::default();
        let mut m = HashMap::new();
        m.insert("exits".to_string(), 1000u64);
        m.insert("halt_exits".to_string(), 200u64);
        totals.per_vcpu = vec![m];
        let json = serde_json::to_string(&totals).unwrap();
        let loaded: KvmStatsTotals = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.per_vcpu.len(), 1);
        assert_eq!(loaded.avg("exits"), 1000);
        assert_eq!(loaded.avg("halt_exits"), 200);
    }

    #[test]
    fn sum_and_avg() {
        use crate::vmm::KvmStatsTotals;
        let mut d = KvmStatsTotals::default();
        let mut m1 = HashMap::new();
        m1.insert("exits".to_string(), 100u64);
        let mut m2 = HashMap::new();
        m2.insert("exits".to_string(), 200u64);
        d.per_vcpu = vec![m1, m2];
        assert_eq!(d.sum("exits"), 300);
        assert_eq!(d.avg("exits"), 150);
        assert_eq!(d.sum("nonexistent"), 0);
        assert_eq!(d.avg("nonexistent"), 0);
    }

    #[test]
    fn avg_empty() {
        use crate::vmm::KvmStatsTotals;
        let d = KvmStatsTotals::default();
        assert_eq!(d.avg("exits"), 0);
    }
}
