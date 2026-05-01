//! Host-side per-vCPU hardware performance counters via
//! `perf_event_open(2)` with `exclude_host=1`.
//!
//! The PMU is a partitioned resource: when a counter is configured with
//! `exclude_host=1`, the kernel toggles the EVENTSEL_ENABLE bit on/off
//! at every guest entry and exit so the counter only ticks while the
//! vCPU thread is executing inside the guest. Verified at
//! `arch/x86/events/intel/core.c:5080` (`if (event->attr.exclude_host)
//! arr[idx].host &= ~ARCH_PERFMON_EVENTSEL_ENABLE;`) and
//! `arch/x86/events/intel/core.c:5090-5093` (`core_pmu_enable_event`
//! skips enable when `exclude_host` is set). KVM's
//! `intel_guest_get_msrs` is what flips the bit at VMENTER/VMEXIT.
//!
//! Practical use: combined with per-CPU schedstat (run_delay,
//! sched_count, ttwu_count) this distinguishes the three "vCPU is
//! consuming time" failure modes that look identical from the
//! host-stat view alone:
//!   - IPC ≈ 0 → guest is halted / idle (CPU not really busy).
//!   - IPC ≈ 0.5 → guest is spinning (lock contention, `cpu_relax`).
//!   - IPC ≥ 1 → guest is doing productive work.
//!
//! Cost is zero in-guest: hardware PMU counters are partitioned at the
//! VMCS/VMCB level. The host-side cost is one read per (vCPU, counter)
//! per monitor tick — an `lseek+read` syscall pair returning 24 bytes.
//!
//! Per-fd `read()` returns three u64s when
//! `PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING`
//! is set in `read_format` (see `tools/include/uapi/linux/perf_event.h`
//! `enum perf_event_read_format`). When `time_enabled > time_running`
//! the kernel multiplexed the counter; the consumer scales raw counts
//! by `time_enabled / time_running` to recover the "what would have
//! been counted if pinned" estimate. We store all three values raw —
//! scaling is the consumer's choice.

#![cfg(target_os = "linux")]

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use perf_event_open_sys as pes;
use pes::bindings::{
    PERF_COUNT_HW_BRANCH_MISSES, PERF_COUNT_HW_CACHE_MISSES, PERF_COUNT_HW_CPU_CYCLES,
    PERF_COUNT_HW_INSTRUCTIONS, PERF_FORMAT_TOTAL_TIME_ENABLED, PERF_FORMAT_TOTAL_TIME_RUNNING,
    PERF_TYPE_HARDWARE, perf_event_attr,
};

/// One per-vCPU sample for the four hardware counters. Fields are raw
/// values as returned by `read(fd, ...)`; scaling against
/// `time_enabled` / `time_running` is the consumer's choice (see the
/// module-level doc on multiplexing).
///
/// `time_enabled` and `time_running` are per-counter-set globals: when
/// reading multiple events that share the same vCPU+task target, the
/// kernel reports one pair across all four reads on the same monitor
/// tick — but we capture per-counter to avoid a fragile assumption
/// about kernel grouping. Consumers that scale should pick one
/// (typically `cycles.{time_enabled,time_running}`) and apply
/// uniformly.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct VcpuPerfSample {
    /// `PERF_COUNT_HW_CPU_CYCLES` raw count.
    pub cycles: u64,
    /// `PERF_COUNT_HW_INSTRUCTIONS` raw count.
    pub instructions: u64,
    /// `PERF_COUNT_HW_CACHE_MISSES` raw count.
    pub cache_misses: u64,
    /// `PERF_COUNT_HW_BRANCH_MISSES` raw count.
    pub branch_misses: u64,
    /// Wall-clock ns the kernel allocated to this counter (sum across
    /// all counters; reflects how long the event was attached to the
    /// task). When equal to `time_running` the counter was never
    /// multiplexed.
    pub time_enabled_ns: u64,
    /// ns the counter was actually scheduled on a hardware PMU slot.
    /// `< time_enabled_ns` indicates multiplexing — multiply raw
    /// counts by `time_enabled_ns / time_running_ns` to recover the
    /// scaled estimate.
    pub time_running_ns: u64,
}

impl VcpuPerfSample {
    /// `instructions / cycles`, the standard "instructions per cycle"
    /// metric. `0.0` when `cycles == 0` (counter wasn't running or
    /// guest never executed). The caller decides whether to scale by
    /// `time_enabled / time_running` first; for the IPC ratio scaling
    /// is unnecessary because both numerator and denominator scale by
    /// the same factor.
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.instructions as f64 / self.cycles as f64
        }
    }
}

/// Four `perf_event_open` file descriptors targeting one vCPU thread
/// (cycles, instructions, cache-misses, branch-misses) — all
/// configured with `exclude_host=1` so they only count while the
/// vCPU is executing inside the guest.
///
/// `Drop` closes the four fds via `OwnedFd`.
pub struct VcpuPerfCounters {
    cycles: OwnedFd,
    instructions: OwnedFd,
    cache_misses: OwnedFd,
    branch_misses: OwnedFd,
}

impl VcpuPerfCounters {
    /// Open the four counters bound to `tid` (Linux thread ID — the
    /// value `gettid()` returns inside the vCPU thread, NOT a
    /// `pthread_t`). `cpu = -1` lets the counter follow the thread
    /// across host CPUs.
    ///
    /// Returns `Err` when `perf_event_open(2)` fails on any of the
    /// four counters. The first error short-circuits — already-opened
    /// fds are dropped/closed by `OwnedFd`'s `Drop`. Common failure
    /// modes: `EACCES` (perf_event_paranoid too high or
    /// CAP_PERFMON missing), `ENODEV` (counter not available on this
    /// hardware), `ESRCH` (`tid` no longer exists), `EOPNOTSUPP`
    /// (running under a hypervisor that virtualizes the PMU
    /// differently — `exclude_host` may be rejected when KVM is the
    /// guest hypervisor). Caller treats `Err` as "perf data
    /// unavailable for this vCPU" and continues without it.
    pub fn open(tid: libc::pid_t) -> io::Result<Self> {
        let cycles = open_one(tid, PERF_COUNT_HW_CPU_CYCLES as u64)?;
        let instructions = open_one(tid, PERF_COUNT_HW_INSTRUCTIONS as u64)?;
        let cache_misses = open_one(tid, PERF_COUNT_HW_CACHE_MISSES as u64)?;
        let branch_misses = open_one(tid, PERF_COUNT_HW_BRANCH_MISSES as u64)?;
        Ok(Self {
            cycles,
            instructions,
            cache_misses,
            branch_misses,
        })
    }

    /// Read all four counters into a single [`VcpuPerfSample`].
    /// Returns `Err` only if a `read(2)` syscall fails (kernel
    /// shouldn't error on a properly opened perf fd; surfaced for
    /// diagnostic visibility). Each `read` returns 24 bytes:
    /// `[value, time_enabled, time_running]` as native-endian u64s
    /// because `read_format` was set to
    /// `PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING`.
    /// A short read (less than 24 bytes) is treated as failure —
    /// the kernel always returns the full record or fails.
    pub fn read(&self) -> io::Result<VcpuPerfSample> {
        let (cycles, time_enabled_ns, time_running_ns) = read_one(&self.cycles)?;
        let (instructions, _, _) = read_one(&self.instructions)?;
        let (cache_misses, _, _) = read_one(&self.cache_misses)?;
        let (branch_misses, _, _) = read_one(&self.branch_misses)?;
        Ok(VcpuPerfSample {
            cycles,
            instructions,
            cache_misses,
            branch_misses,
            time_enabled_ns,
            time_running_ns,
        })
    }
}

fn open_one(tid: libc::pid_t, config: u64) -> io::Result<OwnedFd> {
    let mut attr = perf_event_attr::default();
    attr.size = std::mem::size_of::<perf_event_attr>() as u32;
    attr.type_ = PERF_TYPE_HARDWARE;
    attr.config = config;
    attr.read_format =
        (PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING) as u64;
    // disabled=0 — start counting immediately on open. We don't gate
    // with PERF_EVENT_IOC_ENABLE because the monitor opens these on
    // a steady-state vCPU that is already running.
    attr.set_disabled(0);
    // exclude_host=1 — see module-level doc. The verified
    // arch/x86/events/intel/core.c paths zero EVENTSEL_ENABLE during
    // host execution and KVM toggles it on guest entry/exit.
    attr.set_exclude_host(1);
    // exclude_user=0, exclude_kernel=0 — count both guest userspace
    // and guest kernel. Together with exclude_host=1 this means
    // "everything that runs inside the VM, regardless of CPL".
    attr.set_exclude_user(0);
    attr.set_exclude_kernel(0);
    // exclude_hv=0 — there is no hypervisor below us in the typical
    // baremetal-host case; setting this would be wrong on
    // platforms where the host kernel itself runs as a guest of a
    // higher hypervisor (rare on the dev/CI hosts ktstr targets).
    attr.set_exclude_hv(0);
    // pinned=0 — share PMU slots with other tools (perf record,
    // bpftop). Multiplexing is signaled via time_enabled vs
    // time_running and the consumer scales accordingly.
    attr.set_pinned(0);
    // SAFETY: `attr` is a valid initialized perf_event_attr; the
    // syscall reads it as a `*const`. `cpu = -1` means "follow the
    // thread"; `group_fd = -1` makes this a standalone leader.
    // Returning fd is checked below.
    let fd = unsafe { pes::perf_event_open(&mut attr, tid, -1, -1, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the kernel returned a valid fd; we own it from now on.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Read one perf fd into `(value, time_enabled, time_running)`.
fn read_one(fd: &OwnedFd) -> io::Result<(u64, u64, u64)> {
    let mut buf = [0u64; 3];
    // SAFETY: buf is a valid 24-byte writable region; the kernel
    // writes the read_format record into it. read(2) returns the
    // number of bytes; we check below.
    let n = unsafe {
        libc::read(
            fd.as_raw_fd(),
            buf.as_mut_ptr() as *mut libc::c_void,
            std::mem::size_of_val(&buf),
        )
    };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    if (n as usize) < std::mem::size_of_val(&buf) {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "short read on perf fd: got {n} bytes, expected {}",
                std::mem::size_of_val(&buf)
            ),
        ));
    }
    Ok((buf[0], buf[1], buf[2]))
}

/// All-vCPU perf counter capture. The monitor thread opens one
/// [`VcpuPerfCounters`] per vCPU TID at startup, then reads all of
/// them on every sample. `Drop` closes every fd through `OwnedFd`.
pub struct PerfCountersCapture {
    /// One counter set per vCPU. Index = vCPU id.
    pub per_vcpu: Vec<VcpuPerfCounters>,
}

impl PerfCountersCapture {
    /// Open counters for every TID in `tids`. `tids[i]` is the
    /// Linux TID of vCPU `i` (BSP or AP). On any open failure the
    /// already-opened fds are closed by `OwnedFd`'s `Drop` as the
    /// partial vec drops.
    ///
    /// Caller decides what to do on partial failure. The current
    /// monitor integration falls back to "no perf data" wholesale —
    /// individual per-vCPU error tracking would clutter the
    /// `CpuSnapshot.perf` field's "all-or-nothing per sample"
    /// invariant.
    pub fn open(tids: &[libc::pid_t]) -> io::Result<Self> {
        let mut per_vcpu = Vec::with_capacity(tids.len());
        for &tid in tids {
            per_vcpu.push(VcpuPerfCounters::open(tid)?);
        }
        Ok(Self { per_vcpu })
    }

    /// Read every vCPU's counter set into a `Vec<VcpuPerfSample>`
    /// indexed by vCPU. Read errors propagate as `Err`; the caller
    /// drops the whole sample rather than returning a partial
    /// vec — a half-populated vec would produce misleading deltas
    /// in any timeline analysis.
    pub fn read_all(&self) -> io::Result<Vec<VcpuPerfSample>> {
        self.per_vcpu.iter().map(|p| p.read()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `perf_event_open(2)` requires either `CAP_PERFMON` or
    /// `kernel.perf_event_paranoid <= 2` for hardware counters bound
    /// to a non-self thread. The CI runner always runs as root so
    /// neither gate fires; in this unit test we exercise the
    /// caller-side error propagation by opening against the current
    /// thread's tid (which always has permission to monitor itself
    /// regardless of paranoid level).
    #[test]
    fn open_self_then_read_returns_consistent_fields() {
        // Read this thread's tid via the gettid syscall — libc
        // versions older than 0.2.156 don't expose `libc::gettid` as
        // a stable function on every target, so use the syscall
        // wrapper directly.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        let counters = match VcpuPerfCounters::open(tid) {
            Ok(c) => c,
            Err(e) => {
                // Skip when the kernel rejected the open — covers
                // CI containers that disable perf_event_open or
                // kernels missing the requested counter (e.g. some
                // VM-on-VM nesting drops PERF_COUNT_HW_BRANCH_MISSES).
                eprintln!("perf_event_open unavailable in this env: {e}; skipping");
                return;
            }
        };
        // Burn a few hundred million instructions so cycles and
        // instructions are nonzero; the loop body is NOT optimized
        // away by `volatile` reads.
        let mut acc: u64 = 0;
        for i in 0u64..1_000_000 {
            unsafe { std::ptr::read_volatile(&i) };
            acc = acc.wrapping_add(i);
        }
        std::hint::black_box(acc);

        let sample = counters.read().expect("read perf counters");
        // exclude_host=1 means *zero* host-side counts — when this
        // test runs on baremetal (not inside a VM), every counter
        // should be 0 because no guest entry happened. We verify
        // the structural invariants instead: each counter read
        // succeeded, time_running <= time_enabled, and the values
        // are <= u64::MAX (trivially true; here for documentation).
        assert!(
            sample.time_running_ns <= sample.time_enabled_ns,
            "time_running ({}) > time_enabled ({})",
            sample.time_running_ns,
            sample.time_enabled_ns,
        );
        // Cycles fit in u63 in any sane scenario — guard against an
        // accidental sign-extension bug in the read path.
        assert!(sample.cycles < (1u64 << 63));
    }

    #[test]
    fn ipc_zero_when_cycles_zero() {
        let s = VcpuPerfSample::default();
        assert_eq!(s.ipc(), 0.0);
    }

    #[test]
    fn ipc_computes_instructions_over_cycles() {
        let s = VcpuPerfSample {
            cycles: 200,
            instructions: 100,
            ..Default::default()
        };
        assert!((s.ipc() - 0.5).abs() < 1e-9);
    }

    // -- Verdict API integration coverage -------------------------------
    //
    // The three IPC-band classifications named in the module-level doc
    // (≈0 = halted, ≈0.5 = spinning, ≥1 = productive) become Verdict
    // claims when a scenario test asserts on a captured VcpuPerfSample.
    // Pin the integration shape so a future change to `ipc()`'s
    // numerator-denominator order or the f64 precision routing does
    // not silently produce passing claims that should fail.

    /// Authorial claim: a productive vCPU has IPC ≥ 1.0 — pin the
    /// claim! macro routing through ClaimBuilder<f64>::at_least and
    /// the Verdict accumulator. Failing-branch siblings cover the
    /// idle and spinning cases.
    #[test]
    fn vcpu_perf_sample_ipc_productive_claim_passes() {
        use crate::assert::Verdict;
        let s = VcpuPerfSample {
            cycles: 1_000,
            instructions: 1_500,
            ..Default::default()
        };
        let mut v = Verdict::new();
        let ipc = s.ipc();
        crate::claim!(v, ipc).at_least(1.0);
        crate::claim!(v, ipc).is_finite();
        let r = v.into_result();
        assert!(
            r.passed,
            "productive IPC=1.5 must satisfy at_least(1.0): {:?}",
            r.details,
        );
    }

    /// Idle / halted vCPU: cycles=0 produces ipc()=0.0 by the
    /// guard. A claim demanding ipc ≥ 1.0 must fail with a labeled
    /// detail naming the threshold.
    #[test]
    fn vcpu_perf_sample_idle_ipc_fails_productive_claim() {
        use crate::assert::Verdict;
        let s = VcpuPerfSample::default();
        let ipc = s.ipc();
        let mut v = Verdict::new();
        crate::claim!(v, ipc).at_least(1.0);
        let r = v.into_result();
        assert!(!r.passed, "idle vCPU's ipc=0 must fail at_least(1.0)");
        let msg = &r.details[0].message;
        assert!(msg.contains("at least 1"), "msg must name threshold: {msg}");
        assert!(msg.contains("ipc"), "msg must include the label: {msg}");
    }

    /// Multiplexed counter: time_running < time_enabled means the
    /// PMU was multiplexed. A scenario test that asserts the
    /// counter wasn't multiplexed can claim
    /// `time_enabled == time_running` via `claim!(eq)`. Pin the
    /// failure path — multiplexing surfaces with the observed and
    /// expected values both visible.
    #[test]
    fn vcpu_perf_sample_multiplex_detect_via_eq_claim() {
        use crate::assert::Verdict;
        let s = VcpuPerfSample {
            cycles: 500,
            instructions: 800,
            cache_misses: 10,
            branch_misses: 2,
            time_enabled_ns: 1_000_000,
            time_running_ns: 600_000,
        };
        let mut v = Verdict::new();
        crate::claim!(v, s.time_running_ns).eq(s.time_enabled_ns);
        let r = v.into_result();
        assert!(!r.passed, "multiplexed sample must fail eq claim");
        let msg = &r.details[0].message;
        assert!(
            msg.contains("expected 1000000"),
            "msg must reflect expected value: {msg}",
        );
        assert!(
            msg.contains("was 600000"),
            "msg must reflect observed value: {msg}",
        );
    }
}
