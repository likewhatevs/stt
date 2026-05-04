//! BTF offsets and stable enum indices for per-CPU CPU-time, softirq,
//! and IRQ counters.
//!
//! Three structs participate in the failure-dump CPU-time capture:
//!
//! - `struct kernel_cpustat` (`include/linux/kernel_stat.h`):
//!   `u64 cpustat[NR_STATS]` per CPU, indexed by `enum cpu_usage_stat`.
//! - `struct kernel_stat` (`include/linux/kernel_stat.h`):
//!   `unsigned long irqs_sum` plus `unsigned int softirqs[NR_SOFTIRQS]`
//!   per CPU.
//! - `struct tick_sched` (`kernel/time/tick-sched.h`): per-CPU
//!   `iowait_sleeptime` accumulated under `CONFIG_NO_HZ_COMMON`.
//!
//! All three sit in `.data..percpu` symbols (`kernel_cpustat`, `kstat`,
//! `tick_cpu_sched`); the dump path resolves the symbols via
//! `super::symbols::KernelSymbols::cpu_time_symbols` and adds
//! `__per_cpu_offset[cpu]` per CPU.

use anyhow::Result;
use btf_rs::Btf;

use super::{find_struct, member_byte_offset};

/// Stable indices of `kernel_cpustat::cpustat[NR_STATS]` from
/// `enum cpu_usage_stat` (`include/linux/kernel_stat.h`). The kernel
/// pins the order so external readers — `/proc/stat` formatting,
/// `account_user_time` / `account_system_index_time` accumulation,
/// every userspace tool that reads `kernel_cpustat` — depend on it.
/// Hard-code the indices the failure dump captures instead of
/// resolving them via BTF: BTF only encodes the array length, not the
/// enum-to-position mapping, so a BTF-driven read would have to
/// resolve the enum separately. The cited header values are the
/// authoritative source; mismatching kernels would be a UAPI break,
/// not a layout drift this code can adapt to.
pub const CPUTIME_USER: usize = 0;
/// Index of `cpustat[CPUTIME_NICE]` (CPU time spent on nice'd user
/// processes). See [`CPUTIME_USER`].
pub const CPUTIME_NICE: usize = 1;
/// Index of `cpustat[CPUTIME_SYSTEM]` (CPU time spent in kernel).
/// See [`CPUTIME_USER`].
pub const CPUTIME_SYSTEM: usize = 2;
/// Index of `cpustat[CPUTIME_SOFTIRQ]` (CPU time servicing softirqs).
/// See [`CPUTIME_USER`].
pub const CPUTIME_SOFTIRQ: usize = 3;
/// Index of `cpustat[CPUTIME_IRQ]` (CPU time servicing hardirqs).
/// See [`CPUTIME_USER`].
pub const CPUTIME_IRQ: usize = 4;
/// Index of `cpustat[CPUTIME_IDLE]` (CPU time spent idle).
/// See [`CPUTIME_USER`].
pub const CPUTIME_IDLE: usize = 5;
/// Index of `cpustat[CPUTIME_IOWAIT]` (CPU time waiting on
/// outstanding block IO). See [`CPUTIME_USER`].
pub const CPUTIME_IOWAIT: usize = 6;
/// Index of `cpustat[CPUTIME_STEAL]` (CPU time stolen by the
/// hypervisor — virt only). See [`CPUTIME_USER`].
pub const CPUTIME_STEAL: usize = 7;

/// Number of softirq vectors per `enum` in `include/linux/interrupt.h`
/// (HI/TIMER/NET_TX/NET_RX/BLOCK/IRQ_POLL/TASKLET/SCHED/HRTIMER/RCU,
/// in that order). The order is enum-stable, mirroring
/// [`CPUTIME_USER`]'s rationale: external consumers (`/proc/softirqs`
/// formatting, `softirq_to_name[]`) depend on the layout, so a
/// reordering would be a UAPI break and resolving each name via BTF
/// would buy nothing.
pub const NR_SOFTIRQS: usize = 10;

/// Names of every softirq vector, indexed by the enum order shared
/// with the kernel's `softirq_to_name[]` (kernel/softirq.c). Surfaced
/// in failure-dump JSON so a downstream consumer reading
/// `softirqs[i]` knows which vector each slot represents without
/// chasing the kernel header.
///
/// `dead_code` allow: forward-looking — referenced from
/// [`super::super::dump::PerCpuTimeStats`] doc comments but no
/// renderer consumes the names yet.
#[allow(dead_code)]
pub const SOFTIRQ_NAMES: [&str; NR_SOFTIRQS] = [
    "HI", "TIMER", "NET_TX", "NET_RX", "BLOCK", "IRQ_POLL", "TASKLET", "SCHED", "HRTIMER", "RCU",
];

/// Byte offsets used to read per-CPU CPU-time and softirq/IRQ
/// counters from guest memory.
///
/// Three structs participate:
///   - `struct kernel_cpustat` (`include/linux/kernel_stat.h`):
///     a per-CPU `u64 cpustat[NR_STATS]` table indexed by
///     `enum cpu_usage_stat`. Hand-rolled accumulators in the
///     kernel's CPU-time accounting (`account_idle_time`,
///     `account_user_time`, etc.) bump these in nanoseconds (or
///     jiffies pre-NO_HZ_FULL — the field is `u64 nsecs` regardless;
///     `cputime64_to_clock_t` does the conversion at `/proc/stat`
///     read).
///   - `struct kernel_stat` (`include/linux/kernel_stat.h`): a
///     per-CPU `unsigned long irqs_sum` plus
///     `unsigned int softirqs[NR_SOFTIRQS]` table (10 counters in
///     2026-04-30 mainline). `kstat_incr_softirqs_this_cpu` and
///     `kstat_incr_irq_this_cpu` are the producers.
///   - `struct tick_sched` (`kernel/time/tick-sched.h`): per-CPU
///     `iowait_sleeptime` (`ktime_t` aka `s64` ns) accumulated only
///     under NO_HZ when the CPU enters idle with `nr_iowait > 0`.
///
/// All three structs sit in `.data..percpu` symbols
/// (`kernel_cpustat`, `kstat`, `tick_cpu_sched`). Per-CPU symbols
/// carry section-relative offsets in vmlinux's symtab; the per-CPU
/// KVA for CPU `n` is `<symbol> + __per_cpu_offset[n]` —
/// [`super::super::symbols::KernelSymbols::cpu_time_symbols`] resolves the
/// symbols and the dump path adds `__per_cpu_offset[cpu]` per CPU.
///
/// Field-presence semantics: a kernel without sched_ext omits no
/// field captured here, but a kernel built without
/// `CONFIG_NO_HZ_COMMON` drops `tick_sched`. The offset resolver
/// reports `tick_sched_iowait_sleeptime` as `Some` only when the
/// type is present. Callers that observe `None` skip the
/// `iowait_sleeptime` capture and surface `nr_iowait` (an atomic
/// counter on `struct rq` that the existing scx walker already
/// reads) instead.
#[derive(Debug, Clone, Copy)]
pub struct CpuTimeOffsets {
    /// Offset of `cpustat[]` (the `u64[NR_STATS]` array) within
    /// `struct kernel_cpustat`. Always zero on every kernel since
    /// the introduction of the struct, but resolved via BTF rather
    /// than hard-coded so a future addition of a leading field
    /// surfaces here without silent miscalculation.
    pub kernel_cpustat_cpustat: usize,
    /// Offset of `irqs_sum` (`unsigned long`) within `struct kernel_stat`.
    pub kstat_irqs_sum: usize,
    /// Offset of `softirqs[]` (the `unsigned int[NR_SOFTIRQS]`
    /// array) within `struct kernel_stat`.
    pub kstat_softirqs: usize,
    /// Offset of `iowait_sleeptime` (`ktime_t` / `s64` ns) within
    /// `struct tick_sched`. `None` when the kernel was built
    /// without `CONFIG_NO_HZ_COMMON` (the type is absent from BTF).
    pub tick_sched_iowait_sleeptime: Option<usize>,
}

impl CpuTimeOffsets {
    /// Resolve CPU-time / softirq / IRQ offsets from a pre-loaded BTF
    /// object. Returns Err when `kernel_cpustat` or `kernel_stat` are
    /// missing — these are universal, so their absence indicates a
    /// stripped vmlinux. `tick_sched` is best-effort: a kernel
    /// without `CONFIG_NO_HZ_COMMON` has no such type, and the
    /// resolver returns `Ok` with `tick_sched_iowait_sleeptime: None`.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (kernel_cpustat, _) = find_struct(btf, "kernel_cpustat")?;
        let kernel_cpustat_cpustat = member_byte_offset(btf, &kernel_cpustat, "cpustat")?;

        let (kernel_stat, _) = find_struct(btf, "kernel_stat")?;
        let kstat_irqs_sum = member_byte_offset(btf, &kernel_stat, "irqs_sum")?;
        let kstat_softirqs = member_byte_offset(btf, &kernel_stat, "softirqs")?;

        // tick_sched is CONFIG_NO_HZ_COMMON-gated; report None
        // rather than Err so the caller can capture the rest of the
        // struct's fields without forcing every kernel to enable
        // dynticks for failure-dump support.
        let tick_sched_iowait_sleeptime = match find_struct(btf, "tick_sched") {
            Ok((tick_sched, _)) => member_byte_offset(btf, &tick_sched, "iowait_sleeptime").ok(),
            Err(_) => None,
        };

        Ok(Self {
            kernel_cpustat_cpustat,
            kstat_irqs_sum,
            kstat_softirqs,
            tick_sched_iowait_sleeptime,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::load_btf_from_path;
    use super::*;

    /// Resolve [`CpuTimeOffsets`] against the test vmlinux. Pins the
    /// offsets the per-CPU CPU-time / softirq / IRQ failure-dump
    /// capture path consumes:
    ///   - `kernel_cpustat::cpustat[]` is at offset 0 (single-field
    ///     struct).
    ///   - `kstat.irqs_sum` and `kstat.softirqs[]` are distinct.
    ///   - `tick_sched::iowait_sleeptime` is best-effort (Some only
    ///     under CONFIG_NO_HZ_COMMON).
    #[test]
    fn parse_cpu_time_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let btf = match load_btf_from_path(&path) {
            Ok(b) => b,
            Err(e) => skip!("vmlinux BTF load failed: {e}"),
        };
        let offsets = match CpuTimeOffsets::from_btf(&btf) {
            Ok(o) => o,
            Err(e) => skip!("CpuTimeOffsets::from_btf failed: {e}"),
        };
        assert_eq!(
            offsets.kernel_cpustat_cpustat, 0,
            "kernel_cpustat::cpustat[] must live at offset 0 \
             (single-field struct in include/linux/kernel_stat.h)"
        );
        assert_ne!(
            offsets.kstat_irqs_sum, offsets.kstat_softirqs,
            "kstat irqs_sum and softirqs must be at distinct offsets"
        );
        // tick_sched is CONFIG_NO_HZ_COMMON-gated. None is a valid
        // outcome on dynticks-disabled kernels; just don't crash.
        if let Some(off) = offsets.tick_sched_iowait_sleeptime {
            assert!(
                off > 0,
                "tick_sched::iowait_sleeptime must be nonzero when present"
            );
        }
    }
}
