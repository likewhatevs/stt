use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::workload::{WorkType, WorkerReport};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

/// MAP_SHARED futex word. All workers forked from the same parent
/// inherit this mapping and contend on the same physical page.
static mut FUTEX_PTR: *mut u32 = std::ptr::null_mut();

/// Allocate the shared futex word before forking workers.
/// Must be called exactly once from the parent process.
fn init_shared_futex() {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            std::mem::size_of::<u32>(),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED, "mmap failed for shared futex");
    unsafe {
        std::ptr::write_bytes(ptr as *mut u8, 0, std::mem::size_of::<u32>());
        FUTEX_PTR = ptr as *mut u32;
    }
}

/// Combined lock-holding + page-faulting workload.
///
/// Each iteration: spin work, then CAS acquire a shared futex mutex
/// (FUTEX_WAIT on contention), touch random cold pages while holding
/// the lock, then atomic Release store + FUTEX_WAKE(1). When the lock
/// holder is preempted during page faults, all contenders stall — the
/// cascade that caused PostgreSQL regressions under preemption-heavy
/// schedulers. Neither MutexContention (no memory pressure under lock)
/// nor PageFaultChurn (no contention) alone reproduces this interaction.
fn fault_under_lock(stop: &AtomicBool) -> WorkerReport {
    let tid: libc::pid_t = unsafe { libc::getpid() };
    let start = Instant::now();
    let mut work_units = 0u64;
    let mut iterations = 0u64;

    let futex_ptr = unsafe { FUTEX_PTR };
    if futex_ptr.is_null() {
        return zeroed_report(tid, start);
    }
    let atom = unsafe { &*(futex_ptr as *const AtomicU32) };

    let region_size: usize = 256 * 1024; // 256 KB
    let page_count = region_size / 4096;
    let touches_per_hold = 32usize;

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            region_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return zeroed_report(tid, start);
    }
    unsafe {
        libc::madvise(ptr, region_size, libc::MADV_NOHUGEPAGE);
    }

    let mut rng_state = (tid as u64) | 1;
    let xorshift64 = |state: &mut u64| -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    };

    while !stop.load(Ordering::Relaxed) {
        // Spin work between lock acquisitions.
        for _ in 0..256u64 {
            work_units = std::hint::black_box(work_units.wrapping_add(1));
            std::hint::spin_loop();
        }

        // Acquire: CAS 0 -> 1, FUTEX_WAIT on contention.
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if atom
                .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            let ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 100_000_000, // 100ms
            };
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    futex_ptr,
                    libc::FUTEX_WAIT,
                    1u32,
                    &ts as *const libc::timespec,
                    std::ptr::null::<u32>(),
                    0u32,
                );
            }
        }

        // Critical section: fault cold pages while holding the lock.
        for _ in 0..touches_per_hold {
            let page_idx = (xorshift64(&mut rng_state) as usize) % page_count;
            let page_ptr = unsafe { (ptr as *mut u8).add(page_idx * 4096) };
            unsafe { std::ptr::write_volatile(page_ptr, 1u8) };
            work_units = work_units.wrapping_add(1);
        }

        // Release: atomic store + FUTEX_WAKE(1).
        atom.store(0, Ordering::Release);
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                futex_ptr,
                libc::FUTEX_WAKE,
                1,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<u32>(),
                0u32,
            );
        }

        // Zap PTEs so next iteration faults again.
        unsafe {
            libc::madvise(ptr, region_size, libc::MADV_DONTNEED);
        }

        iterations += 1;
    }

    unsafe {
        libc::munmap(ptr, region_size);
    }

    let wall_time_ns = start.elapsed().as_nanos() as u64;
    WorkerReport {
        tid,
        work_units,
        cpu_time_ns: 0,
        wall_time_ns,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        wake_latencies_ns: vec![],
        iterations,
        schedstat_run_delay_ns: 0,
        schedstat_ctx_switches: 0,
        schedstat_cpu_time_ns: 0,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
    }
}

fn zeroed_report(tid: libc::pid_t, start: Instant) -> WorkerReport {
    WorkerReport {
        tid,
        work_units: 0,
        cpu_time_ns: 0,
        wall_time_ns: start.elapsed().as_nanos() as u64,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        wake_latencies_ns: vec![],
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_ctx_switches: 0,
        schedstat_cpu_time_ns: 0,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
    }
}

/// Reproduces the preemption-under-lock regression pattern observed in
/// PostgreSQL workloads. Multiple cgroups run the combined fault+lock
/// workload alongside pure CPU workers competing for the same CPUs.
///
/// To compare across kernel versions: run this test on both kernels,
/// compare `total_iterations` and `schedstat_run_delay_ns` from the
/// worker reports. A regression shows as lower throughput and higher
/// run delay on the affected kernel.
#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn preempt_regression_fault_under_load(ctx: &Ctx) -> Result<AssertResult> {
    init_shared_futex();
    let fault_lock_wt = WorkType::custom("fault_under_lock", fault_under_lock);

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("fault_lock_workers")
                .workers(4)
                .work_type(fault_lock_wt),
            CgroupDef::named("cpu_contenders")
                .workers(4)
                .work_type(WorkType::CpuSpin),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}
