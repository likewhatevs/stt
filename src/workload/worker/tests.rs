//! Worker-side tests: worker_main helpers (clock, schedstat, IO
//! backings, set_sched_policy, reservoir_push, matrix_multiply,
//! resolve_alu_width). Co-located with their production code in
//! `worker/mod.rs`.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::spawn::*;
use super::super::types::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// `clock_gettime_ns(CLOCK_MONOTONIC)` must never observe time
/// moving backwards between two sequential calls on the same
/// thread. Pins the non-decreasing contract the wake-latency
/// reservoirs depend on: the messenger stamps `wake_ns` into
/// shared memory and the worker subtracts to compute
/// `now_ns - wake_ns`; a backward step would saturate to zero
/// in the subtractor and silently discard a valid sample, or
/// (without the saturator) wrap to `u64::MAX`.
///
/// A 2-sample test would miss a backward step that only
/// appears under load; the 1000-sample tight loop here burns
/// a few microseconds of CPU and catches any regression that
/// makes the clock non-monotonic under reasonable contention
/// (timer drift on a virtualised guest, or a helper swap
/// from `CLOCK_MONOTONIC` to `CLOCK_REALTIME` which is NOT
/// monotonic). Every adjacent pair in the 999-element diff
/// list is checked for non-decreasing order so a mid-run
/// regression is localised to the offending index, not just
/// "some pair somewhere".
#[test]
fn clock_gettime_ns_monotonic_non_decreasing() {
    const N: usize = 1000;
    let samples: Vec<u64> = (0..N)
        .map(|i| {
            clock_gettime_ns(libc::CLOCK_MONOTONIC).unwrap_or_else(|| {
                panic!(
                    "CLOCK_MONOTONIC must be readable on any Linux host; \
                     sample {i}/{N} returned None"
                )
            })
        })
        .collect();
    for i in 1..N {
        assert!(
            samples[i] >= samples[i - 1],
            "CLOCK_MONOTONIC went backwards at sample {i}: \
             prev={prev} curr={curr} (delta={delta})",
            prev = samples[i - 1],
            curr = samples[i],
            delta = samples[i - 1] - samples[i],
        );
    }
}
// -- matrix_multiply --

#[test]
fn matrix_multiply_1x1_produces_product() {
    // Size=1: A=[a], B=[b], expected C=[a*b]. The `black_box` calls
    // prevent constant folding, so the test directly exercises the
    // wrapping_mul path without any compiler optimization eating
    // the multiplication.
    let mut data = vec![0u64; 3];
    data[0] = 3; // A
    data[1] = 5; // B
    let mut work_units = 0u64;
    matrix_multiply(&mut data, 1, &mut work_units);
    assert_eq!(data[2], 15, "C = A * B for 1x1 matrix");
    // Read-back sink consumed C[0] (= 15) into work_units.
    assert_eq!(work_units, 15, "post-loop sink folds C[0] into work_units");
}
#[test]
fn matrix_multiply_2x2_against_reference() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = A * B = [[19, 22], [43, 50]]
    let size = 2;
    let stride = size * size;
    let mut data = vec![0u64; 3 * stride];
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    data[3] = 4;
    data[stride] = 5;
    data[stride + 1] = 6;
    data[stride + 2] = 7;
    data[stride + 3] = 8;
    let mut work_units = 0u64;
    matrix_multiply(&mut data, size, &mut work_units);
    assert_eq!(data[2 * stride], 19);
    assert_eq!(data[2 * stride + 1], 22);
    assert_eq!(data[2 * stride + 2], 43);
    assert_eq!(data[2 * stride + 3], 50);
}
#[test]
fn matrix_multiply_3x3_diagonal() {
    // Identity-like: A = diag(2, 3, 5), B = diag(1, 1, 1) = I.
    // Expected C = A = diag(2, 3, 5).
    let size = 3;
    let stride = size * size;
    let mut data = vec![0u64; 3 * stride];
    data[0] = 2;
    data[4] = 3;
    data[8] = 5;
    data[stride] = 1;
    data[stride + 4] = 1;
    data[stride + 8] = 1;
    let mut work_units = 0u64;
    matrix_multiply(&mut data, size, &mut work_units);
    let c = &data[2 * stride..3 * stride];
    // Diagonal entries carry A's diagonal because B = I.
    assert_eq!(c[0], 2);
    assert_eq!(c[4], 3);
    assert_eq!(c[8], 5);
    // All 6 off-diagonal entries must be 0 for A*I. Sparse
    // coverage (just c[1], c[3]) left 4 positions unverified,
    // which would mask a transposition bug that mis-writes
    // rows/columns of an identity product — this assertion
    // fingerprints the full matrix identity.
    assert_eq!(c[1], 0);
    assert_eq!(c[2], 0);
    assert_eq!(c[3], 0);
    assert_eq!(c[5], 0);
    assert_eq!(c[6], 0);
    assert_eq!(c[7], 0);
}
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "assertion")]
fn matrix_multiply_mismatched_len_panics_in_debug() {
    // debug_assert_eq!(data.len(), 3 * size * size) guards the
    // bounds contract. Under cfg(debug_assertions) this panics.
    // Release builds skip the assert (no panic), so the test
    // itself is gated on `cfg(debug_assertions)` — otherwise
    // `cargo nextest run --release` would run the test expecting
    // a panic the release binary can't raise.
    let mut data = vec![0u64; 5]; // 3 * 2 * 2 = 12, so 5 is wrong.
    let mut work_units = 0u64;
    matrix_multiply(&mut data, 2, &mut work_units);
}
// -- RAII unit tests for the IO scratch / backing wrappers --
//
// Pin the contracts each Drop documents: DirectIoBuf returns a
// logical-block-aligned heap buffer, IoBacking unlinks its
// tempfile path on Drop (only when one was set), and
// PhaseIoTempfile unlinks unconditionally. The `ensure_*`
// helpers must be lazy-init — second call returns the same fd /
// pointer rather than re-opening / re-allocating.

/// `DirectIoBuf::alloc` returns a 4 KiB buffer aligned to the
/// logical-block boundary `O_DIRECT` requires. Writing the full
/// region and reading it back proves the allocation is mapped
/// for both reads and writes; the 0xAA pattern is arbitrary,
/// any non-zero bit pattern that survives the round trip is
/// sufficient.
#[test]
fn direct_io_buf_alloc_aligned() {
    let buf = DirectIoBuf::alloc()
        .expect("DirectIoBuf::alloc must succeed under normal allocator pressure");
    let addr = buf.as_ptr() as usize;
    assert_eq!(
        addr % IO_BLOCK_SIZE,
        0,
        "DirectIoBuf must be IO_BLOCK_SIZE-aligned (got addr={addr:#x})"
    );
    // SAFETY: alloc returned a non-null pointer to IO_BLOCK_SIZE
    // bytes of writable heap; the slice is fully covered by the
    // allocation, no aliasing live, and the pointer is unique
    // to this test scope.
    let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_ptr(), IO_BLOCK_SIZE) };
    slice.fill(0xAA);
    assert!(
        slice.iter().all(|&b| b == 0xAA),
        "round-trip pattern must persist across the buffer",
    );
    // Drop runs at end of scope and dealloc's the layout. The
    // test itself can't observe dealloc; it only proves no
    // panic / no UB on the freed pointer.
}
/// `IoBacking` with `tempfile_path: Some(_)` unlinks the path on
/// Drop. Constructs a real on-disk file with a unique name,
/// drops the wrapper inside a scope, then asserts the path no
/// longer exists.
#[test]
fn io_backing_tempfile_unlinked_on_drop() {
    let path = std::env::temp_dir()
        .join(format!(
            "ktstr_iobacking_unlink_{}_{}",
            std::process::id(),
            unsafe { libc::syscall(libc::SYS_gettid) },
        ))
        .to_string_lossy()
        .to_string();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .expect("create real tempfile for IoBacking test");
    assert!(
        std::path::Path::new(&path).exists(),
        "precondition: file exists"
    );
    {
        let _backing = IoBacking {
            file,
            capacity_bytes: 0,
            tempfile_path: Some(path.clone()),
        };
        // Path still exists inside scope.
        assert!(std::path::Path::new(&path).exists());
    }
    assert!(
        !std::path::Path::new(&path).exists(),
        "IoBacking::Drop must unlink {path}",
    );
}
/// `IoBacking` with `tempfile_path: None` (the `/dev/vda` case)
/// must NOT call `remove_file` — block devices are never
/// deleted. Drop in this shape only closes the file fd. Use a
/// host tempfile as the backing fd so the test is self-contained
/// (running outside a VM where /dev/vda exists), but pass
/// `tempfile_path: None` to exercise the "block device" arm.
#[test]
fn io_backing_none_path_no_unlink() {
    let path = std::env::temp_dir()
        .join(format!(
            "ktstr_iobacking_nounlink_{}_{}",
            std::process::id(),
            unsafe { libc::syscall(libc::SYS_gettid) },
        ))
        .to_string_lossy()
        .to_string();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .expect("create stand-in for /dev/vda");
    {
        let _backing = IoBacking {
            file,
            capacity_bytes: 0,
            tempfile_path: None,
        };
        // Drop fires here.
    }
    // File still exists because tempfile_path was None.
    assert!(
        std::path::Path::new(&path).exists(),
        "IoBacking::Drop must NOT unlink when tempfile_path is None",
    );
    // Cleanup the stand-in we created for the test.
    let _ = std::fs::remove_file(&path);
}
/// `PhaseIoTempfile::Drop` unconditionally unlinks `path`. Same
/// shape as the IoBacking test, simpler invariants (no
/// optional path).
#[test]
fn phase_io_tempfile_unlinked_on_drop() {
    let path = std::env::temp_dir()
        .join(format!(
            "ktstr_phaseio_unlink_{}_{}",
            std::process::id(),
            unsafe { libc::syscall(libc::SYS_gettid) },
        ))
        .to_string_lossy()
        .to_string();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .expect("create real tempfile for PhaseIoTempfile test");
    assert!(std::path::Path::new(&path).exists(), "precondition");
    {
        let _tf = PhaseIoTempfile {
            file,
            path: path.clone(),
        };
    }
    assert!(
        !std::path::Path::new(&path).exists(),
        "PhaseIoTempfile::Drop must unlink {path}",
    );
}
/// `ensure_io_disk` is lazy-init — calling it twice on the same
/// `Option<IoBacking>` slot opens the backing once and returns
/// the same fd on the second call. Compares `as_raw_fd()` across
/// both calls.
#[test]
fn ensure_io_disk_lazy_init() {
    use std::os::unix::io::AsRawFd;
    let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
    let mut io_disk: Option<IoBacking> = None;
    // First call: opens the backing.
    assert!(
        ensure_io_disk(&mut io_disk, 0, tid),
        "first ensure_io_disk must succeed (host can open tempfile fallback)",
    );
    let fd1 = io_disk
        .as_ref()
        .expect("io_disk Some after first call")
        .file
        .as_raw_fd();
    // Second call: must be a no-op, return the same fd.
    assert!(ensure_io_disk(&mut io_disk, 0, tid));
    let fd2 = io_disk.as_ref().unwrap().file.as_raw_fd();
    assert_eq!(
        fd1, fd2,
        "ensure_io_disk must be lazy-init — second call must not re-open",
    );
    // Drop unlinks the tempfile (if fallback path was used).
}
/// `ensure_io_buf` is lazy-init — calling it twice on the same
/// `Option<DirectIoBuf>` allocates once and returns the same
/// pointer on the second call.
#[test]
fn ensure_io_buf_lazy_init() {
    let mut io_buf: Option<DirectIoBuf> = None;
    assert!(
        ensure_io_buf(&mut io_buf),
        "first ensure_io_buf must succeed under normal allocator pressure",
    );
    let ptr1 = io_buf
        .as_ref()
        .expect("io_buf Some after first call")
        .as_ptr();
    assert!(ensure_io_buf(&mut io_buf));
    let ptr2 = io_buf.as_ref().unwrap().as_ptr();
    assert_eq!(
        ptr1, ptr2,
        "ensure_io_buf must be lazy-init — second call must not re-allocate",
    );
}
#[test]
fn thread_cpu_time_positive() {
    // Do some work so CPU time is non-zero
    let mut x = 0u64;
    for i in 0..100_000 {
        x = x.wrapping_add(i);
    }
    std::hint::black_box(x);
    let t = super::thread_cpu_time_ns();
    assert!(t > 0);
}
#[test]
fn set_sched_policy_normal_succeeds() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Normal);
    assert!(result.is_ok());
}
#[test]
#[ignore]
fn set_sched_policy_fifo_returns_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Fifo(1));
    // SCHED_FIFO requires CAP_SYS_NICE; succeeds when the runner holds it.
    assert!(
        result.is_ok(),
        "SCHED_FIFO should succeed with CAP_SYS_NICE"
    );
    restore_normal(pid);
}
#[test]
#[ignore]
fn set_sched_policy_rr_returns_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::RoundRobin(1));
    // SCHED_RR requires CAP_SYS_NICE; succeeds when the runner holds it.
    assert!(result.is_ok(), "SCHED_RR should succeed with CAP_SYS_NICE");
    restore_normal(pid);
}
// -- SchedPolicy variants --

/// Restore SCHED_NORMAL via the raw syscall. `set_sched_policy(Normal)`
/// is a no-op, so tests that change policy must use this to restore.
fn restore_normal(pid: libc::pid_t) {
    let param = libc::sched_param { sched_priority: 0 };
    unsafe { libc::sched_setscheduler(pid, libc::SCHED_OTHER, &param) };
}
#[test]
fn set_sched_policy_batch_returns_valid_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Batch);
    // SCHED_BATCH does NOT require CAP_SYS_NICE:
    // `user_check_sched_setscheduler` routes only rt_policy /
    // dl_policy / negative-nice / leaving-IDLE through
    // req_priv; a fair-policy → fair-policy transition that
    // does not reduce nice never reaches the capable() check.
    // `scx_check_setscheduler` (kernel/sched/ext.c) does not
    // reject BATCH either — it only rejects transitions INTO
    // SCHED_EXT when `p->scx.disallow` is set, which BATCH is
    // not. Failure is therefore expected only on environments
    // that introduce extra LSM / security-module gates; the
    // test tolerates both outcomes.
    match result {
        Ok(()) => {
            let pol = unsafe { libc::sched_getscheduler(pid) };
            // Under sched_ext switch-all (`task_should_scx`
            // returns true for any policy when
            // `scx_switching_all` is set), `__setscheduler_class`
            // routes BATCH to `ext_sched_class`. Reading back
            // via `sched_getscheduler` returns the requested
            // policy regardless — this just sanity-checks the
            // syscall returned a non-negative policy id.
            assert!(
                pol >= 0,
                "sched_getscheduler must return a valid policy, got {pol}",
            );
            restore_normal(pid);
        }
        Err(ref e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("sched_setscheduler"),
                "error must name the syscall: {msg}"
            );
        }
    }
}
#[test]
fn set_sched_policy_idle_returns_valid_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Idle);
    // SCHED_IDLE does NOT require CAP_SYS_NICE for *entering*
    // IDLE: `user_check_sched_setscheduler` gates the
    // IDLE-related capability check on `task_has_idle_policy(p)
    // && !idle_policy(policy)` — i.e. CAP_SYS_NICE is required
    // only when *leaving* SCHED_IDLE for a non-idle class
    // without RLIMIT_NICE permission, not when entering it.
    // `scx_check_setscheduler` (kernel/sched/ext.c) does not
    // reject IDLE either — same reasoning as the BATCH test
    // above. Failure is expected only on environments with
    // extra LSM / security-module gates.
    match result {
        Ok(()) => {
            let pol = unsafe { libc::sched_getscheduler(pid) };
            // Same switch-all reasoning as the BATCH test —
            // IDLE routes to `ext_sched_class` under switch-all
            // but the syscall return is the requested policy id.
            assert!(
                pol >= 0,
                "sched_getscheduler must return a valid policy, got {pol}",
            );
            restore_normal(pid);
        }
        Err(ref e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("sched_setscheduler"),
                "error must name the syscall: {msg}"
            );
        }
    }
}
// -- SCHED_DEADLINE validation tests --
//
// The five rejection tests below exercise the structural
// pre-validation that `set_sched_policy` performs before
// issuing the `sched_setattr` syscall. Each invariant mirrors
// a `__checkparam_dl` clause (`kernel/sched/deadline.c`); the
// tests pin user-space rejection so a malformed `Deadline`
// surfaces a named field rather than a generic kernel
// `EINVAL`. None of these tests require `CAP_SYS_NICE`
// because the bail!s fire before the syscall.

/// `deadline == Duration::ZERO` must be rejected:
/// `__checkparam_dl` returns false on `attr->sched_deadline ==
/// 0`. The runtime floor is satisfied here so the failure
/// pins the zero-deadline check, not the DL_SCALE check.
#[test]
fn set_sched_policy_deadline_zero_deadline_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            deadline: Duration::ZERO,
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("zero deadline must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("deadline"),
        "error must name deadline field: {msg}"
    );
    assert!(
        msg.contains("must be > 0") || msg.contains("zero"),
        "error must explain zero rejection: {msg}"
    );
}
/// `runtime` shorter than 1024 ns must be rejected per the
/// `DL_SCALE` floor in `__checkparam_dl`.
#[test]
fn set_sched_policy_deadline_runtime_below_dl_scale_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1023),
            deadline: Duration::from_nanos(100_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("runtime below DL_SCALE must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("runtime"),
        "error must name runtime field: {msg}"
    );
    assert!(
        msg.contains("DL_SCALE") || msg.contains("1024"),
        "error must reference the floor: {msg}"
    );
}
/// `runtime > deadline` must be rejected per the
/// `runtime <= deadline` clause of `__checkparam_dl`.
#[test]
fn set_sched_policy_deadline_runtime_exceeds_deadline_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(200_000),
            deadline: Duration::from_nanos(100_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("runtime > deadline must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("runtime") && msg.contains("deadline"),
        "error must name both fields: {msg}"
    );
}
/// `deadline > period` must be rejected when `period` is
/// non-zero. Pairs with
/// `set_sched_policy_deadline_period_zero_passes_validation`
/// which proves the gate is conditional on a non-zero period.
#[test]
fn set_sched_policy_deadline_deadline_exceeds_period_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            deadline: Duration::from_nanos(2_000_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("deadline > period must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("deadline") && msg.contains("period"),
        "error must name both fields: {msg}"
    );
}
/// A `deadline` whose nanosecond count exceeds `i64::MAX` must
/// be rejected. The kernel's `__checkparam_dl` clause `if
/// (attr->sched_deadline & (1ULL << 63)) return false;`
/// requires bit 63 to be clear; `duration_to_kernel_ns`
/// enforces this as a single i64::MAX overflow check on
/// `Duration::as_nanos()` (u128). The error message names the
/// offending field via the `field` argument so the diagnostic
/// points at `deadline` and not `runtime`/`period`.
#[test]
fn set_sched_policy_deadline_top_bit_set_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            // 1e12 seconds = 1e21 ns >> i64::MAX (≈ 9.2e18 ns)
            // — guaranteed to trip the overflow guard. Picked
            // far above the threshold so any future tweak to
            // the constraint still fires this test.
            deadline: Duration::from_secs(1_000_000_000_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("deadline exceeding i64::MAX must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("deadline") && (msg.contains("i64::MAX") || msg.contains("63 bits")),
        "error must name deadline field and the bit-63 / i64::MAX bound: {msg}"
    );
    // Per-field message: must NOT name `period` since only
    // `deadline` overflowed. `period` ordering matters —
    // `duration_to_kernel_ns` is called runtime → deadline →
    // period, so a deadline overflow short-circuits before
    // period is touched.
    assert!(
        !msg.contains("period"),
        "deadline-only overflow error must not mention period: {msg}"
    );
}
/// Happy-path: a structurally valid `Deadline` with
/// `period == Duration::ZERO` reaches the `sched_setattr`
/// syscall. The kernel substitutes `deadline` for the period
/// in this case (see `if (!period) period = attr->sched_deadline;`
/// in `__checkparam_dl`). Without `CAP_SYS_NICE` the syscall
/// fails with EPERM at the kernel-side capability check;
/// either Ok(()) or an Err whose message names
/// `sched_setattr` confirms we cleared the user-space
/// pre-validation. Marked `#[ignore]` so unprivileged CI
/// doesn't see EPERM as a hard failure — runners with
/// CAP_SYS_NICE can opt in.
#[test]
#[ignore]
fn set_sched_policy_deadline_period_zero_passes_validation() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            deadline: Duration::from_nanos(200_000),
            period: Duration::ZERO,
        },
    );
    match result {
        Ok(()) => {
            // Restore SCHED_NORMAL so the test process leaves
            // its run with default policy.
            restore_normal(pid);
        }
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("sched_setattr"),
                "validation must have passed (error from kernel must name sched_setattr): {msg}"
            );
        }
    }
}
// -- reservoir_push tests --

#[test]
fn reservoir_push_empty_buf() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    reservoir_push(&mut buf, &mut count, 42, 10);
    assert_eq!(buf, vec![42]);
    assert_eq!(count, 1);
}
#[test]
fn reservoir_push_under_cap() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    for i in 0..5 {
        reservoir_push(&mut buf, &mut count, i * 100, 10);
    }
    assert_eq!(buf.len(), 5);
    assert_eq!(count, 5);
    assert_eq!(buf, vec![0, 100, 200, 300, 400]);
}
#[test]
fn reservoir_push_at_cap() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    for i in 0..10 {
        reservoir_push(&mut buf, &mut count, i, 10);
    }
    assert_eq!(buf.len(), 10);
    assert_eq!(count, 10);
    // All values should be present since we're exactly at cap.
    for i in 0..10 {
        assert!(buf.contains(&i), "missing {i}");
    }
}
#[test]
fn reservoir_push_over_cap_maintains_size() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    let cap = 5;
    for i in 0..1000 {
        reservoir_push(&mut buf, &mut count, i, cap);
    }
    assert_eq!(buf.len(), cap);
    assert_eq!(count, 1000);
}
#[test]
fn reservoir_push_uniform_sampling() {
    // Statistical test: push 10000 values into cap=100 reservoir.
    // Each value should have roughly equal probability of being present.
    // We test that the reservoir contains values from the full range.
    let mut buf = Vec::new();
    let mut count = 0u64;
    let cap = 100;
    let total = 10_000u64;
    for i in 0..total {
        reservoir_push(&mut buf, &mut count, i, cap);
    }
    assert_eq!(buf.len(), cap);
    assert_eq!(count, total);
    // The reservoir should contain values from different parts of the range.
    let has_early = buf.iter().any(|&v| v < total / 4);
    let has_late = buf.iter().any(|&v| v > total * 3 / 4);
    assert!(has_early, "reservoir should contain early values");
    assert!(has_late, "reservoir should contain late values");
}
#[test]
fn reservoir_push_cap_zero() {
    // Zero-capacity reservoir: buf.len() < 0 is never true (usize),
    // falls through to else branch where random_range(0..1) returns 0,
    // and 0 < 0 is false — sample is discarded.
    let mut buf = Vec::new();
    let mut count = 0u64;
    for i in 0..10 {
        reservoir_push(&mut buf, &mut count, i, 0);
    }
    assert!(buf.is_empty(), "cap=0 should never store samples");
    assert_eq!(count, 10, "count incremented regardless");
}
#[test]
fn reservoir_push_cap_one() {
    // Single-element reservoir. First sample always stored.
    // Subsequent samples replace with probability 1/count.
    let mut buf = Vec::new();
    let mut count = 0u64;
    reservoir_push(&mut buf, &mut count, 42, 1);
    assert_eq!(buf, vec![42]);
    assert_eq!(count, 1);
    // Push more — buf stays length 1.
    for i in 1..100 {
        reservoir_push(&mut buf, &mut count, i * 100, 1);
    }
    assert_eq!(buf.len(), 1);
    assert_eq!(count, 100);
}
// -- read_schedstat tests --

#[test]
fn read_schedstat_returns_finite_triple() {
    // The calling thread has been scheduled at least once by the
    // time this test runs (it's executing right now), so cpu_time
    // and timeslices must be strictly positive. run_delay can
    // legitimately be zero on an idle host where the test thread
    // never waited for a runqueue slot, so it is left unchecked.
    //
    // `None` is a legitimate outcome when the host kernel is
    // built without `CONFIG_SCHEDSTATS` — treat that as a skip
    // rather than a test failure.
    let Some((cpu_time, _run_delay, timeslices)) = read_schedstat(None) else {
        eprintln!("skipping: /proc/self/schedstat not available (CONFIG_SCHEDSTATS off)");
        return;
    };
    assert!(cpu_time > 0);
    assert!(timeslices > 0);
}
#[test]
fn parse_schedstat_line_happy_path() {
    // A well-formed line has at least three whitespace-separated
    // u64 fields; extra trailing fields are ignored.
    let (cpu_time, run_delay, timeslices) = parse_schedstat_line("100 200 300 999 extra").unwrap();
    assert_eq!(cpu_time, 100);
    assert_eq!(run_delay, 200);
    assert_eq!(timeslices, 300);
}
#[test]
fn parse_schedstat_line_tab_and_newline_separators() {
    // `split_whitespace` treats any run of whitespace as one
    // separator, so tabs and trailing newlines must parse.
    let parsed = parse_schedstat_line("1\t2\t3\n").unwrap();
    assert_eq!(parsed, (1, 2, 3));
}
#[test]
fn parse_schedstat_line_missing_field_returns_none() {
    // Two fields is one short — the third `?` bails.
    assert!(parse_schedstat_line("100 200").is_none());
    // One field short of two.
    assert!(parse_schedstat_line("100").is_none());
    // Empty input — zero fields.
    assert!(parse_schedstat_line("").is_none());
    // Whitespace-only input — zero tokens after split.
    assert!(parse_schedstat_line("   \t\n  ").is_none());
}
#[test]
fn parse_schedstat_line_non_u64_token_returns_none() {
    // Any non-u64 token fails the `.parse::<u64>().ok()?` chain.
    assert!(parse_schedstat_line("not-a-number 200 300").is_none());
    assert!(parse_schedstat_line("100 abc 300").is_none());
    assert!(parse_schedstat_line("100 200 nan").is_none());
    // Negative numbers parse to u64 as an error.
    assert!(parse_schedstat_line("-1 200 300").is_none());
    // Overflow beyond u64::MAX.
    assert!(parse_schedstat_line("99999999999999999999 2 3").is_none());
}
#[test]
fn warn_schedstat_unavailable_once_does_not_panic_on_repeat() {
    // `std::sync::Once::call_once` guarantees at most one
    // eprintln regardless of how many times the gate fires.
    // Smoke-check that repeated calls don't panic — direct
    // stderr-emission assertions require a process-global
    // capture gate (`#[test]` threads share fd 2), which is
    // out of scope for this unit test.
    for _ in 0..10 {
        warn_schedstat_unavailable_once();
    }
}
/// [`resolve_alu_width`] never returns `Widest`. Every
/// concrete variant resolves to itself (when the host
/// supports it) or to a downgrade — the sentinel must
/// disappear before reaching [`alu_hot_chain`].
#[test]
fn alu_width_resolve_never_returns_widest() {
    for &w in &[
        AluWidth::Scalar,
        AluWidth::Vec128,
        AluWidth::Vec256,
        AluWidth::Vec512,
        AluWidth::Amx,
        AluWidth::Widest,
    ] {
        let r = resolve_alu_width(w);
        assert!(
            !matches!(r, AluWidth::Widest),
            "resolve_alu_width({w:?}) returned Widest; \
             caller invariant violated",
        );
    }
}
