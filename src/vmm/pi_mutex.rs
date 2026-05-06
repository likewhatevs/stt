//! Priority-inheritance mutex via `pthread_mutex` + `PTHREAD_PRIO_INHERIT`.
//!
//! Used wherever the host VMM holds a lock that may be contended between
//! SCHED_FIFO and SCHED_OTHER threads. CONFIG_FUTEX_PI is a hard
//! requirement on the host kernel — see [`PiMutex`] for the failure
//! mode when it is not available.

/// Mutex that uses the kernel's priority-inheritance protocol to avoid
/// priority inversion between RT and non-RT threads.
///
/// When a SCHED_FIFO thread blocks on a PiMutex held by a SCHED_OTHER
/// thread, the kernel temporarily boosts the holder to the waiter's
/// priority, ensuring the critical section completes without unbounded
/// delay.
///
/// `pthread_mutexattr_setprotocol(PTHREAD_PRIO_INHERIT)` operates on
/// a userspace `pthread_mutexattr_t`; it does not enter the kernel and
/// therefore does not observe the running kernel's config. The kernel
/// boundary is crossed at lock time via the futex syscall family
/// (`FUTEX_LOCK_PI` / `FUTEX_LOCK_PI2`), which returns `-ENOSYS` when
/// the kernel is built without `CONFIG_FUTEX_PI` (kernel/futex/pi.c,
/// `futex_lock_pi` early return; same gate covers the PI-related
/// syscalls in `kernel/futex/syscalls.c::do_futex`). When that
/// surfaces as a nonzero return from `pthread_mutex_lock`, the assert
/// in [`PiMutex::lock`] panics. There is no graceful degradation in
/// this path: a host without `CONFIG_FUTEX_PI` cannot run ktstr.
///
/// # Panics
///
/// * `PiMutex::new` panics if `pthread_mutexattr_init` or
///   `pthread_mutex_init` fails, or if
///   `pthread_mutexattr_setprotocol(PTHREAD_PRIO_INHERIT)` returns any
///   nonzero value other than `ENOTSUP`. The `ENOTSUP` branch covers
///   only a libc-level refusal of the protocol value (e.g. a non-glibc
///   libc that does not implement PI mutexes); it is not reached on
///   Linux/glibc when the kernel lacks CONFIG_FUTEX_PI, because that
///   condition is invisible to glibc until lock time.
///   The alternative on real init failures (a partially initialized
///   mutex) would have undefined lock/unlock semantics.
/// * `PiMutex::lock` panics if `pthread_mutex_lock` returns nonzero.
///   The expected failure mode in practice is `ENOSYS` on a host
///   kernel without `CONFIG_FUTEX_PI` (the kernel-side gate cited
///   above). Returning a guard on an unlocked mutex would let the
///   caller obtain `&mut T` without exclusive access — a data race
///   and undefined behaviour — so this mirrors `std::sync::Mutex`.
/// * `PiMutexGuard::drop` calls `libc::abort()` if
///   `pthread_mutex_unlock` fails (typical cause: `EPERM` — the
///   current thread does not own the mutex, indicating a violated
///   guard-ownership invariant elsewhere). Drop cannot propagate
///   errors; releasing the `&mut T` contract on a still-locked
///   mutex is worse than abort, because another thread could then
///   observe the interior mutably while we also reference it.
pub(crate) struct PiMutex<T> {
    inner: std::cell::UnsafeCell<T>,
    mutex: std::cell::UnsafeCell<libc::pthread_mutex_t>,
}

// SAFETY: PiMutex provides mutual exclusion via pthread_mutex_lock/unlock.
// The UnsafeCell<T> is only accessed while the mutex is held.
unsafe impl<T: Send> Send for PiMutex<T> {}
unsafe impl<T: Send> Sync for PiMutex<T> {}

impl<T> PiMutex<T> {
    /// Create a new PI mutex wrapping `value`.
    pub(crate) fn new(value: T) -> Self {
        unsafe {
            let mut attr: libc::pthread_mutexattr_t = std::mem::zeroed();
            let rc = libc::pthread_mutexattr_init(&mut attr);
            assert_eq!(rc, 0, "pthread_mutexattr_init failed: {rc}");
            let rc = libc::pthread_mutexattr_setprotocol(&mut attr, libc::PTHREAD_PRIO_INHERIT);
            // pthread_mutexattr_setprotocol writes a field in the
            // userspace pthread_mutexattr_t; it does not enter the
            // kernel and so it cannot observe whether the running
            // kernel was built with CONFIG_FUTEX_PI. The CONFIG_FUTEX_PI
            // gate fires at lock time via the FUTEX_LOCK_PI futex op
            // (kernel/futex/pi.c, futex_lock_pi early return), which
            // surfaces as a nonzero return from pthread_mutex_lock and
            // trips the assert in lock() — the process panics.
            //
            // The ENOTSUP branch below therefore does not catch the
            // missing-CONFIG_FUTEX_PI case; it covers only a libc
            // that refuses PTHREAD_PRIO_INHERIT outright (e.g. a
            // pthread implementation that lacks PI support). On such
            // a libc, falling back to the default PRIO_NONE protocol
            // preserves mutual exclusion and lets the process come
            // up. Any other nonzero rc is a programmer error (EINVAL
            // from a bad attr pointer) and is still asserted.
            if rc == libc::ENOTSUP {
                tracing::warn!(
                    "PTHREAD_PRIO_INHERIT unsupported by libc (errno {}); PiMutex degrading to non-PI protocol",
                    rc
                );
                // Make the fallback explicit: a libc that rejected
                // PRIO_INHERIT may have left the attr's protocol field
                // in an unspecified state. Force PRIO_NONE so the
                // resulting mutex has a well-defined protocol. PRIO_NONE
                // is the POSIX default and is required to be supported
                // by every conforming pthread implementation, so this
                // setprotocol call cannot itself return ENOTSUP.
                let rc_none =
                    libc::pthread_mutexattr_setprotocol(&mut attr, libc::PTHREAD_PRIO_NONE);
                assert_eq!(
                    rc_none, 0,
                    "pthread_mutexattr_setprotocol(PTHREAD_PRIO_NONE) failed: {rc_none}"
                );
            } else {
                assert_eq!(
                    rc, 0,
                    "pthread_mutexattr_setprotocol(PTHREAD_PRIO_INHERIT) failed: {rc}"
                );
            }
            let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
            let rc = libc::pthread_mutex_init(&mut mutex, &attr);
            libc::pthread_mutexattr_destroy(&mut attr);
            assert_eq!(rc, 0, "pthread_mutex_init failed: {rc}");
            PiMutex {
                inner: std::cell::UnsafeCell::new(value),
                mutex: std::cell::UnsafeCell::new(mutex),
            }
        }
    }

    /// Lock the mutex and return a guard providing `&mut T`.
    pub(crate) fn lock(&self) -> PiMutexGuard<'_, T> {
        unsafe {
            let rc = libc::pthread_mutex_lock(self.mutex.get());
            assert_eq!(rc, 0, "pthread_mutex_lock failed: {rc}");
        }
        PiMutexGuard { mutex: self }
    }
}

impl<T> Drop for PiMutex<T> {
    fn drop(&mut self) {
        unsafe {
            libc::pthread_mutex_destroy(self.mutex.get());
        }
    }
}

pub(crate) struct PiMutexGuard<'a, T> {
    mutex: &'a PiMutex<T>,
}

impl<T> std::ops::Deref for PiMutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.inner.get() }
    }
}

impl<T> std::ops::DerefMut for PiMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.inner.get() }
    }
}

impl<T> Drop for PiMutexGuard<'_, T> {
    fn drop(&mut self) {
        unsafe {
            let rc = libc::pthread_mutex_unlock(self.mutex.mutex.get());
            if rc != 0 {
                eprintln!("pthread_mutex_unlock failed: {rc}");
                libc::abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn pi_mutex_lock_unlock() {
        let m = PiMutex::new(42u32);
        {
            let mut guard = m.lock();
            assert_eq!(*guard, 42);
            *guard = 99;
        }
        assert_eq!(*m.lock(), 99);
    }

    #[test]
    fn pi_mutex_cross_thread() {
        let m = Arc::new(PiMutex::new(0u32));
        let m2 = m.clone();
        let handle = std::thread::spawn(move || {
            *m2.lock() += 1;
        });
        handle.join().unwrap();
        assert_eq!(*m.lock(), 1);
    }

    #[test]
    fn pi_mutex_concurrent_increment() {
        let m = Arc::new(PiMutex::new(0u64));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let m = m.clone();
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        *m.lock() += 1;
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(*m.lock(), 8000);
    }

    #[test]
    fn pi_mutex_contention_10_threads_increments_correctly() {
        // N-thread contention: 10 × 1000 increments through
        // PiMutexGuard's DerefMut. If `lock()` ever returned a guard
        // without holding the mutex (the pre-fix debug_assert bug in
        // release builds), concurrent increments would race and the
        // final count would be < 10_000. The unconditional assert
        // panics on lock failure and the abort() in Drop panics on
        // unlock failure, so any guard-violation surfaces loudly.
        let m = Arc::new(PiMutex::new(0u64));
        let threads: Vec<_> = (0..10)
            .map(|_| {
                let m = m.clone();
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        *m.lock() += 1;
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("worker thread panicked");
        }
        assert_eq!(*m.lock(), 10_000);
    }

    #[test]
    fn pi_mutex_protocol_is_inherit() {
        // Verify PTHREAD_PRIO_INHERIT is supported on this system.
        unsafe {
            let mut attr: libc::pthread_mutexattr_t = std::mem::zeroed();
            assert_eq!(libc::pthread_mutexattr_init(&mut attr), 0);
            assert_eq!(
                libc::pthread_mutexattr_setprotocol(&mut attr, libc::PTHREAD_PRIO_INHERIT),
                0,
            );
            let mut protocol: libc::c_int = 0;
            assert_eq!(libc::pthread_mutexattr_getprotocol(&attr, &mut protocol), 0);
            assert_eq!(protocol, libc::PTHREAD_PRIO_INHERIT);
            libc::pthread_mutexattr_destroy(&mut attr);
        }
    }
}
