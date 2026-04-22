//! Panic-hook shim for vCPU worker threads.
//!
//! The crate runs with `panic = "abort"` in release (Cargo.toml), so
//! a panic on any thread tears down the entire VM process without
//! unwinding — Drop impls do not run, and `std::panic::catch_unwind`
//! cannot observe the failure. That leaves a single window between
//! "thread panics" and "libc::abort" during which the user-registered
//! panic hook (`std::panic::set_hook`) runs synchronously on the
//! panicking thread. This module uses that window to flip the
//! per-VM `kill` flag and the per-thread `exited` flag so the
//! watchdog/monitor threads observe a classified shutdown instead of
//! an opaque abort (panic=abort calls `libc::abort`, which raises
//! SIGABRT — not SIGKILL — but the outward signal an observer sees
//! is "process terminated with no cleanup").
//!
//! Primary benefit is *ordering* — the `kill` / `exited` flip
//! happens before `libc::abort`, so any observer that polls those
//! atomics (watchdog, parent join loop) sees a classified shutdown
//! rather than an unexplained abort. User-facing diagnostics
//! (panic message, backtrace) come from the preserved previous-hook
//! chain, not from this module.
//!
//! Scope of work done inside the hook:
//! - Atomic `store(true)` on `kill` and `exited` — non-blocking,
//!   allocation-free, and correct under the panicking-thread
//!   constraint (any lock acquisition here risks deadlocking against
//!   the same thread if it held the lock at the point of panic; any
//!   allocation risks triggering a nested panic).
//! - Nothing else. Serial-buffer flush is *not* performed here: the
//!   serial state lives behind a `PiMutex` and `PiMutex::lock`'s
//!   non-try path would assert-fail if the panic struck mid-`lock`.
//!   On a normal exit path the VM cleanup code drains serial; on
//!   panic=abort, final serial bytes are intentionally sacrificed for
//!   hook correctness.
//!
//! Registration model: `install_once` sets the process-wide hook
//! exactly once (a single test process can spawn multiple VMs
//! sequentially — the hook is installed on the first call and
//! reused). Per-thread context (`VcpuPanicCtx`) is stashed in a
//! `thread_local` from inside the vCPU thread body; the hook reads
//! the thread-local to decide what to signal. Threads that never
//! register leave the thread-local at `None` and the hook falls
//! through to the default.
//!
//! The previous (already-installed) hook is captured at install time
//! and called after our hook runs so standard panic messages /
//! backtraces still reach stderr.
//!
//! Limitation: `std::panic::set_hook` is process-wide. If a future
//! caller of this crate installs their own hook after ours, the
//! previous-hook chain is broken and our signaling is bypassed for
//! any vCPU panic that happens after that. Callers that embed ktstr
//! must install their own hook before spawning vCPU threads, or
//! accept the fall-through.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    /// Per-thread context consulted by the panic hook. `Some` only
    /// for threads that passed through [`with_vcpu_panic_ctx`].
    static VCPU_PANIC_CTX: RefCell<Option<VcpuPanicCtx>> = const { RefCell::new(None) };
}

/// Flags the panic hook will flip on behalf of a panicking vCPU
/// thread. Clone-cheap — each field is an `Arc<AtomicBool>` shared
/// with the main VM thread and the monitor/watchdog. Fields are
/// `pub(crate)` to match the container's `pub(crate)` visibility;
/// nothing outside the `vmm` module observes this struct directly.
#[derive(Clone)]
pub(crate) struct VcpuPanicCtx {
    /// VM-wide kill signal. Flipping this unblocks the monitor loop
    /// and lets the parent thread observe a clean shutdown path
    /// instead of treating the abort as an unexplained termination.
    pub(crate) kill: Arc<AtomicBool>,
    /// Per-thread exited marker. The parent's
    /// `VcpuThread::exited.load()` polling sees this flip before the
    /// `libc::abort` call returns to the kernel, so the parent can
    /// record "vcpu-N exited" in its failure ledger.
    pub(crate) exited: Arc<AtomicBool>,
}

static HOOK_ONCE: Once = Once::new();

/// Install the vCPU panic hook if it has not already been installed.
/// Idempotent — safe to call from every `spawn_ap_threads`
/// invocation; `Once` gates the actual registration.
pub(crate) fn install_once() {
    HOOK_ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            VCPU_PANIC_CTX.with(|slot| {
                if let Some(ctx) = slot.borrow().as_ref() {
                    ctx.kill.store(true, Ordering::Release);
                    ctx.exited.store(true, Ordering::Release);
                }
            });
            prev(info);
        }));
    });
}

/// Register `ctx` for the current thread, run `body`, then clear the
/// registration. Any panic inside `body` is observed by the hook
/// installed by [`install_once`]; on normal return the thread-local
/// is reset so a future reuse of this OS thread (unusual but
/// possible if a runtime recycles threads) does not carry stale
/// context into an unrelated panic.
///
/// INVARIANT: `body()` must not hold a `borrow` or `borrow_mut` on
/// `VCPU_PANIC_CTX` across a potential panic — the hook needs a
/// `borrow()` to read the context, and an outstanding borrow would
/// make that `borrow()` panic (turning the hook into a nested
/// panic, which under panic=abort aborts without signaling). The
/// two `borrow_mut` windows in this function scope strictly to the
/// `set` / `clear` statements and release before/after `body()`, so
/// the panic window inside `body()` never overlaps a mutable borrow.
/// Callers inside `body` must not re-enter this module's thread
/// local.
pub(crate) fn with_vcpu_panic_ctx<R>(ctx: VcpuPanicCtx, body: impl FnOnce() -> R) -> R {
    VCPU_PANIC_CTX.with(|slot| {
        *slot.borrow_mut() = Some(ctx);
    });
    let result = body();
    VCPU_PANIC_CTX.with(|slot| {
        *slot.borrow_mut() = None;
    });
    result
}

#[cfg(test)]
mod tests {
    //! The `[profile.test]` profile inherits `[profile.dev]` and does
    //! NOT set `panic = "abort"`, so the default unwind behavior is
    //! active inside cargo test / nextest runs. That is what allows
    //! `catch_unwind` below to observe the panic without tearing down
    //! the test process — the release-profile panic=abort semantic
    //! this module targets is itself NOT under test here (it's
    //! outside rustc's testable surface).
    //!
    //! Tests run on freshly `std::thread::spawn`ed threads so the
    //! `VCPU_PANIC_CTX` thread-local begins at its `None` init for
    //! every case. That isolates state between tests even under the
    //! parallel test runner (nextest).
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// Ten `install_once` calls must neither panic nor deadlock;
    /// `Once` guarantees the body runs exactly once.
    #[test]
    fn install_once_is_idempotent() {
        for _ in 0..10 {
            install_once();
        }
    }

    /// After `with_vcpu_panic_ctx` returns normally, the thread-local
    /// must be reset to `None` so a later unrelated panic on the
    /// same OS thread does not surface stale kill/exited atomics.
    #[test]
    fn with_vcpu_panic_ctx_clears_thread_local_on_return() {
        install_once();
        let ctx = VcpuPanicCtx {
            kill: Arc::new(AtomicBool::new(false)),
            exited: Arc::new(AtomicBool::new(false)),
        };
        std::thread::spawn(move || {
            with_vcpu_panic_ctx(ctx, || {});
            VCPU_PANIC_CTX.with(|slot| {
                assert!(
                    slot.borrow().is_none(),
                    "thread-local must be None after normal return",
                );
            });
        })
        .join()
        .unwrap();
    }

    /// A panic inside the `with_vcpu_panic_ctx` body must flip both
    /// `kill` and `exited` via the installed hook. `catch_unwind`
    /// observes the unwind (test profile); under release panic=abort
    /// the same hook would run immediately before `libc::abort`.
    #[test]
    fn panic_inside_ctx_flips_flags() {
        install_once();
        let kill = Arc::new(AtomicBool::new(false));
        let exited = Arc::new(AtomicBool::new(false));
        let ctx = VcpuPanicCtx {
            kill: kill.clone(),
            exited: exited.clone(),
        };
        let kill_c = kill.clone();
        let exited_c = exited.clone();
        let (kill_r, exited_r) = std::thread::spawn(move || {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                with_vcpu_panic_ctx(ctx, || panic!("test: intended panic"));
            }));
            (
                kill_c.load(Ordering::Acquire),
                exited_c.load(Ordering::Acquire),
            )
        })
        .join()
        .unwrap();
        assert!(kill_r, "kill must be flipped by the panic hook");
        assert!(exited_r, "exited must be flipped by the panic hook");
    }

    /// A panic on a thread that never registered a context must NOT
    /// touch any external flags — the hook's thread-local read
    /// returns `None`.
    #[test]
    fn panic_outside_ctx_leaves_flags_alone() {
        install_once();
        let kill = Arc::new(AtomicBool::new(false));
        let exited = Arc::new(AtomicBool::new(false));
        let kill_c = kill.clone();
        let exited_c = exited.clone();
        let (kill_r, exited_r) = std::thread::spawn(move || {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                panic!("test: intended panic without registered ctx");
            }));
            (
                kill_c.load(Ordering::Acquire),
                exited_c.load(Ordering::Acquire),
            )
        })
        .join()
        .unwrap();
        assert!(!kill_r, "kill must stay false when no ctx registered");
        assert!(!exited_r, "exited must stay false when no ctx registered");
    }
}
