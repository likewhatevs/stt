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

#[cfg(test)]
use std::sync::atomic::AtomicUsize;

/// Panic-hook callable type. Matches the signature accepted by
/// [`std::panic::set_hook`] and returned by [`std::panic::take_hook`].
type PanicHook = dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync + 'static;

/// Build the chained panic hook: flip per-thread kill/exited flags (if
/// this thread registered a [`VcpuPanicCtx`]) and then delegate to
/// `prev`. Factored out of [`install_once`] so tests can install a
/// custom `prev` and observe that the chain is not silently dropped.
///
/// # RefCell borrow invariant
///
/// The hook takes a shared `slot.borrow()` on `VCPU_PANIC_CTX`, the
/// per-thread `RefCell<Option<VcpuPanicCtx>>`. That borrow is safe
/// only because the hook is never re-entered while another borrow is
/// active on the same thread's RefCell:
///
/// - `with_vcpu_panic_ctx` scopes its two `borrow_mut()` windows
///   strictly to the set / clear statements and drops them before
///   running `body()`, per that function's documented INVARIANT. Any
///   panic raised inside `body()` therefore finds the RefCell
///   unborrowed, and the hook's `borrow()` cannot conflict.
/// - A panic is delivered to `std`'s hook machinery synchronously on
///   the panicking thread. `std::panic::set_hook` serializes hook
///   registration, and `catch_unwind` / runtime unwinding calls the
///   hook exactly once per `panic!` site before unwinding continues.
///   There is no concurrent second entry into this closure on the
///   same thread to hold a conflicting borrow.
/// - `prev(info)` is the previously-installed process-wide hook
///   captured at `install_once` time. By construction it does not
///   re-enter this module's thread-local (no ktstr code path inside a
///   `prev` hook touches `VCPU_PANIC_CTX`), so the delegation tail
///   cannot recursively panic into our hook.
///
/// If any of those preconditions breaks — a caller holds a `borrow`
/// across a panic site, a runtime gains re-entrant hook dispatch, or
/// a downstream `prev` hook calls back into this module — the
/// `borrow()` here panics, the panic hook double-panics, and under
/// `panic = "abort"` the process aborts without emitting the classified
/// shutdown signal `VcpuPanicCtx` exists to produce. Preserve the
/// invariant.
fn make_hook(prev: Box<PanicHook>) -> Box<PanicHook> {
    Box::new(move |info| {
        VCPU_PANIC_CTX.with(|slot| {
            if let Some(ctx) = slot.borrow().as_ref() {
                ctx.kill.store(true, Ordering::Release);
                ctx.exited.store(true, Ordering::Release);
            }
        });
        prev(info);
    })
}

/// Count of times the `HOOK_ONCE` body executed. The `Once` contract
/// guarantees this reaches 1 and stays there regardless of how many
/// callers invoke [`install_once`], giving tests a stronger assertion
/// than "no panic / no deadlock" for install idempotency.
#[cfg(test)]
static INSTALL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Serialize tests that install a custom panic hook via
/// [`install_hook_with_prev_for_test`]. The hook is process-wide, so
/// concurrent manipulation would race. Tests that only rely on the
/// standard `install_once` hook do NOT need this lock — they observe
/// a stable hook via `Once`.
#[cfg(test)]
static HOOK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
///
/// INVARIANT: Every field's `Drop` must be panic-free. `VcpuPanicCtx`
/// is owned by the `VCPU_PANIC_CTX` thread-local slot and dropped
/// when that slot is cleared — potentially during unwinding of a
/// `body()` panic in `with_vcpu_panic_ctx`. A panicking `Drop` in
/// that window produces a double-panic: the unwind-in-progress plus
/// the Drop panic → `std` aborts the process (even under the
/// test-profile `panic = "unwind"` setting), and the classified
/// shutdown signal this type exists to produce never reaches the
/// watchdog. `Arc<AtomicBool>` satisfies the invariant — its Drop is
/// an atomic decrement + optional `Box` deallocation, neither of
/// which panics. Any new field must uphold the same guarantee.
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
///
/// Install convention: callers MUST ensure no other panic hook is
/// installed process-wide AFTER this call. Any later
/// [`std::panic::set_hook`] replaces ours as the active hook; the
/// replacement sees ours as its `prev`, but for vCPU-thread panics our
/// hook is no longer invoked first, so the classified-shutdown
/// signaling documented on [`VcpuPanicCtx`] is bypassed. Embedders
/// with their own panic hook must install it BEFORE `install_once` so
/// our hook sits on top of theirs in the chain.
pub(crate) fn install_once() {
    HOOK_ONCE.call_once(|| {
        #[cfg(test)]
        INSTALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let prev = std::panic::take_hook();
        std::panic::set_hook(make_hook(prev));
    });
}

/// Install a custom `prev` hook wrapped by [`make_hook`] directly,
/// bypassing [`HOOK_ONCE`]. Test-only helper for verifying the
/// prev-hook chain fires on a panic that enters the vCPU hook.
/// Callers must hold [`HOOK_TEST_LOCK`] and restore the previous hook
/// via [`std::panic::set_hook`] before releasing the lock.
#[cfg(test)]
fn install_hook_with_prev_for_test(prev: Box<PanicHook>) {
    std::panic::set_hook(make_hook(prev));
}

/// Register `ctx` for the current thread, run `body`, then clear the
/// registration. Any panic inside `body` is observed by the hook
/// installed by [`install_once`]; regardless of whether `body`
/// returns normally or unwinds, a Drop guard clears the thread-local
/// before this function's stack frame exits so a future reuse of
/// this OS thread (unusual but possible if a runtime recycles
/// threads) does not carry stale context into an unrelated panic.
///
/// INVARIANT: `body()` must not hold a `borrow` or `borrow_mut` on
/// `VCPU_PANIC_CTX` across a potential panic — the hook needs a
/// `borrow()` to read the context, and an outstanding borrow would
/// make that `borrow()` panic (turning the hook into a nested
/// panic, which under panic=abort aborts without signaling). The
/// set- and clear-site `borrow_mut` windows scope strictly to one
/// statement each: the `set` releases before `body()` runs, and the
/// guard's `clear` runs after the hook has already fired and
/// released its shared borrow. The panic window inside `body()`
/// never overlaps a mutable borrow. Callers inside `body` must not
/// re-enter this module's thread local.
///
/// RAII via `CtxGuard`: the previous formulation cleared the slot
/// with an unconditional statement after `body()`, which was skipped
/// when `body()` unwound under the test profile (`panic = "unwind"`).
/// That left stale context in the thread-local for the next reuse
/// of this OS thread; if the runtime recycled the thread onto an
/// unrelated panic, the hook would fire flags that weren't meant for
/// it. Clearing via a Drop guard closes that window — Drop runs on
/// both the normal-return path and the unwinding path.
pub(crate) fn with_vcpu_panic_ctx<R>(ctx: VcpuPanicCtx, body: impl FnOnce() -> R) -> R {
    /// Clear-on-drop helper so the `VCPU_PANIC_CTX` slot is always
    /// reset, whether `body()` returns normally or unwinds. Drop is
    /// panic-free: the `borrow_mut()` is safe because the panic hook
    /// (which takes `borrow()`) has already fired and released by
    /// the time any unwinding Drop runs; writing `None` into the
    /// slot drops the stored `VcpuPanicCtx`, whose `Arc<AtomicBool>`
    /// fields are themselves panic-free per the type's INVARIANT.
    struct CtxGuard;
    impl Drop for CtxGuard {
        fn drop(&mut self) {
            VCPU_PANIC_CTX.with(|slot| {
                *slot.borrow_mut() = None;
            });
        }
    }

    VCPU_PANIC_CTX.with(|slot| {
        *slot.borrow_mut() = Some(ctx);
    });
    let _guard = CtxGuard;
    body()
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

    /// RAII guard in `with_vcpu_panic_ctx` must clear the thread-
    /// local on the unwind path too, not just normal return. Before
    /// the guard landed, the clear statement ran AFTER `body()`, so
    /// a panicking `body()` skipped it entirely — leaving stale ctx
    /// in the slot for the next reuse of this OS thread. Under the
    /// test profile (`panic = "unwind"`), `catch_unwind` here
    /// observes the panic; the slot-is-None assertion that follows
    /// proves the guard's Drop ran during unwind.
    #[test]
    fn with_vcpu_panic_ctx_clears_thread_local_on_unwind() {
        install_once();
        let ctx = VcpuPanicCtx {
            kill: Arc::new(AtomicBool::new(false)),
            exited: Arc::new(AtomicBool::new(false)),
        };
        std::thread::spawn(move || {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                with_vcpu_panic_ctx(ctx, || panic!("test: intended panic"));
            }));
            VCPU_PANIC_CTX.with(|slot| {
                assert!(
                    slot.borrow().is_none(),
                    "thread-local must be None after unwind — RAII guard must clear on drop, not just after body()",
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

    /// Stronger idempotency check: [`HOOK_ONCE`]'s body must run at
    /// most once regardless of how many [`install_once`] calls land,
    /// including across concurrent tests. Asserts the underlying
    /// [`INSTALL_COUNT`] counter is 1 after repeated calls — the
    /// original `install_once_is_idempotent` only proved absence of
    /// panic/deadlock, which does not distinguish "body ran once"
    /// from "body ran every call".
    #[test]
    fn install_once_body_runs_exactly_once() {
        install_once();
        let after_first = INSTALL_COUNT.load(Ordering::Relaxed);
        for _ in 0..20 {
            install_once();
        }
        let after_many = INSTALL_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            after_many, after_first,
            "HOOK_ONCE body ran more than once under repeated install_once calls",
        );
        assert!(
            after_many >= 1,
            "INSTALL_COUNT must reach 1 after install_once",
        );
    }

    /// A panic inside a registered context must still chain to the
    /// previously-installed panic hook. Guards against a regression
    /// where the tail `prev(info)` call is removed or skipped — the
    /// classified-shutdown signaling is harmless on its own, but the
    /// user-facing panic message / backtrace only reach stderr via
    /// the preserved prev chain.
    #[test]
    fn panic_inside_ctx_still_runs_prev_hook() {
        let _guard = HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::panic::take_hook();

        let prev_ran = Arc::new(AtomicBool::new(false));
        let prev_ran_c = prev_ran.clone();
        install_hook_with_prev_for_test(Box::new(move |_info| {
            prev_ran_c.store(true, Ordering::Release);
        }));

        let kill = Arc::new(AtomicBool::new(false));
        let exited = Arc::new(AtomicBool::new(false));
        let ctx = VcpuPanicCtx {
            kill: kill.clone(),
            exited: exited.clone(),
        };
        std::thread::spawn(move || {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                with_vcpu_panic_ctx(ctx, || panic!("test: prev-hook chain"));
            }));
        })
        .join()
        .unwrap();

        std::panic::set_hook(saved);

        assert!(
            prev_ran.load(Ordering::Acquire),
            "prev hook must run after our hook in the chain",
        );
        assert!(
            kill.load(Ordering::Acquire),
            "our hook must flip kill before delegating to prev",
        );
        assert!(
            exited.load(Ordering::Acquire),
            "our hook must flip exited before delegating to prev",
        );
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
