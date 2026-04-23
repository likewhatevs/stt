//! Raw-mode terminal guard with RAII restore and signal-safe cleanup.
//!
//! Used by [`KtstrVm::run_interactive`](super::KtstrVm::run_interactive)
//! to put stdin into raw mode for the lifetime of the shell session,
//! restoring the original termios on Drop and on a SA_RESETHAND
//! signal-handler chain. See [`TerminalRawGuard`] for the full catch
//! list, bypass list, and per-signal rationale.

use anyhow::{Context, Result};

/// Stdin fd for signal handler. Set by TerminalRawGuard::enter, cleared by Drop.
///
/// The signal handler's Acquire load on this fd is the single gate for
/// touching `SAVED_TERMIOS`: the `fd >= 0` observation happens-after
/// the Release stores of the termios pointer and INSTALLED flag.
static SAVED_TERMIOS_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Pointer to a boxed termios used by the signal handler for restore.
/// Written exclusively by `TerminalRawGuard::enter` after the
/// `SAVED_TERMIOS_INSTALLED` CAS succeeds; read by the signal handler.
/// Drop does NOT free the backing allocation — a signal may be
/// dispatched concurrently on another thread between our
/// `SAVED_TERMIOS_FD = -1` store and the handler's Acquire load, and
/// freeing would create a use-after-free.
///
/// The leak is bounded per enter/drop cycle: each successful
/// `enter()` allocates exactly one `sizeof(libc::termios)` and Drop
/// never frees it, so total process-lifetime allocation grows linearly
/// with the number of raw-mode install cycles. In practice that count
/// is small (each full `ktstr` invocation installs raw mode at most
/// once for interactive I/O), so the leak is negligible — but it is
/// O(N) in cycles, not O(1). If a future caller drives repeated
/// enter/drop cycles in a tight loop, the leak becomes observable and
/// a more sophisticated reclamation scheme (hazard pointer, epoch
/// reclamation) would be required.
static SAVED_TERMIOS: std::sync::atomic::AtomicPtr<libc::termios> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Single-writer guard for SAVED_TERMIOS. `enter()` performs a
/// CAS(false → true) before any state installation; re-entry fails
/// with an error rather than stomping a previously-installed termios
/// (which would leak the prior pointer AND desynchronize the signal
/// handler). Drop clears this so a subsequent `enter()` may install.
static SAVED_TERMIOS_INSTALLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Signal handler that restores terminal state then re-raises.
/// Async-signal-safe: uses only libc::tcsetattr (POSIX async-signal-safe)
/// and libc::raise. SA_RESETHAND restores SIG_DFL before entry, so the
/// re-raised signal terminates normally.
extern "C" fn terminal_restore_signal_handler(sig: libc::c_int) {
    // Gate 1: FD >= 0. Acquire-loads the fd stored last in enter(); if
    // negative (never installed, or Drop already cleared it) we have
    // nothing to restore and skip straight to re-raise.
    let fd = SAVED_TERMIOS_FD.load(std::sync::atomic::Ordering::Acquire);
    if fd >= 0 {
        // Gate 2: non-null termios pointer. The Acquire load pairs with
        // the Release store in enter() (ordered before the FD store),
        // so observing fd >= 0 implies a valid termios write
        // happens-before the pointer load. Drop stores null here AFTER
        // clearing FD, so a concurrent handler that saw fd >= 0 is
        // guaranteed to observe a still-valid pointer (the leak policy
        // on SAVED_TERMIOS ensures the allocation outlives any
        // in-flight handler invocation).
        //
        // SA_RESETHAND ensures this handler runs at most once per
        // delivered signal; we cannot race with ourselves on the same
        // thread.
        let ptr = SAVED_TERMIOS.load(std::sync::atomic::Ordering::Acquire);
        if !ptr.is_null() {
            // SAFETY: ptr was produced by Box::into_raw in enter() and
            // is never freed (see SAVED_TERMIOS static doc). tcsetattr
            // is POSIX async-signal-safe.
            unsafe {
                libc::tcsetattr(fd, libc::TCSANOW, ptr);
            }
        }
    }
    // SA_RESETHAND already restored SIG_DFL; re-raise terminates.
    // SAFETY: libc::raise is POSIX async-signal-safe.
    unsafe {
        libc::raise(sig);
    }
}

/// Sets stdin to raw mode on creation, restores original termios on drop.
/// Handles panic paths via Drop. Installs signal handlers for SIGINT,
/// SIGTERM, SIGQUIT, SIGABRT, and SIGFPE with SA_RESETHAND that
/// restore termios via raw libc::tcsetattr (async-signal-safe) before
/// the default handler runs. SIGABRT catches the `panic = "abort"`
/// release-build path where an unwind-less panic calls `abort(3)`
/// instead of unwinding Drop. SIGFPE catches a synchronous integer
/// divide-by-zero / invalid FP operation: the process memory state
/// is still coherent (unlike SIGSEGV/SIGBUS/SIGILL), so the tcsetattr
/// from the handler is safe; the re-raised default handler then runs
/// the core dump with the terminal already back in cooked mode.
///
/// # Guard bypass paths
///
/// Handled set: `SIGINT`, `SIGTERM`, `SIGQUIT`, `SIGABRT`, `SIGFPE`.
/// Any signal not in that set is bypassed — the guard does NOT
/// install a handler for it, so termios is not restored on delivery.
/// Neither the Drop path nor the SA_RESETHAND handler fires in these
/// cases, leaving the terminal in raw mode after process exit:
/// - **SIGSEGV / SIGBUS / SIGILL**: synchronous hardware-fault
///   signals that fire from corrupted process state where
///   `tcsetattr` is unsafe; deliberately left uncaught. The kernel's
///   default action produces a core dump without running userspace
///   recovery.
/// - **Any other signal** (`SIGHUP`, `SIGPIPE`, `SIGALRM`, `SIGUSR1`,
///   `SIGUSR2`, `SIGTSTP`, etc.): bypassed by design. Only the five
///   signals above arm a restore-on-delivery handler; every other
///   signal runs under whatever disposition was installed before the
///   guard was entered.
/// - **`std::process::exit` / `libc::_exit`**: skip Drop entirely.
/// - **SIGKILL**: uncatchable by design, no handler runs.
///
/// If any of these fires, the shell will appear broken (no echo, no
/// line editing). Run `stty sane` (or `reset`) in the shell to
/// restore cooked mode.
pub(crate) struct TerminalRawGuard {
    original: nix::sys::termios::Termios,
    fd: std::os::unix::io::RawFd,
    /// Previous signal actions, restored on drop.
    prev_sigint: libc::sigaction,
    prev_sigterm: libc::sigaction,
    prev_sigquit: libc::sigaction,
    prev_sigabrt: libc::sigaction,
    prev_sigfpe: libc::sigaction,
}

impl TerminalRawGuard {
    /// Set stdin to raw mode. Returns the guard that restores on drop.
    pub(crate) fn enter() -> Result<Self> {
        use nix::sys::termios::{self, SetArg};
        use std::os::unix::io::AsRawFd;

        // Structural single-writer enforcement: only one
        // TerminalRawGuard may hold the termios-restore statics at a
        // time. CAS false → true before touching ANY terminal state,
        // so a second concurrent enter() fails here instead of
        // silently leaking the prior boxed termios.
        if SAVED_TERMIOS_INSTALLED
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            anyhow::bail!(
                "TerminalRawGuard already installed; only one raw-mode guard may be live at a time"
            );
        }

        let fd = std::io::stdin().as_raw_fd();
        // SAFETY: stdin fd is valid for the lifetime of this process.
        let borrowed = unsafe { std::os::unix::io::BorrowedFd::borrow_raw(fd) };
        let original = match termios::tcgetattr(borrowed).context("tcgetattr") {
            Ok(t) => t,
            Err(e) => {
                // Release the installation guard so a subsequent caller
                // can try again; we haven't written any statics yet.
                SAVED_TERMIOS_INSTALLED.store(false, std::sync::atomic::Ordering::Release);
                return Err(e);
            }
        };
        let mut raw = original.clone();
        termios::cfmakeraw(&mut raw);
        if let Err(e) = termios::tcsetattr(borrowed, SetArg::TCSANOW, &raw).context("tcsetattr raw")
        {
            SAVED_TERMIOS_INSTALLED.store(false, std::sync::atomic::Ordering::Release);
            return Err(e);
        }

        // Leak the termios into the AtomicPtr. Drop does not free
        // (see static doc on SAVED_TERMIOS) — a concurrent signal
        // handler may still be reading the pointer when Drop runs.
        let boxed: Box<libc::termios> = Box::new(original.clone().into());
        let ptr = Box::into_raw(boxed);
        // Store pointer and fd with Release so the handler's Acquire
        // load of FD happens-after the pointer is visible. INSTALLED
        // was already set by the CAS above.
        SAVED_TERMIOS.store(ptr, std::sync::atomic::Ordering::Release);
        SAVED_TERMIOS_FD.store(fd, std::sync::atomic::Ordering::Release);

        // Install signal handlers with SA_RESETHAND. Matches the raw libc
        // pattern used by register_vcpu_signal_handler.
        let mut prev_sigint: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut prev_sigterm: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut prev_sigquit: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut prev_sigabrt: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut prev_sigfpe: libc::sigaction = unsafe { std::mem::zeroed() };
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = terminal_restore_signal_handler as *const () as usize;
            sa.sa_flags = libc::SA_RESETHAND;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGINT, &sa, &mut prev_sigint);
            libc::sigaction(libc::SIGTERM, &sa, &mut prev_sigterm);
            libc::sigaction(libc::SIGQUIT, &sa, &mut prev_sigquit);
            // SIGABRT covers the `panic = "abort"` release path:
            // `abort(3)` raises SIGABRT on the panicking thread, which
            // terminates the process after the default handler runs.
            // Installing our restore handler with SA_RESETHAND means
            // the termios is restored before the default SIGABRT
            // handler generates the core dump / exits the process.
            libc::sigaction(libc::SIGABRT, &sa, &mut prev_sigabrt);
            // SIGFPE fires on integer divide-by-zero or invalid FP
            // operation. Unlike SIGSEGV/SIGBUS/SIGILL the process
            // memory state is coherent, so tcsetattr from the handler
            // is safe; the re-raised SIGFPE then runs the default
            // core dump with the terminal already restored.
            libc::sigaction(libc::SIGFPE, &sa, &mut prev_sigfpe);
        }

        Ok(Self {
            original,
            fd,
            prev_sigint,
            prev_sigterm,
            prev_sigquit,
            prev_sigabrt,
            prev_sigfpe,
        })
    }
}

impl Drop for TerminalRawGuard {
    fn drop(&mut self) {
        // Disable the signal handler before restoring termios to prevent
        // a stale restore racing with our own restore below.
        SAVED_TERMIOS_FD.store(-1, std::sync::atomic::Ordering::Release);

        // Restore original termios.
        // SAFETY: fd was valid at construction, stdin persists for process lifetime.
        let borrowed = unsafe { std::os::unix::io::BorrowedFd::borrow_raw(self.fd) };
        let _ = nix::sys::termios::tcsetattr(
            borrowed,
            nix::sys::termios::SetArg::TCSANOW,
            &self.original,
        );

        // Clear the pointer and installation guard, but DO NOT free
        // the boxed termios: a signal handler dispatched on another
        // thread may still be executing between its Acquire load of
        // SAVED_TERMIOS_FD (before the store above retired) and its
        // subsequent load of SAVED_TERMIOS. Freeing here would create
        // a use-after-free window. The leak is one termios per
        // enter/drop cycle — total process-lifetime allocation grows
        // O(N) in cycles, not O(1). See the SAVED_TERMIOS static doc
        // for the full policy.
        SAVED_TERMIOS.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
        SAVED_TERMIOS_INSTALLED.store(false, std::sync::atomic::Ordering::Release);

        // Restore previous signal handlers.
        unsafe {
            libc::sigaction(libc::SIGINT, &self.prev_sigint, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &self.prev_sigterm, std::ptr::null_mut());
            libc::sigaction(libc::SIGQUIT, &self.prev_sigquit, std::ptr::null_mut());
            libc::sigaction(libc::SIGABRT, &self.prev_sigabrt, std::ptr::null_mut());
            libc::sigaction(libc::SIGFPE, &self.prev_sigfpe, std::ptr::null_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_raw_guard_double_enter_fails() {
        // Allocate a pty pair and redirect stdin to the slave so
        // TerminalRawGuard::enter()'s tcgetattr/tcsetattr calls see a
        // real tty. Each nextest test runs in its own process, so
        // dup2 on fd 0 is test-isolated.
        let mut master: libc::c_int = 0;
        let mut slave: libc::c_int = 0;
        let rc = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert_eq!(rc, 0, "openpty failed: {}", std::io::Error::last_os_error());
        let saved_stdin = unsafe { libc::dup(0) };
        assert!(saved_stdin >= 0);
        assert_eq!(unsafe { libc::dup2(slave, 0) }, 0);

        let first = TerminalRawGuard::enter().expect("first enter must succeed");
        let second = TerminalRawGuard::enter();
        let err_msg = match second {
            Ok(_) => panic!("second concurrent enter must fail the INSTALLED CAS"),
            Err(e) => e.to_string(),
        };
        assert!(
            err_msg.contains("already installed"),
            "error message should name the double-install condition, got: {err_msg}"
        );
        drop(first);
        let third = TerminalRawGuard::enter()
            .expect("third enter must succeed after Drop clears INSTALLED");
        drop(third);

        unsafe {
            libc::dup2(saved_stdin, 0);
            libc::close(saved_stdin);
            libc::close(slave);
            libc::close(master);
        }
    }

    #[test]
    fn terminal_raw_guard_enter_drop_cycle() {
        // Verify enter/drop can be repeated without getting stuck in
        // the INSTALLED=true state. Each iteration allocates a fresh
        // boxed termios (leaked by design — see the static doc on
        // SAVED_TERMIOS) and transitions INSTALLED through
        // false→true→false.
        let mut master: libc::c_int = 0;
        let mut slave: libc::c_int = 0;
        let rc = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert_eq!(rc, 0, "openpty failed: {}", std::io::Error::last_os_error());
        let saved_stdin = unsafe { libc::dup(0) };
        assert!(saved_stdin >= 0);
        assert_eq!(unsafe { libc::dup2(slave, 0) }, 0);

        for i in 0..3 {
            let guard = TerminalRawGuard::enter()
                .unwrap_or_else(|e| panic!("enter iteration {i} must succeed, got: {e}"));
            drop(guard);
        }

        unsafe {
            libc::dup2(saved_stdin, 0);
            libc::close(saved_stdin);
            libc::close(slave);
            libc::close(master);
        }
    }

    /// Regression pin for the SIGABRT arm of the signal-handler set.
    /// Before `enter()`, install `SIG_IGN` as a sentinel disposition
    /// for SIGABRT. After `enter()`, the live disposition must NO
    /// LONGER be `SIG_IGN` — proof `enter()` replaced it with
    /// [`terminal_restore_signal_handler`]. After `drop(guard)`, the
    /// disposition must return to the `SIG_IGN` sentinel — proof
    /// Drop stored and restored the previous sigaction rather than
    /// clobbering it to `SIG_DFL` (which would re-enable SIGABRT's
    /// default core-dump on a later abort the caller did not raise
    /// themselves).
    #[test]
    fn terminal_raw_guard_installs_and_restores_sigabrt_handler() {
        // Same openpty + dup2 stdin dance the other tests use so
        // enter()'s tcgetattr / tcsetattr see a real tty.
        let mut master: libc::c_int = 0;
        let mut slave: libc::c_int = 0;
        let rc = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert_eq!(rc, 0, "openpty failed: {}", std::io::Error::last_os_error());
        let saved_stdin = unsafe { libc::dup(0) };
        assert!(saved_stdin >= 0);
        assert_eq!(unsafe { libc::dup2(slave, 0) }, 0);

        // Install SIG_IGN as the pre-enter sentinel and save the
        // ORIGINAL disposition so the test can hand the process back
        // to the test runner with whatever was in place before.
        let mut pre_test: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut ign: libc::sigaction = unsafe { std::mem::zeroed() };
        ign.sa_sigaction = libc::SIG_IGN;
        unsafe {
            libc::sigemptyset(&mut ign.sa_mask);
            libc::sigaction(libc::SIGABRT, &ign, &mut pre_test);
        }

        // Sanity-check the sentinel is in place.
        let mut current: libc::sigaction = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigaction(libc::SIGABRT, std::ptr::null(), &mut current);
        }
        assert_eq!(
            current.sa_sigaction,
            libc::SIG_IGN,
            "test setup: SIG_IGN sentinel must be installed before enter()",
        );

        let guard = TerminalRawGuard::enter().expect("enter must succeed");

        // After enter() the disposition MUST NOT be SIG_IGN (or
        // SIG_DFL) — it must be a real handler function pointer,
        // i.e. [`terminal_restore_signal_handler`].
        unsafe {
            libc::sigaction(libc::SIGABRT, std::ptr::null(), &mut current);
        }
        assert_ne!(
            current.sa_sigaction,
            libc::SIG_IGN,
            "enter() must replace the SIGABRT SIG_IGN sentinel with its own handler",
        );
        assert_ne!(
            current.sa_sigaction,
            libc::SIG_DFL,
            "enter() must not leave SIGABRT at SIG_DFL",
        );
        let expected = terminal_restore_signal_handler as *const () as usize;
        assert_eq!(
            current.sa_sigaction, expected,
            "enter() must point SIGABRT at terminal_restore_signal_handler",
        );

        drop(guard);

        // Drop must restore the SIG_IGN sentinel verbatim.
        unsafe {
            libc::sigaction(libc::SIGABRT, std::ptr::null(), &mut current);
        }
        assert_eq!(
            current.sa_sigaction,
            libc::SIG_IGN,
            "Drop must restore the previous SIGABRT SIG_IGN sentinel",
        );

        // Hand the process back to whatever disposition was in
        // place before the test ran. Close the pty pair + restore
        // stdin last so any assertion failure above still surfaces
        // the original tty.
        unsafe {
            libc::sigaction(libc::SIGABRT, &pre_test, std::ptr::null_mut());
            libc::dup2(saved_stdin, 0);
            libc::close(saved_stdin);
            libc::close(slave);
            libc::close(master);
        }
    }

    /// Regression pin for the SIGFPE arm of the signal-handler set.
    /// Mirrors `terminal_raw_guard_installs_and_restores_sigabrt_handler`
    /// for the SIGFPE handler added so a synchronous FP trap (integer
    /// divide-by-zero, invalid FP op) restores cooked mode before the
    /// kernel's default core dump runs. Install `SIG_IGN` as a
    /// sentinel, verify `enter()` overwrites it with
    /// [`terminal_restore_signal_handler`], then verify `drop(guard)`
    /// restores the sentinel verbatim.
    #[test]
    fn terminal_raw_guard_installs_and_restores_sigfpe_handler() {
        let mut master: libc::c_int = 0;
        let mut slave: libc::c_int = 0;
        let rc = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert_eq!(rc, 0, "openpty failed: {}", std::io::Error::last_os_error());
        let saved_stdin = unsafe { libc::dup(0) };
        assert!(saved_stdin >= 0);
        assert_eq!(unsafe { libc::dup2(slave, 0) }, 0);

        let mut pre_test: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut ign: libc::sigaction = unsafe { std::mem::zeroed() };
        ign.sa_sigaction = libc::SIG_IGN;
        unsafe {
            libc::sigemptyset(&mut ign.sa_mask);
            libc::sigaction(libc::SIGFPE, &ign, &mut pre_test);
        }

        let mut current: libc::sigaction = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigaction(libc::SIGFPE, std::ptr::null(), &mut current);
        }
        assert_eq!(
            current.sa_sigaction,
            libc::SIG_IGN,
            "test setup: SIG_IGN sentinel must be installed before enter()",
        );

        let guard = TerminalRawGuard::enter().expect("enter must succeed");

        // After enter() the disposition MUST NOT be SIG_IGN (or
        // SIG_DFL) — it must be a real handler function pointer,
        // i.e. [`terminal_restore_signal_handler`]. Mirrors the gate
        // pattern in the SIGABRT test so the same regression shape
        // (enter() leaves sentinel in place, or clobbers to SIG_DFL
        // re-enabling the core dump behavior the handler is meant
        // to sequence around) is pinned for both signals.
        unsafe {
            libc::sigaction(libc::SIGFPE, std::ptr::null(), &mut current);
        }
        assert_ne!(
            current.sa_sigaction,
            libc::SIG_IGN,
            "enter() must replace the SIGFPE SIG_IGN sentinel with its own handler",
        );
        assert_ne!(
            current.sa_sigaction,
            libc::SIG_DFL,
            "enter() must not leave SIGFPE at SIG_DFL",
        );
        let expected = terminal_restore_signal_handler as *const () as usize;
        assert_eq!(
            current.sa_sigaction, expected,
            "enter() must point SIGFPE at terminal_restore_signal_handler",
        );

        drop(guard);

        unsafe {
            libc::sigaction(libc::SIGFPE, std::ptr::null(), &mut current);
        }
        assert_eq!(
            current.sa_sigaction,
            libc::SIG_IGN,
            "Drop must restore the previous SIGFPE SIG_IGN sentinel",
        );

        unsafe {
            libc::sigaction(libc::SIGFPE, &pre_test, std::ptr::null_mut());
            libc::dup2(saved_stdin, 0);
            libc::close(saved_stdin);
            libc::close(slave);
            libc::close(master);
        }
    }
}
