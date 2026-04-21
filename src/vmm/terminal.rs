//! Raw-mode terminal guard with RAII restore and signal-safe cleanup.
//!
//! Used by [`KtstrVm::run_interactive`](super::KtstrVm::run_interactive)
//! to put stdin into raw mode for the lifetime of the shell session,
//! restoring the original termios on every exit path — including
//! panics and process-killing signals (SIGINT/SIGTERM/SIGQUIT).

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
/// SIGTERM, SIGQUIT with SA_RESETHAND that restore termios via raw
/// libc::tcsetattr (async-signal-safe) before the default handler runs.
pub(crate) struct TerminalRawGuard {
    original: nix::sys::termios::Termios,
    fd: std::os::unix::io::RawFd,
    /// Previous signal actions, restored on drop.
    prev_sigint: libc::sigaction,
    prev_sigterm: libc::sigaction,
    prev_sigquit: libc::sigaction,
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
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = terminal_restore_signal_handler as *const () as usize;
            sa.sa_flags = libc::SA_RESETHAND;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGINT, &sa, &mut prev_sigint);
            libc::sigaction(libc::SIGTERM, &sa, &mut prev_sigterm);
            libc::sigaction(libc::SIGQUIT, &sa, &mut prev_sigquit);
        }

        Ok(Self {
            original,
            fd,
            prev_sigint,
            prev_sigterm,
            prev_sigquit,
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
}
