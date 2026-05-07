/// Rust init (PID 1) for the VM guest.
///
/// When the test binary is
/// packed as `/init` in the initramfs, `ktstr_guest_init()` is called
/// from the ctor when PID 1 is detected.
/// It never returns — it mounts filesystems, then either dispatches
/// a test (start scheduler, run test, reboot) or drops into an
/// interactive shell (when `KTSTR_MODE=shell` is on the kernel
/// cmdline).
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::sync::Latch;

use nix::mount::{MsFlags, mount};
use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::pty::openpty;
use nix::sys::reboot::{RebootMode, reboot};
use nix::sys::termios::{SetArg, cfmakeraw, tcgetattr, tcsetattr};

/// COM2 device path for sentinel and diagnostic output.
const COM2: &str = "/dev/ttyS1";
/// COM1 device path for kernel console / trace output.
const COM1: &str = "/dev/ttyS0";
/// Virtio-console device path. Used for shell I/O when available.
const HVC0: &str = "/dev/hvc0";

/// tracefs enable gate for the `sched_ext_dump` tracepoint. Writing
/// `"1"` activates the event, `"0"` deactivates it.
const TRACE_SCHED_EXT_DUMP_ENABLE: &str =
    "/sys/kernel/tracing/events/sched_ext/sched_ext_dump/enable";
/// Global tracefs on/off switch. Writing `"0"` stops new events from
/// being recorded into the ring buffer (`ring_buffer_record_off`); the
/// userspace trace_pipe reader still has to drain whatever is already
/// buffered before reboot. Disabling the producer side first is what
/// makes the reader's drain window terminate — once no new events
/// arrive, poll eventually returns 0 and the drain_deadline elapses.
const TRACE_TRACING_ON: &str = "/sys/kernel/tracing/tracing_on";
/// tracefs streaming endpoint for the active trace. The trace_pipe
/// reader opens this once per boot and forwards every line to COM1.
const TRACE_PIPE: &str = "/sys/kernel/tracing/trace_pipe";

/// sysfs attribute exposing the active sched_ext root scheduler's
/// name. Empty / absent when no scheduler is registered; populated
/// (with a trailing newline) when registration has completed.
/// Kernel-side owner: `kernel/sched/ext.c` creates this via
/// `kobject_init_and_add` under the `sched_ext` kset after
/// `sch->ops.name` is set.
const SYSFS_SCHED_EXT_ROOT_OPS: &str = "/sys/kernel/sched_ext/root/ops";

/// Reboot immediately. Used for fatal init errors and normal shutdown.
fn force_reboot() -> ! {
    let _ = reboot(RebootMode::RB_AUTOBOOT);
    // The kernel is rebooting — no event will ever fire. Park the
    // thread forever; this is cheaper than a sleep loop because
    // `park` blocks in the kernel without a wake-up timer attached.
    // No `unpark` call exists in this path; the process dies when
    // the reboot syscall completes.
    loop {
        std::thread::park();
    }
}

/// Side channel for the scheduler PID published by [`start_scheduler`]
/// once `Command::spawn` returns. The guest test-dispatch path
/// (e.g. [`crate::test_support`] consumers that need the scheduler's
/// pid for cgroup attach / kill / probe) reads it via [`sched_pid`].
///
/// Replaces a previous `std::env::set_var("SCHED_PID", ...)` write.
/// Mutating glibc's global `__environ` array while another thread is
/// live (the Phase A probe thread spawned in `start_probe_phase_a`
/// runs concurrently with `start_scheduler`) is documented UB on
/// Linux — see
/// [`crate::test_support::propagate_rust_env_from_cmdline`] for the
/// mirroring rationale. An atomic side channel is the
/// data-race-free alternative.
///
/// Sentinel: `0` means "no scheduler started". `pid_t` is a signed
/// integer in glibc; the kernel never returns `0` from `fork(2)` to
/// the parent, so `0` is a safe "unset" marker for the producer to
/// initialise with and the consumer to filter on.
static SCHED_PID: AtomicI32 = AtomicI32::new(0);

/// Read the scheduler PID published by [`start_scheduler`]. Returns
/// `None` when the scheduler has not been spawned yet (the atomic
/// reads as `0`, the sentinel for "unset"). `Acquire` synchronises
/// against the producer's `Release` store so any side effects
/// `start_scheduler` performed before the publish are visible to the
/// reader.
pub(crate) fn sched_pid() -> Option<libc::pid_t> {
    let v = SCHED_PID.load(Ordering::Acquire);
    if v == 0 { None } else { Some(v) }
}

/// RAII guard that flips SIGCHLD to a target disposition on
/// construction and restores the previous handler on drop. Used by
/// [`with_sigchld_default`] so a panic inside the closure cannot
/// leak `SIG_DFL` into the rest of the guest's lifetime — Drop
/// runs even on unwind.
///
/// `libc::signal` returns the previous handler on every call, so
/// the snapshot we capture in `install` is the authoritative value
/// to restore in `Drop`. Re-installing the snapshot makes the
/// guard idempotent across nested calls (an outer guard's restore
/// observes the inner guard's restore as a no-op rebind to the
/// same handler).
struct SigchldDispositionGuard {
    prev: libc::sighandler_t,
}

impl SigchldDispositionGuard {
    /// Install `handler` as the SIGCHLD disposition and capture
    /// the previous handler for restoration on drop.
    ///
    /// SAFETY: signal disposition is a process-wide property. PID
    /// 1 owns the disposition for the whole guest, so no other
    /// thread can race the signal install. `libc::signal` is
    /// async-signal-safe per POSIX.1-2008 TC2.
    ///
    /// # Panics
    ///
    /// Panics if `libc::signal` returns `SIG_ERR` — the libc
    /// failure indicator (`!0 as sighandler_t`) for an invalid
    /// signal number or other install failure. Without the check,
    /// `SIG_ERR` would be captured into `prev` as if it were a
    /// valid handler, and Drop would then attempt to install
    /// `SIG_ERR` (which the kernel rejects with `EINVAL`,
    /// surfacing as a separate `SIG_ERR` return that the no-check
    /// Drop also drops on the floor — silently leaking the
    /// install error). For SIGCHLD the failure path is
    /// implausible in practice (the signal number is valid and
    /// `SIG_DFL`/`SIG_IGN` are always-installable handlers), but
    /// the library invariant is general — `signal(2)` returning
    /// `SIG_ERR` is a programming error, not a runtime condition,
    /// so panicking is the right discipline.
    fn install(handler: libc::sighandler_t) -> Self {
        let prev = unsafe { libc::signal(libc::SIGCHLD, handler) };
        assert_ne!(
            prev,
            libc::SIG_ERR,
            "failed to install SIGCHLD handler — libc::signal returned SIG_ERR; \
             check signum / handler validity",
        );
        Self { prev }
    }
}

impl Drop for SigchldDispositionGuard {
    fn drop(&mut self) {
        // SAFETY: `self.prev` was returned by an earlier
        // `libc::signal` call on the same signal number, so
        // re-installing it is the documented restore pattern. The
        // `Drop` runs on both the normal-return and panic-unwind
        // paths, so a panic inside the protected closure cannot
        // leak the temporary disposition into the rest of the
        // process.
        unsafe {
            libc::signal(libc::SIGCHLD, self.prev);
        }
    }
}

/// Run `f` with SIGCHLD temporarily restored to `SIG_DFL` so the
/// kernel does not auto-reap any child spawned inside the closure.
/// `Command::status()` calls `waitpid(2)`, which returns `ECHILD`
/// when SIGCHLD is `SIG_IGN` (the default installed by
/// [`ktstr_guest_init`] for zombie prevention) — losing the real
/// exit status. Restoring `SIG_DFL` for the closure's lifetime
/// re-enables `waitpid` reaping; the post-closure restore puts
/// the previous disposition back so subsequent guest children
/// continue to be auto-reaped without leaking zombies.
///
/// Mirrors the inline save/restore pattern formerly open-coded at
/// the [`ktstr_guest_init`] shell `--exec` site (now also routed
/// through this helper). Both call sites share the same
/// SIGCHLD-vs-`waitpid` hazard; centralising the helper prevents
/// drift between the two implementations.
///
/// Restore is panic-safe via [`SigchldDispositionGuard`]: a panic
/// in `f` runs the guard's `Drop`, which re-installs the previous
/// SIGCHLD handler before unwinding past the helper boundary.
/// Without the guard, a panicking child-spawn site would leak
/// `SIG_DFL` into the rest of the guest, breaking PID 1's zombie
/// reaping for every subsequent fork.
///
/// The closure must reap every child it spawns before returning.
/// Leaving an unreaped child at the boundary where `SIG_IGN` is
/// restored would orphan the zombie until the next reaper cycle.
/// `Command::status()` waits synchronously, so the typical caller
/// satisfies this invariant by construction.
fn with_sigchld_default<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = SigchldDispositionGuard::install(libc::SIG_DFL);
    f()
}

/// Whether `/proc/{pid}` exists. Used as a `waitpid`-free liveness
/// probe: under SIGCHLD `SIG_IGN` the kernel auto-reaps children, so
/// `waitpid` returns `ECHILD` even when the child exited cleanly.
/// `/proc/{pid}` removal is signal-disposition-independent — the
/// directory disappears the moment the kernel finishes
/// `release_task` for the pid (see kernel/exit.c
/// `release_task` → `proc_flush_pid`), regardless of whether
/// `waitpid` ever ran.
///
/// Returns `true` when `/proc/{pid}` exists (process alive or
/// pre-reap), `false` when it does not (process exited and the
/// kernel has dropped the procfs entry).
fn proc_pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Async-signal-safe rendering of `value` as lowercase hex (no `0x`
/// prefix, no leading-zero trim) into the tail of `buf`. Returns the
/// byte slice covering the rendered digits.
///
/// Used by [`fatal_signal_handler`], where every libc allocator
/// boundary is forbidden — `format!`, `write!`, and even
/// `core::fmt::Display` formatters can pull in heap or thread-local
/// state. A hand-rolled nibble walk over a stack buffer is the only
/// AS-safe way to surface the faulting address.
///
/// 16 hex digits cover the full `u64` range. The caller passes a
/// `[u8; 16]` and uses the returned subslice (always exactly 16
/// bytes) directly.
fn u64_to_hex_asm(value: u64, buf: &mut [u8; 16]) -> &[u8] {
    static HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, slot) in buf.iter_mut().enumerate() {
        let nibble = (value >> ((15 - i) * 4)) & 0xf;
        *slot = HEX[nibble as usize];
    }
    &buf[..]
}

/// AS-safe write of every byte in `bytes` to fd `fd`. Loops on partial
/// writes; bails on the first error or zero-byte return so a closed/
/// faulted fd cannot wedge the handler.
fn write_all_asm(fd: libc::c_int, bytes: &[u8]) {
    let mut off = 0;
    while off < bytes.len() {
        // SAFETY: `write(2)` is async-signal-safe per signal-safety(7)
        // on Linux. `bytes.as_ptr().add(off)` is in-bounds because
        // `off < bytes.len()`. The write is best-effort — any
        // failure short-circuits the loop and the handler proceeds
        // to `reboot(2)`.
        let n = unsafe {
            libc::write(
                fd,
                bytes.as_ptr().add(off) as *const libc::c_void,
                bytes.len() - off,
            )
        };
        if n <= 0 {
            return;
        }
        off += n as usize;
    }
}

/// Async-signal-safe handler for SIGSEGV / SIGBUS / SIGILL.
///
/// The Rust panic hook installed in [`ktstr_guest_init`] does NOT
/// fire for native CPU faults: the kernel raises these signals with
/// `SIG_DFL` disposition, which calls `do_coredump` and terminates
/// the process. Inside guest init that means PID 1 dies, the kernel
/// observes "init exited", and the host sees the VM force-reboot
/// without any guest-side diagnostic on COM2.
///
/// This handler closes the gap by emitting a `PANIC:`-prefixed line
/// — matching the prefix [`extract_panic_message`] anchors on — that
/// names the signal and the faulting address before driving
/// [`force_reboot`]. The host crash-classification pipeline then
/// surfaces native faults through the same code path as Rust panics.
///
/// Constraints, all enforced inside the handler:
///
/// - Async-signal-safety per `signal-safety(7)`. No `fs::write`, no
///   `format!`, no `Backtrace::force_capture` — all of those touch
///   the heap, locks, or per-thread formatter state. Only `open(2)`,
///   `write(2)`, `tcdrain(2)`, and `reboot(2)` (all in the AS-safe
///   list) are invoked, plus pure stack arithmetic.
/// - No thread-local state. Worker threads spawned later
///   (`hvc0_poll_loop`, `start_trace_pipe`) inherit the parent's
///   sigaction disposition because Linux signal dispositions are
///   process-wide; this handler runs on whichever thread faulted.
/// - Bounded recursion. `SA_RESETHAND` is set so a fault inside this
///   handler reverts to `SIG_DFL`, which terminates immediately
///   instead of looping.
unsafe extern "C" fn fatal_signal_handler(
    sig: libc::c_int,
    info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    // Static prefixes per signal. Hard-coded because signal-name
    // formatting via `strsignal(3)` allocates / touches locale
    // state and is not AS-safe.
    let prefix: &[u8] = match sig {
        libc::SIGSEGV => b"PANIC: fatal signal SIGSEGV at addr 0x",
        libc::SIGBUS => b"PANIC: fatal signal SIGBUS at addr 0x",
        libc::SIGILL => b"PANIC: fatal signal SIGILL at addr 0x",
        _ => b"PANIC: fatal signal (unknown) at addr 0x",
    };

    // Faulting address from `siginfo_t.si_addr`. `siginfo_t` field
    // access in Rust requires going through the libc bindings;
    // `si_addr()` is the canonical accessor that handles the union
    // layout differences between glibc and musl. Falls back to 0
    // when `info` is null. Defensive null check; Linux always
    // populates info for SA_SIGINFO handlers (see kernel/signal.c
    // `force_sig_fault_to_task` → `force_sig_info_to_task` and the
    // arch `setup_rt_frame` paths, which unconditionally pass
    // `&frame->info` to the handler).
    let addr: u64 = if info.is_null() {
        0
    } else {
        // SAFETY: `info` is non-null here; `si_addr()` reads the
        // address-fault arm of the siginfo union, which is the
        // valid arm for SIGSEGV / SIGBUS / SIGILL per the kernel's
        // `force_sig_fault` path (`kernel/signal.c`).
        let p = unsafe { (*info).si_addr() };
        p as u64
    };

    let mut hex_buf = [0u8; 16];
    let hex = u64_to_hex_asm(addr, &mut hex_buf);

    // Open COM2 first (canonical destination), then COM1. Both with
    // `O_WRONLY | O_NONBLOCK` so the open and the `write_all_asm`
    // loop never block on guest-side flow control. `tcdrain(2)`
    // does NOT honor `O_NONBLOCK` — it is a separate ioctl that
    // waits for the kernel tty layer's write queue to drain — but
    // the wait is bounded by UART FIFO drain time (microseconds at
    // worst) because PIO commits each byte inside `KVM_RUN` before
    // userspace returns; the kernel sees its own output queue empty
    // almost immediately after the final `write(2)` returns.
    //
    // SAFETY: `open(2)`, `write(2)`, `tcdrain(2)`, and `close(2)`
    // are all in the signal-safety(7) AS-safe set. The path
    // strings are static C strings with explicit NUL terminators.
    for path in [c"/dev/ttyS1", c"/dev/ttyS0"] {
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_NONBLOCK) };
        if fd < 0 {
            continue;
        }
        write_all_asm(fd, prefix);
        write_all_asm(fd, hex);
        write_all_asm(fd, b"\n");
        // Seal the contract: tcdrain waits for the kernel's output
        // queue to drain before we issue `reboot(2)`. PIO commits
        // per byte so the wait is effectively immediate; tcdrain
        // ignores `O_NONBLOCK` but the drain time is bounded by
        // UART FIFO depth, not by host-side back-pressure.
        unsafe {
            libc::tcdrain(fd);
            libc::close(fd);
        }
    }

    // `reboot(LINUX_REBOOT_CMD_RESTART)` is the AS-safe analogue of
    // `force_reboot()`'s nix wrapper. The syscall does not return
    // on success; if it somehow does (CAP_SYS_BOOT missing,
    // already rebooting), `_exit(1)` ensures the handler does
    // NOT fall through to user code with a corrupt stack /
    // mid-fault state.
    unsafe {
        libc::reboot(libc::LINUX_REBOOT_CMD_RESTART);
        libc::_exit(1);
    }
}

/// Install [`fatal_signal_handler`] for SIGSEGV, SIGBUS, and SIGILL.
///
/// `SA_SIGINFO` makes the handler receive the `siginfo_t *` whose
/// `si_addr` carries the faulting address. `SA_RESETHAND` reverts
/// the disposition to `SIG_DFL` after the first delivery so a fault
/// inside the handler terminates cleanly instead of looping. `SA_ONSTACK`
/// directs the kernel to run the handler on the alternate stack
/// registered via `sigaltstack(2)` below — without it a stack-overflow
/// SIGSEGV faults again on the overflowed stack and the kernel
/// terminates the process before any diagnostic reaches the host.
///
/// `sa_mask` adds SIGSEGV / SIGBUS / SIGILL so that while one fatal-
/// signal handler is executing, the other two cannot interrupt it.
/// Cross-signal nesting (e.g. SIGBUS arriving while the SIGSEGV
/// handler is mid-write to COM2) would scribble interleaved bytes
/// onto the serial output and lose the diagnostic. The signal being
/// delivered is also masked by default; combined with `SA_RESETHAND`
/// a re-fault of the same signal terminates under `SIG_DFL` instead
/// of looping back into this handler.
///
/// Failures are silently ignored: if `sigaction(2)` rejects the
/// install (returns -1), the previous disposition (typically
/// `SIG_DFL`) remains in place — which is exactly the pre-fix
/// behavior. There's no user-visible regression on failure, just
/// the unchanged gap the panic hook also doesn't cover. `mmap(2)` /
/// `sigaltstack(2)` failures are similarly tolerated: the handler
/// stays installed without `SA_ONSTACK`, which only loses the
/// stack-overflow diagnostic — every other fatal-signal path keeps
/// working.
fn install_fatal_signal_handlers() {
    // SAFETY: `std::mem::zeroed::<libc::sigaction>()` produces a
    // valid all-zero `sigaction` (all libc fields are integer or
    // pointer-typed, zero is valid for all of them). The
    // `sa_sigaction` field is then set to a function pointer with
    // the correct `extern "C"` signature, and `sa_flags` is set
    // to a valid combination of POSIX `SA_*` constants.
    let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
    act.sa_sigaction = fatal_signal_handler as *const () as usize;
    act.sa_flags = libc::SA_SIGINFO | libc::SA_RESETHAND;
    // Initialize the mask, then add every fatal signal so that one
    // handler in flight cannot be interrupted by another fatal
    // signal — see fn doc for why interleaved handlers corrupt the
    // diagnostic.
    unsafe {
        libc::sigemptyset(&mut act.sa_mask);
        libc::sigaddset(&mut act.sa_mask, libc::SIGSEGV);
        libc::sigaddset(&mut act.sa_mask, libc::SIGBUS);
        libc::sigaddset(&mut act.sa_mask, libc::SIGILL);
    }

    // Allocate and register a signal alternate stack so a stack-
    // overflow SIGSEGV runs the handler on a separate stack instead
    // of faulting again on the overflowed one. `SIGSTKSZ` is the
    // platform's recommended minimum; clamp to 64 KiB so older libc
    // headers (where SIGSTKSZ is 8 KiB) still leave headroom for the
    // backtrace-free handler frame plus `write(2)` / `tcdrain(2)` /
    // `reboot(2)` syscall trampolines.
    //
    // `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` is the AS-safe-allocation
    // analogue to a heap allocation: pages are zero-initialised on
    // first touch, so no separate clear is needed. The mapping is
    // intentionally leaked — `sigaltstack` keeps the kernel pointing
    // at it for the lifetime of the process, and PID 1 never returns
    // from `ktstr_guest_init`.
    //
    // SAFETY: `mmap(2)` and `sigaltstack(2)` are both POSIX-defined
    // syscalls. The pointers / lengths supplied are well-formed
    // (NULL hint, fd=-1 for anonymous mappings, offset=0). Failure
    // returns `MAP_FAILED`; on that path we skip `sigaltstack` and
    // leave `SA_ONSTACK` unset on `sa_flags` — see fn doc for the
    // failure-mode rationale.
    let stack_size = libc::SIGSTKSZ.max(65536);
    let stack = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            stack_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if stack != libc::MAP_FAILED {
        let ss = libc::stack_t {
            ss_sp: stack,
            ss_flags: 0,
            ss_size: stack_size,
        };
        // SAFETY: `ss` is a fully-initialised `stack_t` with a
        // valid mmap'd buffer and matching size. Passing
        // `null_mut()` for `oss` discards the previous alternate
        // stack — PID 1 has no prior alternate stack at this
        // call site (signal handling has not been touched yet).
        unsafe {
            libc::sigaltstack(&ss, std::ptr::null_mut());
        }
        act.sa_flags |= libc::SA_ONSTACK;
    }

    for sig in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL] {
        // SAFETY: `sigaction(2)` with a valid `struct sigaction`
        // and a NULL old-action pointer is well-defined.
        // Failures are silently swallowed (see fn doc).
        let _ = unsafe { libc::sigaction(sig, &act, std::ptr::null_mut()) };
    }
}

/// Full guest init lifecycle. Called from the ctor when PID 1 is
/// detected. Mounts filesystems, then either runs the test lifecycle
/// (scheduler + dispatch + reboot) or drops into an interactive
/// shell. Never returns.
pub(crate) fn ktstr_guest_init() -> ! {
    let t0 = std::time::Instant::now();

    // Crash diagnostic capture has two arms because they have
    // disjoint trigger surfaces:
    //
    // 1. Native fatal signals (`install_fatal_signal_handlers`,
    //    installed first): SIGSEGV / SIGBUS / SIGILL invoke the
    //    kernel's `do_coredump` under SIG_DFL — they bypass the
    //    panic hook entirely. Without a sigaction handler the
    //    kernel terminates init, which the parent kernel observes
    //    as "init exited" and force-reboots without any guest-side
    //    diagnostic reaching the host. Installing this arm before
    //    the panic hook minimises the window where an early fault
    //    (heap setup, mount syscalls, anything before the hook
    //    registers) escapes capture.
    // 2. Rust panic hook (below): fires on `panic!`, `unwrap`,
    //    assertion failures, and any other invocation of the Rust
    //    panic machinery (both `panic = "unwind"` and
    //    `panic = "abort"` runtimes invoke the hook before
    //    unwinding/aborting).
    //
    // Both arms write a `PANIC:`-prefixed line to COM2 (and COM1)
    // so the host-side `extract_panic_message` picks them up
    // through the same code path. COM2 is the canonical crash-
    // diagnostic transport, surviving a wedged virtio port: the
    // bulk-virtio path is intentionally NOT used here because the
    // kernel `virtio_console` TX can block on host backpressure
    // and blocking inside a fault handler would deadlock the
    // guest before the diagnostic reached the host. COM2 (16550
    // UART) PIO writes commit synchronously inside `KVM_RUN`
    // before userspace returns, so the host's serial capture
    // sees every byte even on a wedged guest.
    install_fatal_signal_handlers();
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        let msg = format!("PANIC: {info}\n{bt}\n");
        // COM2 / COM1 serial. COM2 is the canonical crash log
        // destination for the host's serial-capture path; the
        // host parses the `PANIC:` prefix via
        // `extract_panic_message` to reconstruct the crash
        // diagnostic.
        let _ = fs::write(COM2, &msg);
        let _ = fs::write(COM1, &msg);
        // Push any buffered Rust-side bytes into the underlying pipe
        // before reboot. After stdio redirect, fd 1 / fd 2 are
        // pipe write ends drained by `redirect_stdio_to_bulk_port`'s
        // forwarder threads — `tcdrain` is unavailable here (the
        // pipe is not a tty, the syscall returns ENOTTY silently).
        // `flush()` is the equivalent: it commits any
        // BufWriter-buffered bytes into the pipe's kernel buffer
        // where the forwarder thread can pick them up. The
        // forwarder threads are not joined before `force_reboot`;
        // bytes that have not yet been read out of the pipe and
        // shipped over the bulk port at the moment of reboot are
        // lost — see the queue task on joining the forwarders for
        // the residual gap. The COM1/COM2 `fs::write` above remains
        // the synchronous-PIO path that guarantees the panic
        // diagnostic itself reaches the host before reboot.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        force_reboot();
    }));

    // Ignore SIGCHLD so child processes don't become zombies.
    // PID 1 is the reaper — without this, zombie processes accumulate.
    unsafe {
        libc::signal(libc::SIGCHLD, libc::SIG_IGN);
    }

    // Phase 1: Mounts.
    mount_filesystems();
    let t_mounts = t0.elapsed();

    // Verify initramfs extraction completed. The sentinel file is the
    // last entry written by build_initramfs_base — its absence means
    // the kernel ran out of memory during cpio extraction. The memory
    // formula should prevent this; hitting it indicates an estimation bug.
    if !Path::new("/.ktstr_init_ok").exists() {
        // Dump dmesg to serial so the host sees the kernel OOM messages.
        if let Ok(raw) = rmesg::logs_raw(rmesg::Backend::Default, false) {
            let _ = fs::write(COM2, &raw);
            let _ = fs::write(COM1, &raw);
        }
        let msg = "FATAL: initramfs extraction incomplete — kernel ran out of \
                   memory during cpio extraction. This indicates a bug in ktstr's \
                   memory estimation. Please report this issue. As a workaround, \
                   try `--memory N` with a larger value.";
        let _ = fs::write(COM2, msg);
        let _ = fs::write(COM1, msg);
        eprintln!("{msg}");
        force_reboot();
    }

    // Boot-complete signal. The host monitor's pre-sample
    // `epoll_wait` blocks on a sys_rdy eventfd; the freeze
    // coordinator's bulk-drain dispatch promotes a CRC-valid
    // `MSG_TYPE_SYS_RDY` frame into that eventfd. Sending here —
    // after `mount_filesystems()` brought up devtmpfs so
    // `/dev/vport0p1` exists, and after the initramfs-extraction
    // sentinel confirms userspace is sound — guarantees the
    // host's first sample observes a fully-booted guest with
    // `setup_per_cpu_areas` populated and KASLR randomization
    // already complete (both kernel-boot prerequisites for the
    // monitor's `__per_cpu_offset[]` / `page_offset_base`
    // reads). Replaces the earlier trigger that fired on the
    // first port-0 TX byte (kernel printk via `/dev/hvc0`),
    // which depended on incidental console traffic rather than
    // an explicit readiness signal.
    //
    // The kernel virtio_console driver's multiport handshake
    // (DEVICE_READY → PORT_ADD → PORT_READY → PORT_OPEN, see
    // `drivers/char/virtio_console.c`) completes asynchronously
    // and is independent of devtmpfs being mounted. On a fast
    // boot the handshake can still be in flight when this
    // statement runs, so `send_sys_rdy()`'s lazy
    // `/dev/vport0p1` open returns `None` and the call returns
    // `false`. Retry up to 100 × 100 ms (10 s) — generous
    // enough to absorb cold-cache TRY 1 boots where the
    // multiport handshake may take several seconds. The host
    // monitor's pre-sample wait is bounded at 5 s; once that
    // expires the monitor falls through to its `data_valid`
    // gate and starts sampling, while THIS retry continues
    // running in the guest's init thread to deliver SYS_RDY
    // as soon as the device appears. Late delivery still
    // promotes the eventfd, but the freeze coordinator's
    // `Option::take` makes the promotion fire-once so a late
    // SYS_RDY past the host wait is harmless. If the full 10 s
    // budget exhausts, the guest continues with the rest of
    // init and the monitor's `data_valid` gate keeps reads
    // safe — the BSS-zero rejection in
    // [`super::super::monitor::reader`]'s sample loop tolerates
    // pre-boot zeros for as long as needed.
    let kern_phys_base = crate::vmm::guest_comms::read_phys_base_from_iomem().unwrap_or(0);
    for attempt in 0..100 {
        crate::vmm::guest_comms::send_kern_addrs(kern_phys_base, 0);
        if crate::vmm::guest_comms::send_sys_rdy() {
            break;
        }
        if attempt == 99 {
            tracing::warn!("ktstr-init: send_sys_rdy retry budget exhausted (10 s)");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Phase 1.5: Auto-mount the user data disk at /mnt/disk0 if the
    // host pre-formatted it (KTSTR_DISK0_FS=<tag> on the cmdline).
    // Runs BEFORE `disk_template_mode_requested()` is checked below
    // — but the template-build cmdline never carries
    // `KTSTR_DISK0_FS` (the host emits it only for non-Raw disks
    // and the template-build VM attaches a Raw disk because the
    // whole point is to format it), so this call is a no-op
    // during template-build and the build path is unaffected.
    auto_mount_data_disks();
    // Enable per-program BPF runtime stats (cnt, nsecs). The kernel
    // only populates bpf_prog_stats when bpf_stats_enabled_key is set.
    let _ = fs::write("/proc/sys/kernel/bpf_stats_enabled", "1");

    // Phase 2: Lifecycle event + stdio redirect. The lifecycle frame
    // is for the test harness on the host; shell mode doesn't need it
    // and would route the InitStarted phase into the operator's
    // bulk-port-backed transcript otherwise.
    if !shell_mode_requested() {
        crate::vmm::guest_comms::send_lifecycle(crate::vmm::wire::LifecyclePhase::InitStarted, "");
    }
    redirect_stdio_to_bulk_port();
    let t_stdio = t0.elapsed();

    // Extract RUST_LOG from kernel cmdline before installing the
    // tracing subscriber so EnvFilter picks it up.
    if let Ok(cmdline) = fs::read_to_string("/proc/cmdline")
        && let Some(val) = cmdline
            .split_whitespace()
            .find(|s| s.starts_with("RUST_LOG="))
            .and_then(|s| s.strip_prefix("RUST_LOG="))
    {
        // SAFETY: single-threaded PID 1 context.
        unsafe { std::env::set_var("RUST_LOG", val) };
    }

    // Install tracing subscriber so tracing calls in guest code produce
    // output on stderr (COM2). Without this, they are silently dropped.
    // EnvFilter respects RUST_LOG when set; default is `warn` so
    // teardown diagnostics (`tracing::warn!`, `tracing::error!`)
    // surface without requiring RUST_LOG to be plumbed through the
    // guest cmdline. `from_default_env()` alone would collapse to
    // the implicit `error` level and swallow warn-level output —
    // exactly the diagnostics needed to debug teardown failures.
    let t_pre_subscriber = t0.elapsed();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let t_subscriber = t0.elapsed();

    tracing::debug!(
        mount_ms = t_mounts.as_millis() as u64,
        stdio_ms = t_stdio.as_millis() as u64,
        pre_subscriber_ms = t_pre_subscriber.as_millis() as u64,
        subscriber_ms = t_subscriber.as_millis() as u64,
        "guest_init_timing",
    );

    // Set environment variables.
    // SAFETY: single-threaded context — PID 1 before any threads spawn.
    unsafe {
        std::env::set_var("PATH", build_include_path());
        // Mark this process tree as running under guest init (PID 1).
        // Workers forked inside the guest legitimately have
        // `getppid() == 1` because init IS their parent, so the
        // host-side orphan-detection fast-path in `workload.rs` must
        // skip the `_exit(0)` branch when this variable is present.
        // The variable is inherited across fork/exec, so every
        // descendant of guest init (including workloads that re-exec
        // /init to run scenarios) observes it.
        std::env::set_var("KTSTR_GUEST_INIT", "1");
    }

    // Disk-template build mode: format /dev/vda with the embedded
    // mkfs binary, then reboot. No scheduler load, no test dispatch,
    // no shell. Must run before shell_mode_requested() so a future
    // operator-facing shell command cannot accidentally trip the
    // template path. See [`crate::vmm::disk_template`] for the host
    // side that drives this mode.
    if disk_template_mode_requested() {
        let _span = tracing::debug_span!("disk_template_mode").entered();
        let code = run_disk_template_mode();
        // Match the post-test exit semantics: push buffered stdio
        // bytes into the pipe (the forwarder threads then ship them
        // over the bulk port), emit the binary exit code over the
        // bulk data port so the host knows we're done, reboot.
        // `flush()` replaces the broken `tcdrain(1/2)`
        // which returned ENOTTY against the pipe write ends; the
        // forwarder threads aren't joined here, so bytes still in
        // the pipe at reboot time are lost — see the queue task
        // for forwarder-join plumbing.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        crate::vmm::guest_comms::send_exit(code);
        // The bulk-port write inside `send_exit` commits via MMIO
        // before userspace returns from KVM_RUN — the EXIT frame is
        // in the host's port-1 RX buffer the moment `send_exit`
        // returns. No additional wait needed before reboot.
        force_reboot();
    }

    // Shell mode: interactive busybox shell instead of test dispatch.
    if shell_mode_requested() {
        let _shell_span = tracing::debug_span!("shell_mode").entered();
        let console_dev = shell_console_device();
        redirect_all_stdio_to(console_dev);

        // Create busybox applet symlinks.
        {
            let _s = tracing::debug_span!("busybox_install").entered();
            let _ = Command::new("/bin/busybox")
                .args(["--install", "-s", "/bin"])
                .status();
        }

        // Mount devpts so PTY allocation works.
        mount_devpts();

        // --exec mode: run a command non-interactively instead of
        // dropping into an interactive shell. Inherits stdio from init
        // which redirect_all_stdio_to() already pointed at the console
        // device (virtio-console /dev/hvc0 when available, COM2
        // otherwise). The host stdout writer thread drains virtio TX.
        // Checked before MOTD so exec output is not polluted.
        if let Some(cmd) = shell_exec_cmd() {
            tracing::debug!(cmd = %cmd, "shell exec mode");
            // Disable OPOST on stdout so the tty layer does not
            // convert \n to \r\n. Without this, every newline in
            // command output gains a spurious \r visible to the host.
            let stdout_fd = unsafe { BorrowedFd::borrow_raw(1) };
            if let Ok(mut termios) = tcgetattr(stdout_fd) {
                termios
                    .output_flags
                    .remove(nix::sys::termios::OutputFlags::OPOST);
                let _ = tcsetattr(stdout_fd, SetArg::TCSANOW, &termios);
            }
            // [`with_sigchld_default`] flips SIGCHLD to SIG_DFL
            // for the closure body so `Command::status()` (which
            // calls `waitpid(2)`) reaps the child and reports the
            // real exit code. The `SIG_IGN` disposition installed
            // earlier in [`ktstr_guest_init`] for zombie
            // prevention is restored on closure return — and on
            // panic unwind, via the helper's RAII guard.
            let status = with_sigchld_default(|| {
                Command::new("/bin/busybox")
                    .args(["sh", "-c", &cmd])
                    .status()
            });
            let code = match status {
                Ok(s) => s.code().unwrap_or(1),
                Err(e) => {
                    eprintln!("ktstr-init: exec failed: {e}");
                    1
                }
            };
            // Exit code travels via the bulk data port so it does
            // not pollute captured command output on stdout.
            crate::vmm::guest_comms::send_exec_exit(code as i32);
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            // tcdrain is synchronous on the vCPU exit: when these
            // syscalls return, every byte is already in the host's
            // serial writer Vec (or virtio-console TX path). No
            // additional wait needed before reboot.
            unsafe {
                libc::tcdrain(1);
            }
            unsafe {
                libc::tcdrain(2);
            }
            force_reboot();
        }

        // MOTD (printed to console before PTY proxy takes over).
        // Skipped in exec mode (handled above).
        let kernel_version = fs::read_to_string("/proc/version")
            .ok()
            .and_then(|v| v.split_whitespace().nth(2).map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());
        let mem_mb = fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
                .map(|kb| kb / 1024)
        });
        println!("ktstr shell");
        println!("  kernel:    {kernel_version}");
        if let Some(mb) = mem_mb {
            println!("  memory:    {mb} MB");
        }
        print_topology_line();
        print_includes_line();
        println!("  tools:     busybox (ls, ps, top, dmesg, ip, vi, ...)");
        println!("  mounts:    /proc /sys /dev /sys/fs/cgroup /sys/fs/bpf /tmp");
        println!("             /sys/kernel/debug /sys/kernel/tracing /dev/pts");
        println!("  type `exit` for clean shutdown, Ctrl+A X to force-kill");
        let _ = std::io::stdout().flush();

        // Allocate a PTY pair so busybox sh gets a controlling terminal
        // (required for job control: Ctrl+Z, bg, fg).
        tracing::debug!("spawning interactive shell with PTY");
        spawn_shell_with_pty();

        force_reboot();
    }

    // Read test args from /args early so Phase 2b can parse
    // --ktstr-probe-stack for probe setup before the scheduler starts.
    let args: Vec<String> = {
        let content = fs::read_to_string("/args").unwrap_or_default();
        let mut a = vec!["/init".to_string()];
        a.extend(content.lines().map(|s| s.to_string()));
        a
    };
    tracing::debug!(args = ?args, "parsed /args");

    // Propagate RUST_BACKTRACE and RUST_LOG from the kernel cmdline to
    // the process environment BEFORE Phase A spawns its probe thread.
    // `std::env::set_var` mutates glibc's `__environ` without locking;
    // calling it while the probe thread is live is UB on Linux.
    crate::test_support::propagate_rust_env_from_cmdline();

    // Phase 2b: Probe Phase A (before scheduler starts).
    // Attaches kprobes + trigger + kernel fexit so the one-shot
    // sched_ext_exit tracepoint is captured even if the scheduler
    // crashes immediately on startup.
    let _s_phase2b = tracing::debug_span!("phase2b_probe_phase_a").entered();
    let probe_phase_a = crate::test_support::start_probe_phase_a(&args);
    let probes_active = probe_phase_a.is_some();
    drop(_s_phase2b);

    // Phase 3: Cgroup parent + Scheduler.
    // Create the cgroup parent directory before starting the scheduler
    // so it exists when the scheduler looks for it.
    let _s_phase3 = tracing::debug_span!("phase3_scheduler_start").entered();
    create_cgroup_parent_from_sched_args();
    exec_shell_script("/sched_enable");
    // Plumb the probe pipeline's `stop` + `output_done` into
    // `start_scheduler` so the early-bail paths (Died / not
    // attached / spawn error) can drain probe JSON to COM2 before
    // calling `force_reboot()`. Without the drain, every path that
    // crashes the scheduler before the test dispatches loses its
    // probe payload to the reboot — exactly the diagnostic the
    // probes were attached to capture.
    let probe_drain = probe_phase_a.as_ref().map(|pa| ProbeDrain {
        stop: pa.pipeline.stop.clone(),
        output_done: pa.pipeline.output_done.clone(),
    });
    let (mut sched_child, sched_log_path) = start_scheduler(probe_drain);
    drop(_s_phase3);

    // Phase 4: hvc0 polling + trace pipe (background threads).
    let _s_phase4 = tracing::debug_span!("phase4_vc_poll").entered();
    let (trace_stop, trace_handle) = start_trace_pipe();
    let vc_poll_stop = start_hvc0_poll(trace_stop.clone());
    drop(_s_phase4);

    // Phase 4b: Scheduler death monitor.
    // Spawn a thread that polls /proc/{pid}. If the scheduler exits during
    // the test, the thread writes MSG_TYPE_SCHED_EXIT via bulk port so the host
    // can detect early death without waiting for the watchdog.
    //
    // When probes are active, suppress COM2 log dump to avoid
    // interleaving with probe JSON output on the same serial port.
    let suppress_com2 = Arc::new(AtomicBool::new(probes_active));
    let probe_output_done = probe_phase_a
        .as_ref()
        .map(|pa| pa.pipeline.output_done.clone());
    let sched_exit_stop = start_sched_exit_monitor(
        sched_child.as_ref().map(|c| c.id()),
        sched_log_path.as_deref(),
        suppress_com2,
        probe_output_done,
    );

    // Phase 5: Dispatch.
    let _s_phase5 = tracing::debug_span!("phase5_dispatch").entered();
    tracing::debug!("dispatching test");
    crate::vmm::guest_comms::send_lifecycle(crate::vmm::wire::LifecyclePhase::PayloadStarting, "");
    let code = if let Some(pa) = probe_phase_a {
        crate::test_support::maybe_dispatch_vm_test_with_phase_a(&args, pa).unwrap_or(1)
    } else {
        crate::test_support::maybe_dispatch_vm_test_with_args(&args).unwrap_or(1)
    };
    drop(_s_phase5);

    // Flush test output before teardown. Rust's BufWriter on stdout
    // holds data until flushed; without this the host may not see the
    // test result before reboot.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    crate::test_support::try_flush_profraw();

    // Phase 6: Scheduler cleanup.
    let _s_phase6 = tracing::debug_span!("phase6_cleanup").entered();
    if let Some(ref mut child) = sched_child {
        let _ = child.kill();
        let _ = child.wait();
        if let Some(ref log_path) = sched_log_path {
            dump_sched_output(log_path);
        }
    }
    exec_shell_script("/sched_disable");

    // Stop background threads.
    if let Some(ref stop) = vc_poll_stop {
        stop.store(true, Ordering::Release);
    }
    if let Some(ref handle) = sched_exit_stop {
        // Order: store `true` first so the monitor's `Acquire` load
        // at the top of its loop sees the stop flag. Then write the
        // wake eventfd to drop the monitor's `poll(2)` wait latency
        // from the legacy 250 ms cadence to microseconds. A
        // monitor that races the eventfd write into its
        // `Acquire` load still observes `stop=true` immediately —
        // the eventfd is the wake edge, not the source of truth.
        handle.stop.store(true, Ordering::Release);
        handle.wake();
    }

    // Flush COM1 trace data before reboot. The reader thread runs on
    // a poll(POLLIN, 200ms) cadence over a non-blocking trace_pipe fd
    // (see start_trace_pipe), so setting `stop` is what bounds
    // `handle.join()` — the thread observes the flag at the next poll
    // wake and enters its 5s drain window. Effective shutdown latency
    // is up to ~5.2s in the worst case: the 200ms poll cadence elapses
    // before the thread notices the stop flag, then the 5s drain
    // deadline begins. Disabling the tracepoint and writing 0 to
    // `tracing_on` first quiesces the producer side so the drain
    // window terminates promptly: no new events are recorded into the
    // ring buffer, the reader sees POLLIN until the buffer is empty,
    // then poll returns 0 each cycle and the drain_deadline elapses
    // cleanly. Trace events arriving after the 5s deadline are dropped
    // by design — bounded drain is the explicit tradeoff that
    // guarantees cleanup completes (a faulty producer that never
    // pauses cannot wedge teardown).
    //
    // tracing_on=0 alone does NOT wake a trace_pipe reader stuck at
    // `iter->pos == 0` — the kernel wake fires `ring_buffer_wake_waiters`
    // but the trace_pipe wait uses `wait_pipe_cond` (not
    // `rb_wait_once`), and that condition only flips when `iter->closed`
    // or `iter->wait_index` change. The non-blocking + poll design
    // sidesteps this by never blocking in the kernel wait at all.
    let _ = fs::write(TRACE_SCHED_EXT_DUMP_ENABLE, "0");
    if let Some(ref stop) = trace_stop {
        stop.store(true, Ordering::Release);
    }
    let _ = fs::write(TRACE_TRACING_ON, "0");
    if let Some(handle) = trace_handle {
        let _ = handle.join();
    }
    if let Ok(com1) = fs::OpenOptions::new().write(true).open(COM1) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::tcdrain(com1.as_raw_fd());
        }
    }

    // Phase 7: Exit.
    // Push buffered stdout/stderr bytes into the pipe write ends so
    // the bulk-port forwarder threads can ship them before reboot.
    // After stdio redirect, fd 1 / fd 2 are pipe write ends
    // (not the COM2 UART) so `tcdrain(1)` would return ENOTTY
    // silently — `flush()` is the equivalent for pipes. The
    // forwarder threads are not joined before `force_reboot`; bytes
    // still resident in the pipe buffer at reboot time are lost
    // (see the queue task for forwarder-join plumbing).
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    // Write exit code via the typed guest API on the bulk data
    // port. The legacy COM2 `SENTINEL_EXIT_PREFIX` fallback is gone
    // — bulk-port backpressure guarantees delivery and the host's
    // `collect_results` walks `guest_messages` for a binary
    // `MSG_TYPE_EXIT` frame as the sole authoritative source.
    crate::vmm::guest_comms::send_exit(code as i32);

    // Drain COM2 UART for any panic-hook bytes that may still be
    // in flight (the panic hook is the one remaining COM2 writer).
    // tcdrain is synchronous on the vCPU exit: when it returns,
    // every byte is already in the host's COM2 writer Vec.
    if let Ok(com2) = fs::OpenOptions::new().write(true).open(COM2) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::tcdrain(com2.as_raw_fd());
        }
    }

    force_reboot()
}

/// Maximum bytes per [`MsgType::Stdout`] / [`MsgType::Stderr`] TLV
/// chunk emitted by the pipe forwarder threads. 4 KiB matches a
/// page-size pipe read; well under the host-side per-frame cap
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`] so a chunk fits
/// comfortably in one frame even with the 16-byte header.
const STDIO_CHUNK_BYTES: usize = 4 * 1024;

/// Redirect stdout and stderr through bulk-port forwarder threads.
///
/// Pre-bincode-migration: dup2'd `/dev/ttyS1` over fd 1 and fd 2 so
/// every `println!` / `eprintln!` reached the host as a stream of
/// COM2 bytes.  The bulk-port migration replaces COM2 with one
/// [`MsgType::Stdout`] / [`MsgType::Stderr`] TLV frame per chunk:
///
///   1. Open a pair of `pipe(2)` pipes (one for stdout, one for
///      stderr).
///   2. `dup2` each pipe's write end over fd 1 / fd 2 so every
///      `println!` / `eprintln!` lands in the pipe.
///   3. Spawn one reader thread per pipe.  Each thread reads up to
///      [`STDIO_CHUNK_BYTES`] at a time from the pipe's read end and
///      ships the chunk via
///      [`crate::vmm::guest_comms::send_stdout_chunk`] /
///      [`crate::vmm::guest_comms::send_stderr_chunk`].
///
/// The threads are detached: they exit cleanly when fd 1 / fd 2 are
/// closed (process exit / `force_reboot`) because the read end then
/// returns EOF.
///
/// Panic diagnostics still go to COM2 — the panic hook in
/// [`ktstr_guest_init`] writes directly to `/dev/ttyS1` because the
/// hook cannot block on virtio backpressure.  Every other guest
/// stream now travels over the bulk port.
///
/// On any pipe / dup2 / thread-spawn failure the function logs via
/// `eprintln!` (fd 2 is still attached to the kernel console at the
/// failure point, so the operator sees the misroute) and returns —
/// stdout/stderr stay attached to whatever fd they pointed at on
/// entry.
fn redirect_stdio_to_bulk_port() {
    use std::io::Read;
    use std::os::unix::io::{AsRawFd, FromRawFd};

    fn make_pipe() -> Option<(std::fs::File, std::fs::File)> {
        let mut fds = [0i32; 2];
        // SAFETY: `fds` is a valid `&mut [i32; 2]`; `pipe(2)` writes
        // exactly two file descriptors on success.  Passing `O_CLOEXEC`
        // would belong on `pipe2`, but we deliberately want the pipe
        // ends to survive across any forks the test may perform — the
        // dup2'd write end carries fd 1 / fd 2 across exec/fork, which
        // is the entire point.
        let r = unsafe { libc::pipe(fds.as_mut_ptr()) };
        if r < 0 {
            return None;
        }
        // SAFETY: `pipe(2)` just returned with the two fds populated.
        // `from_raw_fd` takes ownership of each side; both close on
        // drop.  Held by `File` for the natural Read/Write impls.
        let read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        Some((read_end, write_end))
    }

    fn spawn_forwarder(mut read_end: std::fs::File, name: &'static str, sender: fn(&[u8]) -> bool) {
        let _ = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let mut buf = [0u8; STDIO_CHUNK_BYTES];
                loop {
                    match read_end.read(&mut buf) {
                        Ok(0) => break, // EOF — fd 1/2 closed.
                        Ok(n) => {
                            // Fire-and-forget.  `send_*_chunk`
                            // returns false when the bulk port is
                            // not yet ready; bytes emitted before
                            // the multiport handshake completes are
                            // dropped.  Same caveat as the prior
                            // COM2 path's pre-handshake byte loss.
                            let _ = sender(&buf[..n]);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
            });
    }

    let Some((stdout_r, stdout_w)) = make_pipe() else {
        eprintln!("ktstr-init: redirect_stdio_to_bulk_port: pipe(stdout) failed");
        return;
    };
    let Some((stderr_r, stderr_w)) = make_pipe() else {
        eprintln!("ktstr-init: redirect_stdio_to_bulk_port: pipe(stderr) failed");
        return;
    };

    // Capture errno via `last_os_error` BEFORE any subsequent libc
    // call: errno is per-thread but every libc call may clobber it.
    let (rc1, err1, rc2, err2) = unsafe {
        let r1 = libc::dup2(stdout_w.as_raw_fd(), 1);
        let e1 = std::io::Error::last_os_error();
        let r2 = libc::dup2(stderr_w.as_raw_fd(), 2);
        let e2 = std::io::Error::last_os_error();
        (r1, e1, r2, e2)
    };
    // The dup2 above duplicated each pipe's write end onto fd 1 / fd 2;
    // the originals (`stdout_w` / `stderr_w`) close on this scope's
    // exit.  Without that close, the read end of each pipe would see
    // EOF only after the test process holding fd 1 / fd 2 also dropped
    // those file descriptors — but we want the EOF condition to fire
    // when fd 1 / fd 2 reach their natural close-on-exit, not when
    // some other holder of `stdout_w` closes too.  Letting the
    // originals drop here is correct because `dup2` increments the
    // file's refcount.
    if rc1 < 0 {
        eprintln!("ktstr-init: redirect_stdio_to_bulk_port: dup2(stdout) failed: {err1}");
    }
    if rc2 < 0 {
        eprintln!("ktstr-init: redirect_stdio_to_bulk_port: dup2(stderr) failed: {err2}");
    }

    spawn_forwarder(stdout_r, "ktstr-stdout-fwd", |b| {
        crate::vmm::guest_comms::send_stdout_chunk(b)
    });
    spawn_forwarder(stderr_r, "ktstr-stderr-fwd", |b| {
        crate::vmm::guest_comms::send_stderr_chunk(b)
    });
}

/// Check kernel cmdline for KTSTR_MODE=shell.
fn shell_mode_requested() -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|c| cmdline_contains_token(&c, "KTSTR_MODE=shell"))
        .unwrap_or(false)
}

/// Check kernel cmdline for `KTSTR_MODE=disk_template`. The host
/// asserts this when booting a one-shot template-build VM (see
/// [`crate::vmm::disk_template`]).
fn disk_template_mode_requested() -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|c| cmdline_contains_token(&c, "KTSTR_MODE=disk_template"))
        .unwrap_or(false)
}

/// Pure-function cmdline-token check, factored out of
/// [`shell_mode_requested`] / [`disk_template_mode_requested`] so
/// the precedence-and-multiplicity behavior can be tested without
/// mocking `/proc/cmdline`. Whitespace-separated, exact match (the
/// kernel passes cmdline tokens verbatim — no quoting, no escapes).
fn cmdline_contains_token(cmdline: &str, token: &str) -> bool {
    cmdline.split_whitespace().any(|s| s == token)
}

/// Disk-template build dispatch: exec `/bin/mkfs.btrfs /dev/vda`
/// (the host packed `mkfs.btrfs` into the initramfs at this path),
/// wait for it, return its exit code so the caller emits the exit
/// sentinel on COM2 before rebooting. Returns `0` on success and
/// the binary's exit code (or `1` on spawn failure) otherwise.
///
/// The disk image at `/dev/vda` is the host-side staging file
/// (sparse, sized to the requested capacity); after this function
/// returns and the VM reboots, the host's [`crate::vmm::disk_template::store_atomic`]
/// publishes the now-formatted image into the cache.
///
/// The host never execs `mkfs.btrfs` against a real backing file —
/// driving the format through this guest-side dispatch keeps the
/// kernel under test as the on-disk-format authority, so any btrfs
/// feature regression in that kernel surfaces as a guest format
/// failure here instead of as a host/guest mkfs disagreement that
/// would slip past testing.
fn run_disk_template_mode() -> i32 {
    redirect_stdio_to_bulk_port();
    // The mkfs.btrfs binary is packed at `bin/mkfs.btrfs` by
    // [`crate::vmm::disk_template::build_template_via_vm`] via
    // `include_files`; that function — not `ensure_template` — is
    // the host-side site that assembles the template-VM
    // initramfs.
    const MKFS: &str = "/bin/mkfs.btrfs";
    // `-f` forces overwrite of any existing signature so a leftover
    // ext4 magic from a host that recycled the staging file does
    // not block formatting. `--quiet` keeps the COM2 transcript
    // small. `/dev/vda` is the singleton virtio-blk device the
    // host attached.
    //
    // No `--metadata DUP` override: btrfs picks DUP metadata by
    // default on a single-device fs, which is the desired
    // production format. The 256 MiB minimum capacity (see
    // VIRTIO_BLK_DEFAULT_CAPACITY_BYTES doc) accommodates DUP.
    tracing::info!(mkfs = MKFS, target = "/dev/vda", "running mkfs.btrfs");
    // SIGCHLD is `SIG_IGN` for the rest of this process (installed by
    // [`ktstr_guest_init`] for zombie prevention). `Command::status()`
    // calls `waitpid(2)` internally; under `SIG_IGN` the kernel
    // auto-reaps the child before `waitpid` runs, so the syscall
    // returns `ECHILD`, the std-lib maps it to
    // `Err(io::Error::ECHILD)`, and the original `match status`
    // branch fell into the `Err(_) => 1` arm — surfacing a fixed `1`
    // exit code for every successful `mkfs.btrfs` run. The host
    // would then see "template build failed" for a perfectly
    // formatted image. Restore `SIG_DFL` for the closure's lifetime
    // so `waitpid` reaps and reports the real status; the
    // post-closure restore re-installs `SIG_IGN` for any future
    // child this process spawns.
    let status = with_sigchld_default(|| {
        Command::new(MKFS)
            .args(["-f", "--quiet", "/dev/vda"])
            .status()
    });
    match status {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("ktstr-init: failed to spawn {MKFS}: {e}");
            1
        }
    }
}

/// Read /exec_cmd from the initramfs if present.
/// The host writes this file via build_suffix when --exec is used.
fn shell_exec_cmd() -> Option<String> {
    fs::read_to_string("/exec_cmd")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Extract a KEY=value pair from the kernel cmdline.
fn cmdline_val(key: &str) -> Option<String> {
    let cmdline = fs::read_to_string("/proc/cmdline").ok()?;
    let prefix = format!("{key}=");
    cmdline
        .split_whitespace()
        .find_map(|s| s.strip_prefix(&prefix))
        .map(|s| s.to_string())
}

/// Build PATH with /include-files directories containing executables.
///
/// Walks /include-files recursively, collects directories that contain
/// at least one executable file, prepends them all to PATH. This makes
/// included binaries runnable by name regardless of subdirectory depth
/// (e.g. `-i ../scx/target/release` → `scx_cake` works directly).
fn build_include_path() -> String {
    use std::collections::BTreeSet;
    use std::os::unix::fs::PermissionsExt;
    let include_dir = std::path::Path::new("/include-files");
    let mut dirs = BTreeSet::new();

    if include_dir.is_dir() {
        for entry in walkdir::WalkDir::new(include_dir).follow_links(true) {
            let Ok(entry) = entry else { continue };
            if entry.file_type().is_file()
                && entry
                    .metadata()
                    .is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
                && let Some(parent) = entry.path().parent()
            {
                dirs.insert(parent.to_string_lossy().to_string());
            }
        }
    }

    let mut path_parts: Vec<String> = dirs.into_iter().collect();
    path_parts.push("/bin".to_string());
    path_parts.join(":")
}

/// Redirect stdin, stdout, and stderr to the given device with O_RDWR.
///
/// Shell mode needs all three fds on the console device: stdin for
/// reading input, stdout/stderr for writing output.
///
/// `dup2` failures are logged via `eprintln!`. A failing `dup2`
/// leaves the target fd unchanged, so the eprintln still reaches the
/// pre-redirect stderr (kernel console / COM1) and the operator sees
/// the misroute rather than the failing path silently writing to a
/// wrong device.
fn redirect_all_stdio_to(path: &str) {
    use std::os::unix::io::AsRawFd;

    let Ok(dev) = fs::OpenOptions::new().read(true).write(true).open(path) else {
        return;
    };
    let fd = dev.as_raw_fd();
    // Capture errno per call before the next libc call clobbers
    // it. Run all three syscalls sequentially without aborting on
    // a partial failure — fd 0 redirect failing should not stop us
    // from at least getting stdout/stderr onto the console.
    let (rc0, err0, rc1, err1, rc2, err2) = unsafe {
        let r0 = libc::dup2(fd, 0);
        let e0 = std::io::Error::last_os_error();
        let r1 = libc::dup2(fd, 1);
        let e1 = std::io::Error::last_os_error();
        let r2 = libc::dup2(fd, 2);
        let e2 = std::io::Error::last_os_error();
        (r0, e0, r1, e1, r2, e2)
    };
    if rc0 < 0 {
        eprintln!("ktstr-init: redirect_all_stdio_to({path}): dup2(stdin) failed: {err0}");
    }
    if rc1 < 0 {
        eprintln!("ktstr-init: redirect_all_stdio_to({path}): dup2(stdout) failed: {err1}");
    }
    if rc2 < 0 {
        eprintln!("ktstr-init: redirect_all_stdio_to({path}): dup2(stderr) failed: {err2}");
    }
}

/// Select the console device for shell mode.
/// Prefers /dev/hvc0 (virtio-console) when available, falls back to COM2.
fn shell_console_device() -> &'static str {
    if Path::new(HVC0).exists() { HVC0 } else { COM2 }
}

/// Mount devpts at /dev/pts for PTY allocation.
///
/// Required before `openpty()` — the C library opens `/dev/ptmx` and
/// the slave device lives under `/dev/pts/N`.
fn mount_devpts() {
    mkdir_p("/dev/pts");
    let result = mount(
        Some("devpts"),
        "/dev/pts",
        Some("devpts"),
        MsFlags::empty(),
        None::<&str>,
    );
    if let Err(e) = result {
        eprintln!("ktstr-init: mount devpts on /dev/pts: {e}");
    }
}

/// Spawn busybox sh with a PTY as its controlling terminal.
///
/// Allocates a PTY pair via `openpty()`, spawns sh with the slave as
/// stdin/stdout/stderr and `setsid` + `TIOCSCTTY` in `pre_exec` so sh
/// gets a controlling terminal (job control). The parent proxies data
/// between COM2 (fd 0/1) and the PTY master until the child exits.
///
/// SIGCHLD remains SIG_IGN (set earlier for zombie prevention), so
/// waitpid returns ECHILD after the kernel auto-reaps the child.
/// This is expected and suppressed.
fn spawn_shell_with_pty() {
    let pty = match openpty(None, None) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ktstr-init: openpty failed: {e}");
            return;
        }
    };

    let slave_fd = pty.slave.as_raw_fd();

    // Set PTY size from host terminal dimensions passed via cmdline.
    if let (Some(cols), Some(rows)) = (cmdline_val("KTSTR_COLS"), cmdline_val("KTSTR_ROWS"))
        && let (Ok(cols), Ok(rows)) = (cols.parse::<u16>(), rows.parse::<u16>())
    {
        let ws = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(slave_fd, libc::TIOCSWINSZ, &ws);
        }
    }

    // Set terminal type from host. Default to "linux" if not passed.
    let term = cmdline_val("KTSTR_TERM").unwrap_or_else(|| "linux".to_string());
    let colorterm = cmdline_val("KTSTR_COLORTERM");

    let child = unsafe {
        let mut cmd = Command::new("/bin/busybox");
        cmd.arg("sh")
            .env("TERM", &term)
            .env("PS1", "\x1b[2m^Ax=quit\x1b[0m \\w # ");
        if let Some(ref ct) = colorterm {
            cmd.env("COLORTERM", ct);
        }
        cmd.stdin(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .stdout(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .stderr(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .pre_exec(move || {
                // Create a new session so sh becomes session leader.
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // Acquire a controlling terminal.
                if libc::ioctl(slave_fd, libc::TIOCSCTTY, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            })
            .spawn()
    };

    // Close slave in parent — the child has its own copies.
    drop(pty.slave);

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ktstr-init: spawn shell: {e}");
            return;
        }
    };

    let child_pid = child.id();

    // Set COM2 serial (fd 0) to raw mode so the kernel line discipline
    // passes bytes through without processing. Without this, special
    // characters like tab (0x09) are consumed by the line discipline
    // instead of being forwarded through the proxy to the PTY.
    let stdin_fd = unsafe { BorrowedFd::borrow_raw(0) };
    if let Ok(mut termios) = tcgetattr(stdin_fd) {
        cfmakeraw(&mut termios);
        let _ = tcsetattr(stdin_fd, SetArg::TCSANOW, &termios);
    }

    // Proxy between COM2 (fd 0 for input, fd 1 for output) and PTY master.
    proxy_serial_pty(&pty.master, child_pid);

    // SIGCHLD is SIG_IGN so the kernel auto-reaps the child. waitpid
    // returns ECHILD — expected, not an error.
    match child.wait() {
        Ok(status) => {
            tracing::debug!(?status, "shell exited");
        }
        Err(e) if e.raw_os_error() == Some(libc::ECHILD) => {}
        Err(e) => {
            eprintln!("ktstr-init: wait for shell: {e}");
        }
    }

    // No guest-side exit message — the host prints "Connection to VM
    // closed." after the VM shuts down. Printing here too would
    // duplicate it, and writing to COM2 in raw mode after PTY teardown
    // leaks garbage bytes.
}

/// Proxy data between COM2 serial (fd 0/1) and a PTY master fd.
///
/// Uses poll(2) to multiplex reads from both fds. Exits when the PTY
/// master returns EOF (child closed the slave side) or the child process
/// no longer exists.
fn proxy_serial_pty(master: &OwnedFd, child_pid: u32) {
    let stdin_fd = unsafe { BorrowedFd::borrow_raw(0) };
    let stdout_fd = unsafe { BorrowedFd::borrow_raw(1) };
    let master_fd = master.as_fd();

    let mut buf = [0u8; 4096];

    loop {
        let mut pollfds = [
            PollFd::new(stdin_fd, PollFlags::POLLIN),
            PollFd::new(master_fd, PollFlags::POLLIN),
        ];

        match poll(&mut pollfds, PollTimeout::from(200u16)) {
            Ok(0) => {
                // Timeout — check if child is still alive.
                if !Path::new(&format!("/proc/{child_pid}")).exists() {
                    break;
                }
                continue;
            }
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(_) => break,
        }

        // Serial input -> PTY master (user typing).
        if let Some(revents) = pollfds[0].revents() {
            if revents.contains(PollFlags::POLLIN) {
                match nix::unistd::read(stdin_fd, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = nix::unistd::write(master_fd, &buf[..n]);
                    }
                    Err(nix::errno::Errno::EINTR) => {}
                    Err(_) => break,
                }
            }
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLHUP) {
                break;
            }
        }

        // PTY master -> serial output (shell output).
        // Check POLLHUP/POLLERR before POLLIN: when the shell exits,
        // both flags can arrive in the same poll iteration. Reading
        // after the slave closes produces partial/garbage bytes from
        // the PTY teardown (manifests as a raw U+FFFD on the terminal).
        if let Some(revents) = pollfds[1].revents() {
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLHUP) {
                break;
            }
            if revents.contains(PollFlags::POLLIN) {
                match nix::unistd::read(master_fd, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = nix::unistd::write(stdout_fd, &buf[..n]);
                    }
                    Err(nix::errno::Errno::EINTR) => {}
                    Err(_) => break,
                }
            }
        }
    }
}

/// Print the topology line for the shell MOTD.
///
/// Parses KTSTR_TOPO=N,L,C,T from /proc/cmdline (passed by the host).
/// Falls back to counting online CPUs via /sys/devices/system/cpu/online.
fn print_topology_line() {
    if let Some((n, l, c, t)) = parse_topo_from_cmdline() {
        let total = l * c * t;
        if n > 1 {
            println!(
                "  topology:  {n} NUMA nodes, {l} LLC{}, {c} core{}, {t} thread{} ({total} vCPU{})",
                if l == 1 { "" } else { "s" },
                if c == 1 { "" } else { "s" },
                if t == 1 { "" } else { "s" },
                if total == 1 { "" } else { "s" },
            );
        } else {
            println!(
                "  topology:  {l} LLC{}, {c} core{}, {t} thread{} ({total} vCPU{})",
                if l == 1 { "" } else { "s" },
                if c == 1 { "" } else { "s" },
                if t == 1 { "" } else { "s" },
                if total == 1 { "" } else { "s" },
            );
        }
    } else if let Some(count) = count_online_cpus() {
        println!(
            "  topology:  {count} vCPU{}",
            if count == 1 { "" } else { "s" }
        );
    }
}

/// Parse KTSTR_TOPO=N,L,C,T from /proc/cmdline.
fn parse_topo_from_cmdline() -> Option<(u32, u32, u32, u32)> {
    let val = cmdline_val("KTSTR_TOPO")?;
    let parts: Vec<&str> = val.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let n: u32 = parts[0].parse().ok()?;
    let l: u32 = parts[1].parse().ok()?;
    let c: u32 = parts[2].parse().ok()?;
    let t: u32 = parts[3].parse().ok()?;
    Some((n, l, c, t))
}

/// Count online CPUs from /sys/devices/system/cpu/online.
///
/// The file contains a range list like "0-3" or "0-1,3". Parse and
/// count individual CPUs.
fn count_online_cpus() -> Option<u32> {
    let content = fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    let mut count = 0u32;
    for range in content.trim().split(',') {
        if let Some((start, end)) = range.split_once('-') {
            let s: u32 = start.parse().ok()?;
            let e: u32 = end.parse().ok()?;
            count += e - s + 1;
        } else {
            let _: u32 = range.parse().ok()?;
            count += 1;
        }
    }
    Some(count)
}

/// Print the include-files line for the shell MOTD.
///
/// Scans /include-files/ and lists each entry. Executable files
/// are marked with "(executable)".
fn print_includes_line() {
    let include_dir = Path::new("/include-files");
    if !include_dir.is_dir() {
        return;
    }
    let mut files: Vec<(String, bool)> = Vec::new();
    // Walk recursively to discover files in nested directories.
    for entry in walkdir::WalkDir::new(include_dir)
        .min_depth(1)
        .sort_by_file_name()
    {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(include_dir)
            .unwrap_or(entry.path());
        let name = rel.to_string_lossy().to_string();
        let executable = entry
            .metadata()
            .map(|m| {
                use std::os::unix::fs::PermissionsExt;
                m.permissions().mode() & 0o111 != 0
            })
            .unwrap_or(false);
        files.push((name, executable));
    }
    if files.is_empty() {
        return;
    }
    for (i, (name, executable)) in files.iter().enumerate() {
        let marker = if *executable { " (executable)" } else { "" };
        let path = format!("/include-files/{name}{marker}");
        if i == 0 {
            println!("  includes:  {path}");
        } else {
            println!("             {path}");
        }
    }
}

/// Mount essential filesystems.
fn mount_filesystems() {
    let mounts: &[(&str, &str, &str, bool)] = &[
        ("/proc", "proc", "proc", true),
        ("/sys", "sys", "sysfs", true),
        ("/dev", "dev", "devtmpfs", true),
        ("/sys/kernel/debug", "debugfs", "debugfs", false),
        ("/sys/kernel/tracing", "tracefs", "tracefs", false),
        ("/sys/fs/bpf", "bpffs", "bpf", false),
        ("/sys/fs/cgroup", "none", "cgroup2", false),
        ("/tmp", "tmpfs", "tmpfs", true),
        ("/dev/shm", "tmpfs", "tmpfs", false),
        ("/run", "tmpfs", "tmpfs", false),
    ];

    for &(target, source, fstype, required) in mounts {
        mkdir_p(target);
        let result = mount(
            Some(source),
            target,
            Some(fstype),
            MsFlags::empty(),
            None::<&str>,
        );
        if let Err(e) = result
            && required
        {
            eprintln!("ktstr-init: mount {fstype} on {target}: {e}");
        }
    }

    // Standard /dev/fd symlinks. Needed by bpftrace and shell
    // process substitution (e.g. <(cmd)).
    let _ = std::os::unix::fs::symlink("/proc/self/fd", "/dev/fd");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/0", "/dev/stdin");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/1", "/dev/stdout");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/2", "/dev/stderr");
}

/// Auto-mount the user-configured data disk at `/mnt/disk0` if the
/// host pre-formatted it. Driven by two kernel cmdline tokens
/// emitted by the host's
/// [`crate::vmm::KtstrVmBuilder::build`] cmdline assembly:
///
/// * `KTSTR_DISK0_FS=<tag>` — selects the on-disk filesystem to
///   pass to `mount(2)` (`btrfs` for the only non-Raw variant
///   today). Absence short-circuits this whole function: a `Raw`
///   disk has nothing to mount, and a config with no disk attached
///   never sees a `KTSTR_DISK0_FS` token at all.
/// * `KTSTR_DISK0_RO=1` — set when the host configured the disk
///   `read_only`. The virtio_blk device advertises
///   `VIRTIO_BLK_F_RO` for that case so the guest's gendisk is
///   read-only at the block layer; mounting RW would fail with
///   `-EROFS` (kernel `do_mount` sets the superblock RO from the
///   bdev). Setting `MS_RDONLY` proactively avoids that error path
///   entirely.
///
/// Failure modes are non-fatal: if the mount syscall returns an
/// error (unrecognized fstype tag, kernel `CONFIG_BTRFS_FS=n`,
/// device probe race, ENOMEM), the function logs to COM2 and
/// returns. The test still gets a usable VM; a subsequent test
/// step that depends on `/mnt/disk0` surfaces as a clean
/// userspace filesystem error rather than a confusing init abort.
///
/// Skips entirely when `KTSTR_DISK0_FS` is absent. The cmdline
/// emission on the host side is gated on
/// `disks[0].filesystem != Filesystem::Raw`, so this branch
/// matches the host-side opt-in: every config that requests an
/// on-disk filesystem gets the auto-mount, and every config that
/// doesn't is unaffected.
fn auto_mount_data_disks() {
    let Some(fstype) = cmdline_val("KTSTR_DISK0_FS") else {
        return;
    };
    // Validate the fstype against the known set. Today only
    // `btrfs` is wired (mirroring `Filesystem::Btrfs::cache_tag`);
    // unknown values warn-and-skip rather than handing arbitrary
    // strings to `mount(2)`. A future `Filesystem` variant must
    // add its tag here AND in the disk_config.rs `cache_tag`
    // match — keeping both lists in lockstep is the on-disk-format
    // / cmdline contract.
    let recognized = matches!(fstype.as_str(), "btrfs");
    if !recognized {
        let msg = format!(
            "ktstr-init: KTSTR_DISK0_FS={fstype} not recognized; \
             skipping auto-mount of /dev/vda"
        );
        let _ = fs::write(COM2, &msg);
        eprintln!("{msg}");
        return;
    }
    // RO bit. Absent or any value other than "1" means RW.
    // Strict-`==` rather than truthy-string parsing keeps the
    // contract simple and aligned with the host-side emission
    // (`KTSTR_DISK0_RO=1`).
    let ro = cmdline_val("KTSTR_DISK0_RO").as_deref() == Some("1");
    // Mount path. The host emits `KTSTR_DISK0_MOUNT=<path>` based
    // on `DiskConfig.name` — `/mnt/<name>` when set, `/mnt/disk0`
    // otherwise. Fall back to the default if the host-side value
    // is absent so a future host that emits FS but not MOUNT
    // (e.g. an older binary against a newer kernel) still mounts
    // somewhere sane rather than failing.
    let mount_point_owned =
        cmdline_val("KTSTR_DISK0_MOUNT").unwrap_or_else(|| "/mnt/disk0".to_string());
    let mount_point = mount_point_owned.as_str();
    mkdir_p(mount_point);
    let flags = if ro {
        MsFlags::MS_RDONLY
    } else {
        MsFlags::empty()
    };
    let result = mount(
        Some("/dev/vda"),
        mount_point,
        Some(fstype.as_str()),
        flags,
        None::<&str>,
    );
    if let Err(e) = result {
        let msg = format!(
            "ktstr-init: mount {fstype} on {mount_point} \
             (ro={ro}): {e}"
        );
        let _ = fs::write(COM2, &msg);
        eprintln!("{msg}");
    }
}

/// Recursive mkdir -p equivalent. `DirBuilder::recursive(true)` is
/// idempotent (returns Ok when the path already exists as a
/// directory) and walks parents internally, so the hand-rolled
/// recursion this replaced was redundant. Errors are swallowed to
/// match the previous behavior — the early guest init best-effort
/// creates each mount point and continues regardless, since any
/// real failure surfaces downstream when `mount()` itself fails.
///
/// Directory mode is pinned explicitly at 0o755 via
/// `DirBuilder::mode`. Relying on the default (0o777 & !umask) is
/// fragile: the guest init's umask is process state inherited from
/// the kernel/caller, and a caller that sets umask=0 before exec
/// would produce world-writable mount points. Pinning the mode in
/// the mkdir syscall itself keeps the traversal bit stable
/// regardless of umask.
fn mkdir_p(path: &str) {
    use std::os::unix::fs::DirBuilderExt;
    let _ = fs::DirBuilder::new()
        .recursive(true)
        .mode(0o755)
        .create(path);
}

/// Write a line to COM2 (the application serial port).
/// Falls back to stderr (kernel console) if COM2 is not available.
fn write_com2(msg: &str) {
    if let Ok(mut f) = fs::OpenOptions::new().write(true).open(COM2) {
        let _ = writeln!(f, "{msg}");
    } else {
        // COM2 unavailable (devtmpfs mount failed or device missing).
        // Write to kernel console as fallback so the host sees
        // something on COM1.
        eprintln!("ktstr-init [COM1 fallback]: {msg}");
    }
}

/// Create the cgroup parent directory specified by `--cell-parent-cgroup`
/// in `/sched_args`. The directory must exist before the scheduler starts
/// because the scheduler expects it at startup.
///
/// In cgroup v2, a controller is only visible inside a cgroup when its
/// parent's `cgroup.subtree_control` enables it. The kernel enforces
/// this in `cgroup_subtree_control_write` via `cgroup_control(cgrp)`,
/// which returns `parent->subtree_control` for non-root cgroups. To
/// make `cpuset` and `cpu` available in the leaf, every ancestor from
/// the cgroup root down to (and including) the leaf's immediate parent
/// must enable both controllers. Writes are applied root-to-leaf so
/// each level's prerequisite is already in place when its child is
/// written.
#[tracing::instrument]
fn create_cgroup_parent_from_sched_args() {
    let sched_args = match fs::read_to_string("/sched_args") {
        Ok(s) => s,
        Err(_) => return,
    };
    let args: Vec<&str> = sched_args.split_whitespace().collect();
    for i in 0..args.len() {
        if args[i] == "--cell-parent-cgroup"
            && let Some(&path) = args.get(i + 1)
        {
            let cgroup_dir = format!("/sys/fs/cgroup{path}");
            mkdir_p(&cgroup_dir);
            enable_subtree_controllers_to(&cgroup_dir);
            return;
        }
    }
}

/// Enable `+cpuset +cpu` in `cgroup.subtree_control` at every ancestor
/// from `/sys/fs/cgroup` (inclusive) down to (and including) the
/// immediate parent of `leaf`. Writes are ordered root-first so each
/// level's parent already advertises the controllers when its child is
/// written — without that ordering the kernel rejects the write with
/// `-ENOENT` (see `cgroup_subtree_control_write` /
/// `cgroup_control` in `kernel/cgroup/cgroup.c`).
///
/// `leaf` is expected to live under `/sys/fs/cgroup/...` (the format
/// emitted at the call site). The leaf itself is NOT written: enabling
/// controllers in a cgroup means they are visible inside that cgroup's
/// CHILDREN, so the leaf's own `subtree_control` only matters if the
/// scheduler ever creates sub-cgroups under it. The scheduler attaches
/// tasks to the leaf, so what it needs is `cpuset`/`cpu` enabled IN
/// the leaf — which is achieved by writing to the leaf's parent.
///
/// Failures on individual writes are logged via [`write_com2`] and do
/// not abort the walk: a single intermediate level that already has
/// both controllers enabled returns `0` from kernel side, so most
/// failures observed here will surface a real misconfiguration that
/// the scheduler's own `cgroup_attach` will then re-report with
/// scheduler-specific context.
fn enable_subtree_controllers_to(leaf: &str) {
    let cgroup_root = Path::new("/sys/fs/cgroup");
    let leaf_path = Path::new(leaf);
    // Verify leaf is under the cgroup root before touching anything.
    // A malformed `--cell-parent-cgroup` argument that produces a path
    // outside `/sys/fs/cgroup` (e.g. an empty or missing-leading-slash
    // value) would otherwise walk into `/sys/fs`, `/sys`, or `/`.
    if !leaf_path.starts_with(cgroup_root) || leaf_path == cgroup_root {
        return;
    }
    // `Path::ancestors` yields leaf-first; collect the strict ancestors
    // (skip the leaf itself) up to and including the cgroup root.
    let mut ancestors: Vec<&Path> = leaf_path
        .ancestors()
        .skip(1)
        .take_while(|p| p.starts_with(cgroup_root))
        .collect();
    // Apply root-to-leaf-parent: each level's parent must already
    // enable the controller before the child write is accepted.
    ancestors.reverse();
    for level in ancestors {
        let control = level.join("cgroup.subtree_control");
        if let Err(e) = fs::write(&control, "+cpuset +cpu") {
            write_com2(&format!(
                "ktstr-init: write {} +cpuset +cpu: {}",
                control.display(),
                e
            ));
        }
    }
}

/// Outcome of [`poll_startup`].
#[derive(Debug)]
enum StartupStatus {
    /// Child exited before the poll window closed.
    Died,
    /// Child was still running when the poll window closed.
    Alive,
}

/// Outcome of [`poll_scx_attached`].
#[derive(Debug, PartialEq, Eq)]
enum ScxAttachStatus {
    /// sched_ext root kobject exposes a non-empty `ops` attribute —
    /// scheduler registered and its ops name is populated.
    Attached,
    /// Poll window closed. At least one read of `root/ops` succeeded
    /// (the kernel supports sched_ext and the kset exists), but the
    /// file never became non-empty before the timeout. Typically
    /// means the scheduler process is alive but has not finished
    /// `scx_alloc_and_add_sched` — often a BPF verifier reject, an
    /// ops-mismatch, or a slow userspace init path.
    Timeout,
    /// Every read of `root/ops` returned `Err`. Either the kernel
    /// lacks sched_ext support entirely or the sysfs tree has not
    /// been created for the current kernel — distinct from
    /// [`Timeout`](Self::Timeout), where reads succeed but the file
    /// is empty.
    SysfsAbsent,
}

impl ScxAttachStatus {
    /// True when the scheduler registered successfully. Equivalent to
    /// the pre-enum `bool` return value.
    fn is_attached(&self) -> bool {
        matches!(self, ScxAttachStatus::Attached)
    }
}

/// Poll `/sys/kernel/sched_ext/root/ops` at `interval` cadence for up
/// to `timeout`.
///
/// Returns [`ScxAttachStatus::Attached`] as soon as the file is
/// non-empty (a scheduler is registered and its ops struct has a
/// populated name). When the window closes without a successful
/// attachment, distinguishes [`Timeout`](ScxAttachStatus::Timeout)
/// (reads succeeded but the file never became non-empty — the
/// scheduler did not finish registering) from
/// [`SysfsAbsent`](ScxAttachStatus::SysfsAbsent) (every read
/// errored — the kernel lacks sched_ext sysfs entirely).
///
/// The sysfs path is built in two steps by the kernel:
/// - `kernel/sched/ext.c` creates the `sched_ext` kset under
///   `kernel_kobj` via `kset_create_and_add("sched_ext", ...)` in
///   the scx init path, giving `/sys/kernel/sched_ext/`.
/// - Each `struct scx_sched` allocation assigns `sch->kobj.kset =
///   scx_kset` then calls `kobject_init_and_add(..., NULL, "root")`
///   (or `"sub-%llu"` when `CONFIG_EXT_SUB_SCHED` and a parent is
///   present), yielding `/sys/kernel/sched_ext/root/`. The `ops`
///   attribute is registered on `scx_ktype` via `scx_sched_groups`;
///   `scx_attr_ops_show` emits `sch->ops.name` through `sysfs_emit`.
///
/// Semantics we can claim based on the kernel flow above: a non-empty
/// `root/ops` proves the scheduler completed `scx_alloc_and_add_sched`
/// — the scx_sched struct is allocated, `sch->ops = *ops` has copied
/// the userspace-provided ops (including `name`), and the kobject is
/// registered with the kset. The kobject add happens BEFORE any BPF
/// callback (`ops.init`, `ops.enable`, `ops.runnable`, etc.) runs, so
/// a non-empty read does NOT prove those callbacks validated. Use
/// this poll only to confirm "scheduler registered and name
/// populated"; verify BPF callback success via monitor telemetry or
/// the scheduler's own exit kind.
///
/// Separate from [`poll_startup`] (which watches the child process
/// state): a scheduler can be `Alive` from the process-waitpid
/// perspective and still have zero progress on scx registration.
fn poll_scx_attached(
    interval: std::time::Duration,
    timeout: std::time::Duration,
) -> ScxAttachStatus {
    let start = std::time::Instant::now();
    let mut ever_read_ok = false;
    // Try to open the attribute fd once and use poll(POLLPRI) for
    // sysfs/kernfs notifications. kernfs supports POLLPRI on
    // attribute-content changes via `sysfs_notify` (kernel/fs/kernfs/file.c
    // `kernfs_fop_poll`). The kernel-side `scx_alloc_and_add_sched`
    // path doesn't currently emit `sysfs_notify` for this attribute,
    // but if the kernel ever adds it (or a future patch introduces
    // the call), we get instant wakeup; without it we fall back to
    // the unconditional sleep cadence below — same behaviour as
    // before, with the upper bound on detection latency unchanged.
    //
    // sysfs/kernfs does not reliably emit inotify/epoll events for
    // attribute content changes — the producer (kernel callsite)
    // must explicitly call `sysfs_notify`. Polling at `interval`
    // cadence is the supported mechanism for attributes whose
    // producer doesn't notify, so the fallback is mandatory.
    // Wrap the raw fd in `OwnedFd` so it is closed automatically on
    // every return path — including a panic anywhere inside the
    // loop body. The previous version close()d at each `return`
    // site manually; a panic in `read_to_string`, `Instant::now`
    // arithmetic, or `libc::poll` would have leaked the fd. PID 1's
    // fd budget is small and a leak across repeated calls would be
    // observable.
    //
    // The libc::open returns -1 on failure; turn that into `None`
    // before constructing `OwnedFd` (which requires a valid fd to
    // uphold its safety contract). Subsequent uses gate on
    // `attr_fd.is_some()` exactly like the previous `attr_fd >= 0`
    // checks, but the close on the success / timeout returns is now
    // implicit via Drop.
    let attr_fd: Option<OwnedFd> = {
        let raw = unsafe {
            libc::open(
                c"/sys/kernel/sched_ext/root/ops".as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC,
            )
        };
        if raw < 0 {
            None
        } else {
            // SAFETY: `raw` is a valid fd we just opened and have
            // exclusive ownership of. `OwnedFd::from_raw_fd` takes
            // ownership; the previous manual `close()` calls are
            // now Drop's responsibility.
            Some(unsafe { OwnedFd::from_raw_fd(raw) })
        }
    };
    let interval_ms_clamped = interval.as_millis().min(i32::MAX as u128) as i32;
    loop {
        // The kernel populates `sch->ops.name` before the kobject is
        // added, so the file becomes readable and non-empty the
        // moment registration succeeds. Absent / empty => no
        // registration yet (either no scheduler has reached
        // scx_alloc_and_add_sched or the sysfs tree is still being
        // torn down by a previous scheduler's exit).
        match fs::read_to_string(SYSFS_SCHED_EXT_ROOT_OPS) {
            Ok(contents) => {
                ever_read_ok = true;
                if !contents.trim().is_empty() {
                    // `attr_fd` Drop closes the OwnedFd on return.
                    return ScxAttachStatus::Attached;
                }
            }
            Err(_) => {
                // Leave `ever_read_ok` unchanged — every transient or
                // permanent failure counts toward SysfsAbsent unless
                // at least one success flipped the flag.
            }
        }
        let now = std::time::Instant::now();
        if now.duration_since(start) >= timeout {
            // `attr_fd` Drop closes the OwnedFd on return.
            return if ever_read_ok {
                ScxAttachStatus::Timeout
            } else {
                ScxAttachStatus::SysfsAbsent
            };
        }
        let remaining_ms = (start + timeout - now)
            .as_millis()
            .min(interval_ms_clamped as u128) as i32;
        if let Some(ref fd) = attr_fd {
            // poll(POLLPRI) is the kernfs notification mechanism
            // for attribute content changes. Cap the wait at the
            // requested polling interval so we never exceed the
            // caller's responsiveness contract — kernfs may not
            // emit POLLPRI for this attribute (the kernel-side
            // callsite must explicitly call `sysfs_notify`), in
            // which case poll returns 0 at `interval_ms_clamped`
            // and we re-read.
            //
            // sysfs/kernfs does not reliably emit inotify/epoll
            // events for attribute content changes; this poll is
            // the supported mechanism per `kernfs_fop_poll` plus
            // the read-fallback that catches changes the producer
            // didn't notify on.
            let mut pfd = libc::pollfd {
                fd: fd.as_raw_fd(),
                events: libc::POLLPRI,
                revents: 0,
            };
            // SAFETY: pfd is a single-element pollfd; nfds is 1.
            // Return value not consulted — the loop re-reads the
            // file each iteration regardless of poll outcome.
            let _ = unsafe { libc::poll(&mut pfd, 1, remaining_ms) };
        } else {
            // Open failed (e.g. attribute not present yet). Sleep
            // the polling cadence — sysfs does not provide an
            // event source for "attribute appears", so we have to
            // re-attempt the open via `read_to_string` at the
            // interval the caller requested.
            //
            // sysfs/kernfs does not provide an event source for
            // attribute appearance; polling is the supported
            // mechanism.
            std::thread::sleep(std::time::Duration::from_millis(remaining_ms.max(0) as u64));
        }
    }
}

/// Block on `pidfd` becoming readable for up to `timeout`. Returns
/// as soon as the child exits (pidfd POLLIN edge fires
/// microseconds after the kernel reaps), or when the deadline
/// elapses with the child still alive.
///
/// `pidfd_open` has been available since kernel 5.3 (2019); ktstr
/// targets 6.16+ where it is unconditionally present. The interval
/// parameter is unused here because `poll(2)` blocks until the fd
/// becomes readable or the absolute deadline elapses — there is
/// nothing to "poll faster" inside the wait. The deadline is
/// enforced via `Instant::now()` re-checks across loop iterations
/// because `poll(2)` may return EINTR (e.g. SIGCHLD coalescing); the
/// outer re-check rebuilds the remaining timeout against the
/// absolute deadline.
///
/// Liveness is observed via [`proc_pid_alive`] / pidfd POLLIN, never
/// `Child::try_wait`. PID 1 has SIGCHLD set to `SIG_IGN` for zombie
/// prevention (see [`ktstr_guest_init`]), so the kernel auto-reaps
/// the scheduler child the moment it exits. `try_wait` (which calls
/// `waitpid(pid, ..., WNOHANG)`) then returns `ECHILD`, which the
/// previous implementation mapped to `WaitError` and the caller
/// treated as still-alive — leaving a crashed scheduler undetected.
/// pidfd POLLIN and `/proc/{pid}` removal are signal-disposition
/// independent (the pidfd is readable on exit regardless of who
/// reaps; the procfs entry disappears on `release_task`), so they
/// observe the real state.
fn poll_startup(
    child: &mut Child,
    interval: std::time::Duration,
    timeout: std::time::Duration,
) -> StartupStatus {
    let pid = child.id();
    // SAFETY: `pidfd_open(2)` accepts any process the caller can
    // signal. We just spawned `child`; its pid is owned by this
    // process, so the syscall is safe to issue with no other
    // synchronisation. Failure (rare — e.g. very tight pid reuse,
    // sandbox restriction) falls back to a `proc_pid_alive` loop
    // below.
    let pidfd =
        unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::c_int, 0u32) as libc::c_int };
    if pidfd < 0 {
        // pidfd_open unsupported on this kernel. `proc_pid_alive`
        // is the SIG_IGN-safe fallback: the procfs entry vanishes
        // when the kernel runs `release_task` on the child,
        // regardless of how SIGCHLD is handled. Sleep-poll at the
        // caller's `interval` cadence until the deadline elapses;
        // the upper bound on detection latency is one `interval`.
        let start = std::time::Instant::now();
        loop {
            if !proc_pid_alive(pid) {
                return StartupStatus::Died;
            }
            let now = std::time::Instant::now();
            if now >= start + timeout {
                return StartupStatus::Alive;
            }
            let remaining = (start + timeout) - now;
            std::thread::sleep(remaining.min(interval));
        }
    }
    let start = std::time::Instant::now();
    let result = loop {
        let now = std::time::Instant::now();
        if now >= start + timeout {
            // Deadline elapsed. pidfd POLLIN never fired across
            // the entire window, so the kernel hasn't signalled
            // exit on the pidfd. Re-confirm via /proc to cover
            // the rare race where the child died between the
            // last poll and now (poll cadence is bounded by
            // EINTR-driven loops; a ~microsecond-wide window
            // exists where the child could have exited
            // post-poll-pre-now).
            break if proc_pid_alive(pid) {
                StartupStatus::Alive
            } else {
                StartupStatus::Died
            };
        }
        let remaining_ms = (start + timeout - now).as_millis().min(i32::MAX as u128) as i32;
        let mut pfd = libc::pollfd {
            fd: pidfd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` is a single-element pollfd; nfds is 1.
        // Every poll outcome (ready, timeout, EINTR, error) loops
        // back to the deadline check above, which rebuilds
        // `remaining_ms` against the absolute start+timeout so
        // EINTR cannot extend the wait past the requested
        // duration.
        let rc = unsafe { libc::poll(&mut pfd, 1, remaining_ms) };
        if rc > 0 && pfd.revents & libc::POLLIN != 0 {
            // pidfd POLLIN fires precisely at child exit (kernel
            // `pidfd_poll` in `fs/pidfs.c` checks `exit_state`,
            // woken via `do_notify_pidfd` from `exit_notify`).
            // No `try_wait` follow-up needed — POLLIN itself is
            // the proof.
            break StartupStatus::Died;
        }
        // rc == 0 (timeout) or rc < 0 (EINTR/error) re-checks the
        // deadline at the top of the loop. EINTR with remaining
        // budget loops once more; deadline-exhausted falls into
        // the elapsed branch above.
    };
    // SAFETY: pidfd is owned by this function and not used after
    // close.
    unsafe {
        libc::close(pidfd);
    }
    result
}

/// Probe-pipeline drain handles passed to [`start_scheduler`] so the
/// early-bail paths (scheduler Died, not Attached, spawn Err) can
/// flush probe output to COM2 before calling `force_reboot()`. The
/// success path's drain runs in [`start_sched_exit_monitor`]
/// instead — it sees the scheduler exit notification and waits on
/// `output_done` there.
struct ProbeDrain {
    /// Probe-thread stop request. Setting this wakes the probe
    /// thread out of its ring-buffer poll loop; the thread then
    /// emits its payload and sets `output_done`.
    stop: Arc<AtomicBool>,
    /// One-shot signal: set by the probe thread after writing
    /// `PROBE_PAYLOAD_END` to COM2. Waited on event-driven; the
    /// outer VM wall-clock timeout is the only safety net for a
    /// hung probe (per the queue-management policy: don't add
    /// arbitrary local timeouts when an event source exists).
    output_done: Arc<crate::sync::Latch>,
}

/// Drain the probe pipeline: signal stop, then block on
/// `output_done`. Called from each early-bail path in
/// [`start_scheduler`] before `force_reboot()` so the probe
/// payload (or the diagnostic-only payload the probe thread emits
/// on a forced stop) reaches COM2's host-side capture buffer.
///
/// `drain` is `None` when no probe stack was supplied — every
/// caller is a no-op in that case.
fn drain_probe_pipeline(drain: Option<&ProbeDrain>) {
    let Some(d) = drain else { return };
    d.stop.store(true, Ordering::Release);
    d.output_done.wait();
}

/// Start the scheduler binary if it exists. Returns the child process
/// and the path to its log file.
#[tracing::instrument(skip(probe_drain))]
fn start_scheduler(probe_drain: Option<ProbeDrain>) -> (Option<Child>, Option<String>) {
    if !Path::new("/scheduler").exists() {
        return (None, None);
    }

    let sched_args = fs::read_to_string("/sched_args")
        .unwrap_or_default()
        .trim()
        .to_string();
    let args: Vec<&str> = if sched_args.is_empty() {
        vec![]
    } else {
        sched_args.split_whitespace().collect()
    };

    let log_path = "/tmp/sched.log";
    let log_file = fs::File::create(log_path).ok();

    let stdout = match log_file.as_ref().and_then(|f| f.try_clone().ok()) {
        Some(f) => Stdio::from(f),
        None => Stdio::null(),
    };
    let stderr = match log_file {
        Some(f) => Stdio::from(f),
        None => Stdio::null(),
    };

    // Build RUST_LOG for the scheduler: append libbpf noise suppression
    // to whatever the guest already has. libbpf emits debug/info messages
    // through the `log` crate via scx_utils::libbpf_logger; raising its
    // threshold to warn keeps scheduler output readable.
    let sched_rust_log = match std::env::var("RUST_LOG") {
        Ok(existing) => format!("{existing},scx_utils::libbpf_logger=warn"),
        Err(_) => "info,scx_utils::libbpf_logger=warn".to_string(),
    };

    let child = Command::new("/scheduler")
        .args(&args)
        .env("RUST_LOG", &sched_rust_log)
        .stdout(stdout)
        .stderr(stderr)
        .spawn();

    match child {
        Ok(mut child) => {
            // Publish the scheduler PID via the [`SCHED_PID`] atomic
            // side channel — readers retrieve it through
            // [`sched_pid`]. The previous implementation called
            // `std::env::set_var("SCHED_PID", ...)` here, but the
            // Phase A probe thread spawned earlier in
            // `ktstr_guest_init` (`start_probe_phase_a`) is alive at
            // this point, so mutating glibc's global `__environ`
            // array races with the probe thread's potential
            // `getenv`/`execve` traffic — documented UB on Linux.
            // The atomic store is data-race-free and the published
            // value reaches readers via the same `Acquire`/`Release`
            // synchronisation the [`sched_pid`] reader uses.
            //
            // The `child.id()` value fits in `i32` because Linux pids
            // are `pid_t` (signed 32-bit on every supported arch).
            // `kernel.pid_max` is a 22-bit limit by default and the
            // kernel never returns negative pids from `fork(2)`, so
            // the cast is exact.
            SCHED_PID.store(child.id() as i32, Ordering::Release);

            match poll_startup(
                &mut child,
                std::time::Duration::from_millis(50),
                std::time::Duration::from_secs(1),
            ) {
                StartupStatus::Died => {
                    // Scheduler died during startup. Dump the
                    // scheduler log via the bulk data port — the
                    // SCHED_OUTPUT_START / SCHED_OUTPUT_END markers
                    // travel verbatim inside the chunk bytes so
                    // the host's `parse_sched_output` walker keeps
                    // working unchanged.
                    dump_sched_output(log_path);
                    crate::vmm::guest_comms::send_lifecycle(
                        crate::vmm::wire::LifecyclePhase::SchedulerDied,
                        "",
                    );
                    crate::vmm::guest_comms::send_exit(1);
                    // Drain the probe pipeline so PROBE_OUTPUT_END
                    // hits COM2 before force_reboot rips the VM.
                    // No-op when no probe stack was supplied.
                    drain_probe_pipeline(probe_drain.as_ref());
                    force_reboot();
                }
                StartupStatus::Alive => {
                    // Still running after the liveness window. Now
                    // verify the scheduler actually BOUND to sched_ext
                    // — a scheduler process can be alive but stuck in
                    // its BPF init (verifier reject, ops mismatch),
                    // which would leave the test running against the
                    // default kernel scheduler without the host ever
                    // noticing. `root/ops` is the post-attach marker.
                    let status = poll_scx_attached(
                        std::time::Duration::from_millis(50),
                        std::time::Duration::from_secs(3),
                    );
                    if !status.is_attached() {
                        dump_sched_output(log_path);
                        let reason = match status {
                            ScxAttachStatus::Timeout => "timeout",
                            ScxAttachStatus::SysfsAbsent => "sched_ext sysfs absent",
                            ScxAttachStatus::Attached => unreachable!(),
                        };
                        crate::vmm::guest_comms::send_lifecycle(
                            crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
                            reason,
                        );
                        crate::vmm::guest_comms::send_exit(1);
                        // Drain the probe pipeline before reboot —
                        // see Died-arm comment.
                        drain_probe_pipeline(probe_drain.as_ref());
                        force_reboot();
                    }
                    (Some(child), Some(log_path.to_string()))
                }
            }
        }
        Err(e) => {
            eprintln!("ktstr-init: spawn scheduler: {e}");
            // Synthesize a minimal sched-log payload framed by the
            // existing SCHED_OUTPUT_START/END markers so the host's
            // `parse_sched_output` returns the spawn-failure
            // diagnostic exactly as the prior COM2 path did.
            crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_START.as_bytes());
            send_sched_log_text(&format!("failed to spawn: {e}"));
            crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_END.as_bytes());
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerDied,
                "",
            );
            crate::vmm::guest_comms::send_exit(1);
            // Drain the probe pipeline before reboot — see
            // Died-arm comment.
            drain_probe_pipeline(probe_drain.as_ref());
            force_reboot();
        }
    }
}

/// Maximum scheduler-log chunk emitted in a single
/// [`crate::vmm::guest_comms::send_sched_log`] frame. Sub-cap of
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`] so a chunk fits
/// comfortably inside one TLV frame; chunks above this size are
/// split before emission.
const SCHED_LOG_CHUNK_BYTES: usize = 64 * 1024;

/// Send the scheduler log to the host bracketed by
/// [`crate::verifier::SCHED_OUTPUT_START`] /
/// [`crate::verifier::SCHED_OUTPUT_END`] markers. Replaces the
/// prior COM2 dump path: the markers travel verbatim inside the
/// chunk bytes so the host's `parse_sched_output` walker (which
/// scans for the start/end pair after concatenating chunks) keeps
/// working unchanged. The BPF verifier section embedded in the
/// scheduler's stderr / stdout passes through byte-for-byte so a
/// scheduler author still sees the kernel's verifier rejection
/// text in the host-side failure render.
fn dump_sched_output(log_path: &str) {
    crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_START.as_bytes());
    send_sched_log_file(log_path);
    crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_END.as_bytes());
}

/// Read the scheduler log file and emit it to the host as one or
/// more [`crate::vmm::wire::MsgType::SchedLog`] TLV chunks bounded
/// by [`SCHED_LOG_CHUNK_BYTES`]. Empty / missing file is a silent
/// no-op (mirrors the prior `dump_file_to_com2` behaviour where an
/// `Err` from `read_to_string` skipped the dump rather than
/// emitting a partial marker pair).
fn send_sched_log_file(path: &str) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let bytes = content.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() {
        let end = (start + SCHED_LOG_CHUNK_BYTES).min(bytes.len());
        crate::vmm::guest_comms::send_sched_log(&bytes[start..end]);
        start = end;
    }
}

/// Send a fixed text snippet (e.g. a "failed to spawn" diagnostic)
/// to the host as a single [`crate::vmm::wire::MsgType::SchedLog`]
/// TLV chunk. The snippet is bounded by `SCHED_LOG_CHUNK_BYTES`
/// like every other chunk; oversized snippets would be rejected
/// by the host-side per-frame cap and are guarded here by
/// truncating the input before the call.
fn send_sched_log_text(s: &str) {
    let bytes = s.as_bytes();
    let cap = SCHED_LOG_CHUNK_BYTES.min(bytes.len());
    crate::vmm::guest_comms::send_sched_log(&bytes[..cap]);
}

/// Enable sched_ext_dump trace event and pipe trace_pipe to COM1 in a
/// background thread. Returns the stop flag and thread join handle.
///
/// The reader opens trace_pipe with `O_NONBLOCK` and uses `poll()` on
/// a 200ms cadence so the loop is responsive to `stop` even when the
/// kernel never emits a sched_ext_dump event. A blocking `read(2)` on
/// trace_pipe parks the task in `tracing_wait_pipe` (kernel/trace/trace.c);
/// once that wait is entered with `iter->pos == 0` (no event ever
/// dispatched into the iterator), the kernel re-enters `wait_on_pipe`
/// after every wake because the inner loop in `tracing_wait_pipe` only
/// breaks when `!tracer_tracing_is_on(tr) && iter->pos`. Writing 0 to
/// `tracing_on` does fire `ring_buffer_wake_waiters`, but the
/// trace_pipe path supplies `wait_pipe_cond` (not the default
/// `rb_wait_once`) and that condition only flips when `iter->closed`
/// or `iter->wait_index` change — neither is touched by the trace_pipe
/// fops, so the wake produces a spurious return into `tracing_wait_pipe`
/// which immediately re-sleeps. Going non-blocking sidesteps the kernel
/// wait entirely: every iteration the userspace thread checks the stop
/// flag, polls for data, and drains any pending events without ever
/// parking in the kernel.
fn start_trace_pipe() -> (Option<Arc<AtomicBool>>, Option<std::thread::JoinHandle<()>>) {
    if Path::new(TRACE_SCHED_EXT_DUMP_ENABLE).exists() {
        let _ = fs::write(TRACE_SCHED_EXT_DUMP_ENABLE, "1");

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = std::thread::Builder::new()
            .name("trace-pipe".into())
            .spawn(move || {
                use std::os::unix::fs::OpenOptionsExt;
                let Ok(mut trace) = fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open(TRACE_PIPE)
                else {
                    return;
                };
                let Ok(mut com1) = fs::OpenOptions::new().write(true).open(COM1) else {
                    return;
                };
                let mut buf = [0u8; 4096];
                let mut drain_deadline = None;
                loop {
                    if drain_deadline.is_none() && stop_clone.load(Ordering::Acquire) {
                        drain_deadline =
                            Some(std::time::Instant::now() + std::time::Duration::from_secs(5));
                    }
                    if drain_deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                        break;
                    }

                    let mut pollfds = [PollFd::new(trace.as_fd(), PollFlags::POLLIN)];
                    match poll(&mut pollfds, PollTimeout::from(200u16)) {
                        Ok(0) => {
                            if drain_deadline.is_some() {
                                break;
                            }
                            continue;
                        }
                        Ok(_) => {}
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                    }
                    if let Some(revents) = pollfds[0].revents() {
                        if revents.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL) {
                            break;
                        }
                        if !revents.contains(PollFlags::POLLIN) {
                            // POLLHUP without POLLIN means no buffered
                            // data to drain; with POLLIN, fall through
                            // to read first so events that arrived
                            // before hangup still reach COM1.
                            if revents.contains(PollFlags::POLLHUP) {
                                break;
                            }
                            continue;
                        }
                    }

                    // Drain every byte poll says is ready before
                    // returning to the stop-flag check; otherwise a
                    // continuous trace stream could starve the stop
                    // signal for arbitrarily long. Inner-loop exits use
                    // `break` (not `return`) so the outer poll loop
                    // observes fd state (POLLHUP/POLLERR) and the
                    // drain_deadline check on the next iteration —
                    // terminating the thread from inside the drain
                    // would skip both.
                    loop {
                        match trace.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                let _ = com1.write_all(&buf[..n]);
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(_) => break,
                        }
                    }
                }
            })
            .ok();
        (Some(stop), handle)
    } else {
        (None, None)
    }
}

/// Process-wide latch fired by the guest's `hvc0_poll_loop` when the
/// host's `bpf-map-write` thread pushes `SIGNAL_BPF_WRITE_DONE` through
/// virtio-console RX.
///
/// Producer: [`hvc0_poll_loop`] (this file). Consumer: the scenario
/// executor's [`crate::scenario::Ctx::wait_for_map_write`] gate
/// (in `scenario::ops`). A test that declares `bpf_map_write` on
/// its `KtstrTestEntry` flips `wait_for_map_write=true`; the
/// scenario runner then blocks on this latch's
/// [`Latch::wait_timeout`] before starting the workload phase, so
/// the workload never observes a stale BPF map value.
///
/// `OnceLock` so the first caller materialises the [`Latch`] and
/// every subsequent caller (producer or consumer) shares the same
/// instance. `Arc` so callers can hold the latch across
/// thread-spawn boundaries without re-resolving the static.
static BPF_MAP_WRITE_DONE_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Lazily materialise and return the shared `bpf_map_write_done`
/// latch. Both the producer (`hvc0_poll_loop`) and consumer (scenario
/// `wait_for_map_write` gate) reach for this — the first caller
/// installs the [`Latch`] into [`BPF_MAP_WRITE_DONE_LATCH`], every
/// subsequent caller observes the same instance.
pub(crate) fn bpf_map_write_done_latch() -> Arc<Latch> {
    BPF_MAP_WRITE_DONE_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// Start the hvc0 wake-byte poll loop.
///
/// Spawns a background thread that polls `/dev/hvc0` for host→guest
/// wake bytes and dispatches SysRq-D / shutdown / bpf-map-write-done
/// based on the wake byte. Returns the thread's stop flag so callers
/// can request termination on teardown.
///
/// `trace_stop` is the trace_pipe reader's stop flag. The graceful
/// shutdown handler sets it so the reader enters drain mode.
fn start_hvc0_poll(trace_stop: Option<Arc<AtomicBool>>) -> Option<Arc<AtomicBool>> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    std::thread::Builder::new()
        .name("hvc0-poll".into())
        .spawn(move || {
            hvc0_poll_loop(&stop_clone, trace_stop.as_deref());
        })
        .ok();

    Some(stop)
}

/// Poll `/dev/hvc0` for host→guest wake bytes and dispatch SysRq-D /
/// shutdown / bpf-map-write-done based on the wake byte alone.
///
/// Wake source: opens `/dev/hvc0` non-blocking (`O_NONBLOCK`) and
/// `poll()`s the fd with `POLLIN` at a 1000 ms safety timeout. The
/// host pushes a byte via `VirtioConsole::queue_input` whenever it
/// requests a dump (`SIGNAL_VC_DUMP`), a graceful shutdown
/// (`SIGNAL_VC_SHUTDOWN`), or a `bpf-map-write`-complete notification
/// (`SIGNAL_BPF_WRITE_DONE`). The poll wakes within microseconds of
/// the push.
///
/// On any wake the loop:
///   1. scans every drained hvc0 byte for `SIGNAL_VC_DUMP`; on
///      observing one, triggers SysRq-D via `/proc/sysrq-trigger`.
///   2. scans every drained hvc0 byte for `SIGNAL_BPF_WRITE_DONE`;
///      on observing one, fires [`bpf_map_write_done_latch`] so the
///      scenario's `wait_for_map_write` gate resumes.
///   3. scans every drained hvc0 byte for `SIGNAL_VC_SHUTDOWN`; on
///      observing one, drives graceful shutdown (set `trace_stop`,
///      disable tracing, flush stdio + serial) and breaks.
fn hvc0_poll_loop(stop: &AtomicBool, trace_stop: Option<&AtomicBool>) {
    use std::os::unix::io::AsRawFd;

    // Open the virtio-console wake fd. Failure here used to be
    // `.expect()`d, which panicked the worker thread; the
    // process-wide panic hook installed at PID-1 entry calls
    // `force_reboot()`, so a transient open failure (e.g. devtmpfs
    // not yet populated when the thread spawns) tore the VM down
    // before any test could dispatch. Log + return instead so the
    // poll loop simply doesn't deliver wake bytes for this boot —
    // tests that rely on `bpf_map_write` notification will time out
    // on their `wait_for_map_write` latch with a recoverable error
    // instead of a forced reboot.
    let hvc0 = match fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(HVC0)
    {
        Ok(f) => f,
        Err(e) => {
            write_com2(&format!(
                "ktstr-init: hvc0 poll loop disabled — open {HVC0}: {e}"
            ));
            return;
        }
    };
    let poll_timeout_ms: PollTimeout = 1000u16.into();

    while !stop.load(Ordering::Acquire) {
        let borrowed = unsafe { BorrowedFd::borrow_raw(hvc0.as_raw_fd()) };
        let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
        match poll(&mut fds, poll_timeout_ms) {
            Ok(0) => continue,
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(_) => break,
        }
        // Inspect revents before reading: a host-side virtio-console
        // disconnect raises POLLHUP/POLLERR permanently, and without
        // this guard the bare `read().unwrap_or(0)` below returns
        // Ok(0) every iteration, the next `poll()` returns
        // immediately because the hangup is still latched, and the
        // loop spins burning CPU until `stop` is set. Mirrors the
        // pattern in `start_trace_pipe` (above): break on
        // POLLERR/POLLNVAL, break on POLLHUP-without-POLLIN, and
        // skip the read on a wake without POLLIN.
        if let Some(revents) = fds[0].revents() {
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL) {
                break;
            }
            if !revents.contains(PollFlags::POLLIN) {
                if revents.contains(PollFlags::POLLHUP) {
                    break;
                }
                continue;
            }
        }
        let mut buf = [0u8; 16];
        let mut hvc_ref: &fs::File = &hvc0;
        // Retry on EINTR (the read was interrupted by a signal before
        // returning data). The previous `unwrap_or(0)` collapsed both
        // EINTR and EIO into 0 bytes, masking transient signal races
        // (drops a real wake byte) and permanent device errors (silent
        // hang in the next poll iteration). Treat:
        //   - Ok(n): consume n bytes and dispatch signals below. An
        //     `Ok(0)` here is rare (poll already confirmed POLLIN)
        //     but harmless — the byte-contains checks no-op and the
        //     outer loop iterates normally, same as the original
        //     `unwrap_or(0)` behaviour for that case.
        //   - EINTR: retry the read inline; poll already confirmed
        //     POLLIN, so the wake byte is still in the device's RX
        //     queue waiting to be drained.
        //   - other Err: log via tracing::warn and break the outer
        //     poll loop. A non-EINTR read error after POLLIN means
        //     the device is in an unrecoverable state (host-side
        //     disconnect that didn't surface as POLLHUP, kernel-side
        //     I/O error, fd revoked) and continuing would either
        //     spin on the same error or silently miss every wake
        //     byte for the rest of the run.
        let n = 'read_retry: loop {
            match hvc_ref.read(&mut buf) {
                Ok(n) => break 'read_retry Some(n),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "ktstr-init: hvc0 read failed; aborting poll loop"
                    );
                    break 'read_retry None;
                }
            }
        };
        let Some(n) = n else { break };
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP) {
            let _ = fs::write("/proc/sysrq-trigger", "D");
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_BPF_WRITE_DONE) {
            bpf_map_write_done_latch().set();
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_VC_SHUTDOWN) {
            eprintln!("ktstr-init: shutdown request received, draining");
            if let Some(ts) = trace_stop {
                ts.store(true, Ordering::Release);
            }
            let _ = fs::write(TRACE_TRACING_ON, "0");
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            if let Ok(f) = fs::OpenOptions::new().write(true).open(COM1) {
                unsafe {
                    libc::tcdrain(std::os::unix::io::AsRawFd::as_raw_fd(&f));
                }
            }
            if let Ok(f) = fs::OpenOptions::new().write(true).open(COM2) {
                unsafe {
                    libc::tcdrain(std::os::unix::io::AsRawFd::as_raw_fd(&f));
                }
            }
            break;
        }
    }
}

/// Stop handle for the sched-exit monitor. Carries both the
/// `Arc<AtomicBool>` source-of-truth flag and a writable eventfd handle
/// the cleanup site uses to wake the monitor thread out of `poll(2)`
/// without waiting for the legacy 250 ms cadence.
///
/// Cleanup contract: before reading `stop`, the cleanup site MUST
/// `store(true, Release)` the bool AND call [`SchedExitStop::wake`].
/// The bool is the source of truth; the eventfd write delivers the
/// edge that pulls the thread out of an indefinite `poll`. The
/// eventfd is owned by this struct on the writer side and by the
/// monitor thread on the reader side; both sides drop their fds when
/// the run ends, so the kernel-side counter is reclaimed cleanly.
pub(crate) struct SchedExitStop {
    /// Stop flag the monitor thread polls under `Acquire` ordering at
    /// every loop iteration. Setting `true` is the only way to make
    /// the thread exit through its top-of-loop early-return arm; the
    /// eventfd below is the wake-edge that pairs with this store.
    pub(crate) stop: Arc<AtomicBool>,
    /// Owned eventfd write side. `wake()` writes `1` here; the
    /// monitor's `poll(2)` returns within microseconds. `None` when
    /// `eventfd(2)` failed at monitor spawn (legacy 250 ms timeout
    /// still bounds wake latency in that degraded path).
    wake_fd: Option<OwnedFd>,
}

impl SchedExitStop {
    /// Wake the monitor thread out of its `poll(2)` wait. Idempotent
    /// — eventfd in counter mode coalesces multiple writes into a
    /// single wake. EAGAIN under `EFD_NONBLOCK` (counter saturation —
    /// physically impossible with a single writer + 64-bit counter)
    /// is silently absorbed; the `Acquire`-loaded `stop` bool above
    /// remains the source of truth.
    pub(crate) fn wake(&self) {
        if let Some(ref fd) = self.wake_fd {
            // SAFETY: `fd` is the owned write side of an eventfd
            // created with `EFD_NONBLOCK`; a single 8-byte write of
            // a non-zero u64 advances the counter and edge-fires
            // every reader's `poll(POLLIN)`. The bytes pointer is a
            // 64-bit aligned local; `count` is exactly 8 as
            // eventfd(2) requires.
            let val: u64 = 1;
            let bytes = val.to_ne_bytes();
            let _ = unsafe {
                libc::write(
                    fd.as_raw_fd(),
                    bytes.as_ptr() as *const libc::c_void,
                    bytes.len(),
                )
            };
        }
    }
}

/// Monitor the scheduler child process for unexpected exit.
///
/// Blocks the monitor thread in `poll(2)` against the scheduler's
/// pidfd plus a stop-eventfd; the wait returns when either the
/// child exits (pidfd POLLIN edge from the kernel's `do_notify_pidfd`)
/// or the cleanup site fires the stop-eventfd. `/proc/{pid}` is
/// re-checked post-wake to catch the rare "pidfd opened after kernel
/// reaped" race. When `suppress_com2` is false (normal mode), writes
/// MSG_TYPE_SCHED_EXIT to the bulk port and dumps the scheduler log
/// to COM2. The host detects the bulk message and can terminate the
/// VM early. When `suppress_com2` is true (probes active), both the
/// SCHED_EXIT signal and COM2 dump are suppressed — the probe
/// pipeline handles crash detection via tp_btf/sched_ext_exit
/// instead, and the VM must stay alive for the probe thread to emit
/// output.
///
/// Uses procfs instead of waitpid because SIGCHLD is SIG_IGN (the kernel
/// auto-reaps children, making waitpid return ECHILD).
///
/// The returned [`SchedExitStop`] carries both the `Arc<AtomicBool>` the
/// monitor reads and an eventfd the cleanup site writes via
/// [`SchedExitStop::wake`] to drop wake latency from 250 ms (legacy
/// poll timeout) to microseconds.
///
/// Returns None when no scheduler is running.
fn start_sched_exit_monitor(
    sched_pid: Option<u32>,
    log_path: Option<&str>,
    suppress_com2: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) -> Option<SchedExitStop> {
    let pid = sched_pid?;
    let proc_path = format!("/proc/{pid}");
    let log_path = log_path.map(|s| s.to_string());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // Allocate a stop-eventfd. Two fds are needed: one owned by the
    // monitor thread (read + close on exit), one owned by the
    // [`SchedExitStop`] writer (`wake` writes here). `dup(2)` shares
    // the underlying counter so a write on either fd advances both
    // sides' visibility. EFD_NONBLOCK so a doubled cleanup path can't
    // stall behind a saturated counter; EFD_CLOEXEC so a future
    // `Command::new` from this thread doesn't leak the fd into a
    // child.
    //
    // `eventfd(2)` failure (extremely unlikely on KVM hosts — the
    // syscall is unconditionally available since kernel 2.6.22) falls
    // back to the legacy 250 ms `poll(2)` timeout: stop still works
    // via the `Acquire`-loaded bool, just with a worst-case 250 ms
    // wake latency instead of microseconds.
    let (monitor_fd, writer_fd): (Option<OwnedFd>, Option<OwnedFd>) = {
        let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if raw < 0 {
            let err = std::io::Error::last_os_error();
            tracing::warn!(
                err = %err,
                "ktstr-init: sched-exit-mon eventfd allocation failed; \
                 falling back to 250 ms stop poll cadence"
            );
            (None, None)
        } else {
            // SAFETY: `eventfd(2)` returned a fresh non-negative fd
            // owned by this caller. Wrapping in `OwnedFd` transfers
            // close-on-drop responsibility; `try_clone` issues a
            // `dup` so writer and monitor each carry an independent
            // fd that addresses the same kernel-side counter. A
            // dup failure leaves the monitor fd alive and disables
            // the wake path (degrades to the no-eventfd branch).
            let monitor_fd = unsafe { OwnedFd::from_raw_fd(raw) };
            match monitor_fd.try_clone() {
                Ok(writer_fd) => (Some(monitor_fd), Some(writer_fd)),
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "ktstr-init: sched-exit-mon eventfd dup failed; \
                         falling back to 250 ms stop poll cadence"
                    );
                    (Some(monitor_fd), None)
                }
            }
        }
    };

    std::thread::Builder::new()
        .name("sched-exit-mon".into())
        .spawn(move || {
            // pidfd_open lets us block on SIGCHLD-equivalent
            // notification for the scheduler process exit instead
            // of polling /proc/{pid} on a sleep cadence.
            // SAFETY: pid is the scheduler's stable pid for the
            // run; pidfd_open(2) accepts any process the caller
            // can signal (we are pid 1). pidfd_open has been
            // available since kernel 5.3 (2019); ktstr targets
            // 6.16+ where it is unconditionally present, so the
            // procfs fallback is dead code. A failure here means
            // the kernel rejected the syscall entirely (sandbox /
            // seccomp filter); abort the monitor rather than
            // fabricate a polling fallback that hides the
            // configuration error.
            let pidfd = unsafe {
                libc::syscall(libc::SYS_pidfd_open, pid as libc::c_int, 0u32) as libc::c_int
            };
            if pidfd < 0 {
                eprintln!(
                    "ktstr-init: pidfd_open failed for sched pid {pid}: {} \
                     — sched exit monitor disabled",
                    std::io::Error::last_os_error(),
                );
                return;
            }
            // The monitor-side stop fd's raw value, or `-1` when the
            // caller's eventfd allocation or dup failed. `-1` in a
            // pollfd entry is valid: the kernel ignores the slot
            // (returns revents=0), so the same `poll(2)` call works
            // on the degraded path with a finite timeout that
            // re-checks `stop` periodically.
            let stop_fd = monitor_fd.as_ref().map(|f| f.as_raw_fd()).unwrap_or(-1);
            // Poll timeout policy: when the stop eventfd is live
            // (`stop_fd >= 0`), a stop request fires the eventfd
            // edge and the wait returns within microseconds — so an
            // indefinite `-1` timeout is correct; the loop never has
            // to wake just to re-check `stop`. When the eventfd
            // allocation degraded to `None`, the legacy 250 ms
            // cadence is the only path that pulls the thread out
            // of the wait, so we fall back to that timeout.
            let poll_timeout: i32 = if stop_fd >= 0 { -1 } else { 250 };
            while !stop_clone.load(Ordering::Acquire) {
                let exited = {
                    // pidfd POLLIN fires at child exit (kernel
                    // `pidfd_poll` in `fs/pidfs.c` checks
                    // `exit_state`, woken via `do_notify_pidfd`
                    // from `exit_notify`). Adding the stop eventfd
                    // alongside makes a stop request also wake the
                    // poll, so cleanup latency drops from the
                    // legacy 250 ms (re-checking `stop` after each
                    // `poll` timeout) to the kernel's eventfd
                    // wakeup latency (microseconds).
                    //
                    // Re-checking proc_path post-`poll` is a
                    // belt-and-suspenders against the rare
                    // "pidfd was opened but the kernel reaped
                    // before we entered poll" race — an exited
                    // child's pidfd POLLIN may already be latched
                    // by the time we add it to the poll set;
                    // checking proc_path independently catches
                    // that case.
                    let mut pfds = [
                        libc::pollfd {
                            fd: pidfd,
                            events: libc::POLLIN,
                            revents: 0,
                        },
                        libc::pollfd {
                            fd: stop_fd,
                            events: libc::POLLIN,
                            revents: 0,
                        },
                    ];
                    // SAFETY: pfds is a 2-element pollfd array on
                    // the local stack; nfds matches. A `stop_fd`
                    // value of `-1` is valid per poll(2) — the
                    // kernel skips that slot. Return value not
                    // consulted — the loop re-checks the stop
                    // flag and the proc path each iteration
                    // regardless.
                    let _ = unsafe {
                        libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, poll_timeout)
                    };
                    !Path::new(&proc_path).exists()
                };
                if exited {
                    if suppress_com2.load(Ordering::Acquire) {
                        // Probes active: wait event-driven on the
                        // probe thread's `output_done` latch.
                        // Outer wall-clock VM timeout is the
                        // safety net for a hung probe — adding a
                        // local timer would cap teardown latency
                        // but also truncate slow-but-progressing
                        // probe drains, which is the exact bug
                        // we're avoiding here.
                        if let Some(ref done) = probe_output_done {
                            done.wait();
                        }
                    } else if let Some(ref path) = log_path {
                        dump_sched_output(path);
                    }
                    // Signal SCHED_EXIT after the optional probe
                    // drain (above) and the optional COM2 dump.
                    // The host kills the VM on SCHED_EXIT, so
                    // issuing it AFTER the probe pipeline finishes
                    // ensures probe JSON has hit COM2 before
                    // teardown. The probe thread sets
                    // `output_done` only after writing
                    // PROBE_PAYLOAD_END, so a successful wait
                    // guarantees the marker has landed in COM2's
                    // host-side capture buffer.
                    let exit_code: i32 = 1;
                    crate::vmm::guest_comms::send_sched_exit(exit_code);
                    // SAFETY: pidfd is owned by this thread
                    // and is no longer used after close.
                    unsafe {
                        libc::close(pidfd);
                    }
                    // `monitor_fd` (Option<OwnedFd>) drops here on
                    // function return — the OwnedFd's Drop closes
                    // the read side of the stop eventfd. The
                    // writer-side `OwnedFd` lives on the
                    // SchedExitStop returned to the caller.
                    return;
                }
                // Drain any pending stop-eventfd reads so the next
                // `poll` doesn't immediately re-fire on the same
                // edge. The `stop` AtomicBool is the source of
                // truth (re-checked at the top of the loop); the
                // eventfd is purely a wake-edge, so a missed read
                // is benign — the next iteration's poll wakes
                // either way. EAGAIN under EFD_NONBLOCK (counter
                // already 0 from a racing reader, or no edge
                // arrived) is the steady-state non-stop case.
                if stop_fd >= 0 {
                    let mut buf = [0u8; 8];
                    // SAFETY: `stop_fd` is the borrowed read side
                    // of an eventfd, valid for the lifetime of
                    // this thread (the OwnedFd is owned by the
                    // closure's `monitor_fd` and not dropped
                    // until the closure returns). `buf` is an
                    // 8-byte stack slot matching eventfd(2)'s
                    // 8-byte read requirement.
                    let _ = unsafe {
                        libc::read(stop_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len())
                    };
                }
            }
            // SAFETY: same as above — close on exit path.
            unsafe {
                libc::close(pidfd);
            }
            // `monitor_fd` drops here as the closure returns.
        })
        .ok();

    Some(SchedExitStop {
        stop,
        wake_fd: writer_fd,
    })
}

/// Execute shell-script-like commands from a file.
///
/// Handles the patterns used by sched_enable/sched_disable scripts:
/// - `echo VALUE > /path` (write VALUE to a file)
/// - Lines starting with `#` are comments
/// - Empty lines are ignored
#[tracing::instrument]
fn exec_shell_script(path: &str) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return,
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        exec_shell_line(line);
    }
}

/// Execute a single shell-like command line.
///
/// Supports:
/// - `echo VALUE > /path` — write VALUE followed by newline to /path
fn exec_shell_line(line: &str) {
    if let Some(rest) = line.strip_prefix("echo ")
        && let Some((value, path)) = rest.split_once(" > ")
    {
        let value = value.trim();
        let path = path.trim();
        if let Err(e) = fs::write(path, format!("{value}\n")) {
            eprintln!("ktstr-init: echo '{value}' > {path}: {e}");
        }
        return;
    }
    eprintln!("ktstr-init: unsupported command: {line}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mkdir_p_creates_nested() {
        let base = std::env::temp_dir().join("ktstr-rust-init-test-mkdir");
        let _ = fs::remove_dir_all(&base);
        let nested = base.join("a/b/c");
        mkdir_p(nested.to_str().unwrap());
        assert!(nested.exists());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn mkdir_p_existing_is_noop() {
        let tmp = std::env::temp_dir();
        mkdir_p(tmp.to_str().unwrap());
    }

    #[test]
    fn exec_shell_line_echo_redirect() {
        let tmp = std::env::temp_dir().join("ktstr-rust-init-echo-test");
        let path = tmp.to_str().unwrap();
        exec_shell_line(&format!("echo 42 > {path}"));
        let content = fs::read_to_string(&tmp).unwrap();
        assert_eq!(content, "42\n");
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn exec_shell_line_unsupported_input_no_panic() {
        exec_shell_line("# this is a comment");
    }

    #[test]
    fn shell_mode_not_requested_in_test() {
        // /proc/cmdline exists on the host but won't contain KTSTR_MODE=shell.
        assert!(!shell_mode_requested());
    }

    #[test]
    fn disk_template_mode_not_requested_in_test() {
        // /proc/cmdline on the host won't contain KTSTR_MODE=disk_template.
        assert!(!disk_template_mode_requested());
    }

    #[test]
    fn disk_template_dispatch_precedes_shell_when_both_present() {
        // The dispatch order in `ktstr_guest_init` is:
        //   1. disk_template_mode_requested → run mkfs + reboot, never returns
        //   2. shell_mode_requested → drop into busybox shell
        //   3. test dispatch
        //
        // If both KTSTR_MODE entries appear in /proc/cmdline (e.g.
        // operator typo, host-side cmdline-construction bug), the
        // disk_template path MUST win — running shell mode against
        // a disk that the operator intended to format would skip
        // the formatting step silently. Pin the token-parser
        // semantics so a future refactor that changes the matching
        // logic (regex, prefix-only, or per-token last-wins) does
        // not silently invert the precedence.
        let cmdline = "ro KTSTR_MODE=disk_template KTSTR_MODE=shell console=ttyS0";
        // Both checks see their token in the cmdline.
        assert!(cmdline_contains_token(cmdline, "KTSTR_MODE=disk_template"));
        assert!(cmdline_contains_token(cmdline, "KTSTR_MODE=shell"));
        // The dispatch order in ktstr_guest_init runs the
        // disk_template check FIRST, so the disk_template path is
        // taken and the shell branch is never reached. This test
        // pins the token-parser invariant; the dispatch-order
        // invariant lives in the code at ktstr_guest_init's
        // disk-template-mode block.
        //
        // Reverse-token order produces the same result — the
        // checks are commutative and dispatch-order is the only
        // disambiguator.
        let cmdline_reversed = "ro KTSTR_MODE=shell KTSTR_MODE=disk_template console=ttyS0";
        assert!(cmdline_contains_token(
            cmdline_reversed,
            "KTSTR_MODE=disk_template"
        ));
        assert!(cmdline_contains_token(cmdline_reversed, "KTSTR_MODE=shell"));
    }

    #[test]
    fn cmdline_contains_token_exact_match_not_prefix() {
        // Matching is whole-token, not prefix. A future kernel
        // cmdline that introduces e.g. `KTSTR_MODE=shell_extended`
        // must not accidentally trip the shell-mode dispatch.
        assert!(cmdline_contains_token(
            "KTSTR_MODE=shell",
            "KTSTR_MODE=shell"
        ));
        assert!(!cmdline_contains_token(
            "KTSTR_MODE=shell_extended",
            "KTSTR_MODE=shell"
        ));
        assert!(!cmdline_contains_token(
            "prefix_KTSTR_MODE=shell",
            "KTSTR_MODE=shell"
        ));
        assert!(!cmdline_contains_token("", "KTSTR_MODE=shell"));
    }

    #[test]
    fn count_online_cpus_returns_some() {
        // On any Linux host, /sys/devices/system/cpu/online exists.
        let count = count_online_cpus();
        assert!(count.is_some());
        assert!(count.unwrap() >= 1);
    }

    #[test]
    fn parse_topo_from_cmdline_not_present_on_host() {
        // Host /proc/cmdline won't contain KTSTR_TOPO.
        assert!(parse_topo_from_cmdline().is_none());
    }

    /// A child that exits immediately must be observed as `Died`
    /// well before the poll timeout. This is the regression gate
    /// for the old unconditional `sleep(1s)` — we don't want to
    /// wait a full second to notice an instant crash.
    #[test]
    fn poll_startup_detects_early_death_quickly() {
        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("spawn /bin/true");
        let start = std::time::Instant::now();
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_secs(1),
        );
        let elapsed = start.elapsed();
        assert!(
            matches!(status, StartupStatus::Died),
            "expected Died, got {status:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "early death must be detected fast, took {elapsed:?}"
        );
    }

    /// A child that stays alive past the poll window must be
    /// observed as `Alive` within ~timeout — the caller accepts
    /// this as "scheduler ready" without any longer wait.
    #[test]
    fn poll_startup_reports_alive_after_timeout() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("5")
            .spawn()
            .expect("spawn /bin/sleep");
        let start = std::time::Instant::now();
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(100),
        );
        let elapsed = start.elapsed();
        let _ = child.kill();
        let _ = child.wait();
        assert!(
            matches!(status, StartupStatus::Alive),
            "expected Alive, got {status:?}"
        );
        assert!(
            elapsed >= std::time::Duration::from_millis(100),
            "Alive must wait the full timeout, took only {elapsed:?}"
        );
        // Poll is allowed one extra interval of slack.
        assert!(
            elapsed < std::time::Duration::from_millis(300),
            "Alive should not overshoot timeout significantly, took {elapsed:?}"
        );
    }

    /// SIGCHLD signal disposition is process-wide, so the
    /// `with_sigchld_default_*` and `poll_startup_under_sigign_*`
    /// regression tests must serialize. Without this lock, two
    /// concurrent `libc::signal(SIGCHLD, ...)` calls from different
    /// test threads could leave SIGCHLD in an unexpected state when
    /// either test inspects or restores it. Poison-recovery via
    /// `unwrap_or_else(|e| e.into_inner())` matches the pattern at
    /// `src/vmm/vcpu_panic.rs::HOOK_TEST_LOCK` so a panic in one
    /// signal-aware test does not poison every other one.
    static SIGCHLD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that snapshots the current SIGCHLD disposition on
    /// construction and restores it on drop. Tests that flip
    /// `SIGCHLD` to `SIG_IGN` to reproduce the PID-1 environment
    /// must not bleed that disposition into the rest of the test
    /// run — the cargo nextest binary runs every test in a single
    /// process under threads, so a leaked `SIG_IGN` would make
    /// every subsequent `Child::wait` (in unrelated tests) return
    /// ECHILD. `signal(2)` returns the previous handler; we restore
    /// it verbatim via a second `signal` call.
    struct SigchldGuard {
        prev: libc::sighandler_t,
    }

    impl SigchldGuard {
        fn install(handler: libc::sighandler_t) -> Self {
            // SAFETY: `libc::signal` accepts any process-wide signal
            // disposition; the returned value is the previous
            // handler, captured here for restoration in `Drop`.
            let prev = unsafe { libc::signal(libc::SIGCHLD, handler) };
            Self { prev }
        }
    }

    impl Drop for SigchldGuard {
        fn drop(&mut self) {
            // SAFETY: `self.prev` was returned by an earlier
            // `libc::signal` call on the same signal number;
            // re-installing it is the documented restore pattern.
            unsafe {
                libc::signal(libc::SIGCHLD, self.prev);
            }
        }
    }

    /// Regression: with SIGCHLD set to `SIG_IGN`, a bare
    /// `Command::status()` returns `Err(ECHILD)` because the kernel
    /// auto-reaps the child before `waitpid` can observe it.
    /// `with_sigchld_default` must restore `SIG_DFL` for the
    /// closure's lifetime so `waitpid` reaps and reports a real
    /// status. After the closure returns, `SIG_IGN` must be
    /// restored.
    #[test]
    fn with_sigchld_default_captures_real_exit_status() {
        let _guard = SIGCHLD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        // Sanity: under SIG_IGN, plain Command::status() returns
        // Err(ECHILD) — proves the ambient state matches PID 1.
        let bare = Command::new("/bin/true").status();
        assert!(
            bare.is_err(),
            "under SIG_IGN, Command::status must fail with ECHILD; got {bare:?}",
        );

        // Helper restores SIG_DFL for the closure body, so the same
        // Command::status() succeeds and reports exit code 0.
        let wrapped = with_sigchld_default(|| Command::new("/bin/true").status());
        let status = wrapped.expect("with_sigchld_default must capture status");
        assert_eq!(
            status.code(),
            Some(0),
            "/bin/true must exit 0 under helper; got {status:?}",
        );

        // After the closure returns, SIG_IGN must be back in place
        // so subsequent guest children continue to be auto-reaped.
        // SAFETY: signal(SIG_IGN) reads the previous disposition
        // and re-installs SIG_IGN; we compare the previous value to
        // SIG_IGN to assert nothing changed it underneath us.
        let after = unsafe { libc::signal(libc::SIGCHLD, libc::SIG_IGN) };
        assert_eq!(
            after,
            libc::SIG_IGN,
            "with_sigchld_default must restore SIG_IGN after closure returns",
        );
    }

    /// Regression (non-zero exit propagation): the helper
    /// must surface the child's real non-zero exit code, not the
    /// previous-implementation `Err(_) => 1` mapping that swallowed
    /// every status under SIG_IGN.
    #[test]
    fn with_sigchld_default_captures_nonzero_exit_status() {
        let _guard = SIGCHLD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let wrapped = with_sigchld_default(|| Command::new("/bin/false").status());
        let status = wrapped.expect("with_sigchld_default must capture status");
        // /bin/false on every supported Unix exits with code 1.
        assert_eq!(
            status.code(),
            Some(1),
            "/bin/false must surface non-zero code under helper; got {status:?}",
        );
    }

    /// Regression: under `SIGCHLD = SIG_IGN`, a child that
    /// exits before the poll window closes MUST be observed as
    /// `Died`. The previous implementation called `Child::try_wait`
    /// which internally calls `waitpid(pid, ..., WNOHANG)`; under
    /// SIG_IGN that returns `ECHILD` and the old code mapped it to
    /// `WaitError`, which the caller in `start_scheduler` then
    /// treated as alive — leaving a crashed scheduler undetected.
    /// The fix uses `proc_pid_alive` and pidfd POLLIN, both of
    /// which are signal-disposition independent.
    #[test]
    fn poll_startup_detects_death_under_sigchld_ignore() {
        let _guard = SIGCHLD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("spawn /bin/true");
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_secs(1),
        );
        assert!(
            matches!(status, StartupStatus::Died),
            "under SIG_IGN, an exited child must be observed as Died (was {status:?})",
        );
    }

    /// Regression (Alive arm under SIG_IGN): a child that
    /// is still running when the timeout elapses must be observed
    /// as `Alive` even when SIGCHLD is `SIG_IGN`. This guards the
    /// post-timeout `proc_pid_alive` re-check that replaced the
    /// old `try_wait` call (which would have returned ECHILD-as-
    /// `WaitError` and the caller would have reported alive
    /// anyway, but the new path must not regress that branch).
    #[test]
    fn poll_startup_reports_alive_under_sigchld_ignore() {
        let _guard = SIGCHLD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let mut child = std::process::Command::new("/bin/sleep")
            .arg("5")
            .spawn()
            .expect("spawn /bin/sleep");
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(100),
        );
        // Reap the still-running child via SIGKILL + waitpid. We
        // need to drop SIG_IGN before waiting or `child.wait()`
        // would itself return ECHILD; the SigchldGuard's Drop
        // restores at the end of the test, so flip to SIG_DFL for
        // the cleanup. SAFETY: signal disposition is process-wide
        // but this test holds SIGCHLD_TEST_LOCK, so no other
        // signal-aware test runs concurrently.
        let _ = child.kill();
        unsafe {
            libc::signal(libc::SIGCHLD, libc::SIG_DFL);
        }
        let _ = child.wait();
        assert!(
            matches!(status, StartupStatus::Alive),
            "under SIG_IGN, a running child must be observed as Alive (was {status:?})",
        );
    }

    /// Regression: the [`SCHED_PID`] side channel must
    /// publish the writer's value and `sched_pid()` must return
    /// `Some(pid)` when set, `None` when the sentinel `0` is in
    /// place. Since `SCHED_PID` is a process-wide static, the test
    /// snapshots the current value, exercises both store paths,
    /// and restores the snapshot — so concurrent tests (and the
    /// real producer in `start_scheduler` if some other test ever
    /// drives it) do not see ambient corruption.
    #[test]
    fn sched_pid_side_channel_roundtrips() {
        // Snapshot and restore with `Acquire`/`Release` to mirror
        // the production load/store ordering. The test must hold
        // exclusive access to the static for its lifetime; serial
        // execution under the same process means concurrent
        // `sched_pid()` readers in other tests would race, so this
        // test is annotated to acquire `SIGCHLD_TEST_LOCK` even
        // though it has no signal interaction — the existing lock
        // is already the chokepoint for "tests that touch
        // process-wide state" and serializing through it is
        // cheaper than introducing a second mutex for one test.
        let _guard = SIGCHLD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let snapshot = SCHED_PID.load(Ordering::Acquire);

        // Sentinel 0 must read as None.
        SCHED_PID.store(0, Ordering::Release);
        assert_eq!(sched_pid(), None, "0 must read as None (sentinel)");

        // Non-zero writer publishes, reader observes.
        SCHED_PID.store(12345, Ordering::Release);
        assert_eq!(
            sched_pid(),
            Some(12345),
            "writer must publish via the atomic side channel",
        );

        // Restore so the test does not leak state into peers.
        SCHED_PID.store(snapshot, Ordering::Release);
    }

    /// Regression (no env-var write): the new fix must NOT
    /// touch `std::env::set_var("SCHED_PID", ...)` because
    /// mutating glibc's `__environ` while the probe thread is live
    /// is documented UB. Asserting that the env var is absent
    /// after a fresh atomic store is a proxy for "no rogue
    /// env-mutation snuck back in." If a future refactor brings
    /// `set_var` back, this test fails immediately.
    #[test]
    fn sched_pid_does_not_publish_via_env_var() {
        let _guard = SIGCHLD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Clear any ambient env var — some test harnesses inherit
        // `SCHED_PID` from a parent shell. SAFETY: holding the
        // mutex guarantees no concurrent env reader/writer in this
        // test binary.
        unsafe { std::env::remove_var("SCHED_PID") };

        let snapshot = SCHED_PID.load(Ordering::Acquire);
        SCHED_PID.store(99999, Ordering::Release);
        assert_eq!(sched_pid(), Some(99999));
        assert!(
            std::env::var("SCHED_PID").is_err(),
            "atomic side channel must not publish via env var",
        );
        SCHED_PID.store(snapshot, Ordering::Release);
    }
}
