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
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Full guest init lifecycle. Called from the ctor when PID 1 is
/// detected. Mounts filesystems, then either runs the test lifecycle
/// (scheduler + dispatch + reboot) or drops into an interactive
/// shell. Never returns.
pub(crate) fn ktstr_guest_init() -> ! {
    let t0 = std::time::Instant::now();

    // Panic hook: write crash diagnostic to COM2 then reboot.
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        let msg = format!("PANIC: {info}\n{bt}\n");
        // SHM write (instant memcpy, no serial bottleneck). Uses
        // try_lock to avoid deadlock if the panicking thread already
        // holds SHM_WRITE_LOCK. No-op if SHM is not initialized.
        crate::vmm::shm_ring::write_msg_nonblocking(
            crate::vmm::shm_ring::MSG_TYPE_CRASH,
            msg.as_bytes(),
        );
        // Serial fallback for panics before SHM init.
        let _ = fs::write(COM2, &msg);
        let _ = fs::write(COM1, &msg);
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // tcdrain is synchronous on the vCPU exit: when the syscall
        // returns in the guest, every byte is already in the host's
        // Serial writer Vec (PIO/MMIO is committed inside KVM_RUN
        // before userspace returns). No host-side post-drain wait
        // is needed.
        unsafe {
            libc::tcdrain(1);
            libc::tcdrain(2);
        }
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

    // Phase 2: Sentinel + stdio redirect. The sentinel is for the test
    // harness on the host; shell mode doesn't need it and it would leak
    // to the user's terminal via COM2 stdout drain.
    if !shell_mode_requested() {
        write_com2(crate::test_support::SENTINEL_INIT_STARTED);
    }
    redirect_stdio_to_com2();
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
        // Match the post-test exit semantics: drain stdout/stderr,
        // emit the exit sentinel on COM2 so the host knows we're
        // done, reboot.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe {
            libc::tcdrain(1);
            libc::tcdrain(2);
        }
        write_com2(&format!(
            "{prefix}{code}",
            prefix = crate::test_support::SENTINEL_EXIT_PREFIX,
        ));
        if let Ok(com2) = fs::OpenOptions::new().write(true).open(COM2) {
            unsafe {
                libc::tcdrain(com2.as_raw_fd());
            }
        }
        // tcdrain is synchronous on the vCPU exit — bytes are in the
        // host writer the moment it returns. No additional wait
        // needed before reboot.
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
            // Restore SIGCHLD so waitpid can reap the child and
            // retrieve the real exit code. The default SIG_IGN on
            // SIGCHLD (installed earlier in main for zombie prevention)
            // causes the kernel to auto-reap, making waitpid return
            // ECHILD and losing the exit status. Safe: single-threaded
            // PID 1 context, no other children running in exec mode.
            unsafe {
                libc::signal(libc::SIGCHLD, libc::SIG_DFL);
            }
            let status = Command::new("/bin/busybox")
                .args(["sh", "-c", &cmd])
                .status();
            unsafe {
                libc::signal(libc::SIGCHLD, libc::SIG_IGN);
            }
            let code = match status {
                Ok(s) => s.code().unwrap_or(1),
                Err(e) => {
                    eprintln!("ktstr-init: exec failed: {e}");
                    1
                }
            };
            // Exit code on stderr so it does not pollute captured
            // command output on stdout.
            eprintln!(
                "{prefix}{code}",
                prefix = crate::test_support::SENTINEL_EXEC_EXIT_PREFIX,
            );
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

    // Phase 4: SHM polling + trace pipe (background threads).
    let _s_phase4 = tracing::debug_span!("phase4_shm_trace").entered();
    let (trace_stop, trace_handle) = start_trace_pipe();
    let shm_stop = start_shm_poll(trace_stop.clone());
    drop(_s_phase4);

    // Signal the host that the scheduler is loaded and BPF programs
    // are ready for enumeration.
    crate::vmm::shm_ring::signal(1);

    // Phase 4b: Scheduler death monitor.
    // Spawn a thread that polls /proc/{pid}. If the scheduler exits during
    // the test, the thread writes MSG_TYPE_SCHED_EXIT to SHM so the host
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
    write_com2(crate::test_support::SENTINEL_PAYLOAD_STARTING);
    let code = if let Some(pa) = probe_phase_a {
        // Phase A/B split path: Phase A already attached, dispatch
        // with Phase B for BPF fentry after scheduler is running.
        crate::test_support::maybe_dispatch_vm_test_with_phase_a(&args, pa).unwrap_or(1)
    } else {
        // Non-split path: standard dispatch.
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
    if let Some(ref stop) = shm_stop {
        stop.store(true, Ordering::Release);
    }
    if let Some(ref stop) = sched_exit_stop {
        stop.store(true, Ordering::Release);
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
    // tcdrain stdout (COM2 after redirect) to wait for the UART to
    // finish transmitting all queued bytes.
    unsafe {
        libc::tcdrain(1);
    }

    // Write exit code to SHM (primary) and COM2 (fallback).
    crate::vmm::shm_ring::write_msg(
        crate::vmm::shm_ring::MSG_TYPE_EXIT,
        &(code as i32).to_ne_bytes(),
    );
    write_com2(&format!(
        "{prefix}{code}",
        prefix = crate::test_support::SENTINEL_EXIT_PREFIX,
    ));

    // Drain COM2 UART after writing the exit sentinel.
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

/// Redirect stdout and stderr to COM2 (/dev/ttyS1).
///
/// As PID 1, stdout/stderr initially point to the kernel console (COM1).
/// Test output (println!/eprintln! from the test function and framework)
/// must appear on COM2 so the host-side serial parser sees it.
fn redirect_stdio_to_com2() {
    use std::os::unix::io::AsRawFd;

    let Ok(com2) = fs::OpenOptions::new().write(true).open(COM2) else {
        return;
    };
    let fd = com2.as_raw_fd();
    unsafe {
        libc::dup2(fd, 1); // stdout
        libc::dup2(fd, 2); // stderr
    }
    // com2 is dropped here but fd 1 and 2 keep the file open.
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
/// This dispatch matches the constraint stated in the project
/// CLAUDE.md "disk template lifecycle" section: the kernel's own
/// mkfs (driven by the userspace `mkfs.btrfs` binary running
/// inside the guest) is the on-disk-format authority. The host
/// never execs mkfs against a real backing file.
fn run_disk_template_mode() -> i32 {
    redirect_stdio_to_com2();
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
    let status = Command::new(MKFS)
        .args(["-f", "--quiet", "/dev/vda"])
        .status();
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
fn redirect_all_stdio_to(path: &str) {
    use std::os::unix::io::AsRawFd;

    let Ok(dev) = fs::OpenOptions::new().read(true).write(true).open(path) else {
        return;
    };
    let fd = dev.as_raw_fd();
    unsafe {
        libc::dup2(fd, 0); // stdin
        libc::dup2(fd, 1); // stdout
        libc::dup2(fd, 2); // stderr
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
            // Enable cgroup controllers for the parent.
            let parent = Path::new(&cgroup_dir)
                .parent()
                .unwrap_or(Path::new("/sys/fs/cgroup"));
            let control = parent.join("cgroup.subtree_control");
            let _ = fs::write(&control, "+cpuset +cpu");
            return;
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
    /// `try_wait` returned an error (treated as alive by the caller
    /// so the test can still proceed).
    WaitError(std::io::Error),
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
    let attr_fd = unsafe {
        libc::open(
            c"/sys/kernel/sched_ext/root/ops".as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC,
        )
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
                    if attr_fd >= 0 {
                        // SAFETY: attr_fd opened above by this
                        // function; not used after close.
                        unsafe {
                            libc::close(attr_fd);
                        }
                    }
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
            if attr_fd >= 0 {
                // SAFETY: attr_fd opened above by this function;
                // not used after close.
                unsafe {
                    libc::close(attr_fd);
                }
            }
            return if ever_read_ok {
                ScxAttachStatus::Timeout
            } else {
                ScxAttachStatus::SysfsAbsent
            };
        }
        let remaining_ms = (start + timeout - now)
            .as_millis()
            .min(interval_ms_clamped as u128) as i32;
        if attr_fd >= 0 {
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
                fd: attr_fd,
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
/// Replaces a 50 ms sleep-and-`try_wait` loop. `pidfd_open` has been
/// available since kernel 5.3 (2019); ktstr targets 6.16+ where it
/// is unconditionally present. The interval parameter is unused
/// here because `poll(2)` blocks until the fd becomes readable or
/// the absolute deadline elapses — there is nothing to "poll
/// faster" inside the wait. The deadline is enforced via
/// `Instant::now()` re-checks across loop iterations because
/// `poll(2)` may return EINTR (e.g. SIGCHLD coalescing); the outer
/// re-check rebuilds the remaining timeout against the absolute
/// deadline.
fn poll_startup(
    child: &mut Child,
    _interval: std::time::Duration,
    timeout: std::time::Duration,
) -> StartupStatus {
    // SAFETY: `pidfd_open(2)` accepts any process the caller can
    // signal. We just spawned `child`; its pid is owned by this
    // process, so the syscall is safe to issue with no other
    // synchronisation. Failure (rare — e.g. very tight pid reuse,
    // sandbox restriction) falls back to `try_wait` which is
    // identical to the original poll behaviour.
    let pidfd = unsafe {
        libc::syscall(libc::SYS_pidfd_open, child.id() as libc::c_int, 0u32) as libc::c_int
    };
    if pidfd < 0 {
        // pidfd_open unsupported on this kernel — try_wait once
        // synchronously and report Alive; subsequent observation
        // of an instantly-dead scheduler still reaches the host
        // via the start_sched_exit_monitor pidfd path.
        return match child.try_wait() {
            Ok(Some(_)) => StartupStatus::Died,
            Ok(None) => StartupStatus::Alive,
            Err(e) => StartupStatus::WaitError(e),
        };
    }
    let start = std::time::Instant::now();
    let result = loop {
        let now = std::time::Instant::now();
        if now >= start + timeout {
            break match child.try_wait() {
                Ok(Some(_)) => StartupStatus::Died,
                Ok(None) => StartupStatus::Alive,
                Err(e) => StartupStatus::WaitError(e),
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
            break match child.try_wait() {
                Ok(Some(_)) => StartupStatus::Died,
                Ok(None) => StartupStatus::Alive,
                Err(e) => StartupStatus::WaitError(e),
            };
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
            // Set SCHED_PID env var for the test to find.
            // SAFETY: single-threaded context.
            unsafe {
                std::env::set_var("SCHED_PID", child.id().to_string());
            }

            match poll_startup(
                &mut child,
                std::time::Duration::from_millis(50),
                std::time::Duration::from_secs(1),
            ) {
                StartupStatus::Died => {
                    // Scheduler died during startup.
                    write_com2(crate::verifier::SCHED_OUTPUT_START);
                    dump_file_to_com2(log_path);
                    write_com2(crate::verifier::SCHED_OUTPUT_END);
                    write_com2(crate::test_support::SENTINEL_SCHEDULER_DIED);
                    write_com2(&format!(
                        "{prefix}1",
                        prefix = crate::test_support::SENTINEL_EXIT_PREFIX,
                    ));
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
                        write_com2(crate::verifier::SCHED_OUTPUT_START);
                        dump_file_to_com2(log_path);
                        write_com2(crate::verifier::SCHED_OUTPUT_END);
                        match status {
                            ScxAttachStatus::Timeout => write_com2(&format!(
                                "{}: timeout",
                                crate::test_support::SENTINEL_SCHEDULER_NOT_ATTACHED,
                            )),
                            ScxAttachStatus::SysfsAbsent => write_com2(&format!(
                                "{}: sched_ext sysfs absent",
                                crate::test_support::SENTINEL_SCHEDULER_NOT_ATTACHED,
                            )),
                            ScxAttachStatus::Attached => unreachable!(),
                        }
                        write_com2(&format!(
                            "{prefix}1",
                            prefix = crate::test_support::SENTINEL_EXIT_PREFIX,
                        ));
                        // Drain the probe pipeline before reboot —
                        // see Died-arm comment.
                        drain_probe_pipeline(probe_drain.as_ref());
                        force_reboot();
                    }
                    (Some(child), Some(log_path.to_string()))
                }
                StartupStatus::WaitError(e) => {
                    eprintln!("ktstr-init: check scheduler status: {e}");
                    (Some(child), Some(log_path.to_string()))
                }
            }
        }
        Err(e) => {
            eprintln!("ktstr-init: spawn scheduler: {e}");
            write_com2(crate::verifier::SCHED_OUTPUT_START);
            write_com2(&format!("failed to spawn: {e}"));
            write_com2(crate::verifier::SCHED_OUTPUT_END);
            write_com2(crate::test_support::SENTINEL_SCHEDULER_DIED);
            write_com2(&format!(
                "{prefix}1",
                prefix = crate::test_support::SENTINEL_EXIT_PREFIX,
            ));
            // Drain the probe pipeline before reboot — see
            // Died-arm comment.
            drain_probe_pipeline(probe_drain.as_ref());
            force_reboot();
        }
    }
}

/// Dump scheduler output to COM2 between markers.
fn dump_sched_output(log_path: &str) {
    write_com2(crate::verifier::SCHED_OUTPUT_START);
    dump_file_to_com2(log_path);
    write_com2(crate::verifier::SCHED_OUTPUT_END);
}

/// Write a file's contents to COM2.
fn dump_file_to_com2(path: &str) {
    if let Ok(content) = fs::read_to_string(path)
        && let Ok(mut f) = fs::OpenOptions::new().write(true).open(COM2)
    {
        let _ = f.write_all(content.as_bytes());
    }
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
                        Ok(0) => continue,
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

/// Start the SHM polling loop for dump/stall requests.
/// Reads KTSTR_SHM_BASE and KTSTR_SHM_SIZE from /proc/cmdline and polls
/// /dev/mem. Also initializes the SHM signal slot pointer for
/// `shm_ring::wait_for` / `shm_ring::signal`.
///
/// `trace_stop` is the trace_pipe reader's stop flag. The graceful
/// shutdown handler sets it so the reader enters drain mode.
fn start_shm_poll(trace_stop: Option<Arc<AtomicBool>>) -> Option<Arc<AtomicBool>> {
    let cmdline = fs::read_to_string("/proc/cmdline").ok()?;
    let (shm_base, shm_size) = crate::vmm::shm_ring::parse_shm_params_from_str(&cmdline)?;

    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    std::thread::Builder::new()
        .name("shm-poll".into())
        .spawn(move || {
            shm_poll_loop(shm_base, shm_size, &stop_clone, trace_stop.as_deref());
        })
        .ok();

    Some(stop)
}

/// Poll /dev/mem for dump request bytes.
/// Maps the full SHM region so signal slots are accessible via
/// `shm_ring::init_shm_ptr`.
///
/// On graceful shutdown (SIGNAL_SHUTDOWN_REQ), sets `trace_stop` and
/// disables tracing so the trace_pipe reader drains all buffered data
/// before exiting.
///
/// Wake source: opens `/dev/hvc0` non-blocking (`O_NONBLOCK`) and
/// `poll()`s the fd with `POLLIN` at a 1000 ms safety timeout. The
/// host pushes a byte via `VirtioConsole::queue_input` whenever it
/// writes a SHM control byte (`DUMP_REQ`, signal slot 0
/// `SIGNAL_SHUTDOWN_REQ`); the poll wakes within microseconds of
/// that push, drains the byte, and re-checks SHM. The 1 s timeout
/// bounds teardown observation of the external `stop` flag —
/// sufficient because graceful shutdown also pushes a wake byte via
/// virtio-console, and `kill` / teardown cases tolerate sub-second
/// latency. The `STALL_REQ` byte at `STALL_REQ_OFFSET` is also
/// re-checked on every wake but has no production host-side writer
/// in this codebase; it is a debugging affordance for out-of-band
/// SHM pokes.
fn shm_poll_loop(shm_base: u64, shm_size: u64, stop: &AtomicBool, trace_stop: Option<&AtomicBool>) {
    use std::os::unix::io::AsRawFd;

    let devmem = match fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/mem")
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ktstr-init: /dev/mem open failed: {e}");
            return;
        }
    };

    let m = match crate::vmm::shm_ring::mmap_devmem(devmem.as_raw_fd(), shm_base, shm_size) {
        Some(m) => m,
        None => {
            eprintln!(
                "ktstr-init: /dev/mem mmap failed: base={shm_base:#x} size={shm_size:#x} err={}",
                std::io::Error::last_os_error(),
            );
            return;
        }
    };

    let shm_ptr = m.ptr;

    // Initialize the signal slot pointer so shm_ring::wait_for and
    // shm_ring::signal can use this mmap.
    crate::vmm::shm_ring::init_shm_ptr(shm_ptr, shm_size as usize);

    let dump_offset = crate::vmm::shm_ring::DUMP_REQ_OFFSET;
    let stall_offset = crate::vmm::shm_ring::STALL_REQ_OFFSET;

    // Open `/dev/hvc0` for non-blocking reads via `O_NONBLOCK` +
    // `poll(POLLIN)`. The host pushes a byte via
    // `VirtioConsole::queue_input` whenever it writes a SHM
    // control byte (`DUMP_REQ`, signal slot 0
    // `SIGNAL_SHUTDOWN_REQ`); the poll wakes within microseconds
    // of that push. The 1000 ms safety timeout bounds teardown
    // observation of the `stop` flag. `expect()` panics if the
    // device is missing — `ktstr.kconfig` mandates
    // `CONFIG_VIRTIO_CONSOLE=y` so the kernel is required to expose
    // `/dev/hvc0`; an absent device here would mean a broken kernel
    // build, not a runtime fallback condition.
    let hvc0 = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(HVC0)
        .expect("ktstr.kconfig requires CONFIG_VIRTIO_CONSOLE=y; /dev/hvc0 must exist");
    let poll_timeout_ms: PollTimeout = 1000u16.into();

    while !stop.load(Ordering::Acquire) {
        unsafe {
            let dump_byte = *(shm_ptr.add(dump_offset));
            if dump_byte == crate::vmm::shm_ring::DUMP_REQ_SYSRQ_D {
                let _ = fs::write("/proc/sysrq-trigger", "D");
                *(shm_ptr.add(dump_offset)) = 0;
            }

            let stall_byte = *(shm_ptr.add(stall_offset));
            if stall_byte == crate::vmm::shm_ring::STALL_REQ_ACTIVATE {
                let _ = fs::File::create("/tmp/ktstr_stall");
                *(shm_ptr.add(stall_offset)) = 0;
            }
        }

        // Check for graceful shutdown request from host.
        if crate::vmm::shm_ring::read_signal(0) == crate::vmm::shm_ring::SIGNAL_SHUTDOWN_REQ {
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

        // Wake source: `poll()` on `/dev/hvc0` with a 1000 ms
        // safety timeout. The host pushes a byte into virtio-console
        // RX after every SHM control-byte write (`DUMP_REQ`,
        // `SIGNAL_SHUTDOWN_REQ`); this thread wakes within
        // microseconds of the push, drains the byte (so the next
        // iteration's poll re-arms cleanly), and re-checks SHM. The
        // byte VALUE is not inspected — any byte forces the SHM
        // re-read above. `STALL_REQ` is also re-checked but has no
        // production host-side writer (debug-only).
        let borrowed = unsafe { BorrowedFd::borrow_raw(hvc0.as_raw_fd()) };
        let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
        match poll(&mut fds, poll_timeout_ms) {
            Ok(_) | Err(nix::errno::Errno::EINTR) => {}
            Err(_) => break,
        }
        let mut buf = [0u8; 16];
        let mut hvc_ref: &fs::File = &hvc0;
        let _ = hvc_ref.read(&mut buf);
    }

    // Do NOT munmap here. SHM_PTR (OnceLock) retains the mmap pointer
    // and write_msg() in the main thread dereferences it after this
    // function returns (Phase 7: MSG_TYPE_EXIT). The guest reboots
    // immediately after — the kernel frees all mappings on exit.
}

/// Monitor the scheduler child process for unexpected exit.
///
/// Polls `/proc/{pid}` every 200ms. When the directory disappears, the
/// scheduler has exited. When `suppress_com2` is false (normal mode),
/// writes MSG_TYPE_SCHED_EXIT to SHM and dumps the scheduler log to
/// COM2. The host detects the SHM message and can terminate the VM
/// early. When `suppress_com2` is true (probes active), both the SHM
/// signal and COM2 dump are suppressed — the probe pipeline handles
/// crash detection via tp_btf/sched_ext_exit instead, and the VM
/// must stay alive for the probe thread to emit output.
///
/// Uses procfs instead of waitpid because SIGCHLD is SIG_IGN (the kernel
/// auto-reaps children, making waitpid return ECHILD).
///
/// Returns None when no scheduler is running.
fn start_sched_exit_monitor(
    sched_pid: Option<u32>,
    log_path: Option<&str>,
    suppress_com2: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) -> Option<Arc<AtomicBool>> {
    let pid = sched_pid?;
    let proc_path = format!("/proc/{pid}");
    let log_path = log_path.map(|s| s.to_string());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

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
            while !stop_clone.load(Ordering::Acquire) {
                let exited = {
                    // pidfd is readable when the process exits.
                    // Block on poll(POLLIN) so we wake within
                    // microseconds of exit. Re-check stop every
                    // 250 ms via timeout so a stop request
                    // arriving while the scheduler is alive still
                    // unblocks the loop quickly.
                    //
                    // 250 ms timeout: pidfd POLLIN fires at exit
                    // time; the timeout exists only as the
                    // upper-bound cadence at which the loop
                    // re-checks `stop` (no event-source yet for
                    // stop on this thread). Re-checking proc_path
                    // is a belt-and-suspenders against the rare
                    // "pidfd was opened but the kernel reaped
                    // before we entered poll" race.
                    let mut pfd = libc::pollfd {
                        fd: pidfd,
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    // SAFETY: pfd is a single-element pollfd; nfds
                    // is 1. Return value not consulted — the loop
                    // re-checks the stop flag and the proc path
                    // each iteration regardless.
                    let _ = unsafe { libc::poll(&mut pfd, 1, 250) };
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
                    crate::vmm::shm_ring::write_msg(
                        crate::vmm::shm_ring::MSG_TYPE_SCHED_EXIT,
                        &exit_code.to_ne_bytes(),
                    );
                    // SAFETY: pidfd is owned by this thread
                    // and is no longer used after close.
                    unsafe {
                        libc::close(pidfd);
                    }
                    return;
                }
            }
            // SAFETY: same as above — close on exit path.
            unsafe {
                libc::close(pidfd);
            }
        })
        .ok();

    Some(stop)
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
}
