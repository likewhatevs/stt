/// Rust init (PID 1) for the VM guest.
///
/// When the test binary is
/// packed as `/init` in the initramfs, `ktstr_guest_init()` is called
/// from the ctor or test harness `main()` when PID 1 is detected.
/// It never returns — it mounts filesystems, starts the scheduler,
/// dispatches the test, then reboots.
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use nix::mount::{MsFlags, mount};
use nix::sys::reboot::{RebootMode, reboot};
use nix::sys::stat::Mode;
use nix::unistd::mkdir;

/// COM2 device path for sentinel and diagnostic output.
const COM2: &str = "/dev/ttyS1";
/// COM1 device path for kernel console / trace output.
const COM1: &str = "/dev/ttyS0";

/// Returns true when this process is PID 1 (running as /init in a VM).
pub fn is_pid1() -> bool {
    unsafe { libc::getpid() == 1 }
}

/// Reboot immediately. Used for fatal init errors and normal shutdown.
fn force_reboot() -> ! {
    let _ = reboot(RebootMode::RB_AUTOBOOT);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

/// Full guest init lifecycle. Called from the ctor or test harness
/// `main()` when PID 1 is detected. Mounts filesystems, starts the
/// scheduler, dispatches the test, then reboots. Never returns.
pub(crate) fn ktstr_guest_init() -> ! {
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
        unsafe {
            libc::tcdrain(1);
            libc::tcdrain(2);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        force_reboot();
    }));

    // Ignore SIGCHLD so child processes don't become zombies.
    // PID 1 is the reaper — without this, zombie processes accumulate.
    unsafe {
        libc::signal(libc::SIGCHLD, libc::SIG_IGN);
    }

    // Phase 1: Mounts.
    mount_filesystems();

    // Enable per-program BPF runtime stats (cnt, nsecs). The kernel
    // only populates bpf_prog_stats when bpf_stats_enabled_key is set.
    let _ = fs::write("/proc/sys/kernel/bpf_stats_enabled", "1");

    // Phase 2: Sentinel + stdio redirect.
    write_com2("KTSTR_INIT_STARTED");
    redirect_stdio_to_com2();

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
    // EnvFilter respects RUST_LOG when set.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Set environment variables.
    // SAFETY: single-threaded context — PID 1 before any threads spawn.
    unsafe {
        std::env::set_var("PATH", "/bin");
        std::env::set_var("LD_LIBRARY_PATH", "/lib:/lib64:/usr/lib:/usr/lib64");
    }

    // Phase 3: Cgroup parent + Scheduler.
    // Create the cgroup parent directory before starting the scheduler
    // so it exists when the scheduler looks for it.
    create_cgroup_parent_from_sched_args();
    exec_shell_script("/sched_enable");
    let (mut sched_child, sched_log_path) = start_scheduler();

    // Phase 4: SHM polling + trace pipe (background threads).
    let (trace_stop, trace_handle) = start_trace_pipe();
    let shm_stop = start_shm_poll(trace_stop.clone());

    // Signal the host that the scheduler is loaded and BPF programs
    // are ready for enumeration.
    crate::vmm::shm_ring::signal(1);

    // Phase 4b: Scheduler death monitor.
    // Spawn a thread that polls /proc/{pid}. If the scheduler exits during
    // the test, the thread writes MSG_TYPE_SCHED_EXIT to SHM so the host
    // can detect early death without waiting for the watchdog.
    let sched_exit_stop = start_sched_exit_monitor(
        sched_child.as_ref().map(|c| c.id()),
        sched_log_path.as_deref(),
    );

    // Phase 5: Dispatch.
    // Read test args from /args in the initramfs. As PID 1, the kernel
    // passes cmdline args (console=ttyS0 etc.), not the test args.
    let args: Vec<String> = {
        let content = fs::read_to_string("/args").unwrap_or_default();
        let mut a = vec!["/init".to_string()];
        a.extend(content.lines().map(|s| s.to_string()));
        a
    };
    write_com2("KTSTR_PAYLOAD_STARTING");
    let code = crate::test_support::maybe_dispatch_vm_test_with_args(&args).unwrap_or(1);

    // Flush test output before teardown. Rust's BufWriter on stdout
    // holds data until flushed; without this the host may not see the
    // test result before reboot.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    crate::test_support::try_flush_profraw();

    // Phase 6: Scheduler cleanup.
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

    // Flush COM1 trace data before reboot. tracing_on=0 wakes the
    // blocked reader via ring_buffer_wake_waiters and causes EOF after
    // all buffered events are drained.
    let _ = fs::write(
        "/sys/kernel/tracing/events/sched_ext/sched_ext_dump/enable",
        "0",
    );
    if let Some(ref stop) = trace_stop {
        stop.store(true, Ordering::Release);
    }
    let _ = fs::write("/sys/kernel/tracing/tracing_on", "0");
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
    write_com2(&format!("KTSTR_EXIT={code}"));

    // Drain COM2 UART after writing the exit sentinel.
    if let Ok(com2) = fs::OpenOptions::new().write(true).open(COM2) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::tcdrain(com2.as_raw_fd());
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(100));

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
}

/// Recursive mkdir -p equivalent.
fn mkdir_p(path: &str) {
    let p = Path::new(path);
    if p.exists() {
        return;
    }
    if let Some(parent) = p.parent() {
        let ps = parent.to_str().unwrap_or("");
        if !ps.is_empty() && ps != "/" && !parent.exists() {
            mkdir_p(ps);
        }
    }
    let _ = mkdir(p, Mode::from_bits_truncate(0o755));
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

/// Start the scheduler binary if it exists. Returns the child process
/// and the path to its log file.
fn start_scheduler() -> (Option<Child>, Option<String>) {
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

            // Wait 1 second and check if scheduler is alive.
            std::thread::sleep(std::time::Duration::from_secs(1));
            match child.try_wait() {
                Ok(Some(_status)) => {
                    // Scheduler died during startup.
                    write_com2("===SCHED_OUTPUT_START===");
                    dump_file_to_com2(log_path);
                    write_com2("===SCHED_OUTPUT_END===");
                    write_com2("SCHEDULER_DIED");
                    write_com2("KTSTR_EXIT=1");
                    force_reboot();
                }
                Ok(None) => {
                    // Still running.
                    (Some(child), Some(log_path.to_string()))
                }
                Err(e) => {
                    eprintln!("ktstr-init: check scheduler status: {e}");
                    (Some(child), Some(log_path.to_string()))
                }
            }
        }
        Err(e) => {
            eprintln!("ktstr-init: spawn scheduler: {e}");
            write_com2("===SCHED_OUTPUT_START===");
            write_com2(&format!("failed to spawn: {e}"));
            write_com2("===SCHED_OUTPUT_END===");
            write_com2("SCHEDULER_DIED");
            write_com2("KTSTR_EXIT=1");
            force_reboot();
        }
    }
}

/// Dump scheduler output to COM2 between markers.
fn dump_sched_output(log_path: &str) {
    write_com2("===SCHED_OUTPUT_START===");
    dump_file_to_com2(log_path);
    write_com2("===SCHED_OUTPUT_END===");
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
fn start_trace_pipe() -> (Option<Arc<AtomicBool>>, Option<std::thread::JoinHandle<()>>) {
    let trace_enable = "/sys/kernel/tracing/events/sched_ext/sched_ext_dump/enable";
    if Path::new(trace_enable).exists() {
        let _ = fs::write(trace_enable, "1");

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = std::thread::Builder::new()
            .name("trace-pipe".into())
            .spawn(move || {
                let Ok(mut trace) = fs::File::open("/sys/kernel/tracing/trace_pipe") else {
                    return;
                };
                let Ok(mut com1) = fs::OpenOptions::new().write(true).open(COM1) else {
                    return;
                };
                let mut buf = [0u8; 4096];
                let mut draining = false;
                let mut drain_deadline = None;
                loop {
                    if !draining && stop_clone.load(Ordering::Acquire) {
                        draining = true;
                        drain_deadline =
                            Some(std::time::Instant::now() + std::time::Duration::from_secs(5));
                    }
                    if drain_deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                        break;
                    }
                    match trace.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let _ = com1.write_all(&buf[..n]);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
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

/// Poll /dev/mem for dump and stall request bytes.
/// Maps the full SHM region so signal slots are accessible via
/// `shm_ring::init_shm_ptr`.
///
/// On graceful shutdown (SIGNAL_SHUTDOWN_REQ), sets `trace_stop` and
/// disables tracing so the trace_pipe reader drains all buffered data
/// before exiting.
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

    while !stop.load(Ordering::Acquire) {
        unsafe {
            let dump_byte = *(shm_ptr.add(dump_offset));
            if dump_byte == b'D' {
                let _ = fs::write("/proc/sysrq-trigger", "D");
                *(shm_ptr.add(dump_offset)) = 0;
            }

            let stall_byte = *(shm_ptr.add(stall_offset));
            if stall_byte == b'S' {
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
            let _ = fs::write("/sys/kernel/tracing/tracing_on", "0");
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

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    // Do NOT munmap here. SHM_PTR (OnceLock) retains the mmap pointer
    // and write_msg() in the main thread dereferences it after this
    // function returns (Phase 7: MSG_TYPE_EXIT). The guest reboots
    // immediately after — the kernel frees all mappings on exit.
}

/// Monitor the scheduler child process for unexpected exit.
///
/// Polls `/proc/{pid}` every 200ms. When the directory disappears, the
/// scheduler has exited. Writes MSG_TYPE_SCHED_EXIT to SHM with exit
/// code 1, then dumps the scheduler log to COM2. The host monitor thread
/// detects this message via mid-flight SHM drain and can terminate the
/// VM early.
///
/// Uses procfs instead of waitpid because SIGCHLD is SIG_IGN (the kernel
/// auto-reaps children, making waitpid return ECHILD).
///
/// Returns None when no scheduler is running.
fn start_sched_exit_monitor(
    sched_pid: Option<u32>,
    log_path: Option<&str>,
) -> Option<Arc<AtomicBool>> {
    let pid = sched_pid?;
    let proc_path = format!("/proc/{pid}");
    let log_path = log_path.map(|s| s.to_string());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    std::thread::Builder::new()
        .name("sched-exit-mon".into())
        .spawn(move || {
            while !stop_clone.load(Ordering::Acquire) {
                if !Path::new(&proc_path).exists() {
                    // Scheduler process is gone. Signal the host.
                    let exit_code: i32 = 1;
                    crate::vmm::shm_ring::write_msg(
                        crate::vmm::shm_ring::MSG_TYPE_SCHED_EXIT,
                        &exit_code.to_ne_bytes(),
                    );

                    // Dump scheduler log to COM2 for diagnostics.
                    if let Some(ref path) = log_path {
                        dump_sched_output(path);
                    }
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
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
    fn exec_shell_line_ignores_comments() {
        exec_shell_line("# this is a comment");
    }

    #[test]
    fn is_pid1_false_in_test() {
        assert!(!is_pid1());
    }
}
