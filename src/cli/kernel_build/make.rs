//! `make` subprocess invocation.
//!
//! Wall-clock-bounded [`run_make`] (used for non-build invocations
//! like `defconfig` / `olddefconfig` / `mrproper`) and pipe-drained
//! [`run_make_with_output`] (used for the full build path so the
//! merged stdout+stderr can stream through a spinner). The shared
//! [`poll_child_with_timeout`] polling loop is extracted so timeout
//! mechanics can be exercised against synthetic
//! [`std::process::Child`] fixtures without spawning real `make`,
//! and [`drain_lines_lossy`] handles the byte-oriented line-drain
//! that survives non-UTF-8 compiler output.

use std::io::BufRead;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use super::super::util::Spinner;

/// Run make in a kernel directory under a wall-clock timeout.
///
/// Used for non-build make invocations (`defconfig`, `olddefconfig`,
/// `mrproper`, etc.) where the parent inherits stdout/stderr — the
/// pipe-drained sibling [`run_make_with_output`] handles the full-
/// build path with a separate EOF-driven termination.
///
/// The timeout protects against a wedged make holding the calling
/// pipeline forever. Without it, a stuck `olddefconfig` (e.g. an
/// interactive `conf` prompt that the configure_kernel pre-step
/// failed to bypass, or a kernel-tree inconsistency that wedges
/// `make`) would block the parent process indefinitely. The
/// ceiling is intentionally generous — a single `make defconfig`
/// completes in seconds on any hardware, but large WIP kernel
/// trees with many out-of-tree patches can stretch
/// `mrproper` / `olddefconfig` past the typical seconds-scale; 30
/// minutes covers every legitimate caller while still bounding a
/// genuine wedge.
///
/// Polls `try_wait` at 100ms granularity — small enough that a
/// completed make is reaped within one tick, large enough that
/// the polling itself is not measurable load. On timeout, the
/// child is killed (SIGKILL via `kill_on_drop`-style semantics)
/// and reaped before bailing so no zombie outlives the function.
pub fn run_make(kernel_dir: &Path, args: &[&str]) -> Result<()> {
    const RUN_MAKE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
    // Production poll cadence: small enough that a completed
    // make is reaped within one tick, large enough that the
    // polling itself is not measurable load. Tests pass a
    // sub-millisecond override directly to
    // [`poll_child_with_timeout`] so timeout-fires-and-reaps
    // assertions complete quickly.
    const POLL_INTERVAL: Duration = Duration::from_millis(100);

    let child = std::process::Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .spawn()
        .with_context(|| format!("spawn make {}", args.join(" ")))?;

    poll_child_with_timeout(
        child,
        RUN_MAKE_TIMEOUT,
        POLL_INTERVAL,
        &format!("make {}", args.join(" ")),
    )
}

/// Polling-loop body extracted from [`run_make`] so the timeout
/// mechanics can be exercised against synthetic [`std::process::Child`]
/// fixtures with sub-second deadlines (real `make` invocations
/// would burn the full 30-minute production timeout). Production
/// callers funnel through [`run_make`] which spawns `make`,
/// constructs the production deadline, and delegates here.
///
/// `label` is the human-facing name embedded in error messages
/// (e.g. `"make defconfig"`) — pinning a synthetic label in the
/// test surface lets the assertion match the bail wording without
/// depending on `make` being installed on the runner.
///
/// `timeout` is the wall-clock budget AFTER `child` has already
/// spawned (the deadline is computed inside the helper relative
/// to the call instant). `poll_interval` controls the
/// `try_wait` polling cadence — small enough that a completed
/// child is reaped within one tick, large enough that polling
/// itself is not measurable load. Production uses 100ms; tests
/// use 1ms so a sub-second timeout assertion completes quickly.
///
/// On timeout: kill + reap before bailing so no zombie outlives
/// the function. On a `try_wait` error: same kill+reap cleanup
/// before propagating, so a transient probe failure doesn't leak
/// the child.
pub(super) fn poll_child_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
    poll_interval: Duration,
    label: &str,
) -> Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                anyhow::ensure!(status.success(), "{label} failed");
                return Ok(());
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    // Wedged — kill + reap before bailing so no
                    // zombie persists after we return Err.
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!("{label} timed out after {timeout:?}; child killed");
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                // Reap before propagating so a transient try_wait
                // failure doesn't leak the child.
                let _ = child.kill();
                let _ = child.wait();
                return Err(e).with_context(|| format!("wait on {label}"));
            }
        }
    }
}

/// Drain a reader into a `Vec<String>`, one entry per newline-delimited
/// chunk, with a final partial chunk (no trailing newline) emitted
/// with the same lossy-UTF-8 conversion. Byte-oriented so non-UTF-8
/// input survives via `from_utf8_lossy` (U+FFFD replacement) instead
/// of being dropped at the line boundary. Strips the trailing `\n`
/// and an optional preceding `\r` so CRLF input matches LF semantics.
/// Calls `on_line` for each line before appending to the returned
/// `Vec`.
///
/// Returned entries and the `on_line` argument never carry their
/// terminating `\n` (or `\r\n`) — the strip runs before emission, so
/// callers that re-emit with `println!` get clean single-newline
/// formatting and callers that persist the strings do not double-
/// count line terminators. Interior `\r` bytes (lone CR not paired
/// with a trailing LF) pass through verbatim, matching the unit
/// coverage in `drain_lines_lossy_lone_cr_at_eof_is_preserved` and
/// `drain_lines_lossy_interior_cr_is_preserved`.
///
/// Extracted from [`run_make_with_output`] so the read logic is
/// testable with in-memory readers (the caller still owns child
/// kill+wait).
pub(super) fn drain_lines_lossy(
    mut reader: impl BufRead,
    mut on_line: impl FnMut(&str),
) -> std::io::Result<Vec<String>> {
    let mut captured = Vec::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        let mut slice: &[u8] = &buf;
        if let Some(rest) = slice.strip_suffix(b"\n") {
            slice = rest;
            if let Some(rest) = slice.strip_suffix(b"\r") {
                slice = rest;
            }
        }
        let line = String::from_utf8_lossy(slice).into_owned();
        on_line(&line);
        captured.push(line);
    }
    Ok(captured)
}

/// Run make with merged stdout+stderr piped through a spinner.
///
/// Creates a single pipe via `nix::unistd::pipe2(O_CLOEXEC)`, hands
/// the write end to the child's stdout AND stderr (a clone), and
/// reads from the read end. `O_CLOEXEC` prevents the raw pipe fds
/// from leaking into any concurrently-spawned children on other
/// threads — without the flag, a race between `pipe()` and the
/// `Stdio::from()` consumption could let an unrelated `fork+exec`
/// inherit the write end and hold the reader open indefinitely.
/// One pipe, one reader — no threads, no channel, no chance of a
/// deadlock where reading stdout blocks while stderr fills its
/// buffer. Same merged-stream semantics that `sh -c "make … 2>&1"`
/// gives, without the shell-out.
///
/// When a spinner is active, each line is printed via `println()`
/// so the spinner redraws below the output. When no spinner,
/// output is captured and shown only on failure.
///
/// Pipe-read I/O errors propagate via `Err` rather than silently
/// ending the read loop. The prior line-iterator formulation
/// (`.lines()` + `Result::ok`) dropped every error-tagged item —
/// a mid-stream read failure just looked like EOF and the child's
/// tail output disappeared without a diagnostic. The byte-oriented
/// [`drain_lines_lossy`] now surfaces such failures with `anyhow`
/// context naming the merged-stream read, so a broken-pipe or EIO
/// during make's output is caught at the call site.
///
/// Lines observed by `spinner.println()` and retained in the
/// on-failure replay buffer are LF-normalized: `drain_lines_lossy`
/// strips the trailing `\n`, and a preceding `\r` (the CRLF form
/// Make emits on some toolchain + terminal combinations) is
/// stripped too, so every line the caller sees is LF-only and
/// terminator-less. Interior lone `\r` bytes — e.g. a progress
/// bar using carriage-return redraw — pass through verbatim (see
/// `drain_lines_lossy_interior_cr_is_preserved`), which keeps
/// the on-failure replay readable without mangling tools that
/// legitimately use `\r` mid-line.
pub fn run_make_with_output(
    kernel_dir: &Path,
    args: &[&str],
    spinner: Option<&Spinner>,
) -> Result<()> {
    let (read_fd, write_fd) = nix::unistd::pipe2(nix::fcntl::OFlag::O_CLOEXEC)
        .context("create pipe for merged make stdout+stderr")?;
    let write_fd_err = write_fd
        .try_clone()
        .context("clone pipe write end for stderr")?;

    let mut child = std::process::Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .stdout(std::process::Stdio::from(write_fd))
        .stderr(std::process::Stdio::from(write_fd_err))
        .spawn()
        .with_context(|| format!("spawn make {}", args.join(" ")))?;

    // Parent has no remaining writer handles. `Stdio::from(OwnedFd)`
    // consumed `write_fd` and `write_fd_err` into the Command
    // builder; during `.spawn()` the builder installs them as the
    // child's stdout/stderr via `dup2`, then drops its own OwnedFd
    // copies. The child therefore holds the only live write ends
    // (its dup2'd stdout/stderr, fd 1/2). When `make` exits, those
    // fds are closed and the reader here sees EOF naturally.
    //
    // Read as bytes and convert each line via `from_utf8_lossy` at
    // the boundary. Compiler output can include non-UTF-8 bytes —
    // source paths on exotic filesystems, embedded binary fragments
    // from diagnostic tools, locale-encoded text — and a pure-String
    // reader would drop those lines via the `Result::ok` filter,
    // hiding real compiler errors in CI logs. Lossy conversion keeps
    // every line visible with U+FFFD where the bytes were not valid
    // UTF-8.
    let reader = std::io::BufReader::new(std::fs::File::from(read_fd));
    let captured = match drain_lines_lossy(reader, |line| {
        if let Some(sp) = spinner {
            sp.println(line);
        }
    }) {
        Ok(v) => v,
        Err(e) => {
            // On pipe-read I/O failure, kill and reap the child
            // before propagating so `make` doesn't linger as a
            // zombie — stdlib's Child does not auto-wait on drop.
            // Both ops use `.ok()` because the read-side error is
            // the actionable diagnostic; a secondary wait/kill
            // failure should not mask it.
            child.kill().ok();
            child.wait().ok();
            return Err(e).context("read merged make stdout+stderr");
        }
    };

    let status = child.wait()?;
    if !status.success() {
        // Always show captured output on failure so CI logs contain
        // the actual compiler errors, not just "make failed".
        for line in &captured {
            eprintln!("{line}");
        }
        bail!("make {} failed", args.join(" "));
    }
    Ok(())
}

/// Build the kernel with output piped through a spinner.
///
/// `jobs_override` supplies the `-jN` count when set (used by
/// `kernel_build_pipeline` under `--cpu-cap` to keep gcc's
/// parallelism aligned with the reserved CPU count). `None`
/// falls back to `std::thread::available_parallelism`.
pub fn make_kernel_with_output(
    kernel_dir: &Path,
    spinner: Option<&Spinner>,
    jobs_override: Option<usize>,
) -> Result<()> {
    let nproc = jobs_override.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let args = build_make_args(nproc);
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_make_with_output(kernel_dir, &arg_refs, spinner)
}

/// Build the make arguments for a kernel build.
///
/// Returns the argument list that would be passed to `make` for a
/// parallel kernel build: `["-jN", "KCFLAGS=-Wno-error"]`.
pub(super) fn build_make_args(nproc: usize) -> Vec<String> {
    vec![format!("-j{nproc}"), "KCFLAGS=-Wno-error".into()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Whether `name` resolves to a binary on `PATH`. Inlined here
    /// (rather than reaching across to `super::super::resolve::resolve_in_path`)
    /// so the test module is self-contained and cannot regress on
    /// a path change in the resolver helper.
    fn make_in_path() -> bool {
        let Ok(path) = std::env::var("PATH") else {
            return false;
        };
        std::env::split_paths(&path).any(|p| p.join("make").is_file())
    }

    // -- drain_lines_lossy --

    #[test]
    fn drain_lines_lossy_eof_terminated_happy_path() {
        let input: &[u8] = b"alpha\nbeta\ngamma\n";
        let mut seen = Vec::new();
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |line| {
            seen.push(line.to_string())
        })
        .unwrap();
        assert_eq!(captured, vec!["alpha", "beta", "gamma"]);
        assert_eq!(seen, captured);
    }

    #[test]
    fn drain_lines_lossy_strips_crlf() {
        let input: &[u8] = b"one\r\ntwo\r\nthree\r\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["one", "two", "three"]);
    }

    #[test]
    fn drain_lines_lossy_non_utf8_bytes_survive_via_replacement() {
        let input: &[u8] = b"valid\n\xffbroken\ntail\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["valid", "\u{FFFD}broken", "tail"]);
    }

    #[test]
    fn drain_lines_lossy_empty_stream_yields_empty_vec() {
        let input: &[u8] = b"";
        let mut calls = 0usize;
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| calls += 1).unwrap();
        assert!(captured.is_empty());
        assert_eq!(calls, 0);
    }

    #[test]
    fn drain_lines_lossy_single_line_without_trailing_newline() {
        let input: &[u8] = b"no-newline";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["no-newline"]);
    }

    #[test]
    fn drain_lines_lossy_lone_cr_at_eof_is_preserved() {
        let input: &[u8] = b"foo\r";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["foo\r"]);
    }

    #[test]
    fn drain_lines_lossy_interior_cr_is_preserved() {
        let input: &[u8] = b"ab\rcd\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["ab\rcd"]);
    }

    #[test]
    fn drain_lines_lossy_propagates_io_error_after_first_read() {
        use std::io::{BufReader, ErrorKind, Read};

        struct FlakyReader {
            calls: usize,
        }
        impl Read for FlakyReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.calls += 1;
                match self.calls {
                    1 => {
                        let data = b"line1\n";
                        let n = data.len().min(buf.len());
                        buf[..n].copy_from_slice(&data[..n]);
                        Ok(n)
                    }
                    _ => Err(std::io::Error::new(ErrorKind::BrokenPipe, "pipe closed")),
                }
            }
        }

        let err = drain_lines_lossy(BufReader::new(FlakyReader { calls: 0 }), |_| {})
            .expect_err("flaky reader must surface Err");
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
    }

    #[test]
    fn drain_lines_lossy_mixed_lf_and_crlf() {
        let input: &[u8] = b"lf-line\ncrlf-line\r\nlf-again\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["lf-line", "crlf-line", "lf-again"]);
    }

    #[test]
    fn drain_lines_lossy_empty_lines_lf() {
        let input: &[u8] = b"a\n\nb\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["a", "", "b"]);
    }

    #[test]
    fn drain_lines_lossy_empty_lines_crlf() {
        let input: &[u8] = b"\r\n\r\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["", ""]);
    }

    #[test]
    fn drain_lines_lossy_callback_fires_once_per_line_in_order() {
        let input: &[u8] = b"a\nb\nc\n";
        let lens = std::cell::RefCell::new(Vec::<usize>::new());
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_line| {
            let mut v = lens.borrow_mut();
            let current = v.len();
            v.push(current);
        })
        .unwrap();
        assert_eq!(captured, vec!["a", "b", "c"]);
        assert_eq!(lens.into_inner(), vec![0, 1, 2]);
    }

    // -- run_make_with_output --

    /// `Command::current_dir` on a non-existent path causes
    /// `Command::spawn` to fail before exec, with an underlying
    /// `io::Error` of kind `NotFound`. The wrap via
    /// `.with_context(|| format!("spawn make {}", ...))` must surface
    /// BOTH the `"spawn make <args>"` annotation AND the underlying
    /// `io::Error` with `ErrorKind::NotFound` in the anyhow chain.
    /// `ErrorKind::NotFound` is structural and locale-free; matching
    /// on the rendered "No such file or directory" string would
    /// flake under `LANG=fr_FR.UTF-8`.
    #[test]
    fn run_make_with_output_surfaces_actionable_error_when_kernel_dir_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let missing = tmp.path().join("nonexistent_child");
        let err = run_make_with_output(&missing, &["foo"], None)
            .expect_err("nonexistent kernel_dir must surface a spawn failure");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("spawn make foo"),
            "expected `spawn make foo` context layer, got: {rendered}"
        );
        let has_not_found = err.chain().any(|e| {
            e.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
        });
        assert!(
            has_not_found,
            "expected underlying io::Error with ErrorKind::NotFound in anyhow chain, \
             got: {rendered}"
        );
    }

    /// End-to-end exercise of the merged-pipe path against a real
    /// `make` invocation that emits ~200 KiB across stdout+stderr,
    /// past the 64 KiB Linux pipe buffer. Pins both the no-deadlock
    /// invariant (single-pipe + single-reader cannot deadlock) and
    /// the failure-path Err wording (`"make ... failed"` from the
    /// final `bail!`).
    #[test]
    fn run_make_with_output_drains_high_volume_failing_make_without_deadlock() {
        if !make_in_path() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        let stdout_chunk: String = "S".repeat(1024);
        let stderr_chunk: String = "E".repeat(1024);
        let mut recipe = String::new();
        for _ in 0..100 {
            recipe.push_str(&format!("\t@printf '%s\\n' '{stdout_chunk}'\n"));
            recipe.push_str(&format!("\t@printf '%s\\n' '{stderr_chunk}' >&2\n"));
        }
        let makefile = format!("default:\n{recipe}\t@false\n");
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();
        let err = run_make_with_output(dir.path(), &["default"], None)
            .expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording from bail!, got: {rendered}"
        );
    }

    /// Stderr-only high-volume burst: 128 KiB to stderr alone (2x
    /// the default 64 KiB pipe buffer). No stdout writes — buffer
    /// can only drain via the merged-pipe reader. A regression that
    /// wired stderr to a separate unread pipe would deadlock here.
    #[test]
    fn run_make_with_output_drains_stderr_only_high_volume_without_deadlock() {
        if !make_in_path() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        let chunk: String = "X".repeat(1024);
        let mut recipe = String::new();
        for _ in 0..128 {
            recipe.push_str(&format!("\t@printf '%s\\n' '{chunk}' >&2\n"));
        }
        let makefile = format!("default:\n{recipe}\t@false\n");
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();
        let err = run_make_with_output(dir.path(), &["default"], None)
            .expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording, got: {rendered}"
        );
    }

    /// Spawn-failure path must not leak the pipe2 OwnedFds. Counts
    /// `/proc/self/fd` entries before and after a guaranteed-spawn-
    /// failure call; the count must not grow over 128 iterations.
    /// A regression that switched to raw fd integers (no Drop) or
    /// consumed write_fd via a path other than Stdio::from would
    /// surface as a 1-3 fd leak per call (128-384 over the loop).
    #[test]
    fn run_make_with_output_releases_fds_on_spawn_failure() {
        let proc_fd = std::path::Path::new("/proc/self/fd");
        if !proc_fd.is_dir() {
            skip!("/proc/self/fd not available");
        }
        let count_fds = || -> usize {
            std::fs::read_dir(proc_fd)
                .expect("read /proc/self/fd")
                .filter_map(|e| e.ok())
                .count()
        };
        let tmp = tempfile::TempDir::new().unwrap();
        let missing = tmp.path().join("nonexistent_child");
        // Warm-up pass: ignore first-call process-wide allocations.
        let _ = run_make_with_output(&missing, &["foo"], None);
        let before = count_fds();
        const FD_LEAK_ITERATIONS: u32 = 128;
        for _ in 0..FD_LEAK_ITERATIONS {
            let _ = run_make_with_output(&missing, &["foo"], None);
        }
        let after = count_fds();
        assert!(
            after <= before,
            "fd leak on spawn failure: {before} -> {after} \
             ({FD_LEAK_ITERATIONS} calls, expected no growth)"
        );
    }

    // -- poll_child_with_timeout --

    fn spawn_sleeping_child(seconds: u64) -> (std::process::Child, u32) {
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("sleep {seconds}"))
            .spawn()
            .expect("spawn sh -c sleep N");
        let pid = child.id();
        (child, pid)
    }

    fn pid_is_alive(pid: u32) -> bool {
        use nix::sys::signal::kill;
        use nix::unistd::Pid;
        kill(Pid::from_raw(pid as i32), None).is_ok()
    }

    /// Timeout fires when the child outlives the deadline; the
    /// helper bails with the labeled timeout error AND reaps the
    /// child (no zombie persists past helper return). Three pins:
    /// (1) bail wording carries label + `timed out after`,
    /// (2) elapsed wall-clock stays within a small multiple of the
    /// configured timeout (proves deadline check works),
    /// (3) PID slot is reclaimed (proves child.wait() ran).
    #[test]
    fn poll_child_with_timeout_bails_and_reaps_on_timeout() {
        let (child, pid) = spawn_sleeping_child(60);
        assert!(
            pid_is_alive(pid),
            "fixture precondition: spawned child pid {pid} must be \
             alive before the helper runs",
        );

        let start = std::time::Instant::now();
        let result = poll_child_with_timeout(
            child,
            Duration::from_millis(100),
            Duration::from_millis(1),
            "make wedged-target",
        );
        let elapsed = start.elapsed();

        let err = result.expect_err("timed-out child must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make wedged-target"),
            "timeout bail must include the label parameter; got: {rendered}",
        );
        assert!(
            rendered.contains("timed out after"),
            "timeout bail must include the literal `timed out after` \
             phrase so CI log scrapers can pattern-match wedged builds; \
             got: {rendered}",
        );

        assert!(
            elapsed < Duration::from_secs(5),
            "helper must return within a small multiple of the \
             configured timeout (100ms); took {elapsed:?} which \
             suggests the deadline check is broken",
        );

        let zombie_check_deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            if !pid_is_alive(pid) {
                break;
            }
            if std::time::Instant::now() >= zombie_check_deadline {
                panic!(
                    "child pid {pid} still alive 1s after helper returned — \
                     timeout path leaked a zombie (missing child.wait() \
                     after child.kill()?)",
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Successful pre-deadline exit: the helper observes
    /// `Ok(Some(status))` with success, returns Ok, and reaps via
    /// the natural process-exit path. Pins that the timeout
    /// machinery does not false-fire on a fast-exiting child.
    #[test]
    fn poll_child_with_timeout_succeeds_when_child_exits_clean() {
        let child = std::process::Command::new("true")
            .spawn()
            .expect("spawn true");
        let pid = child.id();

        let result = poll_child_with_timeout(
            child,
            Duration::from_secs(5),
            Duration::from_millis(1),
            "make happy-target",
        );
        assert!(
            result.is_ok(),
            "child that exits 0 must surface as Ok; got: {result:?}",
        );
        let zombie_check_deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            if !pid_is_alive(pid) {
                break;
            }
            if std::time::Instant::now() >= zombie_check_deadline {
                panic!(
                    "child pid {pid} still alive 1s after Ok return — \
                     successful-exit path leaked a zombie",
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Failed pre-deadline exit: the helper observes
    /// `Ok(Some(status))` with non-success and surfaces as Err with
    /// `{label} failed`. Distinct from the timeout case because the
    /// bail wording differs (`failed` vs `timed out after`); CI log
    /// scrapers must distinguish wedged-make from build-failed.
    #[test]
    fn poll_child_with_timeout_surfaces_nonzero_exit_as_err() {
        let child = std::process::Command::new("false")
            .spawn()
            .expect("spawn false");
        let result = poll_child_with_timeout(
            child,
            Duration::from_secs(5),
            Duration::from_millis(1),
            "make broken-target",
        );
        let err = result.expect_err("child that exits non-zero must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make broken-target"),
            "non-zero-exit bail must include the label; got: {rendered}",
        );
        assert!(
            rendered.contains("failed"),
            "non-zero-exit bail must use the `failed` wording so it is \
             distinguishable from the timeout-path's `timed out after`; \
             got: {rendered}",
        );
        assert!(
            !rendered.contains("timed out"),
            "non-zero-exit bail must NOT contain `timed out` — that \
             phrase belongs to the deadline-fired path only; got: {rendered}",
        );
    }

    // -- build_make_args --

    #[test]
    fn cli_build_make_args_single_core() {
        let args = build_make_args(1);
        assert_eq!(args, vec!["-j1", "KCFLAGS=-Wno-error"]);
    }

    #[test]
    fn cli_build_make_args_multi_core() {
        let args = build_make_args(16);
        assert_eq!(args, vec!["-j16", "KCFLAGS=-Wno-error"]);
    }
}
