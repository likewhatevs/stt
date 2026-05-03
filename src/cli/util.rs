//! Terminal-output utilities: color detection, table builders,
//! status / success / warn helpers, and the `Spinner` progress bar.
//!
//! Holds the cross-binary helpers that the rest of the CLI surface
//! delegates to for visible output. Lives in its own submodule
//! because the Spinner machinery (panic-hook stash, nesting guard,
//! termios save/restore) is its own contained subsystem unrelated
//! to kernel build / list / resolve dispatch.

use std::time::Duration;

/// Whether stderr supports color (cached per process).
pub fn stderr_color() -> bool {
    use std::io::IsTerminal;
    static COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COLOR.get_or_init(|| std::io::stderr().is_terminal())
}

/// Whether stdout supports color (cached per process). Distinct from
/// [`stderr_color`] because `cargo ktstr stats compare > report.txt`
/// pipes stdout to a file while leaving stderr on the TTY — gating
/// stdout tables on the stderr TTY state would leave ANSI escapes
/// in the file. Table-rendering code paths gate on this reading;
/// diagnostic/status prints use [`stderr_color`].
pub fn stdout_color() -> bool {
    use std::io::IsTerminal;
    static COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COLOR.get_or_init(|| std::io::stdout().is_terminal())
}

/// Build a borderless comfy-table with styling gated on
/// [`stdout_color`]. When stdout is not a TTY (CI, piped-to-file),
/// `force_no_tty` suppresses cell color escapes so a log or grep
/// capture does not land raw `\x1b[...` sequences. The NOTHING preset
/// skips box-drawing characters and keeps whitespace-padded columns,
/// matching the previous hand-rolled `format!("{:<30}…")` look while
/// auto-measuring each column from actual cell contents.
///
/// `ContentArrangement::Disabled` is the default arrangement: columns
/// expand to whatever each cell needs, even when the result spills
/// past the terminal edge. Callers that want terminal-width-aware
/// cell wrapping use [`new_wrapped_table`] (ctprof compare/show
/// reaches it via `--wrap`).
pub fn new_table() -> comfy_table::Table {
    use comfy_table::{ContentArrangement, Table, presets::NOTHING};
    let mut t = Table::new();
    t.load_preset(NOTHING);
    t.set_content_arrangement(ContentArrangement::Disabled);
    if !stdout_color() {
        t.force_no_tty();
    }
    t
}

/// Variant of [`new_table`] that opts into comfy-table's
/// terminal-width-aware [`comfy_table::ContentArrangement::Dynamic`]
/// layout. Cells too wide for the available terminal width wrap
/// inside the cell rather than pushing later columns past the edge,
/// at the cost of taller rows. Used by `ctprof compare` /
/// `ctprof show` under the `--wrap` flag; the existing
/// fixed-column [`new_table`] stays the default for every other
/// caller (locks, verifier, stats) so their output stays
/// byte-stable for shell-pipeline consumers.
///
/// When stdout is not a TTY, comfy-table's terminal-width probe
/// returns `None`. The `Dynamic` arrangement is documented to
/// degrade to `Disabled` in that case; we additionally call
/// [`comfy_table::Table::force_no_tty`] under the same
/// `!stdout_color()` gate as [`new_table`], so a piped stdout that
/// requested `--wrap` still suppresses ANSI escapes. The end-state
/// behaviour under a non-TTY stdout is therefore equivalent to
/// [`new_table`]'s — the wrap request is silently dropped rather
/// than producing unbounded-wrap output without a width.
pub fn new_wrapped_table() -> comfy_table::Table {
    use comfy_table::{ContentArrangement, Table, presets::NOTHING};
    let mut t = Table::new();
    t.load_preset(NOTHING);
    t.set_content_arrangement(ContentArrangement::Dynamic);
    if !stdout_color() {
        t.force_no_tty();
    }
    t
}

/// Restore SIGPIPE to its default action (terminate the process)
/// so piping a ktstr binary's output to a reader that closes
/// early (e.g. `... | head`) does not panic inside `print!` /
/// `println!`. Rust's startup code sets SIGPIPE to `SIG_IGN`,
/// which turns the broken-pipe write into an `io::Error` that
/// `print!` escalates to a panic. Setting `SIG_DFL` restores the
/// POSIX "process terminates on SIGPIPE" convention that Unix
/// CLI tools rely on.
///
/// Call this at the TOP of each of the three user-facing CLIs'
/// `main` — `ktstr`, `cargo-ktstr`, and `ktstr-jemalloc-probe` —
/// before the tracing subscriber installs its stderr handler and
/// before any stdout write. Shared across `src/bin/ktstr.rs`,
/// `src/bin/cargo-ktstr.rs`, and `src/bin/jemalloc_probe.rs` so
/// the three CLIs behave identically under `|` pipelines and a
/// future reword of the SAFETY rationale lands in one place. The
/// `ktstr-jemalloc-alloc-worker` binary does NOT call this — it
/// is a test-fixture target spawned by the probe's closed-loop
/// integration tests, never piped by a human operator, and its
/// stdout emission path prints a single "ready" breadcrumb that
/// the test body ignores, so SIGPIPE restoration there would
/// add noise without benefit.
///
/// No return value; the call is effectively infallible (libc's
/// `signal(2)` can't fail for a standard signal + SIG_DFL
/// handler on a live process).
///
/// # Safety (FFI)
///
/// `libc::signal` is an FFI call with no memory effects (no
/// pointer dereferences, no mutation of Rust state). `SIG_DFL`
/// is a well-known constant handler. Call must run before any
/// stdout writes so the handler is in place by the time
/// `print!` fires.
pub fn restore_sigpipe_default() {
    // SAFETY: see fn-level doc comment.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

/// Print a styled status message to stderr.
pub(crate) fn status(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[1m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Print a green success message to stderr.
pub(crate) fn success(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[32m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Print a blue warning to stderr.
pub(crate) fn warn(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[34m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Stash of the pre-spinner termios for the panic hook's restore
/// path. Populated by [`Spinner::disable_echo`] before the ECHO flag
/// is cleared, and cleared by [`Spinner::teardown`] on normal exit.
/// The panic hook reads this mutex — when populated, it replays the
/// stashed termios to the terminal BEFORE the default panic handler
/// emits its message. Under `panic = "abort"`, `Spinner::Drop` never
/// runs, so without the hook the terminal stays in echo-disabled /
/// non-canonical mode and the multi-line panic message staircases
/// (LF without CR) before SIGABRT kills the process.
static SPINNER_SAVED_TERMIOS: std::sync::Mutex<Option<libc::termios>> = std::sync::Mutex::new(None);

/// Tracks whether a [`Spinner`] is currently alive. `Spinner::start`
/// flips this from `false` to `true`; `Drop` flips it back. A
/// `debug_assert!` at start-time fires when the previous value was
/// already `true`, catching nested `Spinner::start()` calls that
/// would clobber [`SPINNER_SAVED_TERMIOS`]: the second `start` saves
/// the outer spinner's ALREADY-ECHO-disabled termios, and the outer
/// teardown then restores to the disabled state instead of the
/// original. Release builds skip the check (the assertion compiles
/// away) rather than panic in production; the flag is still
/// maintained so a future `debug_assert` → `assert` upgrade would
/// not need a second seam.
static SPINNER_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Install a panic hook that restores stdin termios from
/// [`SPINNER_SAVED_TERMIOS`] before the default panic handler prints.
/// Called via [`std::sync::Once`] from [`Spinner::disable_echo`], so
/// every Spinner that actually mutates termios triggers the install
/// exactly once per process. Idempotent — subsequent calls hit the
/// `Once` guard and no-op.
///
/// The hook delegates to the default `take_hook()` output after
/// restoring, preserving the full panic-message contract (message,
/// location, backtrace under `RUST_BACKTRACE`).
///
/// # Panic-hook stacking convention
///
/// ktstr installs hooks in two places: this spinner-termios restorer
/// and the vCPU classifier (`crate::vmm::vcpu_panic::install_once`).
/// `std::panic::set_hook` is process-wide — whichever site installs
/// LAST wins, and earlier hooks are reached only via the previous-
/// hook chain each site captures at install time. Every ktstr-side
/// installer MUST follow the stacking pattern used here: call
/// `std::panic::take_hook()` to capture the current hook, then
/// `set_hook` a closure that runs its own work AND calls the
/// captured `prev(info)` at the end. Skipping the delegation
/// breaks the chain and silently drops every earlier-installed
/// hook. See the module-level doc on `src/vmm/vcpu_panic.rs` for
/// the full rationale (limitations section) and an alternative
/// `make_hook(prev)` factoring; the pattern is identical, just
/// packaged differently.
fn install_spinner_termios_panic_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // try_lock, not lock: if the panicking thread is the
            // one mid-mutation inside Spinner::disable_echo (holds
            // the mutex across its own libc::tcsetattr call), a
            // blocking lock would deadlock the hook. try_lock
            // failure ≈ "mutex held by someone mid-mutation" — the
            // terminal state is indeterminate and the hook
            // cannot safely restore, so we fall through to the
            // default handler unchanged.
            if let Ok(guard) = SPINNER_SAVED_TERMIOS.try_lock()
                && let Some(termios) = *guard
            {
                unsafe {
                    libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios);
                }
            }
            default(info);
        }));
    });
}

/// Progress spinner for long-running CLI operations.
///
/// When stderr is a TTY, draws an animated spinner via indicatif,
/// ticks in the background, and disables stdin echo to prevent
/// keypress jank. When stderr is not a TTY, skips all indicatif
/// machinery and falls back to plain stderr writes.
/// Call `finish` with a completion message to replace it with a
/// final line, or let it drop to remove it silently; [`Drop`] also
/// restores echo and clears the bar so a panic or early `?`
/// propagation leaves the terminal in a usable state. Under
/// `panic = "abort"`, Drop does NOT run on a panic — the panic hook
/// installed by [`install_spinner_termios_panic_hook`] restores
/// termios instead, so the panic message renders cleanly before
/// SIGABRT kills the process. Note: Drop also does NOT run on
/// SIGINT/SIGTERM kill; if the spinner is interrupted mid-operation,
/// run `stty sane` to restore echo.
pub struct Spinner {
    /// None when stderr is not a TTY — no indicatif overhead.
    pb: Option<indicatif::ProgressBar>,
    /// Saved termios for echo restore. None when stdin is not a tty
    /// or when the spinner is inactive (non-TTY stderr). Owned directly
    /// (not Arc<Mutex>) because Spinner is not Clone.
    saved_termios: Option<libc::termios>,
}

impl Spinner {
    /// Start a spinner with the given message (e.g. "Building kernel...").
    ///
    /// When stderr is not a TTY, no ProgressBar or ticker thread is
    /// created — all output methods fall back to plain `eprintln!`.
    pub fn start(msg: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        // Nesting rejection: a second `Spinner::start()` while
        // another Spinner is still live would overwrite
        // SPINNER_SAVED_TERMIOS with the ALREADY-ECHO-disabled
        // termios that the outer spinner installed; the outer's
        // Drop / teardown would then restore the disabled state
        // instead of the pre-spinner state, leaving the terminal
        // broken after both exit. `debug_assert!` catches the
        // misuse under `cargo test` / `cargo nextest` without
        // paying a release-mode cost. Release builds allow the
        // nesting and accept the terminal-leakage risk (the
        // alternative — panicking release binaries — would be
        // worse than a terminal that needs `reset` after a crash
        // path that was never exercised in testing). If nesting
        // is genuinely needed in the future, flip this guard and
        // add depth-aware save/restore logic to `teardown()`.
        //
        // The flag is swapped unconditionally at start (before the
        // TTY-absence short-circuit) AND cleared in both Drop and
        // the `is_hidden()` early-return below, so the invariant
        // `SPINNER_ACTIVE == true iff a Spinner exists` holds
        // across every exit path.
        debug_assert!(
            !SPINNER_ACTIVE.swap(true, std::sync::atomic::Ordering::SeqCst),
            "Spinner::start called while another Spinner is already \
             active. Nested spinners clobber SPINNER_SAVED_TERMIOS — \
             the outer spinner's restore path would reset to the \
             already-modified termios state instead of the original. \
             If nesting is genuinely needed, refactor the save/restore \
             path to depth-count before lifting this assertion.",
        );

        if !stderr_color() {
            return Spinner {
                pb: None,
                saved_termios: None,
            };
        }

        let pb = indicatif::ProgressBar::new_spinner();
        pb.set_style(
            indicatif::ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .expect("valid template"),
        );
        pb.set_message(msg);
        pb.enable_steady_tick(Duration::from_millis(80));

        // indicatif hides the bar when NO_COLOR is set or TERM is
        // dumb, even on a real TTY. Downgrade to the non-TTY path
        // so println/finish output is not silently dropped.
        if pb.is_hidden() {
            return Spinner {
                pb: None,
                saved_termios: None,
            };
        }

        let saved_termios = Self::disable_echo();

        Spinner {
            pb: Some(pb),
            saved_termios,
        }
    }

    fn disable_echo() -> Option<libc::termios> {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            return None;
        }
        unsafe {
            let fd = libc::STDIN_FILENO;
            let mut termios: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut termios) != 0 {
                return None;
            }
            let saved = termios;
            // Stash the pre-mutation termios for the panic hook's
            // restore path. Under `panic=abort` the Spinner's Drop
            // never runs, so if a panic fires while the spinner is
            // active the terminal stays in echo-disabled mode and
            // the panic message renders with a "staircase" effect
            // (LF without CR). The hook replays the saved termios
            // before the default panic handler prints, producing a
            // readable diagnostic on the way to SIGABRT.
            install_spinner_termios_panic_hook();
            *SPINNER_SAVED_TERMIOS.lock().unwrap() = Some(saved);
            termios.c_lflag &= !libc::ECHO;
            libc::tcsetattr(fd, libc::TCSANOW, &termios);
            Some(saved)
        }
    }

    /// Restore stdin echo if we disabled it, consuming `saved_termios`
    /// via [`Option::take`]. Idempotent — `finish` and the `Drop`
    /// impl both call this; only the first call has any effect. The
    /// old standalone `clear` method was consolidated into `Drop`
    /// (calling `drop(spinner)` produces the same effect).
    fn teardown(&mut self) {
        if let Some(termios) = self.saved_termios.take() {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios);
            }
            // Clear the panic-hook stash — further panics without a
            // live Spinner should NOT try to restore a termios we
            // already restored via the normal path.
            *SPINNER_SAVED_TERMIOS.lock().unwrap() = None;
        }
    }

    /// Update the spinner message.
    pub fn set_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        if let Some(ref pb) = self.pb {
            pb.set_message(msg);
        }
    }

    /// Finish the spinner, replacing it with a completion message.
    ///
    /// In non-TTY mode, prints the message to stderr directly.
    pub fn finish(mut self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.teardown();
        match self.pb.take() {
            Some(pb) => pb.finish_with_message(msg),
            None => eprintln!("{}", msg.into()),
        }
    }

    /// Print a line above the spinner. The spinner redraws below.
    ///
    /// In non-TTY mode, prints directly to stderr.
    pub fn println(&self, msg: impl AsRef<str>) {
        match self.pb {
            Some(ref pb) => pb.println(msg),
            None => eprintln!("{}", msg.as_ref()),
        }
    }

    /// Suspend the spinner tick, execute a closure, then resume.
    /// Use for terminal output that must not race with the spinner.
    ///
    /// In non-TTY mode, calls `f` directly (no spinner to suspend).
    pub fn suspend<F: FnOnce() -> R, R>(&self, f: F) -> R {
        match self.pb {
            Some(ref pb) => pb.suspend(f),
            None => f(),
        }
    }

    /// Run `f` under a spinner that starts with `start_msg`, replaces
    /// itself with `success_msg` on `Ok`, and drops silently on `Err`
    /// so the error propagates without a stale progress bar obscuring
    /// the caller's diagnostics. The closure receives the live
    /// `&Spinner` so it can call [`Self::println`] / [`Self::suspend`]
    /// / [`Self::set_message`] during the operation.
    pub fn with_progress<T, E, F>(
        start_msg: impl Into<std::borrow::Cow<'static, str>>,
        success_msg: impl Into<std::borrow::Cow<'static, str>>,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(&Spinner) -> Result<T, E>,
    {
        let sp = Spinner::start(start_msg);
        let result = f(&sp);
        match result {
            Ok(v) => {
                sp.finish(success_msg);
                Ok(v)
            }
            Err(e) => {
                drop(sp);
                Err(e)
            }
        }
    }
}

impl Drop for Spinner {
    /// Restore terminal echo and clear any live progress bar on drop.
    ///
    /// [`finish`](Self::finish) calls [`Self::teardown`] and takes
    /// `self.pb` via [`Option::take`], so this impl is a no-op after
    /// an explicit end. When the spinner is dropped implicitly
    /// (panic, `?` propagation, `drop(sp)`, or scope exit), this
    /// restores the termios saved in [`Self::disable_echo`] and
    /// clears the live bar so stdin is usable afterwards.
    fn drop(&mut self) {
        self.teardown();
        if let Some(pb) = self.pb.take() {
            pb.finish_and_clear();
        }
        // Release the nesting guard. Paired with the `swap(true)` in
        // `Spinner::start`: Drop fires exactly once per Spinner
        // (owned value), so the flag returns to `false` and the
        // next call to `start` can succeed. Unconditional store
        // rather than a swap — a nested misuse already panicked
        // under `debug_assert`, so the ordering of the counter
        // value on the first observer side is less important than
        // releasing the guard for the next legitimate caller.
        SPINNER_ACTIVE.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_drop_without_finish_does_not_panic_in_non_tty() {
        // Regression: Spinner previously had no Drop impl so early return
        // or panic leaked the disabled-ECHO termios. The added Drop must
        // run cleanly even on the non-TTY path (pb is None, saved_termios
        // is None) that nextest exercises under stderr capture.
        let sp = Spinner::start("test");
        drop(sp);
    }

    #[test]
    fn spinner_finish_then_drop_is_idempotent() {
        // finish() takes pb via Option::take so Drop's pb.take() sees None
        // and is a no-op on the progress bar side. teardown() is
        // idempotent because it consumes saved_termios via Option::take;
        // the second call finds None and does nothing. This test
        // exercises that lifecycle end-to-end.
        let sp = Spinner::start("test");
        sp.finish("done");
    }

    /// Nesting guard pin: starting a second Spinner while another is
    /// live must panic under `debug_assert!`. Exercises the
    /// SPINNER_ACTIVE swap — without the guard, the inner spinner
    /// would stash the outer's already-ECHO-disabled termios into
    /// SPINNER_SAVED_TERMIOS, and the outer's teardown would restore
    /// to that broken state instead of the pre-spinner original.
    ///
    /// `#[should_panic]` is gated on `debug_assertions` because the
    /// assertion compiles away in release builds; running the test
    /// without the debug gate under a release harness would make
    /// the test fail when the expected panic doesn't fire. The
    /// sibling `spinner_start_releases_guard_on_drop` test covers
    /// the happy path (non-nested sequential spinners) and runs
    /// under both profiles.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Spinner::start called while another Spinner is already active")]
    fn spinner_nested_start_panics_under_debug_assertions() {
        let _outer = Spinner::start("outer");
        // This call must fire the debug_assert! — the outer is
        // still live in scope. The test framework captures the
        // panic via `#[should_panic]`.
        let _inner = Spinner::start("inner");
    }

    /// Happy path paired with the nesting-panic test: starting two
    /// spinners SEQUENTIALLY (with the first dropped before the
    /// second starts) must succeed. Guards against a regression that
    /// forgot to clear SPINNER_ACTIVE in Drop and would one-shot the
    /// guard after a single use.
    #[test]
    fn spinner_start_releases_guard_on_drop() {
        {
            let _sp = Spinner::start("first");
            // Drop at end of block.
        }
        // After the first Spinner is dropped, the guard must be
        // cleared so a fresh start succeeds without panicking.
        let _sp = Spinner::start("second");
    }
}
