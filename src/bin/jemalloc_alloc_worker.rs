//! Tiny jemalloc-linked target for the probe's cross-process
//! closed-loop test (see `tests/jemalloc_probe_tests.rs`).
//!
//! # Modes
//!
//! - **Default**: reads `<BYTES>` from argv, allocates on the main
//!   thread, writes the pid-scoped ready marker via
//!   [`ktstr::worker_ready::worker_ready_marker_path`], then parks
//!   forever. Single-threaded so `tid == pid` and the test can
//!   match on `threads[N].tid` without a TID handshake; a
//!   `/proc/self/task` self-check rejects any silent extra thread.
//! - **`--churn`**: after the main allocation and ready-marker,
//!   loops spawn+join on helper threads to exercise the probe's
//!   ESRCH handling in
//!   `jemalloc_probe_survives_thread_churn`. Relaxes the
//!   single-thread self-check.
//!
//! Links `tikv_jemallocator` so the probe's DWARF-backed lookup
//! finds `tsd_s.thread_allocated`. Minimum allocation is 16 KiB (to
//! route through jemalloc's huge/large path and skip tcache
//! deferred updates); the test passes 16 MiB. Keep the binary
//! minimal — it travels in the guest initramfs next to the probe.
//!
//! # Exit codes
//!
//! Process termination is always by external signal; any non-zero
//! exit is a setup failure:
//! - `2`: `bytes == 0` (caller bug — zero-size alloc would panic in `touch`).
//! - `3`: default-mode `/proc/self/task` self-check failed.
//! - `4`: ready-marker write failed.
//! - `5`: argv parse failed (missing `<BYTES>` or not a `usize`).
//!
//! Argv is hand-rolled (one flag, one positional). The tipping
//! point for switching to `clap-derive` is when the surface needs
//! structured help text (`--help`), multiple subcommand-like modes,
//! value validation beyond `str::parse`, or environment-variable
//! fallbacks — any ONE of those lands in under ten lines of
//! derive-macro annotations and would cost about the same in hand
//! code. Adding a single extra `--flag` or positional while the
//! surface still fits in ~10 lines of std code is not enough to
//! pull in the dependency.

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::io::Write;
use std::time::Duration;

// Share the ready-marker path format with the ktstr library
// (`src/worker_ready.rs`) via `#[path]` rather than `use ktstr::...`.
// Linking the ktstr library into this binary would pull in the
// library's early-dispatch ctor (`test_support::dispatch::ktstr_test_early_dispatch`,
// tagged `#[ctor::ctor]`) plus the rest of the crate, bloating the
// initramfs image and adding `.init_array` work that can stall the
// probe's cross-process timing. `#[path]` compiles the same source
// file into this bin crate directly — zero linker cost, same
// single-source-of-truth behavior.
#[path = "../worker_ready.rs"]
mod worker_ready;
use worker_ready::worker_ready_marker_path;

/// Force jemalloc to observe the heap buffer by reading through it.
///
/// LLVM can in principle observe that a freshly-zeroed `Vec`'s heap
/// buffer is never read through the pointer and fold the allocation
/// away. Black-boxing a reference alone is not strong enough. Force
/// an actual heap read through `v[0]` plus black-box the raw pointer
/// and length so the allocation is provably materialized and
/// jemalloc's `thread_allocated` counter is updated before the probe
/// sees it.
///
/// Inlined at every tick of the churn and park loops so the optimizer
/// cannot move the free before the loop iteration.
#[inline]
fn touch(v: &[u8]) {
    let _ = std::hint::black_box(v[0]);
    std::hint::black_box(v.as_ptr());
    std::hint::black_box(v.len());
}

fn main() {
    // Parse args: first non-flag positional is bytes; `--churn`
    // anywhere on the line enables thread-churn mode. See module doc
    // for the argv-parsing rationale (no clap dep).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let churn = args.iter().any(|a| a == "--churn");
    let bytes: usize = match args.iter().find(|a| !a.starts_with("--")) {
        Some(raw) => match raw.parse() {
            Ok(n) => n,
            Err(e) => {
                eprintln!(
                    "jemalloc-alloc-worker: failed to parse BYTES argument {raw:?} as usize: {e}; \
                     usage: jemalloc-alloc-worker [--churn] <BYTES>"
                );
                std::process::exit(5);
            }
        },
        None => {
            eprintln!(
                "jemalloc-alloc-worker: missing required positional <BYTES>; \
                 usage: jemalloc-alloc-worker [--churn] <BYTES>"
            );
            std::process::exit(5);
        }
    };

    // A zero-length allocation would indexing-panic on `v[0]` inside
    // `touch`. Reject with a dedicated exit code so the test can
    // distinguish a bad caller argument from the other failure modes.
    // See module doc's "Exit codes" section for the full enumeration.
    if bytes == 0 {
        eprintln!(
            "jemalloc-alloc-worker: bytes=0 is not a valid allocation size; \
             caller must pass a positive byte count"
        );
        std::process::exit(2);
    }

    // Fail fast if the process is not single-threaded. `/proc/self/task`
    // lists one directory entry per TID; more than one means an
    // unexpected helper thread is running and the `tid == pid`
    // invariant the test body relies on is already broken. Note that
    // the test-side `thread_count(&metrics) != 1` assertion in
    // tests/jemalloc_probe_tests.rs is the authoritative guard — this
    // check only exits early with an actionable diagnostic instead of
    // letting the probe observe a silently-broken worker.
    //
    // Skipped under `--churn`: churn mode intentionally runs many
    // short-lived helper threads, breaking the tid==pid invariant on
    // purpose. The ESRCH-stress test does NOT rely on that invariant
    // — it asserts the probe survives rapid thread exit races
    // without crashing, using ThreadResult shape in probe JSON.
    if !churn {
        match std::fs::read_dir("/proc/self/task") {
            Ok(iter) => {
                let n = iter.filter(|e| e.is_ok()).count();
                if n != 1 {
                    eprintln!(
                        "jemalloc-alloc-worker: /proc/self/task has {n} entries, expected 1; \
                         extra threads break the tid==pid identity"
                    );
                    std::process::exit(3);
                }
            }
            Err(e) => {
                eprintln!("jemalloc-alloc-worker: read_dir(/proc/self/task) failed: {e}");
                std::process::exit(3);
            }
        }
    }

    // Allocate + hold under black_box so the optimizer cannot elide
    // the vector. The allocation runs on the main thread, which is
    // also the thread jemalloc updates `thread_allocated` on. See
    // `touch` doc for the black-box rationale.
    let known: Vec<u8> = vec![0u8; bytes];
    touch(&known);

    // Signal readiness via a pid-scoped marker file. The test body
    // polls for this file's existence — cheaper and tighter than a
    // fixed 500ms sleep, and safe because each `#[ktstr_test]` boots
    // a fresh VM with a clean /tmp (so no stale file from a prior run
    // can race the poll). The file must be written AFTER `touch`
    // above so the test observes a worker that has already
    // materialized the heap buffer; writing it earlier would let the
    // probe race against the allocation. Path format is centralized
    // in `ktstr::worker_ready` so worker and test cannot drift.
    let pid = std::process::id();
    let ready_path = worker_ready_marker_path(pid);
    if let Err(e) = std::fs::write(&ready_path, b"ready\n") {
        eprintln!(
            "jemalloc-alloc-worker: failed to write ready marker {ready_path}: {e}"
        );
        std::process::exit(4);
    }
    // The stdout line is a human-useful breadcrumb if the test
    // fails and the log is inspected; the test body does not parse
    // it (readiness is communicated via the marker file above).
    println!("jemalloc-alloc-worker ready pid={pid} bytes={bytes}");
    let _ = std::io::stdout().flush();

    if churn {
        // Thread-churn mode: in a tight loop, spawn a short-lived
        // helper thread that returns immediately, and join it. Each
        // iteration opens and closes a TID — the probe's external-pid
        // path iterates `/proc/<pid>/task`, issues PTRACE_SEIZE /
        // PTRACE_INTERRUPT per tid, and races thread exit: a tid that
        // dies between the readdir and the ptrace syscall returns
        // ESRCH, which the probe's PtraceSeize / PtraceInterrupt
        // error variants must surface as `ThreadResult::Err` entries
        // rather than panic or abort.
        //
        // No upper bound on iterations: the test body's
        // `PayloadHandle::kill` reaps the worker when probe runs are
        // done. The sleep below is nominally 100µs but the kernel
        // rounds short sleeps up to its clock granularity — under
        // `CONFIG_HZ=250`/`1000` with `hrtimers` that floor is ~1-4ms
        // and under `CONFIG_HZ=100` closer to 10ms, so the tid lives
        // for roughly a tick rather than the literal 100µs. The
        // argument passed to `sleep` matters less than the fact that
        // the tid is live long enough to appear on a readdir and
        // dead by the time PTRACE_SEIZE / PTRACE_INTERRUPT lands —
        // which is the race the probe must tolerate.
        loop {
            touch(&known);
            let handle = std::thread::spawn(|| {
                // Nominal 100µs; actual sleep floors to the kernel's
                // timer granularity (~1-10ms depending on HZ). Still
                // short enough that the helper-tid churn rate stays
                // well above the probe's per-tid attach latency, so
                // the ESRCH race the stress test exercises fires
                // within a handful of probe invocations.
                std::thread::sleep(Duration::from_micros(100));
            });
            let _ = handle.join();
        }
    } else {
        // Park forever, re-touching the allocation each tick so the
        // optimizer cannot move the free before the park. The test
        // body's `PayloadHandle::kill` delivers SIGKILL via `killpg`
        // on the child's process group, reaching us without further
        // cooperation.
        loop {
            touch(&known);
            std::thread::sleep(Duration::from_secs(3600));
        }
    }
}
