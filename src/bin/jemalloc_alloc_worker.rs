//! Tiny jemalloc-linked target for the probe's cross-process
//! closed-loop test (see `tests/jemalloc_probe_tests.rs`).
//!
//! # Default mode
//!
//! Reads a byte count from argv[1], allocates that many bytes on
//! the main thread, writes a pid-scoped ready marker via
//! [`ktstr::worker_ready::worker_ready_marker_path`], and parks in a
//! `std::thread::sleep` loop until SIGKILL (delivered by the test
//! body via `PayloadHandle::kill`). Single-threaded on purpose — a
//! single-thread process has `tid == pid` on Linux, so the test body
//! can derive the worker's TID from the worker's pid and match on
//! `threads[N].tid` in the probe's JSON output without a separate
//! TID handshake. The worker enforces its own single-threadedness at
//! startup by enumerating `/proc/self/task` and bailing with a
//! non-zero exit if more than one TID is present — a silent extra
//! thread (e.g. jemalloc background bg_thd or a tokio runtime pulled
//! in by a future dep) would break the `tid == pid` identity.
//!
//! # `--churn` mode
//!
//! With `--churn` anywhere on the command line, the `tid == pid`
//! single-thread invariant is RELAXED: the `/proc/self/task`
//! self-check is skipped, and after the main-thread allocation +
//! ready marker the worker enters a tight spawn+join loop to
//! maximize the number of thread lifetimes the probe races against.
//! Used by the ESRCH stress test in
//! `tests/jemalloc_probe_tests.rs::jemalloc_probe_survives_thread_churn`,
//! which asserts the probe's `PtraceSeize` / `PtraceInterrupt`
//! error arms surface as `ThreadResult::Err` JSON entries rather
//! than crashing.
//!
//! # Links
//!
//! Links `tikv_jemallocator` as the global allocator so jemalloc's
//! `tsd_tls` symbol is present in the binary and the probe's
//! DWARF-backed offset resolution finds a valid
//! `tsd_s.thread_allocated` to read.
//!
//! Minimum useful allocation size: ≥ 16 KiB so the allocation lands
//! on jemalloc's huge-size / large-size path and unconditionally
//! updates `thread_allocated`. Smaller values may route through
//! tcache with deferred counter updates and race the probe. The test
//! body passes 16 MiB which is well above the threshold.
//!
//! # Exit codes
//!
//! - `0`: normal exit path does not exist — default mode parks
//!   forever and churn mode loops forever; process termination is
//!   always by external signal (typically SIGKILL from the test
//!   body's `PayloadHandle::kill`).
//! - `2`: `bytes == 0` — a zero allocation would indexing-panic on
//!   `v[0]` in the `touch` helper. The test always passes a positive
//!   size, so this indicates a caller-side bug.
//! - `3`: `/proc/self/task` self-check saw more than one TID (or
//!   could not be read) in default mode — the `tid == pid` identity
//!   the test relies on is broken before allocation even begins.
//! - `4`: pid-scoped ready-marker write failed. A worker that cannot
//!   signal readiness leaves the test hanging on its poll deadline;
//!   exiting loudly here is strictly better.
//! - `5`: argument parse failure — missing positional `<BYTES>` or a
//!   non-decimal token that did not parse as `usize`. Replaces the
//!   implicit `expect()` panic (which would have exited 101) so every
//!   caller-visible exit path lives in this table.
//!
//! Keep this binary minimal — the initramfs size cost is
//! proportional to its text + PT_LOAD, not its allocation size, but
//! the probe binary and this worker travel together in the guest
//! initramfs.
//!
//! # Argument parsing
//!
//! Argv is deliberately hand-rolled: one optional flag (`--churn`)
//! and one positional (`<BYTES>`). Pulling in `clap` or any other
//! CLI crate adds a dependency, build time, and initramfs size for
//! a surface that fits in five lines of std code. If a third argv
//! shape lands (for example `--mode=churn|park --bytes=N`), switch
//! to clap-derive at that point — the tipping point is about
//! explicit help text, subcommand-like modes, or value validation.

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
        // done. Tiny sleep inside the helper so the tid is live for
        // ~100µs, maximizing the fraction of probe attempts that see
        // a live tid on readdir and a dead tid on ptrace.
        loop {
            touch(&known);
            let handle = std::thread::spawn(|| {
                // Micro-nap — keeps the tid alive for ~100µs so the
                // probe's readdir(/proc/pid/task) has a non-trivial
                // chance of listing it before PTRACE_SEIZE races
                // thread exit.
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
