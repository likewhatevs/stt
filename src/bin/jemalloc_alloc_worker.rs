//! Tiny jemalloc-linked target for the probe's cross-process
//! closed-loop test (see `tests/jemalloc_probe_tests.rs`).
//!
//! Reads a byte count from argv[1], allocates that many bytes on
//! the main thread, and parks in a `std::thread::sleep` loop
//! until SIGKILL (delivered by the test body via
//! `PayloadHandle::kill`). Single-threaded on purpose — a
//! single-thread process has `tid == pid` on Linux, so the test
//! body can derive the worker's TID from the worker's pid and
//! match on `threads[N].tid` in the probe's JSON output without
//! needing a separate TID handshake over stdout / a side file.
//! The worker enforces its own single-threadedness at startup
//! by enumerating `/proc/self/task` and bailing with a non-zero
//! exit if more than one TID is present — a silent extra thread
//! (e.g. jemalloc background bg_thd or a tokio runtime pulled in
//! by a future dep) would break the `tid == pid` identity and
//! mis-align the test's probe-result match.
//!
//! Links `tikv_jemallocator` as the global allocator so
//! jemalloc's `tsd_tls` symbol is present in the binary and the
//! probe's DWARF-backed offset resolution finds a valid
//! `tsd_s.thread_allocated` to read.
//!
//! Minimum useful allocation size: ≥ 16 KiB so the allocation
//! lands on jemalloc's huge-size / large-size path and
//! unconditionally updates `thread_allocated`. Smaller values
//! may route through tcache with deferred counter updates and
//! race the probe. The test body passes 16 MiB which is well
//! above the threshold.
//!
//! Keep this binary minimal — the initramfs size cost is
//! proportional to its text + PT_LOAD, not its allocation size,
//! but the probe binary and this worker travel together in the
//! guest initramfs.

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::io::Write;
use std::time::Duration;

fn main() {
    let bytes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("usage: jemalloc-alloc-worker <BYTES>");

    // Fail fast if the process is not single-threaded. `/proc/self/task`
    // lists one directory entry per TID; more than one means an
    // unexpected helper thread is running and the `tid == pid`
    // invariant the test body relies on is already broken.
    match std::fs::read_dir("/proc/self/task") {
        Ok(iter) => {
            let n = iter.filter(|e| e.is_ok()).count();
            if n != 1 {
                eprintln!(
                    "jemalloc-alloc-worker: /proc/self/task has {n} entries, expected 1; \
                     extra threads break the tid==pid identity"
                );
                std::process::exit(2);
            }
        }
        Err(e) => {
            eprintln!("jemalloc-alloc-worker: read_dir(/proc/self/task) failed: {e}");
            std::process::exit(2);
        }
    }

    // Allocate + hold under black_box so the optimizer cannot
    // elide the vector. The allocation runs on the main thread,
    // which is also the thread jemalloc updates thread_allocated
    // on.
    let known: Vec<u8> = vec![0u8; bytes];
    std::hint::black_box(&known);

    // Flush stdout with a single readiness line. The test body
    // doesn't parse this (it reads the worker pid from
    // `PayloadHandle::pid()` and derives tid from a
    // single-threaded-process identity), but the line is a
    // human-useful breadcrumb if the test fails and the log is
    // inspected.
    let pid = std::process::id();
    println!("jemalloc-alloc-worker ready pid={pid} bytes={bytes}");
    let _ = std::io::stdout().flush();

    // Park forever, re-touching the allocation each tick so the
    // optimizer cannot move the free before the park. The test
    // body's `PayloadHandle::kill` delivers SIGKILL via `killpg`
    // on the child's process group, reaching us without further
    // cooperation.
    loop {
        std::hint::black_box(&known);
        std::thread::sleep(Duration::from_secs(3600));
    }
}
