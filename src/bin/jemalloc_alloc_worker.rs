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

    // A zero-length allocation would indexing-panic on `known[0]`
    // in the black_box triple below (both post-alloc and park-loop
    // sites dereference index 0 to force a heap read). Reject here
    // with a diagnostic rather than letting the panic surface as a
    // test timeout: the test is always paired with a positive size
    // (see KNOWN_BYTES in tests/jemalloc_probe_tests.rs), so a zero
    // arg indicates a caller-side bug worth surfacing loudly.
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
    // on. Using `black_box(&known)` on a reference is not strong
    // enough — LLVM can in principle observe that the `Vec`'s
    // heap buffer is never read through the pointer and fold the
    // zero-fill away. Force an actual heap read (through `[0]`)
    // and black-box the raw pointer + length so the allocation is
    // provably materialized and jemalloc sees the accounting.
    let known: Vec<u8> = vec![0u8; bytes];
    let _ = std::hint::black_box(known[0]);
    std::hint::black_box(known.as_ptr());
    std::hint::black_box(known.len());

    // Signal readiness via a pid-scoped marker file. The test body
    // polls `/tmp/ktstr-worker-ready-{pid}` for existence — cheaper
    // and tighter than a fixed 500ms sleep, and safe because each
    // `#[ktstr_test]` boots a fresh VM with a clean /tmp (so no
    // stale file from a prior run can race the poll). The file must
    // be written AFTER the allocation + black_box triple above so
    // the test observes a worker that has already materialized the
    // heap buffer; writing it earlier would let the probe race
    // against the allocation. A write failure is fatal — a worker
    // that cannot signal readiness leaves the test hanging on the
    // poll timeout, which is strictly worse than a clean exit code.
    let pid = std::process::id();
    let ready_path = format!("/tmp/ktstr-worker-ready-{pid}");
    if let Err(e) = std::fs::write(&ready_path, b"ready\n") {
        eprintln!(
            "jemalloc-alloc-worker: failed to write ready marker {ready_path}: {e}"
        );
        std::process::exit(2);
    }
    // The stdout line is a human-useful breadcrumb if the test
    // fails and the log is inspected; the test body does not parse
    // it (readiness is communicated via the marker file above).
    println!("jemalloc-alloc-worker ready pid={pid} bytes={bytes}");
    let _ = std::io::stdout().flush();

    // Park forever, re-touching the allocation each tick so the
    // optimizer cannot move the free before the park. The test
    // body's `PayloadHandle::kill` delivers SIGKILL via `killpg`
    // on the child's process group, reaching us without further
    // cooperation. Mirrors the strong black_box pattern above:
    // a heap read through `[0]` plus a pointer/len materialize.
    loop {
        let _ = std::hint::black_box(known[0]);
        std::hint::black_box(known.as_ptr());
        std::hint::black_box(known.len());
        std::thread::sleep(Duration::from_secs(3600));
    }
}
