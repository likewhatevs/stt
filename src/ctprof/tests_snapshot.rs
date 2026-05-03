//! Snapshot zstd round-trip, loader rejection, decompress-bomb guard.
//!
//! Co-located with `super::mod.rs`; one of the topic-grouped
//! split files that replace the monolithic `tests.rs`.

#![cfg(test)]

use super::*;
use crate::metric_types::{CategoricalString, CpuSet, MonotonicNs, OrdinalI32};


fn thread(pcomm: &str, comm: &str, run_time_ns: u64) -> ThreadState {
    ThreadState {
        tid: 1,
        tgid: 1,
        pcomm: pcomm.into(),
        comm: comm.into(),
        cgroup: "/".into(),
        start_time_clock_ticks: 0,
        policy: CategoricalString("SCHED_OTHER".into()),
        nice: OrdinalI32(0),
        cpu_affinity: CpuSet(vec![0, 1]),
        run_time_ns: MonotonicNs(run_time_ns),
        ..ThreadState::default()
    }
}

#[test]
fn snapshot_roundtrip_through_zstd_json() {
    let snap = CtprofSnapshot {
        captured_at_unix_ns: 42,
        host: None,
        threads: vec![
            thread("proc_a", "worker_0", 1_000_000),
            thread("proc_a", "worker_1", 2_000_000),
        ],
        cgroup_stats: BTreeMap::from([("/".into(), {
            let mut cs = CgroupStats::default();
            cs.cpu.usage_usec = 500;
            cs.memory.current = 1 << 20;
            cs
        })]),
        probe_summary: None,
        parse_summary: None,
        taskstats_summary: None,
        psi: Psi::default(),
        sched_ext: None,
    };
    let tmp = tempfile::NamedTempFile::new().unwrap();
    snap.write(tmp.path()).unwrap();
    let back = CtprofSnapshot::load(tmp.path()).unwrap();
    assert_eq!(back.captured_at_unix_ns, 42);
    assert_eq!(back.threads.len(), 2);
    assert_eq!(
        back.threads[1].run_time_ns,
        crate::metric_types::MonotonicNs(2_000_000),
    );
    assert_eq!(back.cgroup_stats["/"].cpu.usage_usec, 500);
}

#[test]
fn load_rejects_non_zstd_payload() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"{\"not\": \"zstd\"}").unwrap();
    let err = CtprofSnapshot::load(tmp.path()).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("zstd"),
        "expected zstd error in context chain, got: {msg}",
    );
}

#[test]
fn load_rejects_zstd_of_garbage_json() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let compressed = zstd::encode_all(&b"not json"[..], 3).unwrap();
    std::fs::write(tmp.path(), compressed).unwrap();
    let err = CtprofSnapshot::load(tmp.path()).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("parse ctprof"),
        "expected parse error in context chain, got: {msg}",
    );
}

/// Decompression-bomb guard: a zstd payload that decompresses
/// past the configured cap surfaces an error tagged with
/// "decompression-bomb guard" — the loader must not allocate
/// past the ceiling. Test uses a small synthetic payload (8
/// KiB of zeros, which compresses to a tiny blob but
/// decompresses to 8192 bytes) against a 1024-byte cap so
/// the test runs in microseconds rather than allocating a
/// production-sized buffer.
#[test]
fn decompress_capped_rejects_decompression_bomb() {
    let payload = vec![0u8; 8192];
    let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
    let cap: u64 = 1024;
    let err = super::decompress_capped(&compressed, cap).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("decompression-bomb guard"),
        "expected decompression-bomb guard error, got: {msg}",
    );
}

/// Boundary case: a payload whose decompressed length is
/// exactly `cap` bytes is accepted (the cap is inclusive).
/// Pins the `>` (not `>=`) discriminator at the cap boundary
/// so a future refactor that flips the comparison surfaces
/// here rather than turning a legal snapshot into a
/// false-positive bomb rejection.
#[test]
fn decompress_capped_accepts_payload_at_cap_boundary() {
    let payload = b"hello world".to_vec();
    let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
    let out = super::decompress_capped(&compressed, payload.len() as u64).unwrap();
    assert_eq!(
        out, payload,
        "payload exactly at the cap must round-trip — \
         cap is inclusive (`>` not `>=`)",
    );
}
