//! Spawn-pipeline tests — futex group.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::types::*;
use super::super::worker::*;
use super::testing::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[test]
fn needs_shared_mem_non_futex() {
    assert!(!WorkType::SpinWait.needs_shared_mem());
    assert!(!WorkType::pipe_io(100).needs_shared_mem());
    assert!(!WorkType::cache_pipe(32, 100).needs_shared_mem());
    assert!(!WorkType::cache_pressure(32, 64).needs_shared_mem());
}
#[test]
fn spawn_futex_ping_pong_produces_work() {
    let reports = spawn_and_collect_after(WorkType::FutexPingPong { spin_iters: 1024 }, 2, 500);
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.work_units > 0,
            "FutexPingPong worker {} did no work",
            r.tid
        );
    }
}
