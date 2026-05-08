// SPDX-License-Identifier: GPL-2.0-only
mod bpf_skel;
mod stats;
pub use bpf_skel::*;

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;
use scx_stats::prelude::StatsServer;
use scx_utils::UserExitInfo;
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::uei_exited;
use scx_utils::uei_report;

use stats::KtstrStats;

fn parse_delay_flag(flag: &str) -> Option<u64> {
    let args: Vec<String> = std::env::args().collect();
    for (i, a) in args.iter().enumerate() {
        if a == flag {
            return args.get(i + 1).and_then(|v| v.parse().ok());
        }
        let prefix = format!("{flag}=");
        if let Some(v) = a.strip_prefix(&prefix) {
            return v.parse().ok();
        }
    }
    None
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

/// Main-loop poll cadence. Delay flags (`--stall-after`,
/// `--degrade-after`) are checked inline at each tick, so a trigger
/// fires at the NEXT poll after `elapsed >= delay`. At 100ms the
/// worst-case jitter between "delay elapsed" and "trigger fires"
/// is bounded by this constant, well within the resolution of even
/// the smallest non-zero delay the CLI accepts (1s) — a requested
/// 1s stall fires between 1.0s and 1.1s of wall clock, not 1.0s
/// to 2.0s as under the prior 1s cadence. Shorter still would add
/// wakeup cost with no observable benefit; 100ms is the smallest
/// value that reliably keeps a test's timing annotation meaningful.
const POLL_CADENCE: Duration = Duration::from_millis(100);

fn run(shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
    let stall_after = parse_delay_flag("--stall-after");
    let degrade_after = parse_delay_flag("--degrade-after");
    let degrade = has_flag("--degrade");
    let fail_verify = has_flag("--fail-verify");
    let scattershot = has_flag("--scattershot");
    let slow = has_flag("--slow");

    let mut open_object = MaybeUninit::uninit();
    let skel_builder = BpfSkelBuilder::default();
    let mut skel = scx_ops_open!(
        skel_builder,
        &mut open_object,
        ktstr_ops,
        None::<libbpf_rs::libbpf_sys::bpf_object_open_opts>
    )?;

    if degrade && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.degrade = 1;
    }
    if fail_verify && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.fail_verify = 1;
    }
    if has_flag("--verify-loop")
        && let Some(rodata) = skel.maps.rodata_data.as_mut()
    {
        rodata.verify_loop = 1;
    }
    if scattershot && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.scattershot = 1;
    }
    if slow && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.slow = 1;
    }

    let mut skel = scx_ops_load!(skel, ktstr_ops, uei)?;
    let _link = scx_ops_attach!(skel, ktstr_ops)?;

    // Bind the scx_stats Unix-socket server at the default path
    // (`/var/run/scx/root/stats`). The server's listener thread
    // accepts connections and dispatches "stats" / "stats_meta"
    // verbs through the channel pair we drain inline below; the
    // main run loop owns the BPF skeleton and is the only thread
    // that reads bss_data, so reads stay serialised against the
    // userspace mutator paths above. ktstr's host-side
    // `SchedStatsClient` reaches this socket through the in-guest
    // stats relay that bridges `/dev/vport0p2` → `/var/run/scx/root/stats`.
    let stats_server: StatsServer<(), KtstrStats> =
        StatsServer::new(stats::server_data()).launch()?;
    let (res_ch, req_ch) = stats_server.channels();

    if degrade {
        eprintln!("scx-ktstr: degrade mode enabled");
    }
    if scattershot {
        eprintln!("scx-ktstr: scattershot mode enabled");
        if slow || degrade {
            eprintln!(
                "scx-ktstr: WARNING: --scattershot bypasses SHARED_DSQ; --slow/--degrade have no effect"
            );
        }
    }
    if slow {
        eprintln!("scx-ktstr: slow mode enabled");
    }

    // `--stall-after` / `--degrade-after` triggers are checked inline
    // in the main poll loop below. Previously they were driven by two
    // fire-and-forget (JoinHandle dropped unused) `thread::spawn`
    // closures that captured `&mut skel` as `usize` and re-cast it
    // under `unsafe` after the sleep; that is UAF-prone — the threads
    // are never joined, and the `BpfSkel<'_>` stack local (plus its
    // owned libbpf `Object`, `BpfLink`, and mmap'd `bss`/`rodata`
    // regions) drops when `run` returns on shutdown or `uei_exited`.
    // Folding the triggers into the main poll loop eliminates the
    // aliasing, removes the `unsafe` cast, and bounds the wake
    // latency to the same granularity the file-triggered path
    // already has. Delay precision is `Duration::from_secs(delay_s)`
    // bounded by [`POLL_CADENCE`] (100ms) — adequate for every
    // non-zero `--stall-after` / `--degrade-after` value the CLI
    // surface accepts.
    //
    // Gating change vs the old timer-thread design: triggers become
    // inert once `uei_exited!(&skel, uei)` fires. A dead scheduler
    // no longer receives stall/degrade signals, because the poll
    // loop exits before the next tick. The previous design would
    // still flip the bss bytes on a dead skel (also visible to no
    // one — the scheduler was already unloaded). No observable
    // regression in practice.
    //
    // Zero-delay handling: `stall_after=0` / `degrade_after=0` would
    // otherwise wait for the first `thread::sleep(POLL_CADENCE)`
    // before the elapsed check fired. Fire those immediately before
    // entering the loop so the semantics match the old "spawn +
    // sleep(0)" path (which fired essentially instantly).
    if let Some(bss) = skel.maps.bss_data.as_mut() {
        if stall_after == Some(0) && bss.stall == 0 {
            bss.stall = 1;
            eprintln!("scx-ktstr: stall enabled after 0s");
        }
        if degrade_after == Some(0) && bss.degrade_rt == 0 {
            bss.degrade_rt = 1;
            eprintln!("scx-ktstr: degrade enabled after 0s");
        }
    }
    let start = Instant::now();
    while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&skel, uei) {
        thread::sleep(POLL_CADENCE);
        let elapsed = start.elapsed();
        if let Some(bss) = skel.maps.bss_data.as_mut() {
            if let Some(delay_s) = stall_after
                && bss.stall == 0
                && elapsed >= Duration::from_secs(delay_s)
            {
                bss.stall = 1;
                eprintln!("scx-ktstr: stall enabled after {delay_s}s");
            }
            if let Some(delay_s) = degrade_after
                && bss.degrade_rt == 0
                && elapsed >= Duration::from_secs(delay_s)
            {
                bss.degrade_rt = 1;
                eprintln!("scx-ktstr: degrade enabled after {delay_s}s");
            }
            if std::path::Path::new("/tmp/ktstr_stall").exists() && bss.stall == 0 {
                bss.stall = 1;
                eprintln!("scx-ktstr: stall enabled via /tmp/ktstr_stall");
            }
            if std::path::Path::new("/tmp/ktstr_degrade").exists() && bss.degrade_rt == 0 {
                bss.degrade_rt = 1;
                eprintln!("scx-ktstr: degrade enabled via /tmp/ktstr_degrade");
            }
        }
        // Drain every queued stats request this tick. The scx_stats
        // server thread sends `()` over `req_ch` for each "stats"
        // verb its accept loop receives; we reply with one fresh
        // BPF .bss read per request so concurrent requesters each
        // get their own snapshot. `try_recv` is non-blocking — the
        // main loop's pacing is owned by the existing
        // `thread::sleep(POLL_CADENCE)` above. Using
        // `recv_timeout` here would either redundantly block on
        // top of the cadence or starve the trigger logic. The
        // BPF .bss read is naturally aligned 64-bit per field and
        // the BPF side uses `__sync_fetch_and_add`, so a
        // concurrent in-flight increment is observed as a single
        // atomic value rather than a torn read.
        loop {
            match req_ch.try_recv() {
                Ok(()) => {
                    let snapshot = current_stats(&skel);
                    if let Err(e) = res_ch.send(snapshot) {
                        eprintln!("scx-ktstr: stats response send failed ({e})");
                        break;
                    }
                }
                Err(crossbeam::channel::TryRecvError::Empty) => break,
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    eprintln!("scx-ktstr: stats request channel disconnected");
                    break;
                }
            }
        }
    }

    uei_report!(&skel, uei)
}

/// Snapshot the BPF .bss counters into a `KtstrStats`. None-bss
/// (skel still loading or maps unmapped) yields an all-zero
/// snapshot — the response is still valid JSON and the host can
/// distinguish "scheduler running but no traffic" from "scheduler
/// not running" via the wire-level errno (the latter never reaches
/// this function because the run loop wouldn't be running).
fn current_stats(skel: &BpfSkel<'_>) -> KtstrStats {
    skel.maps
        .bss_data
        .as_ref()
        .map(|bss| KtstrStats {
            nr_dispatched: bss.nr_dispatched,
            nr_enqueued: bss.nr_enqueued,
            nr_select_cpu: bss.nr_select_cpu,
        })
        .unwrap_or_default()
}

fn main() -> Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    run(shutdown).map(|_| ())
}
