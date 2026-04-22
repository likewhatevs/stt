// SPDX-License-Identifier: GPL-2.0-only
mod bpf_skel;
pub use bpf_skel::*;

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;
use scx_utils::UserExitInfo;
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::uei_exited;
use scx_utils::uei_report;

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

fn run(shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
    let stall_after = parse_delay_flag("--stall-after");
    let degrade_after = parse_delay_flag("--degrade-after");
    // The main poll cadence is `thread::sleep(Duration::from_secs(1))`
    // (see the loop below). Delay flags are checked inline at each tick,
    // so the trigger fires at the NEXT poll after `elapsed >= delay` —
    // i.e. the actual trigger can land up to 1s after the requested
    // delay. Values of 0 fire immediately before the loop (see below)
    // and are unaffected. Warn the operator on `1..=4` so short-delay
    // scenarios do not attribute a late trigger to scheduler behavior.
    for (name, maybe) in [
        ("--stall-after", stall_after),
        ("--degrade-after", degrade_after),
    ] {
        if let Some(delay_s) = maybe
            && (1..5).contains(&delay_s)
        {
            eprintln!(
                "scx-ktstr: WARNING: {name}={delay_s}s can exhibit up to 1s \
                 poll-granularity jitter under load. Delays of 5s or greater \
                 keep the jitter well within the intended delay.",
            );
        }
    }
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
    // Folding the triggers into the existing 1-second poll cadence
    // eliminates the aliasing, removes the `unsafe` cast, and bounds
    // the wake latency to the same granularity the file-triggered
    // path already has. Delay precision is `Duration::from_secs(delay_s)`
    // with 1 s poll granularity — adequate for the test durations
    // these flags are used with (tens of seconds and up).
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
    // otherwise wait for the first `thread::sleep(1s)` before the
    // elapsed check fired. Fire those immediately before entering
    // the loop so the semantics match the old "spawn + sleep(0)"
    // path (which fired essentially instantly).
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
        thread::sleep(Duration::from_secs(1));
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
    }

    uei_report!(&skel, uei)
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
