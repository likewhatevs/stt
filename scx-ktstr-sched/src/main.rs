// SPDX-License-Identifier: GPL-2.0-only
mod bpf_skel;
pub use bpf_skel::*;

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

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
        eprintln!("ktstr-sched: degrade mode enabled");
    }
    if scattershot {
        eprintln!("ktstr-sched: scattershot mode enabled");
        if slow || degrade {
            eprintln!(
                "ktstr-sched: WARNING: --scattershot bypasses SHARED_DSQ; --slow/--degrade have no effect"
            );
        }
    }
    if slow {
        eprintln!("ktstr-sched: slow mode enabled");
    }

    if let Some(delay_s) = stall_after {
        let skel_ptr = &mut skel as *mut _ as usize;
        let shutdown_clone = shutdown.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(delay_s));
            if !shutdown_clone.load(Ordering::Relaxed) {
                // SAFETY: skel outlives this thread (main thread joins via uei_exited loop).
                // The BPF .bss map is mmap'd; writing to it is a single atomic store
                // visible to the BPF program on the next dispatch call.
                let skel = unsafe { &mut *(skel_ptr as *mut BpfSkel<'_>) };
                if let Some(bss) = skel.maps.bss_data.as_mut() {
                    bss.stall = 1;
                }
                eprintln!("ktstr-sched: stall enabled after {delay_s}s");
            }
        });
    }

    if let Some(delay_s) = degrade_after {
        let skel_ptr = &mut skel as *mut _ as usize;
        let shutdown_clone = shutdown.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(delay_s));
            if !shutdown_clone.load(Ordering::Relaxed) {
                let skel = unsafe { &mut *(skel_ptr as *mut BpfSkel<'_>) };
                if let Some(bss) = skel.maps.bss_data.as_mut() {
                    bss.degrade_rt = 1;
                }
                eprintln!("ktstr-sched: degrade enabled after {delay_s}s");
            }
        });
    }

    while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&skel, uei) {
        thread::sleep(Duration::from_secs(1));
        if let Some(bss) = skel.maps.bss_data.as_mut() {
            if std::path::Path::new("/tmp/ktstr_stall").exists() && bss.stall == 0 {
                bss.stall = 1;
                eprintln!("ktstr-sched: stall enabled via /tmp/ktstr_stall");
            }
            if std::path::Path::new("/tmp/ktstr_degrade").exists() && bss.degrade_rt == 0 {
                bss.degrade_rt = 1;
                eprintln!("ktstr-sched: degrade enabled via /tmp/ktstr_degrade");
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
