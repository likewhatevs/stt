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

fn parse_stall_after() -> Option<u64> {
    let args: Vec<String> = std::env::args().collect();
    for (i, a) in args.iter().enumerate() {
        if a == "--stall-after" {
            return args.get(i + 1).and_then(|v| v.parse().ok());
        }
        if let Some(v) = a.strip_prefix("--stall-after=") {
            return v.parse().ok();
        }
    }
    None
}

fn run(shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
    let stall_after = parse_stall_after();

    let mut open_object = MaybeUninit::uninit();
    let skel_builder = BpfSkelBuilder::default();
    let mut skel = scx_ops_open!(
        skel_builder,
        &mut open_object,
        stt_ops,
        None::<libbpf_rs::libbpf_sys::bpf_object_open_opts>
    )?;
    let mut skel = scx_ops_load!(skel, stt_ops, uei)?;
    let _link = scx_ops_attach!(skel, stt_ops)?;

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
                eprintln!("stt-sched: stall enabled after {delay_s}s");
            }
        });
    }

    while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&skel, uei) {
        thread::sleep(Duration::from_secs(1));
        if std::path::Path::new("/tmp/stt_stall").exists() {
            if let Some(bss) = skel.maps.bss_data.as_mut() {
                if bss.stall == 0 {
                    bss.stall = 1;
                    eprintln!("stt-sched: stall enabled via /tmp/stt_stall");
                }
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
