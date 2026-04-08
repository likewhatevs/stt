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

/// Load BPF programs and emit structured results. When --verify-loop
/// or --fail-verify is set, scx_ops_load! fails and libbpf prints the
/// verifier's instruction-level traces to stderr automatically.
fn dump_verifier() -> Result<()> {
    let mut open_object = MaybeUninit::uninit();
    let skel_builder = BpfSkelBuilder::default();
    let mut skel = scx_ops_open!(
        skel_builder,
        &mut open_object,
        stt_ops,
        None::<libbpf_rs::libbpf_sys::bpf_object_open_opts>
    )?;

    if has_flag("--degrade")
        && let Some(rodata) = skel.maps.rodata_data.as_mut()
    {
        rodata.degrade = 1;
    }
    if has_flag("--verify-loop")
        && let Some(rodata) = skel.maps.rodata_data.as_mut()
    {
        rodata.verify_loop = 1;
    }
    if has_flag("--fail-verify")
        && let Some(rodata) = skel.maps.rodata_data.as_mut()
    {
        rodata.fail_verify = 1;
    }

    // Collect pre-load instruction counts.
    let insn_counts = [
        ("stt_enqueue", skel.progs.stt_enqueue.insn_cnt()),
        ("stt_dispatch", skel.progs.stt_dispatch.insn_cnt()),
        ("stt_init", skel.progs.stt_init.insn_cnt()),
        ("stt_exit", skel.progs.stt_exit.insn_cnt()),
    ];

    // Load normally. On failure, libbpf's default print callback emits
    // the verifier log (instruction traces) to stderr.
    let load_ok = scx_ops_load!(skel, stt_ops, uei).is_ok();

    // Emit structured output on stdout.
    for &(name, cnt) in &insn_counts {
        println!("STT_VERIFIER_PROG {} insn_cnt={}", name, cnt);
        if !load_ok {
            println!("STT_VERIFIER_LOG {} FAIL: verification failed", name);
        }
    }
    println!("STT_VERIFIER_DONE");
    Ok(())
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
        stt_ops,
        None::<libbpf_rs::libbpf_sys::bpf_object_open_opts>
    )?;

    if degrade && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.degrade = 1;
    }
    if fail_verify && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.fail_verify = 1;
    }
    if scattershot && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.scattershot = 1;
    }
    if slow && let Some(rodata) = skel.maps.rodata_data.as_mut() {
        rodata.slow = 1;
    }

    let mut skel = scx_ops_load!(skel, stt_ops, uei)?;
    let _link = scx_ops_attach!(skel, stt_ops)?;

    if degrade {
        eprintln!("stt-sched: degrade mode enabled");
    }
    if scattershot {
        eprintln!("stt-sched: scattershot mode enabled");
        if slow || degrade {
            eprintln!(
                "stt-sched: WARNING: --scattershot bypasses SHARED_DSQ; --slow/--degrade have no effect"
            );
        }
    }
    if slow {
        eprintln!("stt-sched: slow mode enabled");
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
                eprintln!("stt-sched: stall enabled after {delay_s}s");
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
                eprintln!("stt-sched: degrade enabled after {delay_s}s");
            }
        });
    }

    while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&skel, uei) {
        thread::sleep(Duration::from_secs(1));
        if let Some(bss) = skel.maps.bss_data.as_mut() {
            if std::path::Path::new("/tmp/stt_stall").exists() && bss.stall == 0 {
                bss.stall = 1;
                eprintln!("stt-sched: stall enabled via /tmp/stt_stall");
            }
            if std::path::Path::new("/tmp/stt_degrade").exists() && bss.degrade_rt == 0 {
                bss.degrade_rt = 1;
                eprintln!("stt-sched: degrade enabled via /tmp/stt_degrade");
            }
        }
    }

    uei_report!(&skel, uei)
}

fn main() -> Result<()> {
    if has_flag("--dump-verifier") {
        return dump_verifier();
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    run(shutdown).map(|_| ())
}
