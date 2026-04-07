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

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

/// BPF_LOG_STATS — stats-only verifier output (processed insns, states,
/// stack depth, verification time).
const BPF_LOG_STATS: u32 = 4;
/// BPF_LOG_LEVEL1 — brief verifier trace.
const BPF_LOG_LEVEL1: u32 = 1;

struct ProgInfo {
    name: &'static str,
    insn_cnt: usize,
    log: String,
    loaded: bool,
}

/// Load BPF programs with verifier logging enabled, print stats, exit.
fn dump_verifier(verbose: bool) -> Result<()> {
    use libbpf_rs::AsRawLibbpf;

    let log_level = if verbose {
        BPF_LOG_STATS | BPF_LOG_LEVEL1
    } else {
        BPF_LOG_STATS
    };
    let log_buf_size: usize = if verbose { 16 * 1024 * 1024 } else { 64 * 1024 };

    let prog_names: &[&str] = &["stt_enqueue", "stt_dispatch", "stt_init", "stt_exit"];
    let mut results: Vec<ProgInfo> = Vec::new();

    for &target_name in prog_names {
        let mut open_object = MaybeUninit::uninit();
        let skel_builder = BpfSkelBuilder::default();
        let mut skel = scx_ops_open!(
            skel_builder,
            &mut open_object,
            stt_ops,
            None::<libbpf_rs::libbpf_sys::bpf_object_open_opts>
        )?;

        let mut log_buf = vec![0u8; log_buf_size];

        // Set log level and buffer on target, disable autoload on others.
        macro_rules! configure_prog {
            ($prog:expr, $name:expr) => {
                if $name == target_name {
                    $prog.set_autoload(true);
                    $prog.set_log_level(log_level);
                    let prog_ptr = $prog.as_libbpf_object().as_ptr();
                    unsafe {
                        libbpf_rs::libbpf_sys::bpf_program__set_log_buf(
                            prog_ptr,
                            log_buf.as_mut_ptr() as *mut i8,
                            log_buf.len() as u64,
                        );
                    }
                } else {
                    $prog.set_autoload(false);
                }
            };
        }

        let insn_cnt = skel.progs.stt_enqueue.insn_cnt();
        configure_prog!(skel.progs.stt_enqueue, "stt_enqueue");
        let dispatch_cnt = skel.progs.stt_dispatch.insn_cnt();
        configure_prog!(skel.progs.stt_dispatch, "stt_dispatch");
        let init_cnt = skel.progs.stt_init.insn_cnt();
        configure_prog!(skel.progs.stt_init, "stt_init");
        let exit_cnt = skel.progs.stt_exit.insn_cnt();
        configure_prog!(skel.progs.stt_exit, "stt_exit");

        let this_cnt = match target_name {
            "stt_enqueue" => insn_cnt,
            "stt_dispatch" => dispatch_cnt,
            "stt_init" => init_cnt,
            "stt_exit" => exit_cnt,
            _ => 0,
        };

        let load_ok = scx_ops_load!(skel, stt_ops, uei).is_ok();

        let log = {
            let nul_pos = log_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(log_buf.len());
            String::from_utf8_lossy(&log_buf[..nul_pos]).into_owned()
        };

        results.push(ProgInfo {
            name: target_name,
            insn_cnt: this_cnt,
            log,
            loaded: load_ok,
        });
    }

    // Emit structured output the host can parse.
    for p in &results {
        println!("STT_VERIFIER_PROG {} insn_cnt={}", p.name, p.insn_cnt);
        if !p.loaded {
            println!("STT_VERIFIER_LOG {} FAIL: verification failed", p.name);
        }
        for line in p.log.lines() {
            println!("STT_VERIFIER_LOG {} {}", p.name, line);
        }
    }
    println!("STT_VERIFIER_DONE");
    Ok(())
}

fn run(shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
    let stall_after = parse_stall_after();
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

    while !shutdown.load(Ordering::Relaxed) && !uei_exited!(&skel, uei) {
        thread::sleep(Duration::from_secs(1));
        if std::path::Path::new("/tmp/stt_stall").exists()
            && let Some(bss) = skel.maps.bss_data.as_mut()
            && bss.stall == 0
        {
            bss.stall = 1;
            eprintln!("stt-sched: stall enabled via /tmp/stt_stall");
        }
    }

    uei_report!(&skel, uei)
}

fn main() -> Result<()> {
    if has_flag("--dump-verifier") {
        let verbose = has_flag("--dump-verifier-verbose");
        return dump_verifier(verbose);
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    run(shutdown).map(|_| ())
}
