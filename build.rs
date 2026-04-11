// Generates vmlinux.h via bpftool from the kernel's BTF.
// Uses $KTSTR_KERNEL/vmlinux if KTSTR_KERNEL is set, otherwise
// falls back to /sys/kernel/btf/vmlinux (world-readable).

use std::env;
use std::path::PathBuf;
use std::process::Command;

use libbpf_cargo::SkeletonBuilder;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Resolve BTF source: $KTSTR_KERNEL/vmlinux or /sys/kernel/btf/vmlinux.
    let vmlinux_h = out_dir.join("vmlinux.h");
    if !vmlinux_h.exists() {
        let btf_source = if let Ok(kernel_dir) = env::var("KTSTR_KERNEL") {
            let p = PathBuf::from(&kernel_dir).join("vmlinux");
            if p.exists() {
                println!("cargo::warning=generating vmlinux.h from $KTSTR_KERNEL/vmlinux");
                p.to_string_lossy().into_owned()
            } else {
                panic!(
                    "KTSTR_KERNEL={kernel_dir} but {}/vmlinux not found. \
                     Build the kernel first.",
                    kernel_dir,
                );
            }
        } else {
            println!(
                "cargo::warning=KTSTR_KERNEL not set, generating vmlinux.h \
                 from /sys/kernel/btf/vmlinux"
            );
            "/sys/kernel/btf/vmlinux".to_string()
        };
        let output = Command::new("bpftool")
            .args(["btf", "dump", "file", &btf_source, "format", "c"])
            .output()
            .expect("bpftool required to generate vmlinux.h");
        assert!(
            output.status.success(),
            "bpftool btf dump failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::write(&vmlinux_h, &output.stdout).expect("write vmlinux.h");
    }
    println!("cargo:rerun-if-env-changed=KTSTR_KERNEL");

    let clang_args = [
        format!("-I{}", out_dir.display()),
        format!("-I{}", "src/bpf"),
    ];

    // Build the kprobe BPF skeleton.
    let skel_path = out_dir.join("probe_skel.rs");
    SkeletonBuilder::new()
        .source("src/bpf/probe.bpf.c")
        .obj(out_dir.join("probe.o"))
        .clang_args(clang_args.clone())
        .reference_obj(true)
        .build_and_generate(&skel_path)
        .expect("build probe BPF skeleton");

    // Build the fentry BPF skeleton (separate for independent loading).
    let fentry_skel_path = out_dir.join("fentry_probe_skel.rs");
    SkeletonBuilder::new()
        .source("src/bpf/fentry_probe.bpf.c")
        .obj(out_dir.join("fentry_probe.o"))
        .clang_args(clang_args)
        .reference_obj(true)
        .build_and_generate(&fentry_skel_path)
        .expect("build fentry probe BPF skeleton");

    println!("cargo::rerun-if-changed=src/bpf/probe.bpf.c");
    println!("cargo::rerun-if-changed=src/bpf/fentry_probe.bpf.c");
    println!("cargo::rerun-if-changed=src/bpf/intf.h");

    // Git info for output directory keying.
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        && output.status.success()
    {
        let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("cargo:rustc-env=KTSTR_GIT_HASH={hash}");
    }
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        && output.status.success()
    {
        let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("cargo:rustc-env=KTSTR_GIT_BRANCH={branch}");
    }
    println!("cargo:rerun-if-changed=.git/HEAD");
}
