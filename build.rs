// NOTE: This build script invokes `sudo bpftool btf dump` to generate
// vmlinux.h from the running kernel's BTF. Requires CAP_SYS_ADMIN.
// You will be prompted for your password on first build (results are
// cached in OUT_DIR for subsequent builds).

use std::env;
use std::path::PathBuf;
use std::process::Command;

use libbpf_cargo::SkeletonBuilder;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate vmlinux.h from the running kernel's BTF.
    let vmlinux_h = out_dir.join("vmlinux.h");
    if !vmlinux_h.exists() {
        println!(
            "cargo::warning=generating vmlinux.h — sudo bpftool requires CAP_SYS_ADMIN (you may be prompted for your password)"
        );
        let output = Command::new("sudo")
            .args([
                "bpftool",
                "btf",
                "dump",
                "file",
                "/sys/kernel/btf/vmlinux",
                "format",
                "c",
            ])
            .output()
            .expect("sudo bpftool required to generate vmlinux.h");
        assert!(
            output.status.success(),
            "bpftool btf dump failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        // Written by the build process (not sudo), so ownership is correct.
        std::fs::write(&vmlinux_h, &output.stdout).expect("write vmlinux.h");
    }

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
}
