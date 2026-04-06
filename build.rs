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

    // Build the probe BPF skeleton.
    let skel_path = out_dir.join("probe_skel.rs");
    SkeletonBuilder::new()
        .source("src/bpf/probe.bpf.c")
        .clang_args(clang_args)
        .build_and_generate(&skel_path)
        .expect("build probe BPF skeleton");

    println!("cargo::rerun-if-changed=src/bpf/probe.bpf.c");
    println!("cargo::rerun-if-changed=src/bpf/intf.h");

    // Build a statically-linked busybox from source for the VMM initramfs.
    // Cloned from GitHub mirror, built with make -j$(nproc).
    // Cached in OUT_DIR — survives incremental builds, rebuilt once per
    // toolchain/target-dir combination.
    let busybox_out = out_dir.join("busybox");
    if !busybox_out.exists() {
        let busybox_src = out_dir.join("busybox-src");
        // Clean up partial clones from interrupted builds
        if busybox_src.exists() && !busybox_src.join(".git/HEAD").exists() {
            std::fs::remove_dir_all(&busybox_src).ok();
        }
        if !busybox_src.exists() {
            let status = Command::new("git")
                .args([
                    "clone",
                    "--depth=1",
                    "--single-branch",
                    "https://github.com/mirror/busybox.git",
                    busybox_src.to_str().unwrap(),
                ])
                .status()
                .expect("git clone busybox");
            assert!(status.success(), "failed to clone busybox from github");
        }
        let status = Command::new("make")
            .arg("defconfig")
            .current_dir(&busybox_src)
            .status()
            .expect("make defconfig");
        assert!(status.success(), "busybox defconfig failed");
        let config_path = busybox_src.join(".config");
        let config = std::fs::read_to_string(&config_path).expect("read .config");
        let config = config
            .replace("# CONFIG_STATIC is not set", "CONFIG_STATIC=y")
            .replace("CONFIG_TC=y", "# CONFIG_TC is not set");
        std::fs::write(&config_path, config).expect("write .config");
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or("4".into());
        let status = Command::new("make")
            .args(["-j", &nproc])
            .current_dir(&busybox_src)
            .status()
            .expect("make busybox");
        assert!(status.success(), "busybox build failed");
        std::fs::copy(busybox_src.join("busybox"), &busybox_out).expect("copy built busybox");
    }
    println!("cargo::rerun-if-changed={}", busybox_out.display());
}
