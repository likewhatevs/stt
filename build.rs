// Generates vmlinux.h from kernel BTF using libbpf's btf_dump API.
// Uses the shared kernel resolver (src/kernel_path.rs) to find the
// BTF source. See resolve_btf() for the full search order.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use libbpf_cargo::SkeletonBuilder;

include!("src/kernel_path.rs");

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Cache invalidation: always track env var and resolved kernel.
    println!("cargo:rerun-if-env-changed=KTSTR_KERNEL");
    println!("cargo:rerun-if-changed=src/kernel_path.rs");
    println!("cargo:rerun-if-changed=src/bpf/vmlinux_gen.c");
    let ktstr_kernel = env::var("KTSTR_KERNEL").ok();
    let kernel = resolve_kernel(ktstr_kernel.as_deref());
    if let Some(ref path) = kernel {
        println!("cargo:rerun-if-changed={}", path.join("vmlinux").display());
    }

    // Generate vmlinux.h from kernel BTF.
    let vmlinux_h = out_dir.join("vmlinux.h");
    if !vmlinux_h.exists() {
        let btf_source = resolve_btf(ktstr_kernel.as_deref()).unwrap_or_else(|| {
            panic!(
                "no BTF source found. Set KTSTR_KERNEL to a kernel build \
                 directory, or ensure /sys/kernel/btf/vmlinux exists."
            );
        });
        println!(
            "cargo::warning=generating vmlinux.h from {}",
            btf_source.display()
        );

        // libbpf-sys (links = "bpf") emits installed headers at
        // DEP_BPF_INCLUDE with bpf/ prefix (bpf/btf.h, bpf/libbpf.h).
        let libbpf_include =
            PathBuf::from(env::var("DEP_BPF_INCLUDE").expect("DEP_BPF_INCLUDE not set"));

        // Compile the C vmlinux generator + driver into a standalone binary.
        let vmlinux_gen_bin = out_dir.join("vmlinux_gen");
        let driver_src = out_dir.join("vmlinux_gen_main.c");
        std::fs::write(
            &driver_src,
            format!(
                r#"
extern int generate_vmlinux_h(const char *, const char *);
int main(void) {{
    return generate_vmlinux_h("{btf}", "{out}") == 0 ? 0 : 1;
}}
"#,
                btf = btf_source.display(),
                out = vmlinux_h.display(),
            ),
        )
        .expect("write driver source");

        // libbpf-sys with vendored feature installs static libraries
        // (libbpf.a, libelf.a, libz.a) in the parent of DEP_BPF_INCLUDE.
        let libbpf_lib_dir = libbpf_include.parent().unwrap();

        let compiler = cc::Build::new().get_compiler();
        let status = Command::new(compiler.path())
            .args([
                "src/bpf/vmlinux_gen.c",
                driver_src.to_str().unwrap(),
                "-o",
                vmlinux_gen_bin.to_str().unwrap(),
                &format!("-I{}", libbpf_include.display()),
                &format!("-L{}", libbpf_lib_dir.display()),
                "-lbpf",
                "-lelf",
                "-lz",
            ])
            .status()
            .expect("compile vmlinux_gen");
        assert!(status.success(), "failed to compile vmlinux_gen");

        let status = Command::new(&vmlinux_gen_bin)
            .status()
            .expect("run vmlinux_gen");
        assert!(
            status.success(),
            "vmlinux_gen failed — check BTF source: {}",
            btf_source.display()
        );
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

    // Git info for output directory keying.
    if let Ok(repo) = gix::discover(".") {
        if let Ok(id) = repo.head_id() {
            let full = id.to_string();
            let short = &full[..full.len().min(7)];
            println!("cargo:rustc-env=KTSTR_GIT_HASH={short}");
        }
        if let Ok(Some(name)) = repo.head_name() {
            println!("cargo:rustc-env=KTSTR_GIT_BRANCH={}", name.shorten());
        }
    }
    println!("cargo:rerun-if-changed=.git/HEAD");
}
