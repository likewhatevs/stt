// Generates vmlinux.h from kernel BTF using libbpf's btf_dump API.
// Uses the shared kernel resolver (src/kernel_path.rs) to find the
// BTF source. See resolve_btf() for the full search order.

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

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
        println!("generating vmlinux.h from {}", btf_source.display());

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

    // arm64 bpf_tracing.h casts pt_regs through struct user_pt_regs,
    // a UAPI type that kernel BTF may omit. Append it if absent so
    // PT_REGS_PARMn_CORE compiles on arm64 hosts.
    if cfg!(target_arch = "aarch64") {
        let content = std::fs::read_to_string(&vmlinux_h).expect("read vmlinux.h");
        if !content.contains("struct user_pt_regs {") {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&vmlinux_h)
                .expect("open vmlinux.h for append");
            writeln!(
                f,
                "\n/* Added by build.rs: arm64 UAPI type needed by bpf_tracing.h */\n\
                 struct user_pt_regs {{\n\
                 \t__u64 regs[31];\n\
                 \t__u64 sp;\n\
                 \t__u64 pc;\n\
                 \t__u64 pstate;\n\
                 }};\n"
            )
            .expect("append user_pt_regs to vmlinux.h");
        }
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

    // Git info for output directory keying and cache invalidation.
    let mut commit_hash = String::from("unknown");
    if let Ok(repo) = gix::discover(".") {
        if let Ok(id) = repo.head_id() {
            let full = id.to_string();
            commit_hash = full.clone();
            let short = &full[..full.len().min(7)];
            println!("cargo:rustc-env=KTSTR_GIT_HASH={short}");
        }
        if let Ok(Some(name)) = repo.head_name() {
            println!("cargo:rustc-env=KTSTR_GIT_BRANCH={}", name.shorten());
        }
    }
    println!("cargo:rustc-env=KTSTR_GIT_FULL_HASH={commit_hash}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");

    // Build busybox from source for guest shell mode.
    // Cache: skip if $OUT_DIR/busybox exists. After build.rs config
    // changes, run `cargo clean` to force a rebuild.
    let busybox_bin = out_dir.join("busybox");
    if !busybox_bin.exists() {
        println!("cargo:warning=compiling busybox (first build only)...");

        // Check required tools before attempting build.
        if Command::new("make").arg("--version").output().is_err() {
            panic!(
                "busybox build requires 'make' — install build-essential \
                 (Debian/Ubuntu) or base-devel (Fedora/Arch)"
            );
        }
        if Command::new("gcc").arg("--version").output().is_err() {
            panic!(
                "busybox build requires 'gcc' — install build-essential \
                 (Debian/Ubuntu) or base-devel (Fedora/Arch)"
            );
        }

        let busybox_src = out_dir.join("busybox-src");

        // Recover from interrupted download: if the directory exists but
        // has no Makefile, the previous extraction was incomplete.
        if busybox_src.exists() && !busybox_src.join("Makefile").exists() {
            std::fs::remove_dir_all(&busybox_src).expect("remove incomplete busybox-src");
        }

        // Download and extract busybox source tarball.
        if !busybox_src.join("Makefile").exists() {
            let url = "https://github.com/mirror/busybox/archive/refs/tags/1_36_1.tar.gz";
            let resp = reqwest::blocking::get(url)
                .and_then(|r| r.error_for_status())
                .unwrap_or_else(|e| {
                    panic!(
                        "failed to download busybox tarball from {url} — \
                         check network connectivity. First build requires internet access: {e}"
                    )
                });
            let gz = flate2::read::GzDecoder::new(resp);
            let mut archive = tar::Archive::new(gz);
            let extract_dir = out_dir.join("busybox-extract");
            archive
                .unpack(&extract_dir)
                .expect("extract busybox tarball");

            // GitHub tarballs extract to busybox-1_36_1/ inside the archive.
            let inner = extract_dir.join("busybox-1_36_1");
            std::fs::rename(&inner, &busybox_src).unwrap_or_else(|e| {
                panic!(
                    "expected extracted directory {} — tarball layout may have changed: {e}",
                    inner.display()
                )
            });
            std::fs::remove_dir_all(&extract_dir).ok();
        }

        // Configure busybox.
        let status = Command::new("make")
            .arg("defconfig")
            .current_dir(&busybox_src)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("make defconfig");
        assert!(status.success(), "busybox make defconfig failed");

        // Enable static linking, disable CONFIG_TC (requires iproute2 headers).
        let config_path = busybox_src.join(".config");
        let config = std::fs::read_to_string(&config_path).expect("read busybox .config");
        let config = config
            .replace("# CONFIG_STATIC is not set", "CONFIG_STATIC=y")
            .replace("CONFIG_TC=y", "# CONFIG_TC is not set");
        std::fs::write(&config_path, config).expect("write patched busybox .config");

        // Build busybox.
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "1".to_string());
        let status = Command::new("make")
            .arg(format!("-j{nproc}"))
            .current_dir(&busybox_src)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("busybox make");
        assert!(status.success(), "busybox build failed");

        // Copy binary to OUT_DIR.
        std::fs::copy(busybox_src.join("busybox"), &busybox_bin)
            .expect("copy busybox binary to OUT_DIR");
    }
}
