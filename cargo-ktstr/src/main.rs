use std::env;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_ktstr::{build_make_args, has_sched_ext};
use clap::{Parser, Subcommand};

/// ktstr.kconfig embedded at compile time so `cargo install` works.
const EMBEDDED_KCONFIG: &str = include_str!("../../ktstr.kconfig");

#[derive(Parser)]
#[command(name = "cargo-ktstr", bin_name = "cargo")]
struct Cargo {
    #[command(subcommand)]
    command: CargoSub,
}

#[derive(Subcommand)]
enum CargoSub {
    /// ktstr dev workflow: build kernel + run tests.
    Ktstr(Ktstr),
}

#[derive(Parser)]
struct Ktstr {
    #[command(subcommand)]
    command: KtstrCommand,
}

#[derive(Subcommand)]
enum KtstrCommand {
    /// Build the kernel with sched_ext support.
    BuildKernel {
        /// Run `make mrproper` first for a full reconfigure + rebuild.
        #[arg(long)]
        clean: bool,
    },
    /// Build the kernel (if needed) and run tests via cargo nextest.
    Test {
        /// Arguments passed through to cargo nextest run.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

/// Write the embedded ktstr.kconfig to a temporary file.
///
/// Returns the path to the temp file. The caller must keep the
/// `TempDir` alive until merge_config.sh finishes.
fn write_embedded_kconfig() -> Result<(tempfile::TempDir, PathBuf), String> {
    let dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e}"))?;
    let path = dir.path().join("ktstr.kconfig");
    std::fs::write(&path, EMBEDDED_KCONFIG).map_err(|e| format!("write embedded kconfig: {e}"))?;
    Ok((dir, path))
}

/// Configure the kernel with sched_ext support.
fn configure_kernel(kernel_dir: &Path, kconfig: &Path) -> Result<(), String> {
    eprintln!("cargo-ktstr: configuring kernel (sched_ext not found in .config)");

    let has_config = kernel_dir.join(".config").exists();
    if !has_config {
        run_make(kernel_dir, &["defconfig"])?;
    }

    let merge_script = kernel_dir.join("scripts/kconfig/merge_config.sh");
    if !merge_script.exists() {
        return Err(format!(
            "merge_config.sh not found at {}",
            merge_script.display()
        ));
    }
    let status = Command::new("sh")
        .arg(&merge_script)
        .args(["-m", ".config"])
        .arg(kconfig)
        .current_dir(kernel_dir)
        .status()
        .map_err(|e| format!("merge_config.sh: {e}"))?;
    if !status.success() {
        return Err("merge_config.sh failed".into());
    }

    run_make(kernel_dir, &["olddefconfig"])?;
    Ok(())
}

/// Run make in the kernel directory with the given args.
fn run_make(kernel_dir: &Path, args: &[&str]) -> Result<(), String> {
    let status = Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .status()
        .map_err(|e| format!("make {}: {e}", args.join(" ")))?;
    if !status.success() {
        return Err(format!("make {} failed", args.join(" ")));
    }
    Ok(())
}

/// Run make in the kernel directory for a parallel build.
fn make_kernel(kernel_dir: &Path) -> Result<(), String> {
    eprintln!("cargo-ktstr: building kernel in {}", kernel_dir.display());
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let args = build_make_args(nproc);
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_make(kernel_dir, &arg_refs)
}

/// Resolve KTSTR_KERNEL, configure if needed, and build.
fn build_kernel(clean: bool) -> Result<PathBuf, String> {
    let kernel_dir = env::var("KTSTR_KERNEL").map_err(|_| {
        "KTSTR_KERNEL is not set. Set it to your kernel source directory:\n  \
         export KTSTR_KERNEL=/path/to/linux"
            .to_string()
    })?;
    let kernel_dir = PathBuf::from(&kernel_dir);
    if !kernel_dir.is_dir() {
        return Err(format!(
            "KTSTR_KERNEL={}: not a directory",
            kernel_dir.display()
        ));
    }

    if clean {
        eprintln!("cargo-ktstr: make mrproper");
        run_make(&kernel_dir, &["mrproper"])?;
    }

    if !has_sched_ext(&kernel_dir) {
        let (_tmpdir, kconfig) = write_embedded_kconfig()?;
        configure_kernel(&kernel_dir, &kconfig)?;
    }

    make_kernel(&kernel_dir)?;
    Ok(kernel_dir)
}

fn run_test(args: Vec<String>) -> Result<(), String> {
    let kernel_dir = build_kernel(false)?;

    eprintln!("cargo-ktstr: running tests");
    let err = Command::new("cargo")
        .args(["nextest", "run"])
        .args(&args)
        .env("KTSTR_KERNEL", &kernel_dir)
        .exec();
    Err(format!("exec cargo nextest run: {err}"))
}

fn main() {
    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = Cargo::parse();

    let result = match ktstr.command {
        KtstrCommand::BuildKernel { clean } => build_kernel(clean).map(|_| ()),
        KtstrCommand::Test { args } => run_test(args),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
