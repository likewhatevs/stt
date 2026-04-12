use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_ktstr::{build_make_args, has_sched_ext, run_test_stats};
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
        /// Path to the kernel source directory.
        #[arg(long)]
        kernel: PathBuf,
        /// Run `make mrproper` first for a full reconfigure + rebuild.
        #[arg(long)]
        clean: bool,
    },
    /// Build the kernel (if needed) and run tests via cargo nextest.
    Test {
        /// Path to the kernel source directory.
        #[arg(long)]
        kernel: PathBuf,
        /// Arguments passed through to cargo nextest run.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Print gauntlet analysis from sidecar JSON files.
    TestStats {
        /// Path to the sidecar directory. Defaults to KTSTR_SIDECAR_DIR
        /// or target/ktstr/{branch}-{hash}/.
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

/// Configure the kernel with sched_ext support.
///
/// Appends the kconfig fragment to .config and runs `make olddefconfig`
/// to resolve dependencies. kconfig uses last-value-wins semantics for
/// duplicate symbols (confdata.c:conf_read_simple), so appending is
/// equivalent to merge_config.sh's delete-then-append approach — both
/// produce the same resolved config after olddefconfig.
fn configure_kernel(kernel_dir: &Path, fragment: &str) -> Result<(), String> {
    eprintln!("cargo-ktstr: configuring kernel (sched_ext not found in .config)");

    let config_path = kernel_dir.join(".config");
    if !config_path.exists() {
        run_make(kernel_dir, &["defconfig"])?;
    }

    let mut config = std::fs::OpenOptions::new()
        .append(true)
        .open(&config_path)
        .map_err(|e| format!("open .config: {e}"))?;
    std::io::Write::write_all(&mut config, fragment.as_bytes())
        .map_err(|e| format!("append kconfig: {e}"))?;

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

/// Configure if needed and build the kernel.
fn build_kernel(kernel_dir: &Path, clean: bool) -> Result<(), String> {
    if !kernel_dir.is_dir() {
        return Err(format!("{}: not a directory", kernel_dir.display()));
    }

    if clean {
        eprintln!("cargo-ktstr: make mrproper");
        run_make(kernel_dir, &["mrproper"])?;
    }

    if !has_sched_ext(kernel_dir) {
        configure_kernel(kernel_dir, EMBEDDED_KCONFIG)?;
    }

    make_kernel(kernel_dir)?;
    gen_compile_commands(kernel_dir)?;
    Ok(())
}

/// Generate compile_commands.json for LSP support.
///
/// Runs `make compile_commands.json` which invokes
/// `scripts/clang-tools/gen_compile_commands.py` over the build
/// artifacts. Requires a completed kernel build (depends on vmlinux.a).
fn gen_compile_commands(kernel_dir: &Path) -> Result<(), String> {
    eprintln!("cargo-ktstr: generating compile_commands.json");
    run_make(kernel_dir, &["compile_commands.json"])
}

fn run_test(kernel_dir: &Path, args: Vec<String>) -> Result<(), String> {
    build_kernel(kernel_dir, false)?;

    eprintln!("cargo-ktstr: running tests");
    let err = Command::new("cargo")
        .args(["nextest", "run"])
        .args(&args)
        .env("KTSTR_KERNEL", kernel_dir)
        .exec();
    Err(format!("exec cargo nextest run: {err}"))
}

fn test_stats(dir: &Option<PathBuf>) -> Result<(), String> {
    let output = run_test_stats(dir.as_deref());
    if !output.is_empty() {
        print!("{output}");
    }
    Ok(())
}

fn main() {
    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = Cargo::parse();

    let result = match ktstr.command {
        KtstrCommand::BuildKernel { kernel, clean } => build_kernel(&kernel, clean),
        KtstrCommand::Test { kernel, args } => run_test(&kernel, args),
        KtstrCommand::TestStats { ref dir } => test_stats(dir),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
