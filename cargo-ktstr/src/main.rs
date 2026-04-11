use std::env;
use std::fs;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Parser, Subcommand};

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
    /// Build the kernel (if needed) and run tests via cargo nextest.
    Test {
        /// Arguments passed through to cargo nextest run.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

/// Locate ktstr.kconfig relative to the cargo-ktstr binary.
///
/// Walks up from the binary's directory looking for ktstr.kconfig.
/// Falls back to the current directory.
fn find_kconfig() -> Option<PathBuf> {
    // Try relative to the binary location (../../ktstr.kconfig from
    // target/debug/cargo-ktstr or target/release/cargo-ktstr).
    if let Ok(exe) = env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        while let Some(d) = dir {
            let candidate = d.join("ktstr.kconfig");
            if candidate.exists() {
                return Some(candidate);
            }
            dir = d.parent().map(Path::to_path_buf);
        }
    }
    // Fallback: current working directory.
    let cwd = Path::new("ktstr.kconfig");
    if cwd.exists() {
        return Some(cwd.to_path_buf());
    }
    None
}

/// Check if .config contains CONFIG_SCHED_CLASS_EXT=y.
fn has_sched_ext(kernel_dir: &Path) -> bool {
    let config = kernel_dir.join(".config");
    fs::read_to_string(config)
        .map(|s| s.lines().any(|l| l == "CONFIG_SCHED_CLASS_EXT=y"))
        .unwrap_or(false)
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

/// Build the kernel.
fn build_kernel(kernel_dir: &Path) -> Result<(), String> {
    eprintln!("cargo-ktstr: building kernel in {}", kernel_dir.display());
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "1".into());
    let j_flag = format!("-j{nproc}");
    run_make(kernel_dir, &[&j_flag, "KCFLAGS=-Wno-error"])
}

fn run_test(args: Vec<String>) -> Result<(), String> {
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

    // Configure kernel if sched_ext is not enabled.
    if !has_sched_ext(&kernel_dir) {
        let kconfig = find_kconfig().ok_or(
            "ktstr.kconfig not found. Run from the ktstr workspace root \
             or install cargo-ktstr from the workspace."
                .to_string(),
        )?;
        configure_kernel(&kernel_dir, &kconfig)?;
    }

    // Build kernel (always — work-conserving, make handles no-op).
    build_kernel(&kernel_dir)?;

    // exec cargo nextest run with KTSTR_KERNEL in the environment.
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
        KtstrCommand::Test { args } => run_test(args),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
