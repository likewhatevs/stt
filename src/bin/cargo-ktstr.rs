//! Cargo-integrated `cargo ktstr <SUB>` binary entry point.
//!
//! This file is the bin target itself: the global jemalloc allocator,
//! tracing init, SIGPIPE restore, top-level [`clap::Parser`] dispatch,
//! and the `KtstrCommand` match arm that fans out to each subcommand
//! handler. The handlers themselves live in submodules under
//! `src/bin/cargo_ktstr/`:
//!
//! - [`cli`]    — clap-derived `Cargo` / `CargoSub` / `Ktstr` /
//!                `KtstrCommand` / `ModelCommand` / `StatsCommand`
//!                types that drive argument parsing and shell
//!                completion generation.
//! - [`kernel`] — `--kernel <SPEC>` resolution shared by the `shell`,
//!                `verifier`, and gauntlet-expansion code paths, plus
//!                the `kernel build` subcommand dispatcher. Pure
//!                wire-format helpers (label emission, `KTSTR_KERNEL_LIST`
//!                encoding, dedup, collision detection) live in the
//!                inner [`kernel::wire_format`] submodule.
//! - [`run_cargo`] — `test`, `coverage`, `llvm-cov` dispatchers that
//!                wrap `cargo nextest` with the kernel/topology
//!                gauntlet wire format.
//! - [`stats`]  — `stats compare` subcommand that diffs
//!                `target/stats/` JSON across two kernel/scheduler
//!                build commits or scheduler-flag profiles.
//! - [`verifier`] — `verifier` subcommand that runs each scheduler
//!                profile under the BPF-stats verifier and renders
//!                per-profile output.
//! - [`misc`]   — smaller subcommand dispatchers, one submodule per
//!                CLI verb: `shell`, `completions`, `funify`,
//!                `model {fetch,status,clean}`, `export`.
//! - `parse_tests` (test-only) — clap parse-shape coverage: every
//!                `KtstrCommand` variant gets at least one test that
//!                pins flag wiring + conflict/requires constraints.
//!
//! Each `mod` declaration uses `#[path = "cargo_ktstr/<file>.rs"]`
//! because rustc derives module file names from the bin's *file*
//! name (`cargo-ktstr`), not the *crate* name. Without `#[path]` it
//! would look for `src/bin/cargo-ktstr/<mod>.rs`, an underscore-vs-hyphen
//! mismatch with the actual `src/bin/cargo_ktstr/` directory.

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[path = "cargo_ktstr/cli.rs"]
mod cli;
#[path = "cargo_ktstr/kernel/mod.rs"]
mod kernel;
#[path = "cargo_ktstr/misc/mod.rs"]
mod misc;
#[path = "cargo_ktstr/run_cargo.rs"]
mod run_cargo;
#[path = "cargo_ktstr/stats.rs"]
mod stats;
#[path = "cargo_ktstr/verifier.rs"]
mod verifier;

#[cfg(test)]
#[path = "cargo_ktstr/parse_tests.rs"]
mod parse_tests;

use clap::Parser;
use ktstr::cli::KernelCommand;

use crate::cli::{Cargo, CargoSub, KtstrCommand, ModelCommand};

fn main() {
    // Restore SIGPIPE so piping `cargo ktstr ... | head` doesn't
    // panic inside `print!`. See `ktstr::cli::restore_sigpipe_default`
    // for the full rationale; shared across all three ktstr bins so
    // the rationale + SAFETY text lives in one place.
    ktstr::cli::restore_sigpipe_default();
    // Mirror `ktstr`'s tracing init (src/bin/ktstr.rs main()) so
    // `tracing::warn!` calls inside `cli::` / `test_support::` surface
    // on stderr instead of being silently dropped. Default to `warn`
    // so normal CLI invocations (kernel build, model fetch, etc.) stay
    // quiet; users who want finer detail set `RUST_LOG=info,debug,...`.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = Cargo::parse();

    // Match-arm order mirrors the `KtstrCommand` enum declaration in
    // `cli.rs`. Keeping the two orderings in lockstep lets a reviewer
    // eyeball "every variant is dispatched" in one linear scan instead
    // of cross-referencing two different orders; a future variant
    // addition then lands in the matching enum position and here
    // without requiring the reader to rebuild the mapping.
    let result = match ktstr.command {
        KtstrCommand::Test {
            kernel,
            no_perf_mode,
            release,
            args,
        } => run_cargo::run_test(kernel, no_perf_mode, release, args),
        KtstrCommand::Coverage {
            kernel,
            no_perf_mode,
            release,
            args,
        } => run_cargo::run_coverage(kernel, no_perf_mode, release, args),
        KtstrCommand::LlvmCov {
            kernel,
            no_perf_mode,
            args,
        } => run_cargo::run_llvm_cov(kernel, no_perf_mode, args),
        KtstrCommand::Stats { ref command } => stats::run_stats(command),
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json, range } => match range {
                Some(r) => {
                    ktstr::cli::kernel_list_range_preview(json, &r).map_err(|e| format!("{e:#}"))
                }
                None => ktstr::cli::kernel_list(json).map_err(|e| format!("{e:#}")),
            },
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
                extra_kconfig,
            } => kernel::kernel_build(
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
                extra_kconfig,
            ),
            KernelCommand::Clean {
                keep,
                force,
                corrupt_only,
            } => ktstr::cli::kernel_clean(keep, force, corrupt_only).map_err(|e| format!("{e:#}")),
        },
        KtstrCommand::Model { command } => match command {
            ModelCommand::Fetch => misc::run_model_fetch(),
            ModelCommand::Status => misc::run_model_status(),
            ModelCommand::Clean => misc::run_model_clean(),
        },
        KtstrCommand::Verifier {
            scheduler,
            scheduler_bin,
            kernel,
            raw,
            all_profiles,
            profiles,
        } => verifier::run_verifier(
            scheduler,
            scheduler_bin,
            kernel,
            raw,
            all_profiles,
            profiles,
        ),
        KtstrCommand::Funify {
            input,
            seed,
            pretty,
        } => misc::run_funify(input, seed, pretty),
        KtstrCommand::Completions { shell, binary } => {
            misc::run_completions(shell, &binary);
            Ok(())
        }
        KtstrCommand::ShowHost => {
            print!("{}", ktstr::cli::show_host());
            Ok(())
        }
        KtstrCommand::ShowThresholds { test } => match ktstr::cli::show_thresholds(&test) {
            Ok(s) => {
                print!("{s}");
                Ok(())
            }
            Err(e) => Err(format!("{e:#}")),
        },
        KtstrCommand::Export {
            test,
            output,
            package,
            release,
        } => misc::run_export(test, output, package, release),
        KtstrCommand::Locks { json, watch } => {
            ktstr::cli::list_locks(json, watch).map_err(|e| format!("{e:#}"))
        }
        KtstrCommand::Shell {
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
            no_perf_mode,
            cpu_cap,
            disk,
        } => misc::run_shell(
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
            no_perf_mode,
            cpu_cap,
            disk,
        ),
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
