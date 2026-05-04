//! Clap parse tests for the cargo-ktstr CLI surface.
//!
//! Lives outside the bin entry file so the entry stays focused on
//! dispatch, and mirrors the shape used elsewhere in the workspace
//! where parse-only test fixtures cluster in their own module. The
//! tests assert the user-visible spelling of every `--flag` and
//! the round-trip of clap-parsed values into the variant fields,
//! so a derive-rename or attribute drop surfaces here at compile-
//! time / test-time rather than in production.
//!
//! # Coverage shape
//!
//! Most [`KtstrCommand`] variants have at least one positive test
//! that round-trips an argument through clap into the variant's
//! fields, plus negative tests for `requires` / `conflicts_with` /
//! `value_parser` constraints clap enforces at parse time.
//!
//! Variants WITHOUT a positive parse test as of this revision:
//! `Model`, `Funify`, `Export`, `Locks`. Their parse-shape
//! coverage is filed as a follow-up; until then a clap regression
//! that reshapes one of those variants will not surface here.
//!
//! Sections are separated by `// -- <theme> --` banners.
//!
//! # External pins
//!
//! The `kconfig_status_*` and `format_entry_row_*` fixtures
//! co-pin the [`ktstr::cache::CacheEntry`] /
//! [`ktstr::cache::KernelMetadata`] shape; if the cache types'
//! constructors, methods, or variants change, those tests must
//! be updated in lockstep with the production types in
//! [`ktstr::cache`].
//!
//! # Why parse-only
//!
//! These tests deliberately do not invoke any subcommand body —
//! they verify that clap parses what we expect into the type-system
//! shape the body matches against. Behaviour-level coverage of each
//! handler lives next to its production code (e.g. `kernel.rs`'s
//! `tests` module exercises label collision detection;
//! `verifier.rs` exercises profile expansion).

#![cfg(test)]

use clap::{CommandFactory, Parser};
use ktstr::cache::{CacheArtifacts, CacheDir, CacheEntry, KernelMetadata};
use ktstr::cli;
use ktstr::cli::KernelCommand;

use crate::cli::{Cargo, CargoSub, KtstrCommand, StatsCommand};

// -- structural validation --

/// Run clap's structural self-check on the entire [`Cargo`] derive tree.
///
/// `clap::Command::debug_assert` walks every subcommand, every
/// arg, every group, and every relationship (`conflicts_with`,
/// `requires`, `default_value_if`, `value_parser`, …) and panics
/// at test time on issues that would otherwise surface as cryptic
/// runtime parse errors or silent UX bugs:
///
///   - duplicate arg / subcommand IDs
///   - dangling references in `conflicts_with` / `requires`
///   - default values that fail the arg's `value_parser`
///   - help/version conflicts with user-defined args
///   - misordered positionals (greedy followed by non-greedy)
///
/// Upstream clap recommends running this helper in a unit test for
/// every derive root; we put it FIRST in the parse-tests file so
/// any structural break stops the rest of the suite immediately
/// rather than producing a wall of less-informative downstream
/// failures from individual `try_parse_from` calls.
#[test]
fn cli_debug_assert() {
    Cargo::command().debug_assert();
}

// -- try_get_matches_from: test subcommand --

#[test]
fn parse_test_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "test"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_test_with_kernel() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "test", "--kernel", "6.14.2"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `--release` on `test` parses to `KtstrCommand::Test { release:
/// true, .. }` so `run_test` prepends `--cargo-profile release`
/// to the cargo nextest invocation. A clap regression that
/// dropped the flag would turn the user-visible `--release` into
/// either a silent no-op (default false) or a passthrough-arg
/// typo — this test pins the clap-level wiring.
#[test]
fn parse_test_with_release_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "test", "--release"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Test { release, .. } => {
            assert!(release, "`--release` must set `release=true`");
        }
        _ => panic!("expected Test"),
    }
}

/// Pin `trailing_var_arg` args forwarded verbatim after `--`.
#[test]
fn parse_test_with_passthrough_args() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "test",
        "--",
        "-p",
        "ktstr",
        "--no-capture",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Test {
            kernel,
            no_perf_mode,
            release,
            args,
        } => {
            assert!(
                kernel.is_empty(),
                "bare `--` passthrough must not spuriously populate --kernel",
            );
            assert!(
                !no_perf_mode,
                "bare `--` passthrough must not spuriously set --no-perf-mode",
            );
            assert!(
                !release,
                "bare `--` passthrough must not spuriously set --release",
            );
            assert_eq!(args, vec!["-p", "ktstr", "--no-capture"]);
        }
        _ => panic!("expected Test"),
    }
}

// -- try_get_matches_from: `test` visible alias `nextest` --

/// `cargo ktstr nextest` resolves to the canonical `Test`
/// variant. `visible_alias = "nextest"` on the variant makes
/// the alias user-facing (shows in --help) and dispatch-
/// transparent (the existing `KtstrCommand::Test` arm handles
/// both spellings). A regression that dropped the attribute
/// would fail this test at runtime.
#[test]
fn parse_nextest_alias_dispatches_to_test() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "nextest"]).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        matches!(k.command, KtstrCommand::Test { .. }),
        "`nextest` alias must dispatch to the Test variant",
    );
}

/// `nextest` alias carries trailing args through the same
/// `trailing_var_arg` pipeline as `test`. Pins the alias's
/// passthrough behaviour byte-exactly so a clap regression
/// that treated the alias as a distinct parse tree surfaces
/// here rather than in runtime dispatch.
#[test]
fn parse_nextest_alias_with_passthrough_args() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "nextest",
        "--",
        "-p",
        "ktstr",
        "--no-capture",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Test { args, .. } => {
            assert_eq!(args, vec!["-p", "ktstr", "--no-capture"]);
        }
        _ => panic!("expected Test (via `nextest` alias)"),
    }
}

/// Verify the `nextest` alias preserves all Test fields in a
/// single invocation: `--kernel`, `--no-perf-mode`, and empty
/// trailing `args`. A clap regression that silently dropped a
/// field on the alias path (e.g. a derive bug that re-generated
/// the subcommand without inheriting the Test variant's args)
/// would surface here.
#[test]
fn parse_nextest_alias_with_kernel_and_no_perf_mode() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "nextest",
        "--kernel",
        "6.14.2",
        "--no-perf-mode",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Test {
            kernel,
            no_perf_mode,
            release,
            args,
        } => {
            assert_eq!(kernel, vec!["6.14.2".to_string()]);
            assert!(no_perf_mode);
            assert!(!release, "bare invocation must default --release to false");
            assert!(args.is_empty());
        }
        _ => panic!("expected Test (via `nextest` alias)"),
    }
}

// -- try_get_matches_from: coverage subcommand --

#[test]
fn parse_coverage_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_coverage_with_kernel() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage", "--kernel", "6.14.2"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `--release` on `coverage` parses to `KtstrCommand::Coverage
/// { release: true, .. }` so `run_coverage` prepends
/// `--cargo-profile release` to the cargo llvm-cov nextest
/// invocation. Same rationale as the sibling
/// `parse_test_with_release_flag` — pins the clap-level wiring
/// against a regression that turns the flag into a no-op.
#[test]
fn parse_coverage_with_release_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "coverage", "--release"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Coverage { release, .. } => {
            assert!(release, "`--release` must set `release=true`");
        }
        _ => panic!("expected Coverage"),
    }
}

/// Pin `trailing_var_arg` args forwarded verbatim after `--`.
#[test]
fn parse_coverage_with_passthrough_args() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "coverage",
        "--",
        "--workspace",
        "--lcov",
        "--output-path",
        "lcov.info",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Coverage { args, .. } => {
            assert_eq!(
                args,
                vec!["--workspace", "--lcov", "--output-path", "lcov.info"]
            );
        }
        _ => panic!("expected Coverage"),
    }
}

/// Combined round-trip for Coverage: `--kernel`, `--no-perf-mode`,
/// AND trailing args all populate on a single invocation. Mirrors
/// `parse_llvm_cov_with_kernel_and_no_perf_mode` — a clap
/// regression that dropped one field on the multi-flag path (or
/// mis-ordered `--` with flags) would surface here for the
/// Coverage variant.
#[test]
fn parse_coverage_with_kernel_and_no_perf_mode() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "coverage",
        "--kernel",
        "6.14.2",
        "--no-perf-mode",
        "--",
        "--workspace",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Coverage {
            kernel,
            no_perf_mode,
            release,
            args,
        } => {
            assert_eq!(kernel, vec!["6.14.2".to_string()]);
            assert!(no_perf_mode);
            assert!(!release, "bare invocation must default --release to false");
            assert_eq!(args, vec!["--workspace"]);
        }
        _ => panic!("expected Coverage"),
    }
}

// -- try_get_matches_from: llvm-cov raw passthrough subcommand --

#[test]
fn parse_llvm_cov_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_llvm_cov_with_kernel() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov", "--kernel", "6.14.2"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::LlvmCov { kernel, .. } => {
            assert_eq!(kernel, vec!["6.14.2".to_string()]);
        }
        _ => panic!("expected LlvmCov"),
    }
}

/// Pin `trailing_var_arg` args forwarded verbatim after `--`.
#[test]
fn parse_llvm_cov_with_passthrough_args() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "llvm-cov",
        "--",
        "report",
        "--lcov",
        "--output-path",
        "lcov.info",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::LlvmCov { args, .. } => {
            assert_eq!(args, vec!["report", "--lcov", "--output-path", "lcov.info"]);
        }
        _ => panic!("expected LlvmCov"),
    }
}

/// Combined round-trip: `--kernel`, `--no-perf-mode`, AND
/// trailing args all populate on a single LlvmCov invocation.
/// A clap regression that dropped one field on the multi-flag
/// path (or mis-ordered `--` with flags) would surface here.
#[test]
fn parse_llvm_cov_with_kernel_and_no_perf_mode() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "llvm-cov",
        "--kernel",
        "6.14.2",
        "--no-perf-mode",
        "--",
        "report",
        "--lcov",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::LlvmCov {
            kernel,
            no_perf_mode,
            args,
        } => {
            assert_eq!(kernel, vec!["6.14.2".to_string()]);
            assert!(no_perf_mode);
            assert_eq!(args, vec!["report", "--lcov"]);
        }
        _ => panic!("expected LlvmCov"),
    }
}

/// Negative pin: the variant is `LlvmCov`, and clap derive's
/// default casing is kebab-case (see clap_derive
/// `DEFAULT_CASING`), so the subcommand name is `llvm-cov`,
/// NOT `llvm_cov`. A regression that switched the derive's
/// rename_all default (or silently aliased the underscore
/// form) would turn this negative pin positive. The parent-
/// level `aliases` slot is empty, so clap rejects the
/// underscore form with an unknown-subcommand error.
#[test]
fn parse_llvm_cov_underscore_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "llvm_cov"]);
    assert!(
        rejected.is_err(),
        "`llvm_cov` (underscore) must be rejected — the \
         canonical name is `llvm-cov` (kebab-case)",
    );
}

/// Positive companion to [`parse_llvm_cov_underscore_rejected`]:
/// the kebab-case form `llvm-cov` MUST resolve to
/// [`KtstrCommand::LlvmCov`] without alias indirection. The
/// existing `parse_llvm_cov_minimal` exercises the spelling but
/// only asserts `is_ok()` — this test pins the variant binding
/// so that a future rename of the derive variant or the
/// subcommand attribute (e.g. `command(name = "llvm-coverage")`)
/// surfaces here as a variant-mismatch panic instead of silently
/// breaking under a renamed-but-still-parseable form.
#[test]
fn parse_llvm_cov_kebab_accepted() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov"]).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        matches!(k.command, KtstrCommand::LlvmCov { .. }),
        "kebab `llvm-cov` must bind to KtstrCommand::LlvmCov",
    );
}

// -- try_get_matches_from: shell subcommand --

#[test]
fn parse_shell_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "shell"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_shell_with_topology() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--topology", "1,2,4,1"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Shell { topology, .. } => {
            assert_eq!(topology, "1,2,4,1");
        }
        _ => panic!("expected Shell"),
    }
}

#[test]
fn parse_shell_default_topology() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell"]).unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Shell { topology, .. } => {
            assert_eq!(topology, "1,1,1,1");
        }
        _ => panic!("expected Shell"),
    }
}

/// Pin `-i` / `--include-files` `ArgAction::Append` round-trip with ordering.
#[test]
fn parse_shell_include_files() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "-i", "/tmp/a", "-i", "/tmp/b"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Shell { include_files, .. } => {
            assert_eq!(
                include_files,
                vec![
                    std::path::PathBuf::from("/tmp/a"),
                    std::path::PathBuf::from("/tmp/b"),
                ],
                "-i flag must accumulate paths in order via ArgAction::Append",
            );
        }
        _ => panic!("expected Shell"),
    }
}

/// `cargo ktstr shell --disk 256mib` parses; the disk arg lands as
/// `Some("256mib")` on the `Shell` variant. The string is parsed
/// into a `DiskConfig` later in `run_shell` via
/// [`ktstr::cli::parse_disk_size_mib`]; the clap stage stores the
/// raw string so a malformed input surfaces with the consistent
/// disk-size diagnostic instead of a generic clap parse error.
#[test]
fn parse_shell_disk_arg() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--disk", "256mib"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Shell { disk, .. } => {
            assert_eq!(disk.as_deref(), Some("256mib"));
        }
        _ => panic!("expected Shell"),
    }
}

/// Omitting `--disk` produces `None`, matching the no-disk default
/// in `run_shell` and `KtstrVm::builder`.
#[test]
fn parse_shell_disk_arg_omitted() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell"]).unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Shell { disk, .. } => {
            assert!(disk.is_none(), "no --disk must produce None");
        }
        _ => panic!("expected Shell"),
    }
}

// -- try_get_matches_from: stats subcommand --

#[test]
fn parse_stats_bare() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "stats"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_stats_list() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `cargo ktstr stats list-metrics` parses (no flags required)
/// and dispatches to the `ListMetrics` variant with `json=false`.
#[test]
fn parse_stats_list_metrics_bare() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ListMetrics { json }),
            ..
        } => {
            assert!(
                !json,
                "bare `list-metrics` must default to text mode (json=false)",
            );
        }
        _ => panic!("expected Stats ListMetrics"),
    }
}

/// `cargo ktstr stats list-metrics --json` sets `json=true`.
/// Pins the flag name so a clap-derive-default rename
/// (kebab-case) cannot drift — `--json` is the same flag name
/// other list-style subcommands use (e.g. `kernel list --json`).
#[test]
fn parse_stats_list_metrics_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics", "--json"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ListMetrics { json }),
            ..
        } => {
            assert!(json, "--json must set the flag true");
        }
        _ => panic!("expected Stats ListMetrics"),
    }
}

/// `list-metrics` takes no positional args — a stray positional
/// must be rejected by clap so a typo like `list-metrics
/// worst_spread` doesn't silently look like success.
#[test]
fn parse_stats_list_metrics_rejects_positional() {
    let rejected =
        Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics", "worst_spread"]);
    assert!(
        rejected.is_err(),
        "list-metrics must reject positional arguments",
    );
}

/// `cargo ktstr stats list-values` parses with no flags and
/// dispatches to the `ListValues` variant with `json=false` and
/// `dir=None`. Pins the bare-call defaults.
#[test]
fn parse_stats_list_values_bare() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ListValues { json, dir }),
            ..
        } => {
            assert!(!json, "bare `list-values` must default to text mode");
            assert!(
                dir.is_none(),
                "bare `list-values` must default to no --dir override"
            );
        }
        _ => panic!("expected Stats ListValues"),
    }
}

/// `cargo ktstr stats list-values --json` sets `json=true`.
/// Pins the flag name so the same `--json` convention used by
/// `list-metrics` and `kernel list` carries here too.
#[test]
fn parse_stats_list_values_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values", "--json"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ListValues { json, .. }),
            ..
        } => {
            assert!(json, "--json must set the flag true");
        }
        _ => panic!("expected Stats ListValues"),
    }
}

/// `cargo ktstr stats list-values --dir PATH` round-trips the
/// path through clap to the dispatch site. Same `--dir`
/// convention as `compare --dir` and `show-host --dir`.
#[test]
fn parse_stats_list_values_with_dir() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "list-values",
        "--dir",
        "/tmp/archived-runs",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ListValues { dir, json }),
            ..
        } => {
            assert_eq!(
                dir.as_deref(),
                Some(std::path::Path::new("/tmp/archived-runs")),
                "--dir must round-trip to Some(PathBuf)",
            );
            assert!(!json, "bare --dir must not spuriously set --json");
        }
        _ => panic!("expected Stats ListValues"),
    }
}

/// `list-values` takes no positional args — clap must reject
/// strays so a typo like `list-values kernel` (intending a
/// per-dim filter) fails loudly rather than getting silently
/// dropped.
#[test]
fn parse_stats_list_values_rejects_positional() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values", "kernel"]);
    assert!(
        rejected.is_err(),
        "list-values must reject positional arguments",
    );
}

#[test]
fn parse_stats_compare() {
    // Minimal partition shape: --a-kernel + --b-kernel define
    // the slicing dimension. The dispatch site rejects empty
    // slicing dims, so a bare `cargo ktstr stats compare`
    // would fail at run time — but the CLI parser accepts
    // it (validation belongs in `compare_partitions`, not
    // clap). This test pins the parse layer only.
    let m = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_stats_compare_with_filter() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
        "-E",
        "cgroup_steady",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    filter,
                    threshold,
                    policy,
                    dir,
                    a_kernel,
                    b_kernel,
                    ..
                }),
            ..
        } => {
            assert_eq!(a_kernel, vec!["6.14"]);
            assert_eq!(b_kernel, vec!["6.15"]);
            assert_eq!(filter.as_deref(), Some("cgroup_steady"));
            assert!(threshold.is_none());
            assert!(policy.is_none());
            assert!(dir.is_none());
        }
        _ => panic!("expected Stats Compare"),
    }
}

#[test]
fn parse_stats_compare_with_threshold() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
        "--threshold",
        "5.0",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    threshold, filter, ..
                }),
            ..
        } => {
            assert_eq!(threshold, Some(5.0));
            assert!(filter.is_none());
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// Proves the `dir: Option<PathBuf>` field is wired on
/// `StatsCommand::Compare` and round-trips through clap's arg
/// parser. A regression that removed the struct field would
/// fail this test at compile time; a regression that dropped
/// the dispatch wiring (cargo-ktstr.rs → cli.rs → stats.rs) is
/// outside parse-scope and covered by the resolver's own
/// tests. The sibling `*_with_filter` test pins the
/// `dir.is_none()` default; this one pins the `Some(PathBuf)`
/// branch byte-exactly. Uses an absolute `/tmp/...` path
/// (synthetic, not required to exist) because the parse path
/// does not touch the filesystem — clap produces the `PathBuf`
/// from the raw argument, full stop.
#[test]
fn parse_stats_compare_with_dir() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
        "--dir",
        "/tmp/archived-runs",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    filter,
                    threshold,
                    policy,
                    dir,
                    ..
                }),
            ..
        } => {
            assert_eq!(
                dir.as_deref(),
                Some(std::path::Path::new("/tmp/archived-runs")),
                "--dir must round-trip to Some(PathBuf); \
                 parse-scope only — resolver coverage lives \
                 with compare_partitions' own tests",
            );
            assert!(
                filter.is_none(),
                "bare --dir must not spuriously populate filter",
            );
            assert!(
                threshold.is_none(),
                "bare --dir must not spuriously populate threshold",
            );
            assert!(
                policy.is_none(),
                "bare --dir must not spuriously populate policy",
            );
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// Positive parse pin: `--policy PATH` round-trips to
/// `StatsCommand::Compare { policy: Some(PathBuf(PATH)),
/// threshold: None, ... }`. Mirrors `parse_stats_compare_with_dir`
/// for the `dir` field. Uses an obviously-synthetic path that
/// does not need to exist — the parse path never touches the
/// filesystem; policy loading happens downstream in the
/// dispatch.
#[test]
fn parse_stats_compare_with_policy() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
        "--policy",
        "/tmp/policy.json",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    threshold, policy, ..
                }),
            ..
        } => {
            assert_eq!(
                policy.as_deref(),
                Some(std::path::Path::new("/tmp/policy.json")),
                "--policy must round-trip to Some(PathBuf); got {policy:?}",
            );
            assert!(
                threshold.is_none(),
                "bare --policy must not populate --threshold",
            );
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// Conflict pin: `--threshold` and `--policy` are mutually
/// exclusive at clap parse time. A regression that dropped the
/// `conflicts_with` attribute on either field would turn the
/// dispatch-level `unreachable!()` branch into a runtime panic
/// instead of a parse-time error.
///
/// Matches on [`clap::error::ErrorKind::ArgumentConflict`] rather
/// than the generic `is_err()` so a regression that produces a
/// DIFFERENT clap error (e.g. `MissingRequiredArgument` from a
/// renamed flag, or `UnknownArgument` from a typo'd attribute)
/// surfaces here as the wrong-kind diagnostic instead of being
/// silently masked by a less-specific success-on-any-error pin.
///
/// Uses a match-on-result form rather than `expect_err`/`unwrap_err`
/// because [`Cargo`] does not derive `Debug` — the unwrap helpers
/// require `T: Debug` for their failure-render path, while a direct
/// match avoids the bound entirely.
#[test]
fn parse_stats_compare_threshold_conflicts_with_policy() {
    let result = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
        "--threshold",
        "5.0",
        "--policy",
        "/tmp/policy.json",
    ]);
    match result {
        Ok(_) => panic!("--threshold + --policy must be rejected at parse time"),
        Err(err) => assert_eq!(
            err.kind(),
            clap::error::ErrorKind::ArgumentConflict,
            "expected ArgumentConflict — a different ErrorKind would \
             signal that the conflicts_with attribute regressed in a way \
             the bare is_err() pin would silently mask. Full err: {err}",
        ),
    }
}

/// Bare `compare` defaults `--no-average` to `false` —
/// averaging is the default. `--no-average` must be opt-in
/// for "keep each sidecar distinct" semantics.
#[test]
fn parse_stats_compare_no_average_default_false() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::Compare { no_average, .. }),
            ..
        } => {
            assert!(
                !no_average,
                "bare compare must default --no-average to false so \
                 averaging-on remains the default — operators get \
                 trial-set folding without an explicit flag.",
            );
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--no-average` parses as a bare flag (no value) and lifts
/// the `no_average: bool` field on `StatsCommand::Compare`
/// to true. Pins the clap binding so a regression that
/// dropped the derive arg, renamed the flag, or accidentally
/// made it take a value lands at parse time.
#[test]
fn parse_stats_compare_with_no_average() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
        "--no-average",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    no_average,
                    threshold,
                    policy,
                    dir,
                    ..
                }),
            ..
        } => {
            assert!(no_average, "--no-average must lift the flag to true");
            assert!(
                threshold.is_none(),
                "bare --no-average must not spuriously populate --threshold",
            );
            assert!(
                policy.is_none(),
                "bare --no-average must not spuriously populate --policy",
            );
            assert!(
                dir.is_none(),
                "bare --no-average must not spuriously populate --dir",
            );
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--project-commit V` round-trips to `Compare { project_commit:
/// vec![V], .. }`. Pins the clap binding for the shared
/// `--project-commit` filter on the stats compare subcommand; a
/// regression that removed the derive arg, renamed the flag, or
/// dropped its `ArgAction::Append` would land here at parse time.
#[test]
fn parse_stats_compare_with_project_commit_single() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--project-commit",
        "abc1234",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    project_commit,
                    a_project_commit,
                    b_project_commit,
                    ..
                }),
            ..
        } => {
            assert_eq!(project_commit, vec!["abc1234"]);
            assert!(
                a_project_commit.is_empty(),
                "shared --project-commit must not populate --a-project-commit",
            );
            assert!(
                b_project_commit.is_empty(),
                "shared --project-commit must not populate --b-project-commit",
            );
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--project-commit A --project-commit B` produces a Vec with two
/// entries — the flag is `ArgAction::Append`, so multiple
/// occurrences accumulate into the OR-combined filter the dispatch
/// applies. A regression that lost the Append action would
/// drop the first occurrence.
#[test]
fn parse_stats_compare_with_project_commit_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--project-commit",
        "a",
        "--project-commit",
        "b",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::Compare { project_commit, .. }),
            ..
        } => {
            assert_eq!(project_commit, vec!["a", "b"]);
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--kernel-commit V` round-trips to `Compare {
/// kernel_commit: vec![V], .. }`. Pins the clap binding for
/// the shared `--kernel-commit` filter on the stats compare
/// subcommand; a regression that removed the derive arg,
/// renamed the flag, or dropped its `ArgAction::Append`
/// would land here at parse time. Mirrors
/// `parse_stats_compare_with_project_commit_single` for the
/// `kernel_commit` dimension.
#[test]
fn parse_stats_compare_with_kernel_commit_single() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--kernel-commit",
        "abc1234",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    kernel_commit,
                    a_kernel_commit,
                    b_kernel_commit,
                    ..
                }),
            ..
        } => {
            assert_eq!(kernel_commit, vec!["abc1234"]);
            assert!(
                a_kernel_commit.is_empty(),
                "shared --kernel-commit must not populate --a-kernel-commit",
            );
            assert!(
                b_kernel_commit.is_empty(),
                "shared --kernel-commit must not populate --b-kernel-commit",
            );
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--kernel-commit A --kernel-commit B` produces a Vec with
/// two entries via `ArgAction::Append`. Mirrors
/// `parse_stats_compare_with_project_commit_repeatable` for the
/// kernel-commit dimension.
#[test]
fn parse_stats_compare_with_kernel_commit_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--kernel-commit",
        "a",
        "--kernel-commit",
        "b",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::Compare { kernel_commit, .. }),
            ..
        } => {
            assert_eq!(kernel_commit, vec!["a", "b"]);
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--scheduler A --scheduler B` produces a Vec with two
/// entries — the flag is `ArgAction::Append` (Vec, not
/// Option), so multiple occurrences accumulate into the
/// OR-combined filter the dispatch applies. Mirrors
/// `parse_stats_compare_with_project_commit_repeatable` for
/// the scheduler dimension. A regression that reverted
/// `scheduler` to `Option<String>` (the pre-conversion shape)
/// would fail this test at parse time — clap's `Option` derive
/// rejects multiple occurrences with a "supplied more than
/// once" diagnostic.
#[test]
fn parse_stats_compare_with_scheduler_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--scheduler",
        "scx_alpha",
        "--scheduler",
        "scx_beta",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::Compare { scheduler, .. }),
            ..
        } => {
            assert_eq!(scheduler, vec!["scx_alpha", "scx_beta"]);
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--topology A --topology B` produces a Vec with two
/// entries via `ArgAction::Append`. Mirrors the scheduler
/// sibling above for the topology dimension. The Display form
/// of `Topology` (e.g. `1n2l4c2t`) is the operator-visible
/// label that flows verbatim through clap into this Vec.
#[test]
fn parse_stats_compare_with_topology_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--topology",
        "1n2l4c2t",
        "--topology",
        "1n4l2c1t",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::Compare { topology, .. }),
            ..
        } => {
            assert_eq!(topology, vec!["1n2l4c2t", "1n4l2c1t"]);
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--work-type A --work-type B` produces a Vec with two
/// entries via `ArgAction::Append`. Mirrors the scheduler /
/// topology siblings above for the work_type dimension.
/// Hyphenated CLI flag (`--work-type`) maps to underscored
/// field name (`work_type`) per clap's default kebab-case
/// rename — pin the field-vs-flag mapping by reading from the
/// underscored field after a hyphenated invocation.
#[test]
fn parse_stats_compare_with_work_type_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--work-type",
        "SpinWait",
        "--work-type",
        "PageFaultChurn",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::Compare { work_type, .. }),
            ..
        } => {
            assert_eq!(work_type, vec!["SpinWait", "PageFaultChurn"]);
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--a-kernel-commit X --b-kernel-commit Y` populates the
/// per-side fields without touching the shared
/// `kernel_commit`. Pins the clap binding for the per-side
/// kernel-commit slicers — required for the
/// `derive_slicing_dims` path to put `KernelCommit` in the
/// slicing-dim set when the operator wants to slice by
/// kernel HEAD.
#[test]
fn parse_stats_compare_with_per_side_kernel_commit() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel-commit",
        "abc1234",
        "--b-kernel-commit",
        "def5678",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    kernel_commit,
                    a_kernel_commit,
                    b_kernel_commit,
                    ..
                }),
            ..
        } => {
            assert!(
                kernel_commit.is_empty(),
                "per-side --a-kernel-commit / --b-kernel-commit must not \
                 populate the shared --kernel-commit vec",
            );
            assert_eq!(a_kernel_commit, vec!["abc1234"]);
            assert_eq!(b_kernel_commit, vec!["def5678"]);
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `cargo ktstr stats show-host --run X` parses to
/// `StatsCommand::ShowHost { run: X, dir: None }`.
#[test]
fn parse_stats_show_host_with_run() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "show-host", "--run", "my-run-id"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ShowHost { run, dir }),
            ..
        } => {
            assert_eq!(run, "my-run-id");
            assert!(dir.is_none(), "bare --run must not populate --dir");
        }
        _ => panic!("expected Stats ShowHost"),
    }
}

/// `cargo ktstr stats show-host --run X --dir PATH` carries
/// both flags through. Same --dir threading contract as
/// `compare` — parse layer preserves the PathBuf; resolution
/// against `runs_root()` is `cli::show_run_host`'s job.
#[test]
fn parse_stats_show_host_with_dir() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "show-host",
        "--run",
        "archive-2024-01-15",
        "--dir",
        "/tmp/archived-runs",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ShowHost { run, dir }),
            ..
        } => {
            assert_eq!(run, "archive-2024-01-15");
            assert_eq!(
                dir.as_deref(),
                Some(std::path::Path::new("/tmp/archived-runs")),
            );
        }
        _ => panic!("expected Stats ShowHost"),
    }
}

/// `cargo ktstr stats show-host` WITHOUT `--run` must fail at
/// parse time — the flag is required and clap's default shape
/// says so. A regression that accidentally made `--run`
/// optional would silently let operators invoke the command
/// with no target, producing a no-op failure.
#[test]
fn parse_stats_show_host_missing_run_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "show-host"]);
    assert!(rejected.is_err(), "stats show-host must require --run",);
}

/// `cargo ktstr stats explain-sidecar --run X` parses to
/// `StatsCommand::ExplainSidecar { run: X, dir: None,
/// json: false }`. Mirrors `parse_stats_show_host_with_run`
/// for the explain-sidecar shape.
#[test]
fn parse_stats_explain_sidecar_with_run() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "explain-sidecar",
        "--run",
        "my-run-id",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ExplainSidecar { run, dir, json }),
            ..
        } => {
            assert_eq!(run, "my-run-id");
            assert!(dir.is_none(), "bare --run must not populate --dir");
            assert!(!json, "default output is text, not json");
        }
        _ => panic!("expected Stats ExplainSidecar"),
    }
}

/// `cargo ktstr stats explain-sidecar --run X --dir PATH
/// --json` carries all three flags. Same --dir threading
/// contract as `show-host`; the `--json` flag toggles the
/// aggregate-by-field output shape.
#[test]
fn parse_stats_explain_sidecar_with_dir_and_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "explain-sidecar",
        "--run",
        "archive-2024-01-15",
        "--dir",
        "/tmp/archived-runs",
        "--json",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command: Some(StatsCommand::ExplainSidecar { run, dir, json }),
            ..
        } => {
            assert_eq!(run, "archive-2024-01-15");
            assert_eq!(
                dir.as_deref(),
                Some(std::path::Path::new("/tmp/archived-runs")),
            );
            assert!(json, "--json must toggle aggregate JSON output");
        }
        _ => panic!("expected Stats ExplainSidecar"),
    }
}

/// `cargo ktstr stats explain-sidecar` WITHOUT `--run` must
/// fail at parse time. Same required-flag contract as
/// `show-host`; without it, an operator could invoke the
/// command with no target.
#[test]
fn parse_stats_explain_sidecar_missing_run_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "explain-sidecar"]);
    assert!(
        rejected.is_err(),
        "stats explain-sidecar must require --run",
    );
}

// -- try_get_matches_from: kernel list --

#[test]
fn parse_kernel_list() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_kernel_list_json() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list", "--json"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `kernel list --range R` round-trips to
/// `KernelCommand::List { range: Some(R), .. }` so the
/// dispatch site routes through `kernel_list_range_preview`
/// rather than the cache-walk path. Pins the clap binding
/// for the new `--range` flag — a regression that dropped
/// the `range` field from the Subcommand enum would surface
/// here as a parse rejection.
#[test]
fn parse_kernel_list_range() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list", "--range", "6.12..6.14"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json, range } => {
                assert!(!json, "bare --range must not enable --json");
                assert_eq!(
                    range.as_deref(),
                    Some("6.12..6.14"),
                    "--range must round-trip the literal spec for \
                     dispatch to pass to `expand_kernel_range`",
                );
            }
            other => panic!("expected KernelCommand::List, got {other:?}"),
        },
        _ => panic!("expected Kernel"),
    }
}

/// `kernel list --range R --json` round-trips both flags.
/// Pins the JSON-output mode is reachable on the range-preview
/// path (a regression that wired `--range` only on the text
/// path would surface here).
#[test]
fn parse_kernel_list_range_with_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "list",
        "--range",
        "6.12..6.14",
        "--json",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json, range } => {
                assert!(json, "--json must round-trip alongside --range");
                assert_eq!(range.as_deref(), Some("6.12..6.14"));
            }
            other => panic!("expected KernelCommand::List, got {other:?}"),
        },
        _ => panic!("expected Kernel"),
    }
}

/// `--run-source V` round-trips to `Compare { run_source: vec![V],
/// .. }`. Pins the clap binding for the shared `--run-source`
/// filter. Mirrors `parse_stats_compare_with_project_commit_single`
/// for the new dimension; per-side `--a-run-source` /
/// `--b-run-source` are covered by the `_per_side` sibling below.
#[test]
fn parse_stats_compare_with_run_source_single() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-kernel",
        "6.14",
        "--b-kernel",
        "6.15",
        "--run-source",
        "ci",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    run_source,
                    a_run_source,
                    b_run_source,
                    ..
                }),
            ..
        } => {
            assert_eq!(
                run_source,
                vec!["ci".to_string()],
                "shared --run-source must populate the shared vec",
            );
            assert!(
                a_run_source.is_empty(),
                "shared --run-source must not populate --a-run-source",
            );
            assert!(
                b_run_source.is_empty(),
                "shared --run-source must not populate --b-run-source",
            );
        }
        _ => panic!("expected Stats Compare"),
    }
}

/// `--a-run-source A --b-run-source B` round-trips to populated
/// per-side vecs with the shared `run_source` left empty. Pins
/// the per-side override path that
/// `BuildCompareFilters::build` consumes — a regression that
/// merged shared and per-side into one bucket would surface
/// here.
#[test]
fn parse_stats_compare_with_run_source_per_side() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "compare",
        "--a-run-source",
        "ci",
        "--b-run-source",
        "local",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Stats {
            command:
                Some(StatsCommand::Compare {
                    run_source,
                    a_run_source,
                    b_run_source,
                    ..
                }),
            ..
        } => {
            assert!(
                run_source.is_empty(),
                "per-side flags must not populate the shared --run-source vec",
            );
            assert_eq!(a_run_source, vec!["ci".to_string()]);
            assert_eq!(b_run_source, vec!["local".to_string()]);
        }
        _ => panic!("expected Stats Compare"),
    }
}

// -- try_get_matches_from: kernel build --

#[test]
fn parse_kernel_build_version() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "6.14.2"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_kernel_build_source() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "--source", "../linux"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// Conflict pin: `--source PATH` and the positional VERSION are
/// mutually exclusive at clap parse time. Catches a regression
/// that drops the `conflicts_with` (or its equivalent
/// `requires_ifs` shape) on the source flag and lets a contradictory
/// `--source ../linux 6.14.2` invocation flow into the dispatcher,
/// where `kernel build` would have to disambiguate a "use this
/// tree" hint from a "fetch this version" hint at runtime.
#[test]
fn parse_kernel_build_source_conflicts_with_version() {
    let result = Cargo::try_parse_from([
        "cargo", "ktstr", "kernel", "build", "--source", "../linux", "6.14.2",
    ]);
    match result {
        Ok(_) => panic!("--source + positional VERSION must be rejected at parse time"),
        Err(err) => assert_eq!(
            err.kind(),
            clap::error::ErrorKind::ArgumentConflict,
            "expected ArgumentConflict — a different ErrorKind would \
             signal that the conflicts_with attribute regressed in a way \
             the bare is_err() pin would silently mask. Full err: {err}",
        ),
    }
}

#[test]
fn parse_kernel_build_git_requires_ref() {
    let result = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--git",
        "https://example.com/linux.git",
    ]);
    match result {
        Ok(_) => panic!("--git without --ref must be rejected at parse time"),
        Err(err) => assert_eq!(
            err.kind(),
            clap::error::ErrorKind::MissingRequiredArgument,
            "expected MissingRequiredArgument — `--git` carries \
             `requires = \"git_ref\"` (clap uses the field name, not \
             the long flag name), so a regression that dropped the \
             attribute would surface as a different ErrorKind that \
             the bare is_err() pin would silently mask. Full err: {err}",
        ),
    }
}

#[test]
fn parse_kernel_build_git_with_ref() {
    let m = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--git",
        "https://example.com/linux.git",
        "--ref",
        "v6.14",
    ]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// Conflict pin: `--git URL --ref REF` and `--source PATH` are
/// mutually exclusive — git-spec triggers a clone, source-spec
/// reuses an existing tree. Both at once would either silently
/// favour one over the other or surface as an inscrutable
/// dispatcher panic; clap's `conflicts_with` pushes the
/// rejection up to parse time so the operator sees a clear
/// argument-conflict error.
#[test]
fn parse_kernel_build_git_conflicts_with_source() {
    let result = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--git",
        "https://example.com/linux.git",
        "--ref",
        "v6.14",
        "--source",
        "../linux",
    ]);
    match result {
        Ok(_) => panic!("--git + --source must be rejected at parse time"),
        Err(err) => assert_eq!(
            err.kind(),
            clap::error::ErrorKind::ArgumentConflict,
            "expected ArgumentConflict — a different ErrorKind would \
             signal that the conflicts_with attribute regressed in a way \
             the bare is_err() pin would silently mask. Full err: {err}",
        ),
    }
}

/// `kernel build VERSION --extra-kconfig PATH` round-trips to
/// `KernelCommand::Build { version: Some(..), extra_kconfig:
/// Some(..), .. }` so the dispatch site forwards the path
/// through `kernel_build` → `kernel_build_one` →
/// `cli::kernel_build_pipeline` with `Some(content)`. Pins the
/// clap binding for the new flag — a regression that dropped
/// the field would surface here as a parse rejection or a None
/// `extra_kconfig`.
#[test]
fn parse_kernel_build_with_extra_kconfig() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "6.14.2",
        "--extra-kconfig",
        "/tmp/extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel {
            command:
                KernelCommand::Build {
                    version,
                    extra_kconfig,
                    ..
                },
        } => {
            assert_eq!(version.as_deref(), Some("6.14.2"));
            assert_eq!(
                extra_kconfig,
                Some(std::path::PathBuf::from("/tmp/extra.kconfig")),
                "--extra-kconfig must round-trip the literal path",
            );
        }
        _ => panic!("expected KernelCommand::Build"),
    }
}

/// Bare `kernel build VERSION` (no `--extra-kconfig`) parses to
/// `extra_kconfig: None`. Pins that the flag is OPTIONAL — a
/// regression that made it required would fail this test.
#[test]
fn parse_kernel_build_without_extra_kconfig_is_none() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "6.14.2"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel {
            command: KernelCommand::Build { extra_kconfig, .. },
        } => {
            assert!(
                extra_kconfig.is_none(),
                "no --extra-kconfig must produce None, got {extra_kconfig:?}",
            );
        }
        _ => panic!("expected KernelCommand::Build"),
    }
}

/// Range expansion + --extra-kconfig composes at the parse
/// layer. A range version + an extra-kconfig path both round-
/// trip to their fields on `KernelCommand::Build`. The dispatch
/// then fans out per version inside `kernel_build`, and the
/// `extra_content` String is read ONCE up front and threaded as
/// `Option<&str>` to every `kernel_build_one` call — so every
/// version in a range observes byte-identical extras. Pin the
/// parse-level composition; the per-version threading is a
/// code-structure invariant of `kernel_build`'s shared read.
#[test]
fn parse_kernel_build_range_with_extra_kconfig() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "6.14.2..6.14.4",
        "--extra-kconfig",
        "/tmp/range-extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel {
            command:
                KernelCommand::Build {
                    version,
                    extra_kconfig,
                    ..
                },
        } => {
            assert_eq!(version.as_deref(), Some("6.14.2..6.14.4"));
            assert_eq!(
                extra_kconfig,
                Some(std::path::PathBuf::from("/tmp/range-extra.kconfig")),
            );
        }
        _ => panic!("expected KernelCommand::Build"),
    }
}

/// --force + --clean + --extra-kconfig orthogonality. None of
/// these flags conflict with each other; pin that all three
/// can co-exist on a single invocation. A regression that
/// introduced a clap `conflicts_with` between any pair would
/// surface here.
#[test]
fn parse_kernel_build_force_clean_and_extra_kconfig_compose() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--source",
        "../linux",
        "--force",
        "--clean",
        "--extra-kconfig",
        "/tmp/extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel {
            command:
                KernelCommand::Build {
                    force,
                    clean,
                    extra_kconfig,
                    ..
                },
        } => {
            assert!(
                force,
                "--force must round-trip alongside --clean and --extra-kconfig"
            );
            assert!(
                clean,
                "--clean must round-trip alongside --force and --extra-kconfig"
            );
            assert_eq!(
                extra_kconfig,
                Some(std::path::PathBuf::from("/tmp/extra.kconfig")),
                "--extra-kconfig must round-trip when combined with --force + --clean",
            );
        }
        _ => panic!("expected KernelCommand::Build"),
    }
}

/// Non-build subcommands that accept `--extra-kconfig` would
/// silently produce wrong cache lookups. The flag is `kernel
/// build`-only at the configuration layer; this test pins the
/// parse-level reject for the subcommands that have CLEAN clap
/// surfaces (no `trailing_var_arg` passthrough).
///
/// Subcommands and their behavior:
/// - `verifier`: REJECTS at parse time (no trailing_var_arg).
///   Pin via `try_parse_from` returning `Err`.
/// - `shell`: REJECTS at parse time (no trailing_var_arg).
///   Pin via `try_parse_from` returning `Err`.
/// - `test` / `coverage` / `llvm-cov`: PASSTHROUGH via
///   `args: Vec<String>` with `trailing_var_arg = true,
///   allow_hyphen_values = true`. Clap forwards `--extra-kconfig
///   ...` as positional args to `cargo nextest run` (or
///   `cargo llvm-cov`), which then rejects it as an unknown
///   cargo flag — but at the cargo subprocess layer, NOT at
///   parse time. This is a structural property of clap's
///   trailing-var-arg shape and is consistent across every
///   passthrough subcommand on `cargo ktstr`. We do NOT pin
///   these as parse errors because that's not where the
///   rejection actually happens.
#[test]
fn parse_extra_kconfig_rejected_on_verifier_subcommand() {
    let m = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "verifier",
        "--scheduler",
        "scx_rustland",
        "--extra-kconfig",
        "/tmp/x.kconfig",
    ]);
    assert!(
        m.is_err(),
        "--extra-kconfig must be rejected on `cargo ktstr verifier` \
         (verifier has no trailing_var_arg, so unknown flags fail at parse time)",
    );
}

#[test]
fn parse_extra_kconfig_rejected_on_shell_subcommand() {
    let m = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "shell",
        "--extra-kconfig",
        "/tmp/x.kconfig",
    ]);
    assert!(
        m.is_err(),
        "--extra-kconfig must be rejected on `cargo ktstr shell` \
         (shell has no trailing_var_arg, so unknown flags fail at parse time)",
    );
}

/// Documents the passthrough behavior on `test` /
/// `coverage` / `llvm-cov`: clap's `trailing_var_arg = true`
/// on `args: Vec<String>` SWALLOWS `--extra-kconfig` as a
/// positional argument forwarded to `cargo nextest run` /
/// `cargo llvm-cov`. The rejection happens later, at the
/// cargo subprocess layer, not at parse time. Pin the
/// shape so a future change to the trailing_var_arg shape
/// (e.g. removing it) surfaces here as a behavior change.
#[test]
fn parse_extra_kconfig_passes_through_test_subcommand_to_args_vec() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "test",
        "--extra-kconfig",
        "/tmp/x.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Test { args, .. } => {
            assert_eq!(
                args,
                vec!["--extra-kconfig", "/tmp/x.kconfig"],
                "--extra-kconfig must passthrough into `args` Vec on test \
                 subcommand (trailing_var_arg = true). The cargo nextest \
                 subprocess will reject it as an unknown flag downstream."
            );
        }
        _ => panic!("expected KtstrCommand::Test"),
    }
}

/// `--extra-kconfig` works alongside `--source` (local source
/// tree path). Pins that the flag is not mutually exclusive
/// with the other source-acquire flags — extra-kconfig is
/// orthogonal to where the kernel SOURCE comes from.
#[test]
fn parse_kernel_build_extra_kconfig_with_source() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--source",
        "../linux",
        "--extra-kconfig",
        "/tmp/extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel {
            command:
                KernelCommand::Build {
                    source,
                    extra_kconfig,
                    ..
                },
        } => {
            assert_eq!(source, Some(std::path::PathBuf::from("../linux")));
            assert_eq!(
                extra_kconfig,
                Some(std::path::PathBuf::from("/tmp/extra.kconfig")),
            );
        }
        _ => panic!("expected KernelCommand::Build"),
    }
}

// -- try_get_matches_from: kernel clean --

#[test]
fn parse_kernel_clean() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_kernel_clean_keep() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean", "--keep", "3"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Kernel {
            command: KernelCommand::Clean { keep, .. },
        } => {
            assert_eq!(keep, Some(3));
        }
        _ => panic!("expected Kernel Clean"),
    }
}

// -- try_get_matches_from: verifier --

#[test]
fn parse_verifier_with_scheduler() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "verifier", "--scheduler", "scx_rustland"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_verifier_with_scheduler_bin() {
    let m = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "verifier",
        "--scheduler-bin",
        "/tmp/sched",
    ]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_verifier_scheduler_conflicts_with_scheduler_bin() {
    let result = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "verifier",
        "--scheduler",
        "scx_rustland",
        "--scheduler-bin",
        "/tmp/sched",
    ]);
    match result {
        Ok(_) => panic!("--scheduler + --scheduler-bin must be rejected at parse time"),
        Err(err) => assert_eq!(
            err.kind(),
            clap::error::ErrorKind::ArgumentConflict,
            "expected ArgumentConflict — a different ErrorKind would \
             signal that the conflicts_with attribute regressed in a way \
             the bare is_err() pin would silently mask. Full err: {err}",
        ),
    }
}

#[test]
fn parse_verifier_all_profiles() {
    let m = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "verifier",
        "--scheduler",
        "scx_rustland",
        "--all-profiles",
    ]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_verifier_profiles_filter() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "verifier",
        "--scheduler",
        "scx_rustland",
        "--profiles",
        "default,llc,llc+steal",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Verifier { profiles, .. } => {
            assert_eq!(profiles, vec!["default", "llc", "llc+steal"]);
        }
        _ => panic!("expected Verifier"),
    }
}

// -- try_get_matches_from: completions --

#[test]
fn parse_completions_bash() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "bash"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_completions_invalid_shell() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "noshell"]);
    assert!(m.is_err());
}

// -- error cases --

#[test]
fn parse_missing_subcommand() {
    let m = Cargo::try_parse_from(["cargo", "ktstr"]);
    assert!(m.is_err());
}

#[test]
fn parse_unknown_subcommand() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "nonexistent"]);
    assert!(m.is_err());
}

// -- completions --

#[test]
fn completions_bash_non_empty() {
    let mut buf = Vec::new();
    let mut cmd = Cargo::command();
    clap_complete::generate(clap_complete::Shell::Bash, &mut cmd, "cargo", &mut buf);
    assert!(!buf.is_empty());
}

#[test]
fn completions_zsh_contains_subcommands() {
    let mut buf = Vec::new();
    let mut cmd = Cargo::command();
    clap_complete::generate(clap_complete::Shell::Zsh, &mut cmd, "cargo", &mut buf);
    let output = String::from_utf8(buf).expect("completions should be valid UTF-8");
    // clap_complete's zsh generator emits each subcommand as a
    // `'NAME:HELP'` describe-list entry (see `add_subcommands`
    // in clap_complete-4.6.1/src/aot/shells/zsh.rs:163). The
    // `'<name>:` prefix pin identifies an actual subcommand
    // completion, not an incidental substring match inside
    // rendered doc text.
    assert!(
        output.contains("'test:"),
        "zsh completions missing 'test:' describe-list entry"
    );
    assert!(
        output.contains("'coverage:"),
        "zsh completions missing 'coverage:' describe-list entry"
    );
    assert!(
        output.contains("'shell:"),
        "zsh completions missing 'shell:' describe-list entry"
    );
    assert!(
        output.contains("'kernel:"),
        "zsh completions missing 'kernel:' describe-list entry"
    );
    // `visible_alias = "nextest"` on the Test variant makes the
    // alias user-facing — clap_complete's zsh generator iterates
    // `get_visible_aliases` (zsh.rs:177) and emits a dedicated
    // describe entry per alias. A regression that dropped the
    // attribute (or silently switched to `alias` which is
    // NON-visible) would drop the entry and fail this assertion.
    assert!(
        output.contains("'nextest:"),
        "zsh completions missing 'nextest:' describe-list \
         entry (visible alias of `test`)"
    );
    // `LlvmCov` variant renders as the kebab-case `llvm-cov`
    // subcommand (clap derive default rename — see
    // clap_derive-4.6.0/src/item.rs:27 `DEFAULT_CASING =
    // CasingStyle::Kebab`). Pinned with the same `'name:`
    // prefix so an accidental doc-text match doesn't mask a
    // missing registration.
    assert!(
        output.contains("'llvm-cov:"),
        "zsh completions missing 'llvm-cov:' describe-list entry"
    );
}

// -- format_entry_row helpers --

fn test_metadata() -> KernelMetadata {
    KernelMetadata::new(
        ktstr::cache::KernelSource::Tarball,
        "x86_64".to_string(),
        "bzImage".to_string(),
        "2026-04-12T10:00:00Z".to_string(),
    )
    .with_version(Some("6.14.2".to_string()))
}

/// Store a fake kernel image and return the CacheEntry.
fn store_test_entry(cache: &CacheDir, key: &str, meta: &KernelMetadata) -> CacheEntry {
    let src = tempfile::TempDir::new().unwrap();
    let image = src.path().join(&meta.image_name);
    std::fs::write(&image, b"fake kernel").unwrap();
    cache
        .store(key, &CacheArtifacts::new(&image), meta)
        .unwrap()
}

// -- format_entry_row --
//
// The (Matches / Stale / Untracked) × (not-EOL / EOL) outcome
// matrix plus the `version == None` → "-" dash-render branch are
// pinned by `format_entry_row_renders_eol_kconfig_matrix` in
// `src/cli/kernel_list.rs` — see that test for the full case
// list. The test below covers a distinct corner the matrix does
// not: `KernelSource::Local` rendering through format_entry_row,
// since the matrix uses `Tarball` exclusively for determinism.

#[test]
fn format_entry_row_no_version() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = KernelMetadata::new(
        ktstr::cache::KernelSource::Local {
            source_tree_path: None,
            git_hash: None,
        },
        "x86_64".to_string(),
        "bzImage".to_string(),
        "2026-04-12T10:00:00Z".to_string(),
    );
    let entry = store_test_entry(&cache, "local-key", &meta);
    let row = cli::format_entry_row(&entry, "hash", &[]);
    // Anchor the dash to the version COLUMN. The row format is
    // `"  {key:<48} {version:<12} {source:<8} {arch:<7} {built_at}{tags}"`
    // (see `format_entry_row` in src/cli/kernel_list.rs). A bare
    // `row.contains("-")` would also match the `-` in the timestamp
    // `2026-04-12T10:00:00Z` even if the version dash were missing.
    // Splitting on whitespace and inspecting the second token isolates
    // the version slot — token 0 is the key, token 1 is the version.
    let tokens: Vec<&str> = row.split_whitespace().collect();
    assert!(
        tokens.len() >= 2,
        "row must have at least key + version columns: {row:?}",
    );
    assert_eq!(
        tokens[1], "-",
        "missing version must render as `-` in the version column: {row:?}",
    );
}

// Corrupt-entry formatting moved inline into the caller iteration
// in cli::kernel_list, so no test on format_entry_row covers it;
// the helper itself now takes only the valid CacheEntry shape.

// -- kconfig_status (via CacheEntry method) --

/// Companion to the stale-kconfig case in
/// `format_entry_row_renders_eol_kconfig_matrix` (in
/// `src/cli/kernel_list.rs`): that test pins the `(stale kconfig)`
/// tag emitted by `cli::format_entry_row` for a hash-mismatch entry;
/// this test pins the enum variant
/// (`KconfigStatus::Stale { cached, current }`) returned by
/// `CacheEntry::kconfig_status` that drives the tag.
#[test]
fn kconfig_status_reports_stale_on_hash_mismatch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = test_metadata().with_ktstr_kconfig_hash(Some("old".to_string()));
    let entry = store_test_entry(&cache, "stale", &meta);
    assert_eq!(
        entry.kconfig_status("new"),
        ktstr::cache::KconfigStatus::Stale {
            cached: "old".to_string(),
            current: "new".to_string(),
        }
    );
}

/// Companion to the matching-kconfig case in
/// `format_entry_row_renders_eol_kconfig_matrix` (in
/// `src/cli/kernel_list.rs`): that test pins the no-tag contract
/// emitted by `cli::format_entry_row` when the hashes agree; this
/// test pins the `KconfigStatus::Matches` variant returned by
/// `CacheEntry::kconfig_status` that drives the no-tag branch.
#[test]
fn kconfig_status_reports_matches_on_hash_equality() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = test_metadata().with_ktstr_kconfig_hash(Some("same".to_string()));
    let entry = store_test_entry(&cache, "fresh", &meta);
    assert_eq!(
        entry.kconfig_status("same"),
        ktstr::cache::KconfigStatus::Matches
    );
}

/// Companion to the untracked-kconfig case in
/// `format_entry_row_renders_eol_kconfig_matrix` (in
/// `src/cli/kernel_list.rs`): that test pins the
/// `(untracked kconfig)` tag emitted by `cli::format_entry_row`
/// when an entry has no recorded hash; this test pins the
/// `KconfigStatus::Untracked` variant returned by
/// `CacheEntry::kconfig_status` that drives the tag.
#[test]
fn kconfig_status_reports_untracked_when_entry_has_no_hash() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = test_metadata();
    let entry = store_test_entry(&cache, "no-hash", &meta);
    assert_eq!(
        entry.kconfig_status("anything"),
        ktstr::cache::KconfigStatus::Untracked
    );
}

// Corrupt entries no longer surface as CacheEntry — they are
// ListedEntry::Corrupt with no metadata-bearing struct — so
// kconfig_status isn't reachable from that state.

/// Differential pin on the three `KconfigStatus` strings that flow
/// into the `kconfig_status` field of `cargo ktstr kernel list
/// --json`. `cli::kernel_list` emits the JSON field via
/// `entry.kconfig_status(&kconfig_hash).to_string()`, so CI scripts
/// that key off the stringified variant break if any of these
/// three words changes. This test exercises the full
/// `CacheEntry::kconfig_status(..).to_string()` chain (not just
/// `KconfigStatus::<variant>.to_string()` in isolation) to pin the
/// end-to-end JSON contract in a single test covering all three
/// variants.
#[test]
fn kconfig_status_json_string_pins_all_three_variants() {
    use ktstr::cache::KconfigStatus;
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));

    let matches_meta = test_metadata().with_ktstr_kconfig_hash(Some("h".to_string()));
    let matches_entry = store_test_entry(&cache, "matches-key", &matches_meta);
    let matches_status = matches_entry.kconfig_status("h");
    assert!(
        matches!(matches_status, KconfigStatus::Matches),
        "hash equality must yield KconfigStatus::Matches"
    );
    assert_eq!(matches_status.to_string(), "matches");

    let stale_meta = test_metadata().with_ktstr_kconfig_hash(Some("old".to_string()));
    let stale_entry = store_test_entry(&cache, "stale-key", &stale_meta);
    let stale_status = stale_entry.kconfig_status("new");
    assert!(
        matches!(stale_status, KconfigStatus::Stale { .. }),
        "hash mismatch must yield KconfigStatus::Stale"
    );
    assert_eq!(stale_status.to_string(), "stale");

    let untracked_meta = test_metadata();
    let untracked_entry = store_test_entry(&cache, "untracked-key", &untracked_meta);
    let untracked_status = untracked_entry.kconfig_status("anything");
    assert!(
        matches!(untracked_status, KconfigStatus::Untracked),
        "entry without hash must yield KconfigStatus::Untracked"
    );
    assert_eq!(untracked_status.to_string(), "untracked");
}

// -- embedded_kconfig_hash --

#[test]
fn embedded_kconfig_hash_deterministic() {
    let h1 = cli::embedded_kconfig_hash();
    let h2 = cli::embedded_kconfig_hash();
    assert_eq!(h1, h2);
}

#[test]
fn embedded_kconfig_hash_is_hex() {
    let h = cli::embedded_kconfig_hash();
    assert_eq!(h.len(), 8, "CRC32 hex should be 8 chars");
    assert!(
        h.chars().all(|c| c.is_ascii_hexdigit()),
        "should be hex digits: {h}"
    );
}

#[test]
fn embedded_kconfig_hash_matches_manual_crc32() {
    let expected = format!("{:08x}", crc32fast::hash(cli::EMBEDDED_KCONFIG.as_bytes()));
    assert_eq!(cli::embedded_kconfig_hash(), expected);
}

// -- show-host --

/// `cargo ktstr show-host` parses with no arguments and maps to
/// the `ShowHost` variant.
#[test]
fn parse_show_host_minimal() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "show-host"]).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(k.command, KtstrCommand::ShowHost));
}

/// A stray positional argument on `show-host` must be rejected at
/// parse time (clap default) so a typo like
/// `cargo ktstr show-host host_context` fails loudly instead of
/// silently looking like success.
#[test]
fn parse_show_host_rejects_positional() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-host", "stray"]);
    assert!(
        rejected.is_err(),
        "show-host must reject positional arguments",
    );
}

/// `cargo ktstr show-thresholds <test>` parses with exactly one
/// positional argument and maps to the `ShowThresholds` variant
/// carrying the test name. Missing argument rejected at parse
/// time; extra argument rejected too. Pins the arg count so a
/// future variadic refactor surfaces here.
#[test]
fn parse_show_thresholds_with_test_arg() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds", "my_test_fn"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::ShowThresholds { test } => {
            assert_eq!(test, "my_test_fn");
        }
        _ => panic!("expected ShowThresholds"),
    }
}

/// `show-thresholds` without the test-name argument must fail
/// at parse time — the positional is required.
#[test]
fn parse_show_thresholds_without_arg_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds"]);
    assert!(
        rejected.is_err(),
        "show-thresholds requires a test-name argument",
    );
}

/// `show-thresholds <a> <b>` is rejected — variadic inputs would
/// silently drop the second arg or reinterpret it as a flag.
#[test]
fn parse_show_thresholds_extra_arg_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds", "a", "b"]);
    assert!(
        rejected.is_err(),
        "show-thresholds must accept exactly one positional arg",
    );
}

/// `cli::show_host` produces a non-empty report under normal
/// Linux CI conditions. Catches a regression in the underlying
/// `HostContext::format_human` (e.g. a panic in the
/// destructuring bind that surfaces every field) before the
/// ShowHost dispatch arm reaches it. Named without a
/// `dispatch_` prefix because this exercises the leaf helper
/// directly; true dispatch-path coverage lives in the parse
/// tests above + the binary's `main` call.
#[test]
fn show_host_helper_produces_non_empty_output() {
    let out = cli::show_host();
    assert!(
        !out.is_empty(),
        "show_host must return a non-empty report under normal Linux CI",
    );
    // Stronger pin: `HostContext::format_human` always includes
    // `kernel_release` even when most other fields are `None`
    // (uname is a syscall, filesystem-independent). Asserting
    // the stable field name catches a regression that returned
    // a non-empty but garbage report (e.g. only comments).
    assert!(
        out.contains("kernel_release"),
        "show_host output must include the stable `kernel_release` row: {out}",
    );
}

/// `cli::show_thresholds` returns `Err` with the actionable
/// "no registered ktstr test named" diagnostic when called with
/// an unknown test name. Named without a `dispatch_` prefix for
/// the same reason as `show_host_helper_produces_non_empty_output`
/// — this exercises the leaf helper, not the dispatch path
/// wrapping it.
#[test]
fn show_thresholds_helper_unknown_test_returns_error() {
    let err = cli::show_thresholds("definitely_not_a_registered_test_xyz").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no registered ktstr test named"),
        "error path must preserve the actionable diagnostic: {msg}",
    );
}

// -- clap argument-parse pins: Shell --cpu-cap requires --no-perf-mode
//
// `#[arg(long, requires = "no_perf_mode", ...)]` on the
// Shell subcommand's `cpu_cap` field enforces the constraint
// that --cpu-cap is only meaningful in no-perf-mode (perf-mode
// already holds every LLC exclusively, so capping under
// perf-mode would double-reserve). These tests pin the
// invariant so a future refactor that drops or renames the
// `requires` attribute trips a unit-test regression instead of
// surfacing as a runtime double-reservation conflict.

/// `cargo ktstr shell --cpu-cap 4 --no-perf-mode` parses
/// successfully with both flags set. Pins the positive path of
/// the `requires = "no_perf_mode"` constraint — the happy-path
/// invocation an operator would type.
#[test]
fn parse_shell_cpu_cap_with_no_perf_mode_succeeds() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "shell",
        "--cpu-cap",
        "4",
        "--no-perf-mode",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Shell {
            cpu_cap,
            no_perf_mode,
            ..
        } => {
            assert_eq!(cpu_cap, Some(4));
            assert!(no_perf_mode, "--no-perf-mode must be set");
        }
        _ => panic!("expected Shell"),
    }
}

/// `cargo ktstr shell --cpu-cap 4` without `--no-perf-mode`
/// must FAIL at parse time because of the `requires =
/// "no_perf_mode"` constraint. Pins the negative path: if
/// the constraint is ever dropped, this test fails so the
/// regression can't reach production where it would cause a
/// silent double-reservation under perf-mode.
#[test]
fn parse_shell_cpu_cap_without_no_perf_mode_fails() {
    // `Cargo` intentionally has no Debug derive, so unwrap
    // helpers that format the Ok variant are unavailable.
    // Match on Err directly to extract the clap error.
    let msg = match Cargo::try_parse_from(["cargo", "ktstr", "shell", "--cpu-cap", "4"]) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("--cpu-cap without --no-perf-mode must fail the parse"),
    };
    // clap renders "the following required arguments were not provided"
    // or similar; lowercase + substring-match is lenient against
    // clap version-to-version message tweaks while still proving
    // the constraint fired.
    assert!(
        msg.to_ascii_lowercase().contains("no-perf-mode")
            || msg.to_ascii_lowercase().contains("no_perf_mode"),
        "clap error must name the missing --no-perf-mode flag, got: {msg}",
    );
}

/// `cargo ktstr shell --no-perf-mode` without `--cpu-cap`
/// parses successfully with `cpu_cap: None`. Pins the shape of
/// the unset sentinel (expanded to the 30%-of-allowed default by
/// the planner) — a user who wants --no-perf-mode with the
/// implicit default must still be able to invoke the shell. A
/// regression that tied --cpu-cap to --no-perf-mode
/// bidirectionally would fail here.
#[test]
fn parse_shell_no_perf_mode_without_cpu_cap_succeeds() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--no-perf-mode"])
        .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Shell {
            cpu_cap,
            no_perf_mode,
            ..
        } => {
            assert_eq!(cpu_cap, None, "no --cpu-cap must produce None");
            assert!(no_perf_mode);
        }
        _ => panic!("expected Shell"),
    }
}

// ---------------------------------------------------------------
// KERNEL_LIST_LONG_ABOUT — range-mode JSON schema discoverability
// ---------------------------------------------------------------
//
// `cargo ktstr kernel list --range R --json` emits a
// structurally-different JSON shape from the cache-walk mode:
// four top-level fields (`range`, `start`, `end`, `versions`)
// with no cache metadata. The help copy is the
// discoverability contract for scripted consumers — without a
// unit-test pin, a JSON emitter that adds, renames, or removes
// a range-mode field could ship without a matching help update
// and silently break dispatch-on-key consumers. The sibling
// `kernel_list_long_about_exposes_json_schema` test in
// `src/cli/kernel_cmd.rs` covers cache-walk mode; this companion
// fills the range-mode gap from the cargo-ktstr binary's
// perspective and exercises the same `pub const` re-exported
// through `ktstr::cli::KERNEL_LIST_LONG_ABOUT`.

/// Pins that every range-mode JSON top-level field name appears
/// in the help copy by its column-aligned row. Range-mode emits
/// `{ range, start, end, versions }` per the schema block in
/// `KERNEL_LIST_LONG_ABOUT` (`src/cli/kernel_cmd.rs`). Each field
/// is pinned against its column-aligned row prefix (e.g.
/// `  range     literal`) rather than the bare word, since
/// `start` / `end` / `range` appear elsewhere in the help copy
/// (e.g. "parsed start endpoint", "the inclusive range") and a
/// bare-word substring would match the prose, masking a regression
/// that dropped the actual schema row.
///
/// Co-update contract: when the JSON schema changes (field
/// added, renamed, removed, or its emission site moves), three
/// updates land in the same commit:
///   1. the JSON emitter — `cli::kernel_list` /
///      `kernel_list_range_preview` in `src/cli/kernel_list.rs`,
///   2. the help-copy schema block — `KERNEL_LIST_LONG_ABOUT`
///      in `src/cli/kernel_cmd.rs` (the column-aligned table
///      this test reads), and
///   3. this test's column-aligned assertions.
/// Updating any one without the others either silently breaks
/// scripted consumers (1 without 2) or surfaces a misleading
/// stale assertion (2 without 3).
#[test]
fn kernel_list_long_about_exposes_range_mode_json_keys() {
    let about = ktstr::cli::KERNEL_LIST_LONG_ABOUT;
    // Column-aligned rows from kernel_cmd.rs's range-mode schema
    // block — each begins with two spaces, the field name, and
    // padding to the description column. Pinning against this
    // exact prefix shape rejects matches inside surrounding prose.
    assert!(
        about.contains("  range     literal"),
        "KERNEL_LIST_LONG_ABOUT must carry the `range` row from the \
         range-mode schema block: got: {about:?}",
    );
    assert!(
        about.contains("  start     parsed start endpoint"),
        "KERNEL_LIST_LONG_ABOUT must carry the `start` row from the \
         range-mode schema block: got: {about:?}",
    );
    assert!(
        about.contains("  end       parsed end endpoint"),
        "KERNEL_LIST_LONG_ABOUT must carry the `end` row from the \
         range-mode schema block: got: {about:?}",
    );
    assert!(
        about.contains("  versions  array of resolved version strings"),
        "KERNEL_LIST_LONG_ABOUT must carry the `versions` row from the \
         range-mode schema block: got: {about:?}",
    );
    // The help copy must explicitly distinguish range-mode from
    // cache-walk-mode by mentioning that the range-mode shape
    // "never carries cache metadata" (the dispatch-on-key contract).
    assert!(
        about.contains("Range-mode output never carries cache metadata"),
        "KERNEL_LIST_LONG_ABOUT must call out the `Range-mode output \
         never carries cache metadata` contract so scripted consumers \
         know to dispatch on the presence of the `range` key versus \
         the `entries` key: got: {about:?}",
    );
    assert!(
        about.contains("--range"),
        "KERNEL_LIST_LONG_ABOUT must reference the `--range` flag \
         so a `kernel list --help` reader sees the range-mode \
         entry point: got: {about:?}",
    );
    // The exact phrase from kernel_cmd.rs:416 splits across a
    // line break (`...range-preview\nmode...`), so pin the
    // unambiguous hyphenated token directly. Plain "range mode"
    // also appears in surrounding prose (e.g. help text — see
    // `the `range` key (range mode) versus `entries` key (list mode)`
    // at kernel_cmd.rs:437) so a disjunction would re-introduce
    // false-positive risk.
    assert!(
        about.contains("range-preview"),
        "KERNEL_LIST_LONG_ABOUT must use the `range-preview` term so \
         scripted consumers know to dispatch on the presence of the \
         `range` key: got: {about:?}",
    );
}
