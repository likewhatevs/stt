//! `cargo ktstr verifier` subcommand placeholder.
//!
//! The prior dispatch path (Stage C) used `--scheduler` /
//! `--scheduler-bin` to point at a scheduler binary and spawned it
//! with `--ktstr-list-schedulers` for metadata discovery. That design
//! was structurally wrong: scheduler binaries under test are not
//! `ktstr-test-support`-linked and won't intercept the flag — at best
//! the dispatch silently falls back to a single-cell run, at worst
//! (host with CAP_BPF + sched_ext kernel) the scheduler binary
//! attaches as the system scheduler and hijacks the host.
//!
//! Replacement design: each `tests/*.rs` integration test crate that
//! contains `declare_scheduler!` declarations emits verifier cells
//! (one per declared scheduler × declared kernel × accepted gauntlet
//! preset) as nextest test names like
//! `verifier/<sched>/<kernel>/<preset>`. `cargo ktstr verifier` will
//! invoke `cargo nextest run` with a verifier-prefix filter; nextest
//! gets per-cell parallelism, retries, and failure isolation for free
//! — same pattern as `cargo ktstr test`'s gauntlet listing in
//! `src/test_support/dispatch.rs::list_tests_all`.
//!
//! The cell enumerator + cell handler still need to land; this stub
//! prints a deliberate diagnostic until then so a user invoking
//! `cargo ktstr verifier` sees the regression explicitly instead of
//! a partial sweep.

/// Stub dispatch for the verifier subcommand.
pub(crate) fn run_verifier(_kernel: Vec<String>, _raw: bool) -> Result<(), String> {
    Err(
        "cargo ktstr verifier: nextest-based cell sweep is pending; \
         the prior `--scheduler` / `--scheduler-bin` dispatch was removed \
         because it required scheduler binaries to link ktstr-test-support \
         (architecturally wrong — and unsafe on hosts with sched_ext where \
         the unmodified scheduler binary would attach and hijack the host). \
         The replacement walks each test binary's `declare_scheduler!` \
         entries and emits one nextest test per cell."
            .to_string(),
    )
}
