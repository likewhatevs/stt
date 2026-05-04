//! `cargo ktstr completions` — emit shell completion script.
//!
//! Wraps [`clap_complete::generate`] against the [`Cargo`] root parser
//! so the user can pipe the output into their shell's completion
//! directory (e.g. `cargo ktstr completions bash > /usr/share/...`).

use clap::CommandFactory;

use crate::cli::Cargo;

/// Generate a shell-completion script for `cargo ktstr` and write it
/// to stdout. `binary` is the leading word the completions register
/// against — `cargo` for the cargo-subcommand wrapping, anything
/// else for direct invocation.
pub(crate) fn run_completions(shell: clap_complete::Shell, binary: &str) {
    let mut cmd = Cargo::command();
    clap_complete::generate(shell, &mut cmd, binary, &mut std::io::stdout());
}
