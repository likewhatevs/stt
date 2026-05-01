//! E2E integration test: real kernel build with `--extra-kconfig`
//! overriding a baked-in symbol must fail at `validate_kernel_config`
//! and skip the cache store.
//!
//! # What this test pins
//!
//! 1. The merge path (`merge_kconfig_fragments`) gives last-wins
//!    precedence to the user fragment, so `# CONFIG_BPF_SYSCALL is
//!    not set` defeats the baked-in `CONFIG_BPF_SYSCALL=y`.
//! 2. `validate_kernel_config` runs AFTER the build and bails with
//!    an actionable error naming `CONFIG_BPF_SYSCALL`. The error is
//!    wrapped with a hint pointing at `--extra-kconfig` because the
//!    extras flag was passed.
//! 3. The cache store is skipped on this failure path: the build
//!    pipeline returns `Err` before reaching `cache.store(...)`, so
//!    no entry lands under the cache key.
//!
//! # Why `#[ignore]`
//!
//! A real kernel build takes 2–10 minutes on a typical workstation
//! and requires `make`, `gcc`, `pahole`, `flex`, `bison`, plus a
//! linux source tree at `../linux`. Running this test in the
//! standard nextest pass would balloon CI time and gate every PR
//! on the build toolchain. The test is therefore
//! `#[ignore = "..."]` and only invoked via `cargo ktstr test`'s
//! gauntlet level (or explicit `cargo nextest run --run-ignored
//! all extra_kconfig_e2e_validate_rejects_disabled_bpf_syscall`).
//!
//! # Prerequisites
//!
//! - `../linux` exists and is a configured kernel source tree
//!   (any version that ktstr's baked-in `ktstr.kconfig` targets).
//! - Build toolchain installed: `make`, `gcc`, `binutils`,
//!   `pahole >= 1.16` (`dwarves` package), `flex`, `bison`,
//!   `libelf-dev`, `libssl-dev`.
//! - `KTSTR_CACHE_DIR` (or default `~/.cache/ktstr/kernels/...`)
//!   is writable. The test creates a temp `.kconfig` fragment under
//!   `tempfile::tempdir()` and removes it on teardown.
//!
//! # Manual reproduction
//!
//! ```bash
//! mkdir -p /tmp/extras
//! printf '# CONFIG_BPF_SYSCALL is not set\n' > /tmp/extras/bad.kconfig
//! cargo ktstr kernel build \
//!     --source ../linux \
//!     --extra-kconfig /tmp/extras/bad.kconfig \
//!     --force
//! # → exit 1, stderr contains "CONFIG_BPF_SYSCALL not set"
//! ```

use std::path::PathBuf;
use std::process::Command;

/// Path to the linux source tree used by the E2E test.
///
/// Resolved at runtime (the binary doesn't move between compile and
/// run) so a missing tree produces a clean skip rather than a
/// `compile_error!`. The team-lead-issued spec uses `../linux`
/// relative to the ktstr crate root.
fn linux_source_dir() -> PathBuf {
    let crate_root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(crate_root).join("..").join("linux")
}

/// Path to the cargo-ktstr binary cargo built for this test pass.
/// `CARGO_BIN_EXE_<name>` is set at compile time for every `[[bin]]`
/// the workspace declares, so the test resolves the absolute path
/// without shelling out to `which cargo-ktstr`.
const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");

/// Build a kernel with a fragment that disables `CONFIG_BPF_SYSCALL`
/// (a baked-in `=y` from `EMBEDDED_KCONFIG`) and prove the post-
/// build validator catches it. Pins three facets of the
/// `--extra-kconfig` pipeline:
///
/// 1. **Last-wins merge.** `merge_kconfig_fragments` interleaves the
///    user fragment AFTER `EMBEDDED_KCONFIG`, so olddefconfig
///    propagates the user's `# CONFIG_BPF_SYSCALL is not set` over
///    the baked-in `CONFIG_BPF_SYSCALL=y`. Without last-wins, this
///    test would not reach the validator's failure path.
/// 2. **Validator catches the baked-in disablement.** After build,
///    `validate_kernel_config` reads `.config` and looks for
///    `CONFIG_BPF_SYSCALL=y`. The user fragment removed it, so the
///    validator must bail. The error message names
///    `CONFIG_BPF_SYSCALL` and includes the actionable hint
///    "required for BPF program loading".
/// 3. **Cache store is skipped.** `kernel_build_pipeline` calls
///    `validate_kernel_config` BEFORE `cache.store(...)`. A failed
///    validation returns `Err` from the pipeline, so the broken
///    kernel image never lands in the cache directory. (Asserted
///    indirectly: the subprocess exits non-zero, and a successful
///    `cargo ktstr kernel build` would have exited 0.)
///
/// The test runs via `[[bin]] cargo-ktstr` and its `kernel build`
/// subcommand — exercising the same dispatch path the user invokes.
/// `--force` defeats any prior cache hit on this version+extras
/// pair, ensuring every run actually reaches the build phase.
#[test]
#[ignore = "long-running E2E (real kernel build, 2-10 min); run via \
            `cargo nextest run --run-ignored all` or `cargo ktstr test` \
            gauntlet level. Requires ../linux and full kernel build \
            toolchain (gcc, make, pahole >= 1.16, flex, bison)."]
fn extra_kconfig_e2e_validate_rejects_disabled_bpf_syscall() {
    let source = linux_source_dir();
    assert!(
        source.is_dir(),
        "../linux source tree missing — see test doc for prerequisites; expected: {}",
        source.display(),
    );

    // Write the malicious fragment to a tempdir so the test cleans
    // up on teardown. Single line: disable a baked-in invariant
    // (`CONFIG_BPF_SYSCALL`). The kbuild last-wins merge ensures
    // this line wins over `EMBEDDED_KCONFIG`'s `=y`.
    let tmp = tempfile::tempdir().expect("create tempdir for kconfig fragment");
    let fragment_path = tmp.path().join("disable_bpf.kconfig");
    std::fs::write(&fragment_path, "# CONFIG_BPF_SYSCALL is not set\n")
        .expect("write malicious kconfig fragment");

    // Spawn `cargo-ktstr kernel build --source ../linux
    // --extra-kconfig <tempfile> --force`. The `--force` defeats the
    // cache lookup so every run reaches the configure + build +
    // validate pipeline.
    let output = Command::new(CARGO_KTSTR_BINARY)
        .arg("ktstr") // cargo-ktstr is invoked via cargo's "cargo-X" -> "X" arg shim
        .arg("kernel")
        .arg("build")
        .arg("--source")
        .arg(&source)
        .arg("--extra-kconfig")
        .arg(&fragment_path)
        .arg("--force")
        .output()
        .expect("spawn cargo-ktstr kernel build");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}");

    assert!(
        !output.status.success(),
        "kernel build with disabled CONFIG_BPF_SYSCALL must fail; got exit={:?}\n{combined}",
        output.status.code(),
    );

    // The validator's diagnostic must name CONFIG_BPF_SYSCALL — that
    // string is the load-bearing identifier letting the operator
    // pinpoint which baked-in invariant their fragment defeated.
    assert!(
        stderr.contains("CONFIG_BPF_SYSCALL"),
        "validator diagnostic must name CONFIG_BPF_SYSCALL:\n{combined}",
    );
    // The hint registered in `VALIDATE_CONFIG_CRITICAL` must reach
    // the operator. Pins that the curated hint table is rendered,
    // not just the raw config name.
    assert!(
        stderr.contains("required for BPF program loading"),
        "validator diagnostic must include the actionable hint from \
         VALIDATE_CONFIG_CRITICAL:\n{combined}",
    );
    // The post-build wrap context names `--extra-kconfig` so the
    // operator knows which input is the likely cause.
    assert!(
        stderr.contains("--extra-kconfig") || stderr.contains("extra-kconfig"),
        "post-build wrap context must point at --extra-kconfig as the \
         likely cause:\n{combined}",
    );
}
