//! Smaller `cargo ktstr` subcommand dispatchers.
//!
//! Houses one helper per subcommand whose implementation is too
//! small to warrant its own module: `shell`, `completions`,
//! `funify`, `model {fetch,status,clean}`, and `export`. The
//! `--kernel` resolution shim used by `shell` (and re-used by
//! the verifier subcommand) lives in [`super::kernel`].

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::CommandFactory;

use ktstr::cli;

use crate::cli::Cargo;
use crate::kernel::resolve_kernel_image;

/// Dispatch the `cargo ktstr shell` subcommand: launch a KVM VM and
/// drop into a busybox shell inside the guest.
///
/// `--cpu-cap` requiring `--no-perf-mode` is enforced at clap parse
/// time via `requires = "no_perf_mode"` on the Shell variant in
/// [`crate::cli`]; this dispatcher trusts the parser's gate and only
/// re-validates the env-level conflict (`KTSTR_BYPASS_LLC_LOCKS=1`)
/// that clap cannot see.
///
/// Both `unsafe std::env::set_var` calls below run on the call chain
/// `main` → match arm → `run_shell` with no thread spawn anywhere —
/// the binary's `tokio` features (`rt` only, no `rt-multi-thread`)
/// rule out runtime-spawned worker threads, and no helper here calls
/// `thread::spawn` before the env write. The single-threaded
/// invariant the SAFETY comments rely on holds for the lifetime of
/// these writes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_shell(
    kernel: Option<String>,
    topology: String,
    include_files: Vec<PathBuf>,
    memory_mb: Option<u32>,
    dmesg: bool,
    exec: Option<String>,
    no_perf_mode: bool,
    cpu_cap: Option<usize>,
    disk: Option<String>,
) -> Result<(), String> {
    if no_perf_mode {
        // SAFETY: single-threaded at this point — main → dispatch →
        // run_shell, no prior thread spawn (ktstr's tokio feature set
        // is `rt` only, no `rt-multi-thread`; no helper above this
        // point calls `thread::spawn`). No concurrent env readers
        // exist, so set_var is sound.
        unsafe { std::env::set_var("KTSTR_NO_PERF_MODE", "1") };
    }
    if let Some(cap) = cpu_cap {
        // Env-level conflict with KTSTR_BYPASS_LLC_LOCKS=1. The
        // CLI-level `--cpu-cap` requires `--no-perf-mode` rule is
        // enforced at clap parse time via `requires = "no_perf_mode"`
        // on the Shell variant in crate::cli, not here.
        if std::env::var("KTSTR_BYPASS_LLC_LOCKS")
            .ok()
            .is_some_and(|v| !v.is_empty())
        {
            return Err(
                "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
                 --cpu-cap is a resource contract; bypass disables the contract entirely."
                    .to_string(),
            );
        }
        // Validate early so a bad cap surfaces at CLI-parse time.
        cli::CpuCap::new(cap).map_err(|e| format!("{e:#}"))?;
        // SAFETY: single-threaded at this point per the chain
        // documented on the function — no concurrent env readers.
        unsafe { std::env::set_var("KTSTR_CPU_CAP", cap.to_string()) };
    }
    // Parse the human-readable disk size into a DiskConfig before the
    // KVM probe so a bad string surfaces at CLI-argument time, not
    // mid-VM-setup. `parse_disk_arg` returns `Ok(None)` when the
    // attribute is absent and applies `DiskConfig::default()` for
    // every knob except `capacity_mb` when present.
    let disk_cfg = cli::parse_disk_arg(disk.as_deref()).map_err(|e| format!("{e:#}"))?;
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    let (numa_nodes, llcs, cores, threads) =
        cli::parse_topology_string(&topology).map_err(|e| format!("{e:#}"))?;

    let resolved_includes =
        cli::resolve_include_files(&include_files).map_err(|e| format!("{e:#}"))?;

    let include_refs: Vec<(&str, &Path)> = resolved_includes
        .iter()
        .map(|(a, p)| (a.as_str(), p.as_path()))
        .collect();

    ktstr::run_shell(
        kernel_path,
        numa_nodes,
        llcs,
        cores,
        threads,
        &include_refs,
        memory_mb,
        dmesg,
        exec.as_deref(),
        disk_cfg,
    )
    .map_err(|e| format!("{e:#}"))
}

pub(crate) fn run_completions(shell: clap_complete::Shell, binary: &str) {
    let mut cmd = Cargo::command();
    clap_complete::generate(shell, &mut cmd, binary, &mut std::io::stdout());
}

/// `cargo ktstr funify <input>` — read a JSON dump, replace every
/// non-metric value (per
/// [`ktstr::fun::Funifier::is_metric_passthrough`](ktstr::fun::Funifier::is_metric_passthrough))
/// with `adjective-animal` petnames, write the result to stdout.
///
/// `seed.is_some()` makes the mapping deterministic across
/// invocations of this binary so a user running `funify` twice on
/// the same dump gets identical fun names (and can correlate fun
/// names across two related dumps). `seed.is_none()` derives a
/// process-fresh ephemeral key — every invocation produces a
/// different fun name for the same input.
///
/// Errors are returned as `String` (matching the existing
/// `Result<(), String>` shape every other run_* helper uses), so
/// the dispatch site's `error: {e:#}` formatter handles them
/// uniformly.
pub(crate) fn run_funify(
    input: Option<PathBuf>,
    seed: Option<String>,
    pretty: bool,
) -> Result<(), String> {
    use std::io::Read;

    // Read input: stdin when no path is given OR when path is the
    // explicit "-" sentinel; otherwise read the file at `input`.
    let from_stdin = match input.as_deref() {
        None => true,
        Some(p) => p.as_os_str() == "-",
    };
    let raw = if from_stdin {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("read stdin: {e}"))?;
        s
    } else {
        let p = input.as_deref().unwrap();
        std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?
    };

    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse JSON input: {e}"))?;

    let funifier = match seed {
        Some(s) => ktstr::fun::Funifier::with_seed(&s),
        None => ktstr::fun::Funifier::ephemeral(),
    };
    let funified = ktstr::fun::funify_json(value, &funifier);

    let out = if pretty {
        serde_json::to_string_pretty(&funified)
    } else {
        serde_json::to_string(&funified)
    }
    .map_err(|e| format!("serialize funified JSON: {e}"))?;
    println!("{out}");
    Ok(())
}

/// `cargo ktstr model fetch` — download + SHA-check the default model
/// into the user's cache. Wraps `ktstr::test_support::ensure` with a
/// human-readable progress line; the status is printed after so
/// users can see the final cache path regardless of whether the
/// fetch did any work.
pub(crate) fn run_model_fetch() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    match ktstr::test_support::ensure(&spec) {
        Ok(path) => {
            println!(
                "ktstr: model '{}' ready at {}",
                spec.file_name,
                path.display()
            );
            Ok(())
        }
        Err(e) => Err(format!("fetch model '{}': {e:#}", spec.file_name)),
    }
}

/// `cargo ktstr model status` — report the cache path and whether a
/// SHA-checked copy of the default model is already present.
pub(crate) fn run_model_status() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    let status = ktstr::test_support::status(&spec).map_err(|e| format!("{e:#}"))?;
    println!("model:    {}", status.spec.file_name);
    println!("path:     {}", status.path.display());
    println!("cached:   {}", status.sha_verdict.is_cached());
    println!("checked:  {}", status.sha_verdict.is_match());
    // Distinguish the four verdict variants so each gets a
    // remediation-specific line: absent cache, I/O failure during
    // the SHA check, successful hash that didn't match, and the
    // all-clear case (no annotation needed). An I/O failure points
    // at the filesystem entry (permissions, truncation); a mismatch
    // points at the bytes themselves.
    // "Re-fetch to replace it" is the shared remediation tail for
    // every non-Matches cached-file branch (both CheckFailed and
    // Mismatches land on the same operator action — the cause
    // differs but the fix does not). Factoring the tail into one
    // string keeps the two arms in lock-step so a wording change
    // lands in both places by construction.
    const RE_FETCH_TAIL: &str = "re-fetch to replace it";
    match &status.sha_verdict {
        ktstr::test_support::ShaVerdict::NotCached => println!(
            "(no cached copy — run `cargo ktstr model fetch` to download {} MiB)",
            status.spec.size_bytes / 1024 / 1024,
        ),
        ktstr::test_support::ShaVerdict::CheckFailed(err) => {
            // Defensively collapse any embedded `\n` into `; `
            // before placing `err` inside the "(single
            // parenthesized note)" wrapper. The alternate
            // anyhow format (`{e:#}`) that produced `err` joins
            // causes with `: ` and is single-line in practice;
            // this replace is a guard against a future error
            // source whose Display impl injects its own
            // newlines (std::io errors wrapping multi-line OS
            // messages, third-party crates formatting
            // call-chain trees). Keeping the output on one
            // line preserves the visual grouping the other
            // match arms use.
            let single_line = err.replace('\n', "; ");
            println!(
                "(cached file could not be checked: {single_line}; \
                 inspect the cache entry or {RE_FETCH_TAIL})",
            );
        }
        ktstr::test_support::ShaVerdict::Mismatches => {
            println!("(cached file failed SHA-256 check; {RE_FETCH_TAIL})",);
        }
        ktstr::test_support::ShaVerdict::Matches => {}
    }
    Ok(())
}

/// `cargo ktstr model clean` — remove the cached GGUF artifact and
/// its `.mtime-size` sidecar. Wraps
/// `ktstr::test_support::clean` and renders the resulting
/// [`ktstr::test_support::CleanReport`] as a per-file deletion
/// log plus a total-freed summary line. Sizes pass through
/// `indicatif::HumanBytes` for IEC-prefixed rendering ("2.34 GiB",
/// "52 B") consistent with the rest of the model surface.
///
/// Empty-cache case (neither file present) prints a single
/// "no cached model found" line so an idempotent re-run after a
/// successful clean produces a clear "nothing to do" outcome
/// rather than two "(absent)" lines.
pub(crate) fn run_model_clean() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    let report =
        ktstr::test_support::clean(&spec).map_err(|e| format!("clean model cache: {e:#}"))?;
    if report.is_empty() {
        println!(
            "no cached model found at {}",
            report.artifact_path.display()
        );
        return Ok(());
    }
    if let Some(bytes) = report.artifact_freed_bytes {
        println!(
            "removed {} ({})",
            report.artifact_path.display(),
            indicatif::HumanBytes(bytes),
        );
    }
    if let Some(bytes) = report.sidecar_freed_bytes {
        println!(
            "removed {} ({})",
            report.sidecar_path.display(),
            indicatif::HumanBytes(bytes),
        );
    }
    println!(
        "freed {} total",
        indicatif::HumanBytes(report.total_freed_bytes()),
    );
    Ok(())
}

/// Route `cargo ktstr export <NAME>` to the test binary that owns
/// the named `#[ktstr_test]` registration. cargo-ktstr cannot embed
/// itself into the .run file because it has no `#[ktstr_test]`
/// entries from the user's crate — only the test binary that
/// links against the user's code carries the registry. The router
/// builds every workspace test binary, then exec's each in turn
/// with `--ktstr-export-test=NAME`. The first binary that exits 0
/// wins; all-fail surfaces the per-binary stderrs so the operator
/// can see why the lookup missed (typically: typoed test name).
///
/// `package` (when `Some`) restricts the build via
/// `cargo build --tests --package <NAME>` — necessary in
/// multi-package workspaces where a test name might exist in
/// multiple packages and the operator wants a deterministic
/// resolution.
///
/// `release: true` builds with `--release` so the embedded test
/// binary matches the profile the operator is running. Mismatched
/// profiles can produce a `.run` whose embedded binary's threshold
/// behavior differs from the test runs the operator is reproducing
/// from.
pub(crate) fn run_export(
    test: String,
    output: Option<PathBuf>,
    package: Option<String>,
    release: bool,
) -> Result<(), String> {
    let bins = build_test_binaries(package.as_deref(), release)?;
    if bins.is_empty() {
        return Err("cargo build --tests produced no executable artifacts; \
             ensure the workspace has at least one [[test]] target or \
             a [lib]/[bin] with #[cfg(test)] tests"
            .to_string());
    }

    // Track per-candidate stderr by exit-code category so we can
    // surface the most useful diagnostic on full miss. Exit 2 means
    // "the test exists in this binary but was rejected by export"
    // (host_only, bpf_map_write, KernelBuiltin, or I/O failure) —
    // ALWAYS the most informative outcome, since every other
    // candidate's exit-1 "not registered" message is uninformative
    // when the test actually exists somewhere.
    let mut rejection_stderr: Option<String> = None;
    let mut last_miss_stderr = String::new();
    for bin in &bins {
        let mut cmd = Command::new(bin);
        cmd.arg(format!("--ktstr-export-test={test}"));
        if let Some(o) = output.as_deref() {
            // Resolve relative paths against cwd before forwarding so
            // the test binary writes to the operator's pwd, not its
            // own (the binary lives under target/debug/deps/...).
            let abs = if o.is_absolute() {
                o.to_path_buf()
            } else {
                std::env::current_dir()
                    .map_err(|e| format!("resolve cwd for --output: {e}"))?
                    .join(o)
            };
            cmd.arg(format!("--ktstr-export-output={}", abs.display()));
        }
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::inherit())
            // Capture stderr so per-candidate "no registered test
            // named X" diagnostics don't spam the operator's terminal
            // for every binary we try. We forward the WINNING
            // binary's stderr (carrying export_test's "wrote ..."
            // confirmation) below; on full miss we surface the
            // exit-2 candidate's stderr (rejection reason) when one
            // exists, falling back to the last exit-1 stderr.
            .stderr(std::process::Stdio::piped());

        let out = cmd
            .output()
            .map_err(|e| format!("exec {}: {e}", bin.display()))?;
        if out.status.success() {
            // Forward the winner's stderr so the "wrote ..." line
            // (and any operator-visible diagnostics) reach the
            // user's terminal.
            std::io::Write::write_all(&mut std::io::stderr(), &out.stderr)
                .map_err(|e| format!("forward winner stderr: {e}"))?;
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        // Exit 2 = "registered but rejected." Save the FIRST one and
        // keep going (other candidates might still succeed if
        // multiple binaries register the same test name and one of
        // them admits it). Exit 1 = "not registered here" — record
        // as last_miss_stderr for the fallback diagnostic.
        if out.status.code() == Some(2) {
            if rejection_stderr.is_none() {
                rejection_stderr = Some(stderr);
            }
        } else {
            last_miss_stderr = stderr;
        }
    }

    if let Some(reason) = rejection_stderr {
        return Err(format!(
            "test '{test}' is registered but cannot be exported:\n{}",
            reason.trim_end(),
        ));
    }
    Err(format!(
        "test '{test}' not found in any workspace test binary ({} candidates tried). \
         Last stderr from a candidate:\n{}",
        bins.len(),
        last_miss_stderr.trim_end(),
    ))
}

/// Compile the workspace's test binaries via
/// `cargo build --tests --message-format=json` and collect the
/// resulting executable paths.
///
/// Filters to artifacts where `executable != null` AND either
/// `target.kind` contains `"test"` (integration tests under
/// `tests/`) or `profile.test == true` (unit-test binaries built
/// from `[lib]` / `[bin]` targets). Both shapes carry the
/// `#[ktstr_test]` distributed-slice registry that the export
/// dispatcher reads, so both are valid candidates.
fn build_test_binaries(package: Option<&str>, release: bool) -> Result<Vec<PathBuf>, String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--tests", "--message-format=json"]);
    if let Some(p) = package {
        cmd.args(["--package", p]);
    }
    if release {
        cmd.arg("--release");
    }
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());

    let out = cmd
        .output()
        .map_err(|e| format!("spawn cargo build --tests: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo build --tests failed (exit {})",
            out.status.code().unwrap_or(-1),
        ));
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut bins: Vec<PathBuf> = Vec::new();
    for line in stdout.lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let Some(exe) = msg.get("executable").and_then(|e| e.as_str()) else {
            continue;
        };
        let kinds: Vec<&str> = msg
            .get("target")
            .and_then(|t| t.get("kind"))
            .and_then(|k| k.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let is_test_target = kinds.contains(&"test");
        let is_unit_test = msg
            .get("profile")
            .and_then(|p| p.get("test"))
            .and_then(|t| t.as_bool())
            == Some(true);
        if is_test_target || is_unit_test {
            bins.push(PathBuf::from(exe));
        }
    }
    bins.sort();
    bins.dedup();
    Ok(bins)
}
