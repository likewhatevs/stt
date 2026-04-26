//! Process-level dispatch and nextest protocol handling.
//!
//! This module owns every code path that runs before (or in lieu of)
//! the user's `main()`:
//!
//! - [`ktstr_test_early_dispatch`]: the `#[ctor]` that fires in every
//!   ktstr-linked binary. Routes the process to guest init, host-side
//!   VM launch, guest-side test execution, or nextest protocol handling.
//! - [`ktstr_main`]: the nextest protocol handler — `--list` returns
//!   `ktstr/` and `gauntlet/` test names, `--exact` runs a single test.
//! - [`run_ktstr_test`]: programmatic entry point used by library
//!   consumers and the macro-generated `#[test]` wrappers.
//! - [`analyze_sidecars`]: collects sidecar JSON from a run directory
//!   and renders the full gauntlet analysis (rows + verifier + callback
//!   profile + KVM stats) into a string.
//!
//! The heavy lifting lives in sibling submodules: `eval` (host-side
//! result judgment — `run_ktstr_test_inner` and `evaluate_vm_result`),
//! `sidecar` (per-run JSON), `probe` (auto-repro + BPF probe pipeline),
//! `args` (CLI extraction), and the [`crate::vmm`] VM launcher.

use std::path::PathBuf;

use anyhow::Result;

use crate::assert::AssertResult;

use super::{
    KTSTR_TESTS, KtstrTestEntry, TopoOverride, collect_sidecars, extract_flags_arg,
    extract_test_fn_arg, extract_topo_arg, find_test, format_callback_profile, format_kvm_stats,
    format_verifier_stats, maybe_dispatch_vm_test, parse_topo_string,
    propagate_rust_env_from_cmdline, record_skip_sidecar, resolve_test_kernel,
    run_ktstr_test_inner, sidecar_dir, try_flush_profraw, validate_entry_flags,
};

/// One resolved kernel entry from `KTSTR_KERNEL_LIST` (the multi-
/// kernel fan-out wire format that `cargo ktstr test --kernel A
/// --kernel B` exports before exec'ing into `cargo nextest`).
///
/// `sanitized` is the nextest-safe identifier appended to test names
/// so `cargo nextest run -E 'test(kernel_6_14_2)'` filters work
/// natively. The producer-side encoder in `cargo-ktstr` emits a
/// semantic, operator-readable label per kernel:
/// - Version / Range expansion: the version string verbatim
///   (`6.14.2`, `6.15-rc3`).
/// - CacheKey: the version prefix (everything before the
///   `-tarball-` / `-git-` source tag).
/// - Git: `git_{owner}_{repo}_{ref}` extracted from the URL.
/// - Path: `path_{basename}_{hash6}` — basename + 6-char crc32 of
///   the canonical path, disambiguating two `linux` directories
///   under different parents.
///
/// [`sanitize_kernel_label`] applies the `kernel_` prefix and
/// `[a-z0-9_]+` normalization downstream.
///
/// `kernel_dir` is the canonical absolute path to the kernel-build
/// directory the per-variant subprocess re-exports as
/// `KTSTR_KERNEL`.
#[derive(Clone, Debug)]
pub(crate) struct KernelEntry {
    pub(crate) sanitized: String,
    pub(crate) kernel_dir: PathBuf,
}

/// Parse the multi-kernel wire format `KTSTR_KERNEL_LIST` into a
/// `Vec<KernelEntry>`. Format: `label1=path1;label2=path2;...`,
/// semicolon-separated entries, `=` separating label from path. Empty
/// / unset env returns an empty vec — callers treat that as
/// "single-kernel mode" and fall through to `KTSTR_KERNEL`.
///
/// Malformed entries (missing `=`, empty label, empty path) are
/// dropped silently — the producer is `cargo ktstr` which encodes
/// the format under our control, so a malformed entry indicates a
/// regression in the producer rather than operator input that
/// deserves a clear error. Silent drop preserves the `len() <= 1` →
/// "treat as single-kernel" invariant in the readers downstream.
pub(crate) fn parse_kernel_list(raw: &str) -> Vec<KernelEntry> {
    raw.split(';')
        .filter_map(|seg| {
            let seg = seg.trim();
            if seg.is_empty() {
                return None;
            }
            let (label, path) = seg.split_once('=')?;
            let label = label.trim();
            let path = path.trim();
            if label.is_empty() || path.is_empty() {
                return None;
            }
            Some(KernelEntry {
                sanitized: sanitize_kernel_label(label),
                kernel_dir: PathBuf::from(path),
            })
        })
        .collect()
}

/// Read [`crate::KTSTR_KERNEL_LIST_ENV`] and parse it into a
/// `Vec<KernelEntry>`. Empty / unset / malformed → empty vec
/// (single-kernel mode at the call site).
pub(crate) fn read_kernel_list() -> Vec<KernelEntry> {
    std::env::var(crate::KTSTR_KERNEL_LIST_ENV)
        .ok()
        .map(|v| parse_kernel_list(&v))
        .unwrap_or_default()
}

/// Sanitise a kernel label (the producer-side identity emitted by
/// `cargo ktstr`'s resolver) into a nextest-safe identifier of the
/// shape `kernel_[a-z0-9_]+`.
///
/// Replaces every `[^A-Za-z0-9]` byte with `_`, lowercases, collapses
/// runs of `_`, and prefixes with `kernel_`. Empty / pathologically-
/// short input collapses to `kernel_` alone, which the parser
/// downstream still recognises as a valid suffix (the empty
/// `sanitized` marker just won't disambiguate two kernels — but the
/// producer side guarantees non-empty labels, so the empty case is
/// defensive only).
///
/// Example mappings:
/// - `6.14.2` → `kernel_6_14_2`
/// - `6.15-rc3` → `kernel_6_15_rc3`
/// - `git_tj_sched_ext_for-next` → `kernel_git_tj_sched_ext_for_next`
/// - `path_linux_a3f2b1` → `kernel_path_linux_a3f2b1`
pub fn sanitize_kernel_label(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 7);
    out.push_str("kernel_");
    let mut last_underscore = true; // suppress leading `_` after `kernel_`
    for ch in raw.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    // Strip a trailing `_` so a label like `for-next-` doesn't
    // produce a dangling separator.
    if out.ends_with('_') && out.len() > "kernel_".len() {
        out.pop();
    }
    out
}

/// Early dispatch for `#[ktstr_test]` test execution.
///
/// Runs before `main()` in any binary that links against ktstr.
///
/// When running as PID 1 (the binary is `/init` in the VM), calls
/// `ktstr_guest_init()` which handles the full init lifecycle and never
/// returns.
///
/// - `--ktstr-test-fn=NAME --ktstr-topo=NnNlNcNt`: host-side dispatch —
///   boots a VM with the specified topology and runs the test inside it.
/// - `--ktstr-test-fn=NAME` (without `--ktstr-topo`): guest-side dispatch —
///   runs the test function directly (inside a VM that was already booted).
/// - nextest protocol (`--list`/`--exact`): intercepted when running
///   under nextest (`NEXTEST` env var set), delegates to [`ktstr_main`].
/// - Otherwise: no-op (falls through to the standard test harness).
#[doc(hidden)]
#[ctor::ctor]
pub fn ktstr_test_early_dispatch() {
    // PID 1: the binary is /init in the VM. Perform full init lifecycle
    // (mounts, scheduler, test dispatch, reboot). Never returns.
    if unsafe { libc::getpid() } == 1 {
        crate::vmm::rust_init::ktstr_guest_init();
    }

    if let Some(code) = maybe_dispatch_host_test() {
        std::process::exit(code);
    }
    // Propagate RUST_BACKTRACE / RUST_LOG from /proc/cmdline before
    // `maybe_dispatch_vm_test` runs: ctor context is single-threaded
    // (`.init_array` runs before any user thread exists), so this
    // `set_var` is sound and the later guest-side code that spawns
    // the probe thread observes the correct env.
    propagate_rust_env_from_cmdline();
    if let Some(code) = maybe_dispatch_vm_test() {
        // The LLVM profiling runtime registers its atexit handler via a
        // .init_array entry (C++ global initializer). Our ctor also lives
        // in .init_array, and the execution order between them is
        // non-deterministic. If our ctor runs first, the atexit handler
        // was never registered, so std::process::exit() won't write the
        // profraw. Serialize profraw to a buffer and write it to the SHM
        // ring for host-side extraction.
        try_flush_profraw();
        std::process::exit(code);
    }

    // nextest protocol: intercept --list and --exact when running under
    // nextest. Under cargo test, fall through to the standard harness
    // which runs the #[test] wrappers generated by #[ktstr_test].
    //
    // Binaries with real #[ktstr_test] entries need the ctor to handle
    // listing (gauntlet expansion) and dispatch (VM booting). The lib
    // test binary has only the dummy entry and no gauntlet variants —
    // skip interception so the standard harness discovers #[cfg(test)]
    // module #[test] functions (unit tests).
    if std::env::var_os("NEXTEST").is_some() {
        let has_real_tests = KTSTR_TESTS.iter().any(|e| !is_test_sentinel(e.name));
        if has_real_tests {
            let args: Vec<String> = std::env::args().collect();
            if args.iter().any(|a| a == "--list" || a == "--exact") {
                ktstr_main();
            }
        }
    } else {
        // cargo-test-direct path: the standard rustc test harness
        // runs only the bare `#[test]` wrappers `#[ktstr_test]`
        // generates. Gauntlet expansion (flag-profile × topology-
        // preset combinations) lives inside `ktstr_main`'s `--list`
        // + `--exact` handlers and is reachable ONLY under nextest.
        // Every real ktstr entry produces topology-preset variants
        // under nextest (`for_each_gauntlet_variant` iterates
        // `crate::vm::gauntlet_presets()` regardless of whether the
        // scheduler declares flags — the flag set only determines
        // the profile half of the `presets × profiles` product).
        // Without nextest those variants would silently not run —
        // coverage loss with no error. Emit a one-shot stderr
        // `warning:` diagnostic (see the `eprintln!` below) when the
        // binary carries any real entry so the user sees the gap
        // instead of trusting a false green. Print once per process
        // (cargo test invokes one test binary per crate; the ctor
        // runs exactly once per test binary) so there is no need to
        // gate with a std::sync::Once.
        let total = KTSTR_TESTS.len();
        let real = KTSTR_TESTS
            .iter()
            .filter(|e| !is_test_sentinel(e.name))
            .count();
        if real > 0 {
            eprintln!(
                "warning: {real} of {total} ktstr test entries registered in this binary \
                 will not generate their flag-profile / topology-preset gauntlet variants — \
                 NEXTEST env var is not set and the standard rustc harness does not expand \
                 them. Use `cargo nextest run` (or `cargo ktstr test`) to exercise the full \
                 gauntlet.",
            );
        }
    }
}

/// Predicate for "this entry is a unit-test sentinel, not a real
/// `#[ktstr_test]` user entry." The lib-test binary registers a
/// single sentinel entry (currently `"__unit_test_dummy__"`) so
/// the dispatch + gauntlet plumbing has something to exercise
/// under `cargo test --lib`; real user entries look like
/// `"module::test_name"` or similar PascalCase-with-dots names.
///
/// Matching the sentinel by convention (`__` prefix + `__`
/// suffix + `_test_` or `_dummy_` infix) rather than by literal
/// equality keeps the filter robust when the sentinel is
/// renamed, or when future scaffolding adds additional
/// sentinel-shaped entries (e.g. `__unit_test_panics__`,
/// `__unit_test_timeout__`). The literal-equality form would
/// silently admit those future sentinels into the real-entry
/// population and double-fire the "NEXTEST env var not set"
/// warning or spuriously enable --list interception.
fn is_test_sentinel(name: &str) -> bool {
    // Real user-authored `#[ktstr_test]` entry names
    // conventionally do not match the `__unit_test_*__` pattern
    // (Rust's reserved-identifier convention for
    // language-implementation and framework-internal names).
    // The `#[ktstr_test]` proc macro does not validate this, so
    // the predicate admits a real user entry in the unlikely
    // case someone names one with the `__unit_test_*__` shape —
    // collision would double-fire the "NEXTEST env var not set"
    // warning / spuriously enable --list interception, but
    // that's a diagnostic glitch, not a correctness failure.
    name.starts_with("__unit_test_") && name.ends_with("__")
}

/// Host-side dispatch: if both `--ktstr-test-fn` and `--ktstr-topo` are
/// present, boot a VM with the specified topology and run the test
/// inside it. Returns `Some(exit_code)` if dispatched, `None` otherwise.
fn maybe_dispatch_host_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_test_fn_arg(&args)?;
    let topo_str = extract_topo_arg(&args)?;

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    let (numa_nodes, llcs, cores, threads) = match parse_topo_string(&topo_str) {
        Some(t) => t,
        None => {
            eprintln!(
                "ktstr_test: invalid --ktstr-topo format '{topo_str}' (expected NnNlNcNt, e.g. 1n2l4c2t)"
            );
            return Some(1);
        }
    };

    let cpus = llcs * cores * threads;
    let memory_mb = (cpus * 64).max(256).max(entry.memory_mb);
    let topo = TopoOverride {
        numa_nodes,
        llcs,
        cores,
        threads,
        memory_mb,
    };

    let active_flags = extract_flags_arg(&args).unwrap_or_default();
    match run_ktstr_test_with_topo_and_flags(entry, &topo, &active_flags) {
        Ok(_) => Some(0),
        Err(e) => {
            eprintln!("ktstr_test: {e:#}");
            Some(1)
        }
    }
}

/// Host-side entry point: build a VM, boot it with `--ktstr-test-fn=NAME`,
/// extract profraw from SHM, and return the test result.
///
/// Validates KVM access and auto-discovers a kernel image via
/// `resolve_test_kernel()` when `KTSTR_TEST_KERNEL` is not set.
pub fn run_ktstr_test(entry: &KtstrTestEntry) -> Result<AssertResult> {
    // Directly-constructed entries bypass the proc-macro's
    // compile-time checks. Call `validate` here so programmatic
    // consumers (library callers pushing into `KTSTR_TESTS`
    // dynamically) hit the same bail messages the macro produces at
    // compile time.
    entry.validate()?;
    if entry.host_only {
        return run_host_only_test_inner(entry);
    }
    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        anyhow::bail!("vmlinux not found, bpf_map_write requires vmlinux");
    }
    // Matches run_named_test: tests declaring `required_flags` must
    // keep them active even on the zero-override library entry point,
    // otherwise scheduler-feature-gated scenarios silently run with
    // the wrong flag profile. Passing &[] here is the bug this
    // replaces.
    let active_flags: Vec<String> = entry.required_flags.iter().map(|s| s.to_string()).collect();
    run_ktstr_test_inner(entry, None, &active_flags)
}

/// Like `run_ktstr_test` but with an explicit topology override and
/// active flags that map to scheduler CLI args via
/// `Scheduler::flag_args()`. Only consumed inside this module by
/// `maybe_dispatch_host_test`; kept as a named helper so the
/// `--ktstr-test-fn` + `--ktstr-topo` dispatch path reads symmetrically
/// with the zero-override [`run_ktstr_test`] library entry point.
fn run_ktstr_test_with_topo_and_flags(
    entry: &KtstrTestEntry,
    topo: &TopoOverride,
    active_flags: &[String],
) -> Result<AssertResult> {
    run_ktstr_test_inner(entry, Some(topo), active_flags)
}

/// Run a test result through expect_err logic and return an exit code.
///
/// Returns 0 on pass, 1 on failure. `ResourceContention` returns
/// 0 — the test never ran, not a real failure. The skip sidecar for
/// this case is written upstream in `run_ktstr_test_inner` at the
/// ResourceContention propagation site so every caller (including
/// the library entry point `run_ktstr_test`) records it, not just
/// the nextest dispatch path.
fn result_to_exit_code(result: Result<AssertResult>, expect_err: bool) -> i32 {
    match result {
        Ok(_) if expect_err => {
            eprintln!("expected error but test passed");
            1
        }
        Ok(_) => 0,
        Err(e)
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some() =>
        {
            let reason = e
                .downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .unwrap()
                .reason
                .clone();
            crate::report::test_skip(format_args!("resource contention: {reason}"));
            0
        }
        Err(_) if expect_err => 0,
        Err(e) => {
            eprintln!("{e:#}");
            1
        }
    }
}

/// Whether a base test entry is "ignored" (skipped by default).
///
/// Tests whose names start with `demo_` are ignored -- they are
/// demonstration/benchmarking tests that require manual opt-in.
fn is_ignored(entry: &KtstrTestEntry) -> bool {
    entry.name.starts_with("demo_")
}

/// Collect test names for nextest discovery (--list --format terse).
///
/// Nextest calls the binary twice:
/// - Without `--ignored`: prints ALL tests (ignored and non-ignored).
/// - With `--ignored`: prints ONLY ignored tests.
///
/// Gauntlet variants are always ignored. Base tests are ignored when
/// their name starts with `demo_`.
///
/// When `KTSTR_BUDGET_SECS` is set, applies greedy coverage maximization
/// to select the subset of tests that maximizes feature coverage within
/// the time budget. Only selected tests are printed.
fn list_tests(ignored_only: bool) {
    let raw = std::env::var("KTSTR_BUDGET_SECS").ok();
    let budget_secs: Option<f64> = raw.as_deref().and_then(|s| match s.parse::<f64>() {
        Ok(v) if v > 0.0 => Some(v),
        Ok(v) => {
            eprintln!("ktstr_test: KTSTR_BUDGET_SECS={v}: must be positive, ignoring");
            None
        }
        Err(e) => {
            eprintln!("ktstr_test: KTSTR_BUDGET_SECS={s:?}: {e}, ignoring");
            None
        }
    });

    if let Some(budget) = budget_secs {
        list_tests_budget(ignored_only, budget);
    } else {
        list_tests_all(ignored_only);
    }
}

/// Host capacity inputs for `TopologyConstraints::accepts`.
///
/// `list_tests_all` and `list_tests_budget` both need the same
/// `(cpus, llcs, max_cpus_per_llc)` triple to filter gauntlet presets
/// against what the host can actually schedule. Reading sysfs here
/// once per listing (instead of per-entry) keeps the signal-to-noise
/// of each lister high and makes the host-query decision explicit.
fn host_capacity() -> (u32, u32, u32) {
    let host_cpus = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    let host_topo = crate::vmm::host_topology::HostTopology::from_sysfs().ok();
    let host_llcs = host_topo
        .as_ref()
        .map(|t| t.llc_groups.len() as u32)
        .unwrap_or(1);
    let host_max_cpus_per_llc = host_topo
        .as_ref()
        .map(|t| t.max_cores_per_llc() as u32)
        .unwrap_or(host_cpus);
    (host_cpus, host_llcs, host_max_cpus_per_llc)
}

/// Iterate (preset, profile) pairs that both fit the host capacity
/// and match the entry's `TopologyConstraints`. Shared between the
/// eager ("print every name") and budgeted ("push a candidate")
/// listers in `list_tests_*`.
fn for_each_gauntlet_variant<F>(
    entry: &KtstrTestEntry,
    presets: &[crate::vm::TopoPreset],
    host_cpus: u32,
    host_llcs: u32,
    host_max_cpus_per_llc: u32,
    mut visit: F,
) where
    F: FnMut(&crate::vm::TopoPreset, &crate::scenario::FlagProfile),
{
    let profiles = entry
        .scheduler
        .generate_profiles(entry.required_flags, entry.excluded_flags);
    for preset in presets {
        if !entry.constraints.accepts(
            &preset.topology,
            host_cpus,
            host_llcs,
            host_max_cpus_per_llc,
        ) {
            continue;
        }
        for profile in &profiles {
            visit(preset, profile);
        }
    }
}

/// List all tests without budget filtering.
///
/// When `KTSTR_KERNEL_LIST` carries 2 or more entries, every test
/// name carries an extra `/{sanitized_kernel_label}` suffix so each
/// (test × kernel) pair becomes a distinct nextest test case;
/// nextest's parallelism, retries, and `-E` filtering all apply
/// natively. Single-kernel mode (0 or 1 entries) preserves the
/// historical `gauntlet/{name}/{preset}/{profile}` shape so existing
/// CI baselines, test-name filters, and per-test config overrides
/// keep matching.
fn list_tests_all(ignored_only: bool) {
    let presets = crate::vm::gauntlet_presets();
    let has_vmlinux = resolve_test_kernel()
        .ok()
        .and_then(|k| crate::vmm::find_vmlinux(&k))
        .is_some();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = host_capacity();

    let kernel_list = read_kernel_list();
    let multi_kernel = kernel_list.len() > 1;
    // Single-kernel mode (no list, or list has exactly one entry)
    // emits one variant per (test × preset × profile) tuple with no
    // kernel suffix. Multi-kernel mode iterates every kernel as an
    // outer loop and appends `/{sanitized}` per variant. The empty-
    // suffix sentinel below is what the single-kernel branch passes
    // to keep the print path uniform.
    let kernel_suffixes: Vec<&str> = if multi_kernel {
        kernel_list.iter().map(|k| k.sanitized.as_str()).collect()
    } else {
        vec![""]
    };

    for entry in KTSTR_TESTS.iter() {
        validate_entry_flags(entry);

        // bpf_map_write tests require vmlinux to resolve BPF map
        // addresses. Don't list them when vmlinux is unavailable —
        // they cannot run and would produce false PASS results.
        if !entry.bpf_map_write.is_empty() && !has_vmlinux {
            continue;
        }

        if !ignored_only || is_ignored(entry) {
            // host_only tests never boot a VM, so the kernel never
            // affects what runs — emit one entry without a kernel
            // suffix even in multi-kernel mode. Otherwise we'd run N
            // identical copies of the same host-side function.
            if entry.host_only {
                println!("ktstr/{}: test", entry.name);
            } else {
                for suffix in &kernel_suffixes {
                    if suffix.is_empty() {
                        println!("ktstr/{}: test", entry.name);
                    } else {
                        println!("ktstr/{}/{suffix}: test", entry.name);
                    }
                }
            }
        }

        // Host-only tests run on the host without a VM -- gauntlet
        // topology variants are meaningless.
        if entry.host_only {
            continue;
        }

        // Gauntlet variants are always ignored — users opt in with
        // --run-ignored. Presets that exceed the host's CPU count or
        // LLC count are filtered from the listing entirely.
        for_each_gauntlet_variant(
            entry,
            &presets,
            host_cpus,
            host_llcs,
            host_max_cpus_per_llc,
            |preset, profile| {
                for suffix in &kernel_suffixes {
                    if suffix.is_empty() {
                        println!(
                            "gauntlet/{}/{}/{}: test",
                            entry.name,
                            preset.name,
                            profile.name()
                        );
                    } else {
                        println!(
                            "gauntlet/{}/{}/{}/{suffix}: test",
                            entry.name,
                            preset.name,
                            profile.name()
                        );
                    }
                }
            },
        );
    }
}

/// List tests with budget-based coverage maximization.
///
/// Collects all eligible tests as candidates, runs greedy selection,
/// and prints only the selected subset. Multi-kernel mode adds the
/// kernel suffix as a feature dimension so the budget selector
/// picks per-kernel coverage; single-kernel mode is unchanged.
fn list_tests_budget(ignored_only: bool, budget_secs: f64) {
    use crate::budget::{TestCandidate, estimate_duration, extract_features, select};

    let presets = crate::vm::gauntlet_presets();
    let has_vmlinux = resolve_test_kernel()
        .ok()
        .and_then(|k| crate::vmm::find_vmlinux(&k))
        .is_some();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = host_capacity();
    let mut candidates: Vec<TestCandidate> = Vec::new();

    let kernel_list = read_kernel_list();
    let multi_kernel = kernel_list.len() > 1;
    let kernel_suffixes: Vec<&str> = if multi_kernel {
        kernel_list.iter().map(|k| k.sanitized.as_str()).collect()
    } else {
        vec![""]
    };

    for entry in KTSTR_TESTS.iter() {
        validate_entry_flags(entry);

        if !entry.bpf_map_write.is_empty() && !has_vmlinux {
            continue;
        }

        let base_ignored = is_ignored(entry);
        let base_topo = entry.topology;

        // Base test
        if !ignored_only || base_ignored {
            // host_only tests never boot a VM, so the kernel never
            // affects what runs — push one candidate without a
            // kernel suffix even in multi-kernel mode. Otherwise the
            // budget selector would consider N identical copies of
            // the same host-side function.
            if entry.host_only {
                candidates.push(TestCandidate {
                    name: format!("ktstr/{}: test", entry.name),
                    features: extract_features(entry, &base_topo, &[], false, entry.name),
                    estimated_secs: estimate_duration(entry, &base_topo),
                });
            } else {
                for suffix in &kernel_suffixes {
                    let name = if suffix.is_empty() {
                        format!("ktstr/{}: test", entry.name)
                    } else {
                        format!("ktstr/{}/{suffix}: test", entry.name)
                    };
                    candidates.push(TestCandidate {
                        name,
                        features: extract_features(entry, &base_topo, &[], false, entry.name),
                        estimated_secs: estimate_duration(entry, &base_topo),
                    });
                }
            }
        }

        if entry.host_only {
            continue;
        }

        for_each_gauntlet_variant(
            entry,
            &presets,
            host_cpus,
            host_llcs,
            host_max_cpus_per_llc,
            |preset, profile| {
                for suffix in &kernel_suffixes {
                    let test_name = if suffix.is_empty() {
                        format!("gauntlet/{}/{}/{}", entry.name, preset.name, profile.name())
                    } else {
                        format!(
                            "gauntlet/{}/{}/{}/{suffix}",
                            entry.name,
                            preset.name,
                            profile.name(),
                        )
                    };
                    candidates.push(TestCandidate {
                        name: format!("{test_name}: test"),
                        features: extract_features(
                            entry,
                            &preset.topology,
                            &profile.flags,
                            true,
                            &test_name,
                        ),
                        estimated_secs: estimate_duration(entry, &preset.topology),
                    });
                }
            },
        );
    }

    let selected = select(&candidates, budget_secs);
    for &i in &selected {
        println!("{}", candidates[i].name);
    }

    let stats = crate::budget::selection_stats(&candidates, &selected, budget_secs);
    eprintln!(
        "ktstr budget: {}/{} tests, {:.0}/{:.0}s used, {}/{} configurations covered",
        stats.selected,
        stats.total,
        stats.budget_used,
        stats.budget_total,
        stats.bits_covered,
        stats.bits_possible,
    );
}

/// Strip an optional `/{sanitized_kernel_label}` suffix from `name`,
/// look up the matching [`KernelEntry`] in the multi-kernel list,
/// and re-export `KTSTR_KERNEL` to that entry's directory. Returns
/// the prefix-only name for the dispatch caller.
///
/// When `KTSTR_KERNEL_LIST` is unset / single-entry, the function
/// is a no-op pass-through: returns `(name, None)` and does not
/// touch the env. When the list has 2+ entries, the suffix is
/// REQUIRED and missing it surfaces as `Err` (the early-dispatch
/// caller turns that into exit code 1 with an actionable message)
/// — the suffix is part of every test name `--list` emitted, so a
/// `--exact` invocation that omits it can only come from operator
/// hand-construction or tooling that hasn't been taught the
/// multi-kernel naming.
fn strip_kernel_suffix<'a>(
    name: &'a str,
    kernel_list: &'a [KernelEntry],
) -> Result<(&'a str, Option<&'a KernelEntry>), String> {
    if kernel_list.len() <= 1 {
        return Ok((name, None));
    }
    // Multi-kernel: every test name carries `/kernel_…` as its
    // final segment. Iterate the labels rather than splitting on
    // `/` — the suffix always has exactly one extra `/` separator
    // before `kernel_…`, but the body of the test name CAN contain
    // `/` (gauntlet variants already do), so a naive
    // `rsplit_once('/')` would accidentally peel the gauntlet's
    // profile segment instead.
    //
    // Distinct kernels in the same `KTSTR_KERNEL_LIST` produce
    // distinct sanitized labels in practice — the producer emits
    // semantic identifiers (version strings, git owner/repo/ref,
    // path basename + 6-char hash) that don't share suffixes
    // among the resolved set. If a future regression DID produce
    // labels where one is a strict suffix of another (e.g.
    // `kernel_6_14` vs `kernel_x_kernel_6_14`), the iterate-and-
    // first-match below would pick whichever appears first in
    // the kernel_list — deterministic but potentially wrong.
    // Producer-side regression detection (#123) would catch that
    // class of collision before it reaches this peeler.
    for entry in kernel_list {
        let needle = format!("/{}", entry.sanitized);
        if let Some(stripped) = name.strip_suffix(&needle) {
            return Ok((stripped, Some(entry)));
        }
    }
    Err(format!(
        "test name {name:?} has no recognised kernel suffix (KTSTR_KERNEL_LIST \
         carries {n} kernels — every test name must end with `/kernel_…`)",
        n = kernel_list.len(),
    ))
}

/// Re-export `KTSTR_KERNEL` to the kernel directory carried by a
/// resolved [`KernelEntry`]. Called when a multi-kernel `--exact`
/// dispatch peels off the per-test kernel suffix.
///
/// SAFETY: nextest invokes the test binary's `--exact` handler in a
/// single-threaded context — there are no other readers of the env
/// at this point. The eventual VM-launch site reads `KTSTR_KERNEL`
/// via `find_kernel` after this returns; that read is sequenced
/// after the write per the program order.
fn export_kernel_for_variant(entry: &KernelEntry) {
    // SAFETY: see fn-level doc — single-threaded ctor / nextest
    // dispatch context.
    unsafe { std::env::set_var(crate::KTSTR_KERNEL_ENV, &entry.kernel_dir) };
}

/// Parse a nextest-style test name and run it.
///
/// Handles base tests (`ktstr/{name}`), gauntlet variants
/// (`gauntlet/{name}/{preset}/{profile}`), and bare names
/// (backward compat). When `KTSTR_KERNEL_LIST` carries 2+ kernels,
/// every test name additionally ends with `/{sanitized_kernel_label}`
/// — that suffix is peeled here and the matching kernel directory
/// is re-exported via [`KTSTR_KERNEL_ENV`] before the dispatch
/// continues. Returns an exit code.
pub(crate) fn run_named_test(test_name: &str) -> i32 {
    let kernel_list = read_kernel_list();
    let (test_name, kernel_entry) = match strip_kernel_suffix(test_name, &kernel_list) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if let Some(entry) = kernel_entry {
        export_kernel_for_variant(entry);
    }

    if let Some(rest) = test_name.strip_prefix("gauntlet/") {
        return run_gauntlet_test(rest);
    }

    let bare_name = test_name.strip_prefix("ktstr/").unwrap_or(test_name);
    let entry = match find_test(bare_name) {
        Some(e) => e,
        None => {
            eprintln!("unknown test: {test_name}");
            return 1;
        }
    };

    if entry.host_only {
        return run_host_only_test(entry);
    }

    // Base tests don't carry a gauntlet profile, but the entry's
    // `required_flags` must still be activated for tests whose
    // scheduler-arg contract requires them (per the invariant
    // enforced by validate_entry_flags: every gauntlet profile
    // contains required_flags). Using an empty flags list would run
    // the scheduler without its required CLI args and produce
    // incorrect sidecar metadata.
    let active_flags: Vec<String> = entry.required_flags.iter().map(|s| s.to_string()).collect();

    if entry.performance_mode && std::env::var("KTSTR_NO_PERF_MODE").is_ok() {
        crate::report::test_skip(format_args!(
            "{}: test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
            bare_name,
        ));
        // See run_ktstr_test_inner for the sidecar-emission rationale.
        record_skip_sidecar(entry, &active_flags);
        return 0;
    }

    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("FAIL: vmlinux not found, bpf_map_write requires vmlinux");
        return 1;
    }

    let result = run_ktstr_test_inner(entry, None, &active_flags);
    result_to_exit_code(result, entry.expect_err)
}

/// Run a host-only test directly without booting a VM.
/// Returns an exit code for nextest dispatch.
fn run_host_only_test(entry: &KtstrTestEntry) -> i32 {
    let result = run_host_only_test_inner(entry);
    result_to_exit_code(result, entry.expect_err)
}

/// Inner host-only dispatch returning `Result<AssertResult>`.
///
/// Builds a minimal Ctx and calls the test function on the host.
/// Used for tests that need host tools (cargo, nested VMs).
fn run_host_only_test_inner(entry: &KtstrTestEntry) -> Result<AssertResult> {
    let topo = crate::topology::TestTopology::from_vm_topology(&entry.topology);
    let cgroups = crate::cgroup::CgroupManager::new("/sys/fs/cgroup/ktstr");
    let workers_per_cgroup = entry.workers_per_cgroup as usize;
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(entry.scheduler.assert())
        .merge(&entry.assert);
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo)
        .duration(entry.duration)
        .workers_per_cgroup(workers_per_cgroup)
        .settle(std::time::Duration::from_millis(500))
        .assert(merged_assert)
        .build();
    (entry.func)(&ctx)
}

/// Run a gauntlet variant test. `rest` is `{name}/{preset}/{profile}`.
pub(crate) fn run_gauntlet_test(rest: &str) -> i32 {
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() != 3 {
        eprintln!("invalid gauntlet test name: gauntlet/{rest}");
        return 1;
    }
    let (test_name, preset_name, profile_name) = (parts[0], parts[1], parts[2]);

    let entry = match find_test(test_name) {
        Some(e) => e,
        None => {
            eprintln!("unknown test: {test_name}");
            return 1;
        }
    };
    validate_entry_flags(entry);

    let presets = crate::vm::gauntlet_presets();
    let preset = match presets.iter().find(|p| p.name == preset_name) {
        Some(p) => p,
        None => {
            eprintln!("unknown gauntlet preset: {preset_name}");
            return 1;
        }
    };

    let t = &preset.topology;
    let cpus = t.total_cpus();

    let memory_mb = (cpus * 64).max(256).max(entry.memory_mb);
    let topo = TopoOverride {
        numa_nodes: t.numa_nodes,
        llcs: t.llcs,
        cores: t.cores_per_llc,
        threads: t.threads_per_core,
        memory_mb,
    };

    let profiles = entry
        .scheduler
        .generate_profiles(entry.required_flags, entry.excluded_flags);
    let flags: Vec<String> = match profiles.iter().find(|p| p.name() == profile_name) {
        Some(p) => p.flags.iter().map(|s| s.to_string()).collect(),
        None => {
            eprintln!("unknown flag profile: {profile_name}");
            return 1;
        }
    };

    if entry.performance_mode && std::env::var("KTSTR_NO_PERF_MODE").is_ok() {
        crate::report::test_skip(format_args!(
            "{}: test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
            test_name,
        ));
        record_skip_sidecar(entry, &flags);
        return 0;
    }

    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("FAIL: vmlinux not found, bpf_map_write requires vmlinux");
        return 1;
    }

    let result = run_ktstr_test_inner(entry, Some(&topo), &flags);
    result_to_exit_code(result, entry.expect_err)
}

/// Collect sidecar JSON files and return the full gauntlet analysis.
///
/// When `dir` is `Some`, reads sidecars from that directory. Otherwise
/// uses the default sidecar directory (`KTSTR_SIDECAR_DIR` override, or
/// `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{timestamp}/`).
///
/// Returns the concatenated output of `analyze_rows`, verifier stats,
/// callback profile, and KVM stats. Returns an empty string when no
/// sidecars are found.
pub fn analyze_sidecars(dir: Option<&std::path::Path>) -> String {
    let default_dir;
    let dir = match dir {
        Some(d) => d,
        None => {
            default_dir = sidecar_dir();
            &default_dir
        }
    };
    let sidecars = collect_sidecars(dir);
    if sidecars.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let rows: Vec<_> = sidecars.iter().map(crate::stats::sidecar_to_row).collect();
    if !rows.is_empty() {
        out.push_str(&crate::stats::analyze_rows(&rows));
    }
    let vstats = format_verifier_stats(&sidecars);
    if !vstats.is_empty() {
        out.push_str(&vstats);
    }
    let cprofile = format_callback_profile(&sidecars);
    if !cprofile.is_empty() {
        out.push_str(&cprofile);
    }
    let kstats = format_kvm_stats(&sidecars);
    if !kstats.is_empty() {
        out.push_str(&kstats);
    }
    out
}

/// Nextest protocol handler.
///
/// Called automatically by [`ktstr_test_early_dispatch`] when running
/// under nextest. Not intended for direct use.
///
/// - `--list --format terse`: output `ktstr/{name}: test\n` for base
///   tests and `gauntlet/{name}/{preset}/{profile}: test\n` for
///   gauntlet variants.
/// - `--exact NAME --nocapture`: run the named test, exit 0/1.
pub fn ktstr_main() -> ! {
    let args: Vec<String> = std::env::args().collect();

    // Discovery mode: --list --format terse [--ignored]
    if args.iter().any(|a| a == "--list") {
        let ignored_only = args.iter().any(|a| a == "--ignored");
        list_tests(ignored_only);
        std::process::exit(0);
    }

    // Execution mode: --exact NAME [--nocapture] [--ignored] [--bench]
    if let Some(pos) = args.iter().position(|a| a == "--exact") {
        if let Some(name) = args.get(pos + 1) {
            let code = run_named_test(name);
            std::process::exit(code);
        }
        eprintln!("--exact requires a test name");
        std::process::exit(1);
    }

    // Fallback: no recognized arguments.
    eprintln!("usage: <binary> --list --format terse [--ignored]");
    eprintln!("       <binary> --exact <test_name> --nocapture");
    std::process::exit(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // is_test_sentinel — convention-based sentinel-name predicate
    // ---------------------------------------------------------------

    /// Accepted shapes: `__unit_test_*__` (the established
    /// sentinel convention — double-underscore prefix with
    /// `unit_test_` tag, arbitrary inner suffix, double-underscore
    /// suffix).
    #[test]
    fn is_test_sentinel_accepts_convention_shaped_names() {
        assert!(is_test_sentinel("__unit_test_dummy__"));
        assert!(is_test_sentinel("__unit_test_panics__"));
        // Any inner body after the prefix is accepted, as long as
        // the `__` suffix is also present.
        assert!(is_test_sentinel("__unit_test_foo_bar_baz__"));
    }

    /// Rejected shapes: real user names, unrelated
    /// double-underscore names, and partial matches.
    #[test]
    fn is_test_sentinel_rejects_non_convention_names() {
        // Real user-authored name.
        assert!(!is_test_sentinel("my_test"));
        // Double-underscore wrapping but not the `__unit_test_` tag.
        assert!(!is_test_sentinel("__foo__"));
        // Empty string.
        assert!(!is_test_sentinel(""));
        // Has the prefix but no `__` suffix (ends with just `_`).
        assert!(!is_test_sentinel("__unit_test_"));
        // Has the prefix, has `__` suffix, but the prefix itself
        // is truncated — missing the trailing `_` of `__unit_test_`.
        assert!(!is_test_sentinel("__unit__"));
    }

    // ---------------------------------------------------------------
    // run_named_test / run_gauntlet_test — nextest dispatch routing
    // ---------------------------------------------------------------
    //
    // These tests cover the `test_name → function` routing without
    // booting a VM. The happy paths require KVM and a kernel image,
    // so the assertions here target the failure branches that return
    // exit code 1 before any VM spawn:
    //   - `ktstr/` prefix with unknown bare name
    //   - `gauntlet/` prefix with malformed parts / unknown preset /
    //     unknown profile
    //   - bare names fall through to `ktstr/` lookup
    //
    // The routing invariant: `gauntlet/` always delegates to
    // `run_gauntlet_test`, every other prefix (including none)
    // delegates to the base-test path inside `run_named_test`.

    #[test]
    fn run_named_test_gauntlet_prefix_routes_to_run_gauntlet_test() {
        // Gauntlet names require three slash-separated parts after
        // the prefix; a name missing them is rejected by
        // `run_gauntlet_test`, proving the prefix routed there and
        // not into the base-test path (which would print
        // `unknown test: gauntlet/...` instead of the gauntlet-
        // specific error and still return 1 but via a different
        // branch).
        let exit = run_named_test("gauntlet/__unit_test_dummy__");
        assert_eq!(exit, 1, "malformed gauntlet names must exit 1");
    }

    #[test]
    fn run_named_test_bare_unknown_exits_nonzero() {
        // `run_named_test` strips `ktstr/` when present; a bare
        // unknown name falls through to `find_test` which returns
        // None, producing exit code 1.
        let exit = run_named_test("__definitely_not_a_real_test__");
        assert_eq!(exit, 1);
    }

    #[test]
    fn run_named_test_ktstr_prefix_unknown_exits_nonzero() {
        // `ktstr/` prefix is stripped; the bare name (also unknown)
        // returns 1 via the find_test None path.
        let exit = run_named_test("ktstr/__definitely_not_a_real_test__");
        assert_eq!(exit, 1);
    }

    #[test]
    fn run_gauntlet_test_rejects_name_with_fewer_than_three_parts() {
        // `rest` must split into exactly 3 parts
        // (`{name}/{preset}/{profile}`). Two parts is a format error.
        let exit = run_gauntlet_test("some_test/some_preset");
        assert_eq!(exit, 1);
    }

    #[test]
    fn run_gauntlet_test_rejects_empty_rest() {
        // Empty rest splits into one empty string — also a format
        // error.
        let exit = run_gauntlet_test("");
        assert_eq!(exit, 1);
    }

    #[test]
    fn run_gauntlet_test_rejects_unknown_test_name() {
        // Well-formed three-part name whose test is not registered
        // in KTSTR_TESTS. Returns 1 via the find_test None branch,
        // never reaching preset lookup or VM spawn.
        let exit = run_gauntlet_test("__not_a_test__/tiny-1llc/default");
        assert_eq!(exit, 1);
    }

    #[test]
    fn run_gauntlet_test_rejects_unknown_preset() {
        // `__unit_test_dummy__` is registered in test_support::tests;
        // combined with a preset name that is not in
        // `gauntlet_presets`, the function returns 1 at the preset-
        // lookup branch.
        let exit = run_gauntlet_test("__unit_test_dummy__/__no_such_preset__/__default__");
        assert_eq!(exit, 1);
    }

    // -- host_capacity --

    #[test]
    fn host_capacity_returns_plausible_triple() {
        // `host_capacity` reads `available_parallelism` and sysfs topology.
        // The exact values depend on the test host, but the invariants
        // hold on any sane Linux machine:
        //   - cpus >= 1
        //   - llcs >= 1 (at least one cache domain)
        //   - max_cpus_per_llc >= 1
        //   - max_cpus_per_llc <= cpus (no LLC wider than the whole host)
        let (cpus, llcs, max_cpus_per_llc) = host_capacity();
        assert!(cpus >= 1, "cpus >= 1, got {cpus}");
        assert!(llcs >= 1, "llcs >= 1, got {llcs}");
        assert!(
            max_cpus_per_llc >= 1,
            "max_cpus_per_llc >= 1, got {max_cpus_per_llc}"
        );
        assert!(
            max_cpus_per_llc <= cpus,
            "max_cpus_per_llc ({max_cpus_per_llc}) must not exceed cpus ({cpus})"
        );
    }

    // -- for_each_gauntlet_variant --

    #[test]
    fn for_each_gauntlet_variant_skips_presets_exceeding_host_capacity() {
        // Pass host_cpus=1/host_llcs=1 against the preset list: every
        // current preset has total_cpus >= 4 (see `gauntlet_presets()`
        // in src/vm.rs), so every preset fails
        // `TopologyConstraints::accepts` and `visit` must never be
        // called. Any entry works since the constraint check runs
        // before the visit — use the test dummy.
        let presets = crate::vm::gauntlet_presets();
        // Precondition for the assertion below: if a future preset
        // with total_cpus <= 1 is added, this test must be updated to
        // account for it instead of silently under-asserting.
        let every_preset_needs_more_than_one_cpu = presets
            .iter()
            .all(|p| p.topology.total_cpus() > 1 || p.topology.llcs > 1);
        assert!(
            presets.is_empty() || every_preset_needs_more_than_one_cpu,
            "test assumes every preset requires >1 CPU or >1 LLC; \
             found a single-CPU preset — update the assertion below"
        );

        let mut visited: Vec<String> = Vec::new();
        for_each_gauntlet_variant(
            find_test("__unit_test_dummy__").unwrap(),
            &presets,
            1,
            1,
            1,
            |preset, _| visited.push(preset.name.to_string()),
        );
        assert!(
            visited.is_empty(),
            "with host_cpus=1 host_llcs=1, no preset should be visited; \
             visited: {visited:?}"
        );
    }

    #[test]
    fn for_each_gauntlet_variant_visits_every_fitting_preset_x_profile() {
        // With generous host capacity (u32::MAX cpus/llcs), every
        // preset that the dummy entry's constraints accept yields at
        // least one `FlagProfile` visit (every scheduler generates a
        // default profile when no flags are required/excluded).
        let presets = crate::vm::gauntlet_presets();
        let mut count = 0;
        for_each_gauntlet_variant(
            find_test("__unit_test_dummy__").unwrap(),
            &presets,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            |_, _| count += 1,
        );
        // `__unit_test_dummy__` defaults to Payload::KERNEL_DEFAULT
        // (which wraps Scheduler::EEVDF) and Scheduler::EEVDF has no
        // flags → exactly one profile per accepted preset.
        // Asserting `count >= 1` proves at least one preset was
        // accepted and visited. Coupling to the exact preset count
        // would fail if the preset list changes.
        if !presets.is_empty() {
            assert!(
                count >= 1,
                "expected at least one visit with unlimited host capacity, got {count}"
            );
        }
    }

    #[test]
    fn for_each_gauntlet_variant_monotonic_in_host_capacity() {
        // Comparative-baseline: giving the function MORE host capacity
        // can only let MORE presets pass the cap-size filter, never
        // fewer. The upper-bound assertion in
        // `for_each_gauntlet_variant_skips_presets_exceeding_host_capacity`
        // and the lower-bound assertion in
        // `..._visits_every_fitting_preset_x_profile` both check one
        // extreme; this test anchors the monotonic relationship
        // between them. A regression that inverted the host-cap
        // comparison (e.g. `host_cpus < preset_cpus` → accept) would
        // pass both endpoint tests but fail here.
        let presets = crate::vm::gauntlet_presets();
        if presets.is_empty() {
            return;
        }
        let entry = find_test("__unit_test_dummy__").unwrap();
        let count_for = |cpus: u32, llcs: u32| {
            let mut n = 0;
            for_each_gauntlet_variant(entry, &presets, cpus, llcs, u32::MAX, |_, _| n += 1);
            n
        };
        let tight = count_for(1, 1);
        let loose = count_for(u32::MAX, u32::MAX);
        assert!(
            loose >= tight,
            "host-capacity monotonicity violated: tight=(1,1) yielded {tight} \
             visits, loose=(u32::MAX,u32::MAX) yielded {loose}; loose \
             must admit at least as many presets as tight",
        );
    }

    // ---------------------------------------------------------------
    // KTSTR_KERNEL_LIST parsing + sanitization + suffix dispatch
    // ---------------------------------------------------------------

    #[test]
    fn parse_kernel_list_empty_returns_empty() {
        assert!(parse_kernel_list("").is_empty());
        assert!(parse_kernel_list(";").is_empty());
        assert!(parse_kernel_list(";;;").is_empty());
        assert!(parse_kernel_list("   ").is_empty());
    }

    #[test]
    fn parse_kernel_list_basic_pair() {
        // Producer emits semantic labels (the version string for
        // Version specs); the parser is shape-agnostic and just
        // splits on `;` and `=` then sanitizes. A version-only
        // label sanitizes to `kernel_6_14_2`.
        let entries = parse_kernel_list("6.14.2=/cache/foo");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kernel_dir, PathBuf::from("/cache/foo"));
        assert_eq!(entries[0].sanitized, "kernel_6_14_2");
    }

    #[test]
    fn parse_kernel_list_two_entries() {
        let entries = parse_kernel_list("6.14.2=/a;6.15.0=/b");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kernel_dir, PathBuf::from("/a"));
        assert_eq!(entries[0].sanitized, "kernel_6_14_2");
        assert_eq!(entries[1].kernel_dir, PathBuf::from("/b"));
        assert_eq!(entries[1].sanitized, "kernel_6_15_0");
    }

    #[test]
    fn parse_kernel_list_drops_malformed() {
        // Missing `=`, empty label, empty path — all silently
        // dropped. Producer is `cargo ktstr` which encodes the
        // format under our control; a malformed entry indicates a
        // regression in the producer rather than operator input
        // that deserves a clear error.
        let entries = parse_kernel_list("noeq;=onlypath;onlylabel=;valid=/foo");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kernel_dir, PathBuf::from("/foo"));
    }

    #[test]
    fn parse_kernel_list_trims_whitespace() {
        let entries = parse_kernel_list("  6.14.2=/a  ;  6.15.0=/b  ");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sanitized, "kernel_6_14_2");
        assert_eq!(entries[1].sanitized, "kernel_6_15_0");
    }

    #[test]
    fn sanitize_kernel_label_pure_version() {
        assert_eq!(sanitize_kernel_label("6.14.2"), "kernel_6_14_2");
    }

    #[test]
    fn sanitize_kernel_label_rc_suffix() {
        assert_eq!(sanitize_kernel_label("6.15-rc3"), "kernel_6_15_rc3");
    }

    /// The sanitizer is shape-agnostic — it normalizes any input
    /// that happens to flow in. The producer-side encoder now
    /// emits semantic labels, but a future regression that
    /// surfaced a raw cache-key basename would still produce a
    /// valid (if uglier) nextest identifier rather than crashing.
    /// Pinned via a synthetic full-cache-key input.
    #[test]
    fn sanitize_kernel_label_handles_full_cache_key_shape() {
        assert_eq!(
            sanitize_kernel_label("6.14.2-tarball-x86_64-kcabc1234"),
            "kernel_6_14_2_tarball_x86_64_kcabc1234",
        );
    }

    /// Git-source semantic label `git_tj_sched_ext_for-next` from
    /// the producer-side encoder maps to the dash-stripped form
    /// the sanitizer produces.
    #[test]
    fn sanitize_kernel_label_git_semantic_label() {
        assert_eq!(
            sanitize_kernel_label("git_tj_sched_ext_for-next"),
            "kernel_git_tj_sched_ext_for_next",
        );
    }

    /// Path-source semantic label `path_linux_a3f2b1` is already
    /// `[a-z0-9_]+` so the sanitizer only adds the `kernel_`
    /// prefix.
    #[test]
    fn sanitize_kernel_label_path_semantic_label() {
        assert_eq!(
            sanitize_kernel_label("path_linux_a3f2b1"),
            "kernel_path_linux_a3f2b1",
        );
    }

    #[test]
    fn sanitize_kernel_label_lowercases() {
        assert_eq!(sanitize_kernel_label("ABC-DEF"), "kernel_abc_def");
    }

    #[test]
    fn sanitize_kernel_label_collapses_repeated_separators() {
        assert_eq!(sanitize_kernel_label("a..b...c"), "kernel_a_b_c");
    }

    #[test]
    fn sanitize_kernel_label_strips_trailing_underscore() {
        assert_eq!(sanitize_kernel_label("for-next-"), "kernel_for_next");
    }

    #[test]
    fn sanitize_kernel_label_empty_input() {
        assert_eq!(sanitize_kernel_label(""), "kernel_");
    }

    /// `strip_kernel_suffix` is a no-op for single-kernel mode (0 or
    /// 1 entries) — returns the input verbatim and signals "no
    /// kernel override needed."
    #[test]
    fn strip_kernel_suffix_single_kernel_passthrough() {
        let kernel_list = vec![KernelEntry {
            sanitized: "kernel_6_14_2".to_string(),
            kernel_dir: PathBuf::from("/a"),
        }];
        let (stripped, entry) =
            strip_kernel_suffix("gauntlet/eevdf/2llc/default", &kernel_list).unwrap();
        assert_eq!(stripped, "gauntlet/eevdf/2llc/default");
        assert!(entry.is_none());

        let (stripped, entry) = strip_kernel_suffix("ktstr/eevdf", &[]).unwrap();
        assert_eq!(stripped, "ktstr/eevdf");
        assert!(entry.is_none());
    }

    /// In multi-kernel mode (2+ entries), the suffix is required and
    /// peeled off. The matching `KernelEntry` is returned.
    #[test]
    fn strip_kernel_suffix_multi_kernel_peels_suffix() {
        let kernel_list = vec![
            KernelEntry {
                sanitized: "kernel_6_14_2".to_string(),
                kernel_dir: PathBuf::from("/a"),
            },
            KernelEntry {
                sanitized: "kernel_6_15_0".to_string(),
                kernel_dir: PathBuf::from("/b"),
            },
        ];
        let (stripped, entry) =
            strip_kernel_suffix("gauntlet/eevdf/2llc/default/kernel_6_14_2", &kernel_list).unwrap();
        assert_eq!(stripped, "gauntlet/eevdf/2llc/default");
        assert_eq!(entry.unwrap().kernel_dir, PathBuf::from("/a"));

        let (stripped, entry) =
            strip_kernel_suffix("gauntlet/eevdf/2llc/default/kernel_6_15_0", &kernel_list).unwrap();
        assert_eq!(stripped, "gauntlet/eevdf/2llc/default");
        assert_eq!(entry.unwrap().kernel_dir, PathBuf::from("/b"));
    }

    /// In multi-kernel mode, a test name that lacks the kernel
    /// suffix surfaces an actionable error rather than silently
    /// using the first kernel — the suffix is part of every test
    /// name `--list` emitted, so a missing suffix indicates
    /// operator hand-construction or stale tooling.
    #[test]
    fn strip_kernel_suffix_multi_kernel_missing_suffix_errors() {
        let kernel_list = vec![
            KernelEntry {
                sanitized: "kernel_6_14_2".to_string(),
                kernel_dir: PathBuf::from("/a"),
            },
            KernelEntry {
                sanitized: "kernel_6_15_0".to_string(),
                kernel_dir: PathBuf::from("/b"),
            },
        ];
        let err = strip_kernel_suffix("gauntlet/eevdf/2llc/default", &kernel_list)
            .expect_err("missing suffix in multi-kernel mode must error");
        assert!(
            err.contains("no recognised kernel suffix"),
            "error must mention missing suffix, got: {err}",
        );
    }

    /// Suffix peeling is anchored at the end of the test name —
    /// gauntlet variants whose body contains `/` (the preset /
    /// profile separator) are not accidentally peeled. A naive
    /// `rsplit_once('/')` would peel the profile segment instead.
    #[test]
    fn strip_kernel_suffix_does_not_peel_profile_segment() {
        let kernel_list = vec![
            KernelEntry {
                sanitized: "kernel_6_14_2".to_string(),
                kernel_dir: PathBuf::from("/a"),
            },
            KernelEntry {
                sanitized: "kernel_6_15_0".to_string(),
                kernel_dir: PathBuf::from("/b"),
            },
        ];
        // The profile name is `default`, NOT `kernel_6_14_2` — the
        // peeler must require an EXACT match against a known
        // sanitized label, not just any `/<word>` ending.
        let (stripped, entry) =
            strip_kernel_suffix("gauntlet/eevdf/2llc/default/kernel_6_14_2", &kernel_list).unwrap();
        // Stripped name still contains all three of the original
        // path segments (eevdf, 2llc, default).
        assert_eq!(stripped, "gauntlet/eevdf/2llc/default");
        assert!(entry.is_some());
    }
}
