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
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, Ordering};

use anyhow::Result;

use crate::assert::AssertResult;

/// Deferred nextest dispatch name set by the ctor's argv rewrite.
/// When the ctor sees `--exact ktstr/...` or `--exact gauntlet/...`,
/// it stores the full prefixed name here and overwrites the argv
/// bytes to the bare test name so libtest can match the `#[test]`
/// wrapper at main-time (after C++ static constructors complete).
/// `run_ktstr_test` checks this global to resolve the gauntlet
/// topology or multi-kernel variant from the original name.
static DEFERRED_DISPATCH: Mutex<Option<String>> = Mutex::new(None);

/// Raw argv pointer captured at process startup via the
/// `.init_array.00001` constructor below. `AtomicPtr` rather than
/// `static mut` so the capture (Release) and the later read in
/// `rewrite_argv_exact` (Acquire) form a happens-before edge — the
/// argv string bytes glibc placed before invoking `.init_array`
/// callbacks are visible to the rewrite via that synchronization
/// edge. The element type is `*mut libc::c_char` (matching argv's
/// `*const *mut c_char` shape); cast back to `*const *mut c_char`
/// before indexing in the consumer.
static RAW_ARGV: AtomicPtr<*mut libc::c_char> = AtomicPtr::new(std::ptr::null_mut());

/// Numbered link section `.init_array.00001` to force CAPTURE_ARGV
/// to run BEFORE the unprioritized dispatch ctor (`.init_array`
/// without a numeric suffix). GNU ld sorts numbered
/// `.init_array.NN` sections numerically and places them before
/// plain `.init_array`. The profraw ctor in profraw.rs uses ctor's
/// `priority = 0` mechanism (which expands to `.init_array.0`) for
/// the same reason; this entry uses `.00001` so profraw still runs
/// first, then CAPTURE_ARGV, then the unprioritized dispatch ctor
/// at `ktstr_test_early_dispatch`.
///
/// Cannot use `#[ctor::ctor(priority = N)]` directly because the
/// ctor crate's macro emits a 0-arg `extern "C" fn() -> CtorRetType`.
/// glibc passes `(argc, argv, envp)` to `.init_array` callbacks and
/// argv capture requires the 2-arg signature. Without the numbered
/// section, link order between this static and the dispatch ctor is
/// unspecified, so the dispatch ctor's argv-rewrite step would race
/// against capture and observe a null `RAW_ARGV`.
#[unsafe(link_section = ".init_array.00001")]
#[used]
static CAPTURE_ARGV: unsafe extern "C" fn(libc::c_int, *const *mut libc::c_char) = {
    unsafe extern "C" fn capture(_argc: libc::c_int, argv: *const *mut libc::c_char) {
        // Release pairs with Acquire in `rewrite_argv_exact`.
        RAW_ARGV.store(argv as *mut *mut libc::c_char, Ordering::Release);
    }
    capture
};

use super::{
    KTSTR_TESTS, KtstrTestEntry, TopoOverride, collect_sidecars, extract_export_output_arg,
    extract_export_test_arg, extract_test_fn_arg, extract_topo_arg, find_test,
    format_callback_profile, format_kvm_stats, format_verifier_stats, maybe_dispatch_vm_test,
    parse_topo_string, propagate_rust_env_from_cmdline, record_skip_sidecar, resolve_test_kernel,
    run_ktstr_test_inner, sidecar_dir, try_flush_profraw,
};

/// Check if an `anyhow::Error` carries a [`ResourceContention`].
///
/// Walks the FULL error chain via `e.chain().any(...)` so a
/// `ResourceContention` wrapped in `.context(...)` (e.g. the
/// `eval.rs` `"build ktstr_test VM"` and `"run ktstr_test VM"`
/// wrappers) is still recognised — the macro's match arm depends on
/// this.
///
/// Used by the `#[ktstr_test]` macro expansion to short-circuit on
/// host-resource contention (LLC slots / CPUs unavailable, KVM fd
/// budget exhausted, ENOMEM): the macro emits the canonical
/// `ktstr: SKIP: resource contention: ...` banner and early-returns
/// so libtest sees pass. The skip sidecar is recorded at every
/// contention site inside `run_ktstr_test_inner`, so stats tooling
/// still sees the skip without a panic-driven nextest retry. `pub`
/// because the macro-generated `#[test]` body in `ktstr-macros`
/// references it by absolute path; `#[doc(hidden)]` keeps it out
/// of rustdoc's public surface — it is plumbing, not user API.
///
/// [`ResourceContention`]: crate::vmm::host_topology::ResourceContention
/// Check if an error is a host topology mismatch (e.g. test
/// requests 2 LLCs but host has 1). String-match because the
/// error is a plain `anyhow::Error`, not a typed error.
#[doc(hidden)]
pub fn is_topology_insufficient(e: &anyhow::Error) -> bool {
    let msg = format!("{e:#}");
    msg.contains("need") && (msg.contains("LLC") || msg.contains("CPU"))
}

#[doc(hidden)]
pub fn is_resource_contention(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::vmm::host_topology::ResourceContention>()
            .is_some()
    })
}

/// Predicate: walks the [`anyhow::Error`] chain looking for a
/// [`KernelUnavailable`] cause. Used by the `#[ktstr_test]`
/// macro's wrapper to distinguish "harness not configured" (skip)
/// from "test failed" (panic).
///
/// The harness signals "I have no kernel to boot, the binary was
/// likely invoked outside `cargo ktstr test`" by surfacing
/// [`KernelUnavailable`] rather than a generic
/// `anyhow::bail!`. The macro wrapper emits the canonical
/// `ktstr: SKIP: harness not configured: ...` banner and
/// early-returns so libtest sees pass — same shape as the
/// resource-contention SKIP arm. `pub` because the macro-generated
/// `#[test]` body in `ktstr-macros` references it by absolute
/// path; `#[doc(hidden)]` keeps it out of rustdoc — plumbing, not
/// user API.
///
/// [`KernelUnavailable`]: crate::test_support::eval::KernelUnavailable
#[doc(hidden)]
pub fn is_kernel_unavailable(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<crate::test_support::eval::KernelUnavailable>()
            .is_some()
    })
}

/// Overwrite the argv string at index `arg_idx` with `replacement`.
///
/// Uses `RAW_ARGV` captured by the `.init_array.00001` constructor
/// above. The replacement MUST be shorter than or equal to the
/// original string — the function null-terminates within the
/// existing buffer.
///
/// When `RAW_ARGV` is still null (the capture ctor did not fire,
/// or fired AFTER this caller — a regression in `.init_array`
/// ordering), emits a stderr diagnostic and returns. Without the
/// warning, the deferred-dispatch path would silently fail to
/// rewrite argv and libtest would not match the bare test name,
/// surfacing as an opaque "no test matches" failure further
/// downstream rather than a clear ordering regression.
fn rewrite_argv_exact(arg_idx: usize, replacement: &str) {
    // Acquire pairs with Release in CAPTURE_ARGV's `capture`.
    let raw = RAW_ARGV.load(Ordering::Acquire) as *const *mut libc::c_char;
    if raw.is_null() {
        eprintln!(
            "ktstr: rewrite_argv_exact called before CAPTURE_ARGV ctor fired \
             (RAW_ARGV is null). Deferred dispatch will not match libtest's \
             test name and the run will likely fail. This indicates a \
             regression in `.init_array` ordering — the `.init_array.00001` \
             section above must be linked before the unprioritized dispatch \
             ctor.",
        );
        return;
    }
    // SAFETY:
    //   (a) RAW_ARGV was set by glibc passing the live `argv` array
    //       to CAPTURE_ARGV — the array is the program's actual
    //       argument vector and remains valid for the lifetime of
    //       the process.
    //   (b) argv strings on Linux live in the high end of the
    //       process stack and are writable by the program (the
    //       `setproctitle(3)`-style argv-overwrite trick relies on
    //       exactly this). POSIX does not require this, but on
    //       Linux/glibc — the only platform ktstr targets — argv
    //       string memory is mutable.
    //   (c) `replacement.len() <= original_len` is checked before
    //       any write, so the in-place overwrite + null terminator
    //       stays inside the original allocation. No bytes past
    //       the original null terminator are touched.
    //   (d) This function is only reachable from
    //       `ktstr_test_early_dispatch`, an `.init_array` ctor.
    //       glibc invokes `.init_array` callbacks on the main
    //       thread before any user thread has spawned, so the
    //       argv overwrite is single-threaded and race-free.
    unsafe {
        let arg = *raw.add(arg_idx);
        if arg.is_null() {
            return;
        }
        let original_len = libc::strlen(arg as *const libc::c_char);
        if replacement.len() > original_len {
            return;
        }
        let dst = arg as *mut u8;
        std::ptr::copy_nonoverlapping(replacement.as_ptr(), dst, replacement.len());
        *dst.add(replacement.len()) = 0;
    }
}

/// A nextest-safe kernel identifier whose construction is gated
/// through [`sanitize_kernel_label`] — once a value of this type
/// exists, the contained string is GUARANTEED to match the
/// `kernel_[a-z0-9_]+` shape that nextest's test-name parsing
/// accepts. The wrapped `String` is private so a future caller
/// cannot bypass [`Self::new`] and stuff a raw label into the
/// invariant.
///
/// Constructed by [`Self::new`] (which always calls
/// [`sanitize_kernel_label`]). Read access is via
/// [`Self::as_str`] / `Display` / `AsRef<str>` — both of which
/// expose the sanitized form unchanged.
///
/// `pub(crate)` because every consumer (this module, the
/// production parser at [`parse_kernel_list`], and the encoder
/// helpers in `cargo-ktstr` that thread labels through
/// `parse_kernel_list`) lives inside the workspace; no external
/// surface is needed today. If a future external consumer needs
/// to construct a `SanitizedKernelLabel` directly, expose
/// `Self::new` as `pub` then — but the private inner stays a
/// private invariant either way.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SanitizedKernelLabel(String);

impl SanitizedKernelLabel {
    /// Sanitize `raw` via [`sanitize_kernel_label`] and wrap the
    /// result in the invariant-preserving newtype. The only path
    /// that produces a `SanitizedKernelLabel`; bypassing it is
    /// impossible because the inner field is private to this
    /// module.
    pub(crate) fn new(raw: &str) -> Self {
        Self(sanitize_kernel_label(raw))
    }

    /// Read access to the sanitized identifier. Returns `&str`
    /// rather than `&String` so callers can compose with
    /// `format!` / `starts_with` / `strip_suffix` without
    /// chaining `.as_str().as_str()`.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SanitizedKernelLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for SanitizedKernelLabel {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// `PartialEq<&str>` and `PartialEq<str>` impls let `assert_eq!`
// against a string literal stay readable in tests
// (`assert_eq!(entries[0].sanitized, "kernel_6_14_2")`) without
// forcing every consumer to chain `.as_str()`. The wrapped
// `String` is private to this module, so impls comparing
// against external `&str` values cannot break the
// "constructor enforces sanitization" invariant — the
// invariant attaches to value PRODUCTION, not to value
// COMPARISON.
impl PartialEq<&str> for SanitizedKernelLabel {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<str> for SanitizedKernelLabel {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

#[cfg(test)]
impl SanitizedKernelLabel {
    /// Test-only escape hatch: wrap a string that's ALREADY in
    /// the sanitized shape (`kernel_[a-z0-9_]+`) without running
    /// the sanitizer. Used by unit-test fixtures that hand-roll
    /// `KernelEntry` values whose `sanitized` field is meant to
    /// be a literal — running [`Self::new`] on `"kernel_6_14_2"`
    /// would double-prefix to `"kernel_kernel_6_14_2"`.
    ///
    /// Production code must NEVER call this — invariant
    /// violation here means callers can stuff arbitrary strings
    /// into the field, defeating the point of the newtype.
    /// `#[cfg(test)]` enforces that at compile time.
    pub(crate) fn from_pre_sanitized_for_test(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// One resolved kernel entry from `KTSTR_KERNEL_LIST` (the multi-
/// kernel fan-out wire format that `cargo ktstr test --kernel A
/// --kernel B` or `cargo ktstr verifier --kernel A --kernel B`
/// exports before exec'ing into `cargo nextest`).
///
/// `label` is the producer-side label string before
/// sanitization — e.g. `"6.14.2"` for Version, `"git_tj_sched_ext_for-next"`
/// for Git, `"6.14.2-tarball-x86_64-kc..."` for CacheKey,
/// `"path_linux_a3f2b1"` for Path. Preserved so the
/// [`crate::test_support::dispatch`] verifier sweep filter can
/// compare against `declare_scheduler!`'s `kernels = [...]`
/// declarations — specifically, range membership
/// (`"6.14..6.16"` vs `"6.14.2"`) needs the raw version string
/// to feed into [`crate::kernel_path::decompose_version_for_compare`],
/// which the sanitized form has lost (slashes / dots → underscores).
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
/// [`SanitizedKernelLabel::new`] (which calls [`sanitize_kernel_label`])
/// applies the `kernel_` prefix and `[a-z0-9_]+` normalization
/// downstream. The newtype on this field makes the invariant
/// compile-checked: a future caller cannot construct a
/// `KernelEntry` whose `sanitized` field skipped sanitization.
///
/// `kernel_dir` is the canonical absolute path to the kernel-build
/// directory the per-variant subprocess re-exports as
/// `KTSTR_KERNEL`.
#[derive(Clone, Debug)]
pub(crate) struct KernelEntry {
    pub(crate) label: String,
    pub(crate) sanitized: SanitizedKernelLabel,
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
                label: label.to_string(),
                sanitized: SanitizedKernelLabel::new(label),
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

    // Export-self dispatch runs BEFORE host/guest test dispatch.
    // `cargo ktstr export` is a router that exec's the test binary
    // with `--ktstr-export-test=NAME`; the binary reads its own
    // `KTSTR_TESTS` registry, embeds itself via `current_exe`, and
    // writes the .run file. Running this check first means the
    // export path never accidentally triggers VM boot if the
    // operator simultaneously passes `--ktstr-test-fn` (the export
    // arg wins because export is a one-shot tool, not a test
    // execution).
    if let Some(code) = maybe_dispatch_export() {
        std::process::exit(code);
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
    //
    // For `--list`, ktstr_main prints the gauntlet/ktstr names and
    // RETURNS so the standard libtest harness can print its own list
    // of `#[test]` items afterward. This makes plain `#[test]`
    // functions inside a ktstr_test integration-test binary visible
    // to nextest — without the fall-through, libtest never runs and
    // those test names are silently dropped from the listing.
    //
    // For `--exact`, ktstr_main runs only when the test name starts
    // with `ktstr/` or `gauntlet/` — names ktstr owns. Other names
    // (libtest #[test] items, including the per-entry wrappers
    // emitted by `#[ktstr_test]` itself) fall through to libtest's
    // dispatch. Without this guard, run_named_test would fail
    // `find_test` for a plain `#[test]` name and exit 1, blocking
    // nextest from running it.
    if std::env::var_os("NEXTEST").is_some() {
        let has_real_tests = KTSTR_TESTS.iter().any(|e| !is_test_sentinel(e.name));
        // A binary may carry only `declare_scheduler!` declarations
        // (no `#[ktstr_test]` entries) — pure verifier-only test
        // binaries. Without the scheduler check below the listing
        // branch would never fire for such a binary and the
        // verifier cells would silently fail to emit under nextest.
        let has_schedulers = !super::KTSTR_SCHEDULERS.is_empty();
        if has_real_tests || has_schedulers {
            let args: Vec<String> = std::env::args().collect();
            if args.iter().any(|a| a == "--list") {
                ktstr_list_only();
                list_verifier_cells_all();
                list_plain_tests();
                std::process::exit(0);
            } else if let Some(pos) = args.iter().position(|a| a == "--exact")
                && let Some(name) = args.get(pos + 1)
                && name.starts_with("verifier/")
            {
                // verifier/<sched>/<kernel>/<preset> cells bypass
                // libtest entirely — the cell handler resolves the
                // scheduler binary, kernel, and preset topology, runs
                // collect_verifier_output, prints the result, and
                // exits. No #[test] wrapper exists for declared
                // schedulers (declare_scheduler! only emits a static),
                // so argv-rewrite + DEFERRED_DISPATCH doesn't apply.
                let code = run_verifier_cell(name);
                try_flush_profraw();
                std::process::exit(code);
            } else if let Some(pos) = args.iter().position(|a| a == "--exact")
                && let Some(name) = args.get(pos + 1)
                && (name.starts_with("ktstr/") || name.starts_with("gauntlet/"))
            {
                let bare = name
                    .strip_prefix("ktstr/")
                    .or_else(|| name.strip_prefix("gauntlet/"))
                    .unwrap_or(name)
                    .split('/')
                    .next()
                    .unwrap_or(name);

                // Reject malformed names like `gauntlet/` (trailing slash,
                // no test name) and `ktstr/`. Writing an empty replacement
                // into argv would null-terminate at offset 0, leaving an
                // empty string libtest would fail to match against any
                // `#[test]` wrapper — surfacing as an opaque "no test
                // matches" error instead of a clear malformed-name error.
                if bare.is_empty() {
                    eprintln!(
                        "ktstr: malformed --exact test name {name:?} \
                         (resolves to an empty bare name after prefix strip)",
                    );
                    std::process::exit(1);
                }

                *DEFERRED_DISPATCH.lock().unwrap() = Some(name.to_string());
                rewrite_argv_exact(pos + 1, bare);
            }
        }
    } else {
        // cargo-test-direct path: the standard rustc test harness
        // runs only the bare `#[test]` wrappers `#[ktstr_test]`
        // generates. Gauntlet expansion (topology-preset variants)
        // lives inside `ktstr_main`'s `--list` + `--exact` handlers
        // and is reachable ONLY under nextest. Every real ktstr
        // entry produces topology-preset variants under nextest
        // (`for_each_gauntlet_variant` iterates
        // `crate::vm::gauntlet_presets()`). Without nextest those
        // variants would silently not run — coverage loss with no
        // error. Emit a one-shot stderr `warning:` diagnostic (see
        // the `eprintln!` below) when the binary carries any real
        // entry so the user sees the gap instead of trusting a
        // false green. Print once per process (cargo test invokes
        // one test binary per crate; the ctor runs exactly once per
        // test binary) so there is no need to gate with a
        // std::sync::Once.
        //
        // `KTSTR_CARGO_TEST_MODE=1` opts out of the warning: the
        // operator deliberately picked the cargo-test-direct path
        // (e.g. for a single-test debug iteration without the
        // nextest harness) and accepts that gauntlet variants
        // won't run. The warning is still emitted under bare
        // `cargo test` without the env var set so unaware users
        // see the coverage gap.
        if !super::runtime::cargo_test_mode_active() {
            let total = KTSTR_TESTS.len();
            let real = KTSTR_TESTS
                .iter()
                .filter(|e| !is_test_sentinel(e.name))
                .count();
            if real > 0 {
                eprintln!(
                    "warning: {real} of {total} ktstr test entries registered in this binary \
                     will not generate their topology-preset gauntlet variants — NEXTEST env \
                     var is not set and the standard rustc harness does not expand them. Use \
                     `cargo nextest run` (or `cargo ktstr test`) to exercise the full gauntlet, \
                     or set KTSTR_CARGO_TEST_MODE=1 to opt into single-variant bare-`cargo test` \
                     mode without this warning.",
                );
            }
            // Verifier cells are emitted by `list_verifier_cells_all`
            // which runs ONLY from the NEXTEST listing branch above.
            // A bare `cargo test` invocation on a binary carrying
            // `declare_scheduler!` declarations gets zero verifier
            // coverage — surface the gap with the same opt-out shape
            // as the gauntlet warning so an unaware operator does not
            // trust a green run that never reached the verifier.
            // Eevdf + KernelBuiltin variants don't produce userspace
            // binaries to verify, so they are excluded from the count
            // (matching the emission-time filter in
            // `list_verifier_cells_all`).
            let verifier_schedulers = super::KTSTR_SCHEDULERS
                .iter()
                .filter(|s| {
                    !matches!(
                        s.binary,
                        super::SchedulerSpec::Eevdf | super::SchedulerSpec::KernelBuiltin { .. }
                    )
                })
                .count();
            if verifier_schedulers > 0 {
                eprintln!(
                    "warning: {verifier_schedulers} `declare_scheduler!` declaration(s) in this \
                     binary will not generate verifier cells — NEXTEST env var is not set and \
                     verifier cells are emitted only by ktstr's `--list` handler under nextest. \
                     Use `cargo ktstr verifier` to exercise the verifier sweep, or set \
                     KTSTR_CARGO_TEST_MODE=1 to acknowledge the verifier-cell-free path without \
                     this warning.",
                );
            }
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

/// Export-self dispatch: if `--ktstr-export-test=NAME` is present in
/// argv, look up `NAME` in the binary's own `KTSTR_TESTS` registry,
/// build a self-extracting `.run` file embedding `current_exe()`
/// (this binary), and exit. Returns `Some(exit_code)` when dispatched,
/// `None` when the flag is absent.
///
/// `cargo ktstr export <NAME>` (the cargo-ktstr binary) is a router
/// that compiles the workspace's tests, locates the test binary that
/// owns `NAME`, and exec's it with this arg. The test binary embeds
/// ITSELF — without that indirection, cargo-ktstr would package its
/// own binary, which has no `#[ktstr_test]` registrations from the
/// user's crate and can't reproduce the test on bare metal.
///
/// `--ktstr-export-output=PATH` overrides the default output path
/// (`<NAME>.run` in the cwd). Both flags are leniently parsed by the
/// helpers in `args.rs`; an empty NAME (`--ktstr-export-test=`)
/// surfaces with diagnostic "requires a non-empty test name" and
/// exit 1 so the router moves on to the next candidate.
///
/// # Exit-code contract
///
/// The router (`cargo-ktstr.rs::run_export`) discriminates between
/// "this binary doesn't know the test" (exit 1) and "this binary
/// has the test but rejects it" (exit 2). When ANY candidate exits
/// 2, the router surfaces THAT candidate's stderr (the rejection
/// reason: host_only, bpf_map_write, KernelBuiltin) rather than
/// the generic "not found in any workspace test binary" message.
/// Without the differentiation, an operator who exports a
/// host_only test would see the misleading "not found" diagnostic
/// even though the test exists.
fn maybe_dispatch_export() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_export_test_arg(&args)?;
    let output = extract_export_output_arg(&args).map(std::path::PathBuf::from);

    // Empty name: surface as a hard error rather than silently
    // succeeding. The router's "first binary that exits 0 wins"
    // protocol relies on the absent-test path returning a non-zero
    // exit so the next candidate is tried.
    if name.is_empty() {
        eprintln!("ktstr export: --ktstr-export-test= requires a non-empty test name");
        return Some(1);
    }

    // Look up the test ourselves so we can discriminate "not
    // registered here" (exit 1, router falls through) from
    // "registered but rejected" (exit 2, router surfaces this
    // stderr). `export_test` itself returns anyhow::Error for both
    // cases, which would conflate them at the exit-code level.
    if find_test(name).is_none() {
        eprintln!("ktstr export: no registered test named '{name}'");
        return Some(1);
    }

    match crate::export::export_test(name, output) {
        Ok(()) => Some(0),
        Err(e) => {
            eprintln!("ktstr export: {e:#}");
            // The test exists in this binary but the export pipeline
            // refused it (host_only / bpf_map_write / KernelBuiltin /
            // I/O error). Exit 2 so the router prefers this stderr
            // over a sibling binary's exit-1 "not registered" miss.
            Some(2)
        }
    }
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

    match run_ktstr_test_with_topo(entry, &topo) {
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

    // Check if the ctor deferred a prefixed dispatch name via argv
    // rewrite. If so, resolve the topology from the full
    // gauntlet/multi-kernel name instead of using the entry defaults.
    if let Some(deferred) = DEFERRED_DISPATCH.lock().unwrap().take() {
        return run_deferred_dispatch(entry, &deferred);
    }

    if entry.host_only {
        return run_host_only_test_inner(entry);
    }
    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        anyhow::bail!("vmlinux not found, bpf_map_write requires vmlinux");
    }
    run_ktstr_test_inner(entry, None)
}

/// Dispatch a test using the full prefixed name that the ctor stored
/// in [`DEFERRED_DISPATCH`] before rewriting argv. Resolves the
/// gauntlet topology preset and multi-kernel suffix from the name,
/// then calls `run_ktstr_test_inner` directly — NOT through the
/// `run_named_test` → `result_to_exit_code` path, so
/// `ResourceContention` propagates as `Err` rather than being
/// swallowed as exit code 0. Called at main-time from the `#[test]`
/// wrapper, so C++ static constructors have completed.
fn run_deferred_dispatch(_entry: &KtstrTestEntry, deferred_name: &str) -> Result<AssertResult> {
    let kernel_list = read_kernel_list();
    let (test_name, kernel_entry) = strip_kernel_suffix(deferred_name, &kernel_list)
        .map_err(|e| anyhow::anyhow!("deferred dispatch for '{deferred_name}': {e}"))?;
    if let Some(ke) = kernel_entry {
        export_kernel_for_variant(ke);
    }

    if let Some(rest) = test_name.strip_prefix("gauntlet/") {
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        anyhow::ensure!(parts.len() == 2, "invalid gauntlet name: gauntlet/{rest}");
        let (bare, preset_name) = (parts[0], parts[1]);
        let entry = find_test(bare).ok_or_else(|| anyhow::anyhow!("unknown test: {bare}"))?;
        let presets = crate::vm::gauntlet_presets();
        let preset = presets
            .iter()
            .find(|p| p.name == preset_name)
            .ok_or_else(|| anyhow::anyhow!("unknown preset: {preset_name}"))?;
        let t = &preset.topology;
        let memory_mb = (t.total_cpus() * 64).max(256).max(entry.memory_mb);
        let topo = TopoOverride {
            numa_nodes: t.numa_nodes,
            llcs: t.llcs,
            cores: t.cores_per_llc,
            threads: t.threads_per_core,
            memory_mb,
        };
        return run_ktstr_test_inner(entry, Some(&topo));
    }

    let bare = test_name.strip_prefix("ktstr/").unwrap_or(test_name);
    let entry = find_test(bare).ok_or_else(|| anyhow::anyhow!("unknown test: {bare}"))?;
    run_ktstr_test_inner(entry, None)
}

/// Like `run_ktstr_test` but with an explicit topology override.
/// Only consumed inside this module by `maybe_dispatch_host_test`;
/// kept as a named helper so the `--ktstr-test-fn` + `--ktstr-topo`
/// dispatch path reads symmetrically with the zero-override
/// [`run_ktstr_test`] library entry point.
fn run_ktstr_test_with_topo(entry: &KtstrTestEntry, topo: &TopoOverride) -> Result<AssertResult> {
    run_ktstr_test_inner(entry, Some(topo))
}

/// Run a test result through expect_err logic and return an exit code.
///
/// Returns 0 on pass, 1 on failure. `ResourceContention` returns
/// 0 — the test never ran, not a real failure. The skip sidecar for
/// this case is written upstream in `run_ktstr_test_inner` at the
/// ResourceContention propagation site so every caller (including
/// the library entry point `run_ktstr_test`) records it, not just
/// the nextest dispatch path.
///
/// `ResourceContention` detection walks the FULL error chain via
/// [`is_resource_contention`] (chain-walk predicate) plus a
/// matching `e.chain().find_map(...)` extraction for the reason
/// string. The eval-side `eval.rs` `"build ktstr_test VM"` and
/// `"run ktstr_test VM"` wrappers nest the contention error under
/// `.context(...)`, so a top-level `downcast_ref` on the outer
/// error misses the inner cause. Without the chain walk a wrapped
/// contention would land in the `Err(e)` arm below as a regular
/// failure (exit 1) rather than the skip path (exit 0), turning
/// every host-resource-exhausted run into a hard test failure.
fn result_to_exit_code(result: Result<AssertResult>, expect_err: bool) -> i32 {
    let no_skip = std::env::var_os("KTSTR_NO_SKIP_MODE").is_some();
    match result {
        Ok(_) if expect_err => {
            eprintln!("expected error but test passed");
            1
        }
        Ok(_) => 0,
        Err(e) if is_resource_contention(&e) => {
            let reason = e
                .chain()
                .find_map(|c| {
                    c.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                        .map(|rc| rc.reason.clone())
                })
                .unwrap_or_else(|| "<unknown>".to_string());
            if no_skip {
                eprintln!(
                    "ktstr: FAIL: resource contention under --no-skip-mode: {reason}. \
                     Either provision hardware that satisfies the test's topology \
                     requirement, or drop --no-skip-mode / KTSTR_NO_SKIP_MODE to \
                     accept the skip."
                );
                1
            } else {
                crate::report::test_skip(format_args!("resource contention: {reason}"));
                0
            }
        }
        Err(e) if is_topology_insufficient(&e) => {
            if no_skip {
                eprintln!(
                    "ktstr: FAIL: host topology insufficient under --no-skip-mode: {e:#}. \
                     Either provision a host with the required CPU / LLC count, or drop \
                     --no-skip-mode / KTSTR_NO_SKIP_MODE to accept the skip."
                );
                1
            } else {
                crate::report::test_skip(format_args!("host topology insufficient: {e:#}"));
                0
            }
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

/// Walk [`KTSTR_TESTS`] once per process and emit a stderr
/// `warning:` line for every duplicate `name` found.
///
/// Two entries with the same name would both match `find_test(name)`
/// (which returns the FIRST match), so the second registration is
/// silently shadowed — `cargo ktstr` would dispatch the first entry
/// and the second entry's body would never run, with no diagnostic
/// surfaced. The warning surfaces the collision so an operator can
/// rename one of the `#[ktstr_test]` functions; discovery itself
/// proceeds (find_test's first-wins behavior continues) so nextest's
/// `--list` output still lands in stdout. A panic here would abort
/// the whole listing — nextest would see no tests at all rather
/// than a partial set with a clear warning. The first-wins
/// shadowing remains a real bug, but the diagnostic is louder than
/// silence and the tradeoff (operator sees the warning AND a
/// usable test list) beats the alternative (operator sees a
/// panic backtrace and no test list).
///
/// `OnceLock<()>` gates the walk to fire EXACTLY ONCE per process:
/// every gauntlet variant resolves through `list_tests` (under
/// nextest's discovery and budget paths), so without the gate a
/// run with N variants would re-walk the slice N times and emit
/// the same warning N times. Each duplicate name surfaces exactly
/// once via the inner `seen`/`warned` HashSet pair so a
/// triple-collision (three entries sharing one name) does not
/// double-print the warning.
///
/// The pure detection logic lives in
/// [`warn_duplicate_test_names_inner`] so the duplicate-walker
/// is testable without process-wide global state. This wrapper
/// only owns the `OnceLock<()>` gate and the
/// `(KTSTR_TESTS, stderr)` plumbing.
fn warn_duplicate_test_names_once() {
    static CHECKED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CHECKED.get_or_init(|| {
        warn_duplicate_test_names_inner(KTSTR_TESTS.iter().map(|e| e.name), &mut std::io::stderr());
    });
}

/// Pure walker behind [`warn_duplicate_test_names_once`]: walks
/// the test-name iterator and emits one `warning:` line per
/// duplicate name to `sink`. Each duplicate name surfaces
/// exactly once (a triple-collision does NOT double-print)
/// via the inner `warned` HashSet.
///
/// Extracted from the OnceLock-gated wrapper so the duplicate
/// detection logic is testable without process-wide global
/// state — the wrapper handles "fire once per process" via its
/// own `OnceLock<()>` gate; this inner is a pure function over
/// `(names, sink)`. The wrapper passes
/// `KTSTR_TESTS.iter().map(|e| e.name)` as the iterator and
/// `std::io::stderr()` as the sink.
///
/// `Result<(), std::io::Error>` is collapsed to ignore-on-write
/// because the production wrapper writes to stderr where IO
/// errors are unrecoverable; tests pass a `Vec<u8>` sink which
/// never errors. The function name says "warn" — diagnostic
/// channel — and matches the wrapper's pre-existing
/// `eprintln!` semantics.
fn warn_duplicate_test_names_inner<'a, W: std::io::Write>(
    names: impl IntoIterator<Item = &'a str>,
    sink: &mut W,
) {
    use std::collections::HashSet;
    let names: Vec<&'a str> = names.into_iter().collect();
    let mut seen: HashSet<&'a str> = HashSet::with_capacity(names.len());
    let mut warned: HashSet<&'a str> = HashSet::new();
    for name in names {
        if !seen.insert(name) && warned.insert(name) {
            let _ = writeln!(
                sink,
                "warning: ktstr_test: duplicate test name {name:?} registered in KTSTR_TESTS — \
                 two `#[ktstr_test]` entries share this name; the SECOND entry is \
                 silently shadowed (find_test returns the first registration). \
                 rename one of the functions to disambiguate.",
            );
        }
    }
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
///
/// Calls [`warn_duplicate_test_names_once`] on the first invocation per
/// process so duplicate registrations surface a stderr `warning:`
/// line BEFORE any test name is printed (discovery itself proceeds
/// — find_test's first-wins behavior continues, but the operator
/// sees which name collided). Subsequent invocations are no-ops via
/// the inner `OnceLock` gate.
fn list_tests(ignored_only: bool) {
    warn_duplicate_test_names_once();
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
// host_capacity moved to crate::test_support::host_capacity for shared
// use by both dispatch.rs (gauntlet variant filter) and
// cargo_ktstr/verifier.rs (verifier sweep filter).

/// Iterate topology presets that both fit the host capacity and
/// match the entry's `TopologyConstraints`. Shared between the
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
    F: FnMut(&crate::vm::TopoPreset),
{
    let no_perf_mode = super::runtime::no_perf_mode_for_entry(entry);
    for preset in presets {
        // No-perf-mode tests run KVM-emulated topology — guest sees the
        // declared NUMA / LLC / per-LLC layout regardless of host
        // hardware — so the host-side LLC count and per-LLC CPU width
        // do not constrain preset eligibility. Only the total-CPU
        // budget survives.
        let accepted = if no_perf_mode {
            entry
                .constraints
                .accepts_no_perf_mode(&preset.topology, host_cpus)
        } else {
            entry.constraints.accepts(
                &preset.topology,
                host_cpus,
                host_llcs,
                host_max_cpus_per_llc,
            )
        };
        if !accepted {
            continue;
        }
        visit(preset);
    }
}

/// List all tests without budget filtering.
///
/// When `KTSTR_KERNEL_LIST` carries 2 or more entries, every test
/// name carries an extra `/{sanitized_kernel_label}` suffix so each
/// (test × kernel) pair becomes a distinct nextest test case;
/// nextest's parallelism, retries, and `-E` filtering all apply
/// natively. Single-kernel mode (0 or 1 entries) emits the
/// `gauntlet/{name}/{preset}` shape with no kernel suffix.
///
/// `KTSTR_CARGO_TEST_MODE=1` skips gauntlet variant emission and
/// the multi-kernel suffix path: each test gets exactly one
/// `ktstr/{name}: test` line. Bare `cargo test` doesn't have
/// access to the cargo-ktstr resolver that produces
/// `KTSTR_KERNEL_LIST`, so the multi-kernel branch can't apply
/// even if it were enabled — pin both behaviors explicitly so
/// the listing matches what the dispatch path will actually run.
fn list_tests_all(ignored_only: bool) {
    let cargo_test_mode = super::runtime::cargo_test_mode_active();
    let presets = crate::vm::gauntlet_presets();
    let has_vmlinux = resolve_test_kernel()
        .ok()
        .and_then(|k| crate::vmm::find_vmlinux(&k))
        .is_some();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = super::host_capacity();

    let kernel_list = read_kernel_list();
    let multi_kernel = kernel_list.len() > 1 && !cargo_test_mode;
    // Single-kernel mode (no list, or list has exactly one entry)
    // emits one variant per (test × preset) tuple with no kernel
    // suffix. Multi-kernel mode iterates every kernel as an outer
    // loop and appends `/{sanitized}` per variant. The empty-suffix
    // sentinel below is what the single-kernel branch passes to keep
    // the print path uniform.
    let kernel_suffixes: Vec<&str> = if multi_kernel {
        kernel_list.iter().map(|k| k.sanitized.as_str()).collect()
    } else {
        vec![""]
    };

    for entry in KTSTR_TESTS.iter() {
        // bpf_map_write tests require vmlinux to resolve BPF map
        // addresses. Don't list them when vmlinux is unavailable —
        // they cannot run and would produce false PASS results.
        if !entry.bpf_map_write.is_empty() && !has_vmlinux {
            continue;
        }

        if !ignored_only || is_ignored(entry) {
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

        // KTSTR_CARGO_TEST_MODE: skip gauntlet expansion. The
        // operator picked the bare-`cargo test` path; emit only
        // the base name so each `#[ktstr_test]` runs once with its
        // declared topology.
        if cargo_test_mode {
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
            |preset| {
                for suffix in &kernel_suffixes {
                    if suffix.is_empty() {
                        println!("gauntlet/{}/{}: test", entry.name, preset.name);
                    } else {
                        println!("gauntlet/{}/{}/{suffix}: test", entry.name, preset.name,);
                    }
                }
            },
        );
    }
}

/// True iff the given operator-resolved kernel `entry` matches one
/// of the `declared` kernel specs from a scheduler's
/// `declare_scheduler!` `kernels = [...]` declaration. Empty
/// `declared` accepts every entry (no per-scheduler filter).
///
/// Match semantics per spec variant (via [`crate::kernel_path::KernelId::parse`]):
/// - [`KernelId::Version`]: raw-label string equality OR sanitized-label match
///   ([`sanitize_kernel_label`] of the spec string equals the entry's
///   sanitized label). Direct match catches the common case where
///   the dispatcher resolved `--kernel 6.14.2` and the scheduler
///   declared `kernels = ["6.14.2"]`.
/// - [`KernelId::Range`]: range-membership check on the entry's raw
///   label via [`crate::kernel_path::decompose_version_for_compare`].
///   Lets schedulers declaring `kernels = ["6.14..6.16"]` match
///   any operator-supplied kernel whose version falls in
///   `[6.14, 6.16]` inclusive.
/// - [`KernelId::Path`] / [`KernelId::CacheKey`] / [`KernelId::Git`]:
///   sanitized-label equality — the producer-side encoder
///   (`cargo_ktstr/kernel/wire_format.rs`) emits a deterministic
///   label per variant (`path_…`, `git_owner_repo_ref`, version
///   prefix from cache key), so identical specs on both sides
///   produce identical sanitized labels.
///
/// [`KernelId`]: crate::kernel_path::KernelId
fn sched_kernel_filter_accepts(declared: &[&'static str], entry: &KernelEntry) -> bool {
    if declared.is_empty() {
        return true;
    }
    declared.iter().any(|spec| entry_matches_spec(entry, spec))
}

/// Single-spec match helper for [`sched_kernel_filter_accepts`].
/// Parses `spec` via [`crate::kernel_path::KernelId::parse`] and
/// dispatches on the variant. Pure logic — no network, no FS.
fn entry_matches_spec(entry: &KernelEntry, spec: &str) -> bool {
    use crate::kernel_path::{KernelId, decompose_version_for_compare};
    match KernelId::parse(spec) {
        KernelId::Version(spec_ver) => {
            entry.label == spec_ver
                || entry.sanitized.as_str() == sanitize_kernel_label(&spec_ver)
        }
        KernelId::Range { start, end } => {
            let Some(entry_t) = decompose_version_for_compare(&entry.label) else {
                return false;
            };
            let Some(start_t) = decompose_version_for_compare(&start) else {
                return false;
            };
            let Some(end_t) = decompose_version_for_compare(&end) else {
                return false;
            };
            entry_t >= start_t && entry_t <= end_t
        }
        KernelId::CacheKey(_) | KernelId::Path(_) | KernelId::Git { .. } => {
            entry.sanitized.as_str() == sanitize_kernel_label(spec)
        }
    }
}

/// Format the `KTSTR_KERNEL_LIST is empty` diagnostic emitted by
/// [`run_verifier_cell`] when a verifier cell name reaches the cell
/// handler with no kernel-list to look the label up in. Extracted
/// from the inline eprintln! so the exact wording can be pinned in
/// unit tests without spawning a process.
fn format_empty_kernel_list_error(full_name: &str) -> String {
    format!(
        "ktstr verifier: cell {full_name}: KTSTR_KERNEL_LIST is empty. \
         Direct `--exact verifier/...` invocation outside `cargo ktstr verifier` \
         is not supported — the dispatcher owns kernel-set resolution. Run \
         `cargo ktstr verifier [--kernel SPEC]` instead.",
    )
}

/// Format the "kernel label not in KTSTR_KERNEL_LIST" diagnostic.
/// `present` is the slice of sanitized labels actually present in
/// the list, in their KTSTR_KERNEL_LIST ordering. Extracted for the
/// same reason as [`format_empty_kernel_list_error`].
fn format_unknown_kernel_label_error(
    full_name: &str,
    kernel_label: &str,
    sched_name: &str,
    present: &[&str],
) -> String {
    format!(
        "ktstr verifier: cell {full_name}: kernel label {kernel_label:?} \
         not in KTSTR_KERNEL_LIST. Present labels: [{}]. \
         Either add --kernel <SPEC> to the dispatcher invocation so it \
         resolves into this label, or remove the matching entry from \
         declare_scheduler!(... kernels = [...]) for {sched_name}.",
        present.join(", "),
    )
}

/// Emit `verifier/<sched>/<kernel>/<preset>: test` lines — one per
/// (declared scheduler × kernel-list entry × accepted gauntlet
/// preset) cell. Mirrors the gauntlet emission pattern in
/// [`list_tests_all`] but walks [`super::KTSTR_SCHEDULERS`] instead
/// of [`KTSTR_TESTS`]. Cells are paired with the
/// [`run_verifier_cell`] handler registered in
/// [`ktstr_test_early_dispatch`]'s `--exact verifier/...` branch.
///
/// The matrix dimension is `KTSTR_KERNEL_LIST` (always populated by
/// the `cargo ktstr verifier` dispatcher — even with a single
/// auto-discovered kernel, the dispatcher synthesizes a one-entry
/// list with a derived label). Each scheduler's
/// `declare_scheduler!` `kernels = [...]` declaration acts as a
/// per-scheduler filter on the matrix — `Version` / `Range`
/// declarations match entries by raw-label equality / range
/// membership; `Path` / `CacheKey` / `Git` declarations match by
/// sanitized-label equality. An empty `kernels = []` declaration
/// accepts every entry in the list (no filter).
///
/// Acceptance filter mirrors the gauntlet branching in
/// [`for_each_gauntlet_variant`]: perf-mode pinning constrains
/// preset eligibility against the host's LLC width AND per-LLC CPU
/// width, while no-perf-mode (KVM-emulated topology) only needs the
/// total-CPU budget to fit. The mode is global for the verifier path
/// — there is no per-cell `performance_mode` attribute analogous to
/// `KtstrTestEntry::no_perf_mode` because every cell shares the same
/// `cargo ktstr verifier` invocation.
///
/// Schedulers declared with [`super::SchedulerSpec::Eevdf`] or
/// [`super::SchedulerSpec::KernelBuiltin`] are skipped at emission
/// time because neither has a userspace binary to load BPF programs
/// from — emitting cells that would always SKIP at execution wastes
/// nextest's per-cell process budget and clutters the run output.
///
/// Cell names with `/` in `sched.name` or `preset.name` would
/// corrupt the splitn-based parse in [`run_verifier_cell`]. The
/// emission elides such cells with a stderr warning so the operator
/// sees the gap rather than silently dropping cells.
///
/// When `KTSTR_KERNEL_LIST` is absent (direct binary invocation
/// outside the `cargo ktstr verifier` dispatcher), no cells emit.
/// Operators who invoke a test binary directly with `--exact
/// verifier/...` will see the cell handler's "kernel label not in
/// KTSTR_KERNEL_LIST" error.
fn list_verifier_cells_all() {
    use super::SchedulerSpec;
    let kernel_list = read_kernel_list();
    if kernel_list.is_empty() {
        return;
    }
    let presets = crate::vm::gauntlet_presets();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = super::host_capacity();
    let no_perf_mode = super::runtime::no_perf_mode_active();

    for sched in super::KTSTR_SCHEDULERS.iter() {
        if matches!(
            sched.binary,
            SchedulerSpec::Eevdf | SchedulerSpec::KernelBuiltin { .. }
        ) {
            continue;
        }
        if sched.name.contains('/') {
            eprintln!(
                "ktstr verifier: scheduler name {:?} contains '/' — skipping cell emission (would corrupt verifier/<sched>/<kernel>/<preset> parse)",
                sched.name,
            );
            continue;
        }
        for kernel_entry in &kernel_list {
            if !sched_kernel_filter_accepts(sched.kernels, kernel_entry) {
                continue;
            }
            for preset in presets.iter() {
                if preset.name.contains('/') {
                    eprintln!(
                        "ktstr verifier: preset name {:?} contains '/' — skipping cell (would corrupt parse)",
                        preset.name,
                    );
                    continue;
                }
                let accepted = if no_perf_mode {
                    sched
                        .constraints
                        .accepts_no_perf_mode(&preset.topology, host_cpus)
                } else {
                    sched.constraints.accepts(
                        &preset.topology,
                        host_cpus,
                        host_llcs,
                        host_max_cpus_per_llc,
                    )
                };
                if !accepted {
                    continue;
                }
                println!(
                    "verifier/{}/{}/{}: test",
                    sched.name,
                    kernel_entry.sanitized,
                    preset.name,
                );
            }
        }
    }
}

/// Parse `verifier/<sched_name>/<kernel_label>/<preset_name>`, look
/// up the declared scheduler in [`super::KTSTR_SCHEDULERS`] + the
/// gauntlet preset in [`crate::vm::gauntlet_presets`] + the kernel
/// in [`KTSTR_KERNEL_LIST_ENV`](crate::KTSTR_KERNEL_LIST_ENV),
/// resolve the scheduler binary path per
/// [`super::SchedulerSpec`], boot the verifier VM via
/// [`crate::verifier::collect_verifier_output`], and print the
/// rendered output. Returns 0 on success, 1 on failure /
/// malformed cell name.
///
/// The per-cell kernel directory is resolved by sanitized-label
/// lookup in `KTSTR_KERNEL_LIST` — the
/// `cargo ktstr verifier` dispatcher always populates the list,
/// even with no `--kernel` flag (it synthesizes a single auto-
/// discovered entry). There is no single-kernel-mode fallback.
/// An unrecognised label or an absent list both surface as an
/// exit-1 diagnostic naming the present labels and pointing at
/// the dispatcher.
///
/// Eevdf + KernelBuiltin scheduler variants are filtered out at
/// emission time in [`list_verifier_cells_all`], so nextest
/// dispatch never reaches the SKIP arms in this function. The
/// SKIP arms remain as defense-in-depth for direct
/// `--exact verifier/<eevdf>/...` invocation outside nextest
/// (the only path that bypasses the emission-time filter); in
/// that case they emit a `SKIP` banner + exit 0.
fn run_verifier_cell(full_name: &str) -> i32 {
    use super::SchedulerSpec;

    let rest = match full_name.strip_prefix("verifier/") {
        Some(r) => r,
        None => {
            eprintln!("ktstr verifier: missing 'verifier/' prefix in {full_name:?}");
            return 1;
        }
    };
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() != 3 {
        eprintln!(
            "ktstr verifier: malformed cell name {full_name:?}; expected verifier/<sched>/<kernel>/<preset>",
        );
        return 1;
    }
    let (sched_name, kernel_label, preset_name) = (parts[0], parts[1], parts[2]);

    // Emit the cell banner BEFORE every SKIP / FAIL branch so the
    // operator always sees which (scheduler, kernel, preset) tuple
    // produced the result. Without it an early-exit SKIP / FAIL would
    // surface as a bare error line nextest tags with the full cell
    // name but no per-axis context.
    println!("\n=== {sched_name} | kernel {kernel_label} | topology {preset_name} ===");

    // Fail-fast on missing KVM with the canonical actionable error
    // (kvm group / kvm-ok hint). Without this preflight the operator
    // gets a deep error inside VM bring-up.
    if let Err(e) = crate::cli::check_kvm() {
        eprintln!("ktstr verifier: cell {full_name}: {e:#}");
        return 1;
    }

    let Some(sched) = super::KTSTR_SCHEDULERS.iter().find(|s| s.name == sched_name) else {
        eprintln!("ktstr verifier: no declared scheduler {sched_name:?} (cell {full_name:?})",);
        return 1;
    };

    let preset_list = crate::vm::gauntlet_presets();
    let Some(preset) = preset_list.iter().find(|p| p.name == preset_name) else {
        eprintln!("ktstr verifier: no gauntlet preset {preset_name:?} (cell {full_name:?})",);
        return 1;
    };

    // Resolve the per-cell kernel directory by looking the cell's
    // sanitized label up in `KTSTR_KERNEL_LIST`. The
    // `cargo ktstr verifier` dispatcher always populates the list —
    // even with no `--kernel` flag it synthesizes a single auto-
    // discovered entry — so the lookup is the single source of
    // truth and there is no single-kernel-mode fallback that would
    // silently run a cell against an unrelated kernel.
    //
    // An empty list reaching this function means the test binary was
    // invoked outside the dispatcher (direct `--exact verifier/...`
    // under a hand-spawned nextest, for instance). Error with an
    // actionable message rather than fall through to auto-discovery.
    let kernel_list = read_kernel_list();
    let Some(kernel_entry) = kernel_list
        .iter()
        .find(|k| k.sanitized.as_str() == kernel_label)
    else {
        if kernel_list.is_empty() {
            eprintln!("{}", format_empty_kernel_list_error(full_name));
        } else {
            let present: Vec<&str> = kernel_list.iter().map(|k| k.sanitized.as_str()).collect();
            eprintln!(
                "{}",
                format_unknown_kernel_label_error(
                    full_name,
                    kernel_label,
                    sched_name,
                    &present,
                ),
            );
        }
        return 1;
    };

    let sched_bin: std::path::PathBuf = match sched.binary {
        SchedulerSpec::Discover(pkg) => match crate::build_and_find_binary(pkg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ktstr verifier: build scheduler {pkg:?}: {e:#}");
                return 1;
            }
        },
        SchedulerSpec::Path(p) => {
            let path = std::path::PathBuf::from(p);
            if !path.exists() {
                eprintln!("ktstr verifier: scheduler binary not found: {p}");
                return 1;
            }
            path
        }
        // Eevdf + KernelBuiltin are filtered at list time in
        // list_verifier_cells_all, so nextest dispatch never reaches
        // these arms. The SKIP arms remain as defense-in-depth for
        // direct `--exact verifier/<eevdf>/...` invocation outside
        // nextest.
        SchedulerSpec::Eevdf => {
            println!(
                "ktstr verifier: SKIP cell {full_name} (Eevdf has no userspace binary to verify)",
            );
            return 0;
        }
        SchedulerSpec::KernelBuiltin { .. } => {
            println!(
                "ktstr verifier: SKIP cell {full_name} (KernelBuiltin has no userspace binary to verify)",
            );
            return 0;
        }
    };

    let ktstr_bin = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "ktstr verifier: locate ktstr binary via current_exe() (required so the \
                 verifier VM can boot the same test binary as /init for guest-side dispatch): {e}",
            );
            return 1;
        }
    };

    let kernel_path = kernel_entry.kernel_dir.clone();
    let topology = super::TopologyJson::from(preset.topology);
    let sched_args: Vec<String> = sched.sched_args.iter().map(|s| s.to_string()).collect();

    // Raw mode is opt-in via the dispatcher's --raw flag, plumbed
    // through KTSTR_VERIFIER_RAW_ENV. Presence (any value, including
    // empty) enables raw rendering — matches the "set to any value"
    // semantics documented on the const and the dispatcher's
    // `cmd.env(KTSTR_VERIFIER_RAW_ENV, "1")` setter.
    let raw = std::env::var_os(crate::KTSTR_VERIFIER_RAW_ENV).is_some();

    match crate::verifier::collect_verifier_output(
        &sched_bin,
        &ktstr_bin,
        &kernel_path,
        &sched_args,
        topology,
    ) {
        Ok(result) => {
            let output = crate::verifier::format_verifier_output("verifier", &result, raw);
            print!("{output}");
            0
        }
        Err(e) => {
            eprintln!("ktstr verifier: cell {full_name} FAILED: {e:#}");
            1
        }
    }
}

/// List tests with budget-based coverage maximization.
///
/// Collects all eligible tests as candidates, runs greedy selection,
/// and prints only the selected subset. Multi-kernel mode adds the
/// kernel suffix as a feature dimension so the budget selector
/// picks per-kernel coverage; single-kernel mode is unchanged.
///
/// `KTSTR_CARGO_TEST_MODE=1` is treated identically to
/// `list_tests_all`: the budget pipeline runs only over base test
/// candidates (no gauntlet-variant candidates, no multi-kernel
/// fan-out). The greedy selector still applies — a low budget
/// can still trim the base list — but the candidate set is the
/// same set that the dispatch path would actually run.
fn list_tests_budget(ignored_only: bool, budget_secs: f64) {
    use crate::budget::{TestCandidate, estimate_duration, extract_features, select};

    let cargo_test_mode = super::runtime::cargo_test_mode_active();
    let presets = crate::vm::gauntlet_presets();
    let has_vmlinux = resolve_test_kernel()
        .ok()
        .and_then(|k| crate::vmm::find_vmlinux(&k))
        .is_some();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = super::host_capacity();
    let mut candidates: Vec<TestCandidate> = Vec::new();

    let kernel_list = read_kernel_list();
    let multi_kernel = kernel_list.len() > 1 && !cargo_test_mode;
    let kernel_suffixes: Vec<&str> = if multi_kernel {
        kernel_list.iter().map(|k| k.sanitized.as_str()).collect()
    } else {
        vec![""]
    };

    for entry in KTSTR_TESTS.iter() {
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
                    features: extract_features(entry, &base_topo, false, entry.name),
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
                        features: extract_features(entry, &base_topo, false, entry.name),
                        estimated_secs: estimate_duration(entry, &base_topo),
                    });
                }
            }
        }

        if entry.host_only {
            continue;
        }

        if cargo_test_mode {
            // No gauntlet candidates in cargo-test mode — the
            // dispatch path will never execute them and including
            // them in the budget candidate set would shift greedy
            // selection toward variants that resolve to "no test"
            // at run time.
            continue;
        }

        for_each_gauntlet_variant(
            entry,
            &presets,
            host_cpus,
            host_llcs,
            host_max_cpus_per_llc,
            |preset| {
                for suffix in &kernel_suffixes {
                    let test_name = if suffix.is_empty() {
                        format!("gauntlet/{}/{}", entry.name, preset.name)
                    } else {
                        format!("gauntlet/{}/{}/{suffix}", entry.name, preset.name)
                    };
                    candidates.push(TestCandidate {
                        name: format!("{test_name}: test"),
                        features: extract_features(entry, &preset.topology, true, &test_name),
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
    // `/` (gauntlet variants already do — `gauntlet/{name}/{preset}`),
    // so a naive `rsplit_once('/')` would accidentally peel the
    // preset segment instead.
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
    // Producer-side regression detection would catch that
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
/// (`gauntlet/{name}/{preset}`), and bare names (backward compat).
/// When `KTSTR_KERNEL_LIST` carries 2+ kernels,
/// VM-bound test names additionally end with
/// `/{sanitized_kernel_label}` — that suffix is peeled here and
/// the matching kernel directory is re-exported via
/// [`KTSTR_KERNEL_ENV`] before the dispatch continues. `host_only`
/// tests are short-circuited BEFORE the suffix peel: they never
/// boot a VM, so the kernel-suffix listing path emits one
/// `ktstr/{name}: test` entry without a kernel suffix regardless
/// of the kernel-list cardinality (see `list_tests_all` /
/// `list_tests_budget`), and routing them through
/// `strip_kernel_suffix` would surface as a "no recognised kernel
/// suffix" exit-1 error. Returns an exit code.
pub(crate) fn run_named_test(test_name: &str) -> i32 {
    let kernel_list = read_kernel_list();

    // host_only short-circuit: in multi-kernel mode, host_only tests
    // are listed without a `/{sanitized_kernel_label}` suffix (see
    // `list_tests_all` / `list_tests_budget`, which emit a single
    // `ktstr/{name}: test` line for host_only entries regardless of
    // the kernel-list cardinality — a host_only test never boots a
    // VM, so the kernel never affects what runs). Calling
    // `strip_kernel_suffix` on such a name in multi-kernel mode
    // would fail with the "no recognised kernel suffix" error and
    // misroute every host_only dispatch to exit 1.
    //
    // Resolve the host_only check from `find_test` BEFORE the
    // suffix peel so the multi-kernel branch only applies to
    // VM-bound tests. Single-kernel mode is unaffected — the
    // pass-through arm in `strip_kernel_suffix` returns the input
    // verbatim either way.
    let bare_for_lookup = test_name.strip_prefix("ktstr/").unwrap_or(test_name);
    if let Some(entry) = find_test(bare_for_lookup)
        && entry.host_only
    {
        return run_host_only_test(entry);
    }

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

    // Defense-in-depth: host_only re-check after suffix peel for the
    // edge case where the bare_for_lookup pre-strip lookup missed
    // (e.g. a future test name shape that doesn't match the
    // pre-strip form but does after the suffix peel).
    if entry.host_only {
        return run_host_only_test(entry);
    }

    if entry.performance_mode && super::runtime::no_perf_mode_active() {
        crate::report::test_skip(format_args!(
            "{}: test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
            bare_name,
        ));
        // See run_ktstr_test_inner for the sidecar-emission rationale.
        record_skip_sidecar(entry);
        return 0;
    }

    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("FAIL: vmlinux not found, bpf_map_write requires vmlinux");
        return 1;
    }

    let result = run_ktstr_test_inner(entry, None);
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
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo)
        .duration(entry.duration)
        .settle(std::time::Duration::ZERO)
        .assert(merged_assert)
        .build();
    (entry.func)(&ctx)
}

/// Run a gauntlet variant test. `rest` is `{name}/{preset}`.
pub(crate) fn run_gauntlet_test(rest: &str) -> i32 {
    let parts: Vec<&str> = rest.splitn(2, '/').collect();
    if parts.len() != 2 {
        eprintln!("invalid gauntlet test name: gauntlet/{rest}");
        return 1;
    }
    let (test_name, preset_name) = (parts[0], parts[1]);

    let entry = match find_test(test_name) {
        Some(e) => e,
        None => {
            eprintln!("unknown test: {test_name}");
            return 1;
        }
    };

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

    if entry.performance_mode && super::runtime::no_perf_mode_active() {
        crate::report::test_skip(format_args!(
            "{}: test requires performance_mode but --no-perf-mode or KTSTR_NO_PERF_MODE is active",
            test_name,
        ));
        record_skip_sidecar(entry);
        return 0;
    }

    if !entry.bpf_map_write.is_empty()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("FAIL: vmlinux not found, bpf_map_write requires vmlinux");
        return 1;
    }

    let result = run_ktstr_test_inner(entry, Some(&topo));
    result_to_exit_code(result, entry.expect_err)
}

/// Collect sidecar JSON files and return the full gauntlet analysis.
///
/// When `dir` is `Some`, reads sidecars from that directory. Otherwise
/// uses the default sidecar directory (`KTSTR_SIDECAR_DIR` override, or
/// `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{project_commit}/`,
/// where `{project_commit}` is the project HEAD short hex with
/// `-dirty` when the worktree differs).
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

/// Discover plain `#[test]` items by re-invoking the binary without
/// NEXTEST, reading libtest's `--list` output, and printing only
/// names that don't match any KTSTR_TESTS entry. This lets plain
/// tests coexist with `#[ktstr_test]` in the same binary without
/// duplicating the ktstr entries.
fn list_plain_tests() {
    use std::collections::HashSet;
    let ktstr_names: HashSet<&str> = KTSTR_TESTS.iter().map(|e| e.name).collect();

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.env_remove("NEXTEST");
    cmd.args(["--list", "--format", "terse"]);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::null());
    let output = match cmd.output() {
        Ok(o) => o,
        Err(_) => return,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let name = line.strip_suffix(": test").unwrap_or(line);
        if !ktstr_names.contains(name) && !name.is_empty() {
            println!("{line}");
        }
    }
}

/// `--list` subprotocol: emit ktstr/gauntlet test names without
/// exiting so the standard libtest harness can also print its own
/// test list afterward. This is what makes plain `#[test]` items
/// inside a ktstr_test integration-test binary visible to nextest.
///
/// Honours `--ignored` the same way [`ktstr_main`] does — when set,
/// only the ignored subset (gauntlet variants and `demo_` base
/// tests) is printed. Unlike `ktstr_main`, this function returns to
/// the caller after listing so the ctor's caller can fall through
/// to libtest's `main`.
fn ktstr_list_only() {
    let args: Vec<String> = std::env::args().collect();
    let ignored_only = args.iter().any(|a| a == "--ignored");
    list_tests(ignored_only);
}

/// Nextest protocol handler.
///
/// Called automatically by [`ktstr_test_early_dispatch`] when running
/// under nextest with `--exact <ktstr_or_gauntlet_name>`.
/// Not intended for direct use.
///
/// - `--list --format terse`: output `ktstr/{name}: test\n` for base
///   tests and `gauntlet/{name}/{preset}: test\n` for gauntlet
///   variants. (Discovery uses [`ktstr_list_only`] instead to allow
///   libtest to print its own list afterward; this branch is
///   preserved for direct callers of `ktstr_main`.)
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
    //   - `gauntlet/` prefix with malformed parts / unknown preset
    //   - bare names fall through to `ktstr/` lookup
    //
    // The routing invariant: `gauntlet/` always delegates to
    // `run_gauntlet_test`, every other prefix (including none)
    // delegates to the base-test path inside `run_named_test`.

    #[test]
    fn run_named_test_gauntlet_prefix_routes_to_run_gauntlet_test() {
        // Gauntlet names require two slash-separated parts after the
        // prefix (`{name}/{preset}`); a single-segment name is
        // rejected by `run_gauntlet_test`, proving the prefix routed
        // there and not into the base-test path (which would print
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
    fn run_gauntlet_test_rejects_name_with_fewer_than_two_parts() {
        // `rest` must split into exactly 2 parts (`{name}/{preset}`).
        // A single-segment name has no preset and is a format error.
        let exit = run_gauntlet_test("some_test_no_preset");
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
        // Well-formed two-part name whose test is not registered
        // in KTSTR_TESTS. Returns 1 via the find_test None branch,
        // never reaching preset lookup or VM spawn.
        let exit = run_gauntlet_test("__not_a_test__/tiny-1llc");
        assert_eq!(exit, 1);
    }

    #[test]
    fn run_gauntlet_test_rejects_unknown_preset() {
        // `__unit_test_dummy__` is registered in test_support::tests;
        // combined with a preset name that is not in
        // `gauntlet_presets`, the function returns 1 at the preset-
        // lookup branch.
        let exit = run_gauntlet_test("__unit_test_dummy__/__no_such_preset__");
        assert_eq!(exit, 1);
    }

    // ---------------------------------------------------------------
    // warn_duplicate_test_names_inner — pure duplicate-walker
    // ---------------------------------------------------------------
    //
    // The OnceLock-gated `warn_duplicate_test_names_once` wrapper is
    // process-wide and reads `KTSTR_TESTS` directly, so its emit-once
    // semantics aren't observable from a unit test (the gate may
    // already have fired from the production listing path during
    // nextest discovery, or from a sibling test in this file). The
    // pure inner walker is exposed exactly so its detection logic
    // is testable without that global state — these tests pin the
    // dedup invariants that actually matter:
    //   1. No duplicates → no output (zero-emit on clean input).
    //   2. Each duplicate name surfaces EXACTLY once even when the
    //      same name appears 3+ times (the warned-set prevents
    //      double-prints).
    //   3. The emitted line embeds the offending name in
    //      double-quoted form (the canonical `{name:?}` debug
    //      format the production message uses) so tooling can
    //      grep operator output for the collision.
    //   4. Distinct duplicate names each produce one line —
    //      independent collision groups must not blur into one.
    //   5. An empty input is a no-op (defensive: exhaust early
    //      before any HashSet alloc).

    /// No duplicates → empty sink. Pins the zero-emit base case.
    #[test]
    fn warn_duplicate_test_names_inner_no_duplicates_writes_nothing() {
        let mut sink = Vec::<u8>::new();
        warn_duplicate_test_names_inner(["alpha", "beta", "gamma"], &mut sink);
        assert!(
            sink.is_empty(),
            "clean input must produce zero diagnostic bytes; got {:?}",
            String::from_utf8_lossy(&sink),
        );
    }

    /// Empty input → no walking, no output. Defensive against a
    /// regression that would crash on a zero-element HashSet
    /// allocation or emit a spurious line on the empty path.
    #[test]
    fn warn_duplicate_test_names_inner_empty_input_writes_nothing() {
        let mut sink = Vec::<u8>::new();
        warn_duplicate_test_names_inner(std::iter::empty::<&str>(), &mut sink);
        assert!(
            sink.is_empty(),
            "empty input must emit nothing; got {:?}",
            String::from_utf8_lossy(&sink),
        );
    }

    /// One duplicate name appearing twice → exactly one warning
    /// line containing the duplicated name. Pins the basic
    /// emit-on-collision contract.
    #[test]
    fn warn_duplicate_test_names_inner_emits_warning_for_duplicate() {
        let mut sink = Vec::<u8>::new();
        warn_duplicate_test_names_inner(["alpha", "beta", "alpha"], &mut sink);
        let out = String::from_utf8(sink).expect("sink is utf-8");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines.len(),
            1,
            "single duplicate must emit exactly one line; got {lines:?}",
        );
        let line = lines[0];
        assert!(
            line.contains("warning: ktstr_test:"),
            "warning prefix must be present (operator-actionable signal); \
             got line: {line:?}",
        );
        // The duplicated name must appear in the canonical `{name:?}`
        // double-quoted form so grep tooling can pull it out.
        assert!(
            line.contains("\"alpha\""),
            "the duplicated name must appear in quoted form; got: {line:?}",
        );
        assert!(
            !line.contains("\"beta\""),
            "non-duplicate names must NOT appear in any warning; got: {line:?}",
        );
    }

    /// A triple-collision (same name appearing 3 times) emits the
    /// warning EXACTLY ONCE — the inner `warned` HashSet
    /// suppresses the second and third occurrences. Pins the
    /// "one warning per duplicated name" contract documented on
    /// the public wrapper.
    #[test]
    fn warn_duplicate_test_names_inner_triple_collision_emits_once() {
        let mut sink = Vec::<u8>::new();
        warn_duplicate_test_names_inner(["dup", "dup", "dup"], &mut sink);
        let out = String::from_utf8(sink).expect("sink is utf-8");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines.len(),
            1,
            "triple-collision must emit exactly one warning, not one-per-extra; \
             got {lines:?} — a regression that drops the warned-set guard would \
             surface here as 2 lines (one for the second, one for the third).",
        );
        assert!(
            lines[0].contains("\"dup\""),
            "warning must name the duplicated entry; got: {:?}",
            lines[0],
        );
    }

    /// Two distinct duplicate names → two warning lines (one per
    /// collision group). A regression that collapses every
    /// duplicate into a single warning would surface here as one
    /// line instead of two.
    #[test]
    fn warn_duplicate_test_names_inner_independent_duplicates_each_warn() {
        let mut sink = Vec::<u8>::new();
        warn_duplicate_test_names_inner(["alpha", "beta", "alpha", "gamma", "beta"], &mut sink);
        let out = String::from_utf8(sink).expect("sink is utf-8");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "two independent collision groups must produce two warnings; \
             got {lines:?}",
        );
        let body = lines.join("\n");
        assert!(
            body.contains("\"alpha\""),
            "first duplicate name must appear in output; got: {body:?}",
        );
        assert!(
            body.contains("\"beta\""),
            "second duplicate name must appear in output; got: {body:?}",
        );
        assert!(
            !body.contains("\"gamma\""),
            "non-duplicate `gamma` must NOT trigger a warning; got: {body:?}",
        );
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
        let (cpus, llcs, max_cpus_per_llc) = super::super::host_capacity();
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
            |preset| visited.push(preset.name.to_string()),
        );
        assert!(
            visited.is_empty(),
            "with host_cpus=1 host_llcs=1, no preset should be visited; \
             visited: {visited:?}"
        );
    }

    #[test]
    fn for_each_gauntlet_variant_visit_count_equals_accepted_preset_count() {
        // With generous host capacity (u32::MAX cpus/llcs/per-LLC),
        // every preset that `entry.constraints.accepts(...)` admits
        // must yield exactly one visit — no profile multiplier, no
        // duplicate visits per preset. Computing the expected count
        // from the same `accepts` predicate the function calls means
        // this assertion catches both directions of regression:
        //
        //   - a regression that double-visits each accepted preset
        //     produces `count == 2 * expected` (the weaker `>= 1`
        //     assertion this test replaced would have silently
        //     passed),
        //   - a regression that skips accepted presets (e.g. an
        //     inverted condition) produces `count < expected`.
        let presets = crate::vm::gauntlet_presets();
        let entry = find_test("__unit_test_dummy__").unwrap();
        let expected: usize = presets
            .iter()
            .filter(|p| {
                entry
                    .constraints
                    .accepts(&p.topology, u32::MAX, u32::MAX, u32::MAX)
            })
            .count();
        let mut count = 0;
        for_each_gauntlet_variant(entry, &presets, u32::MAX, u32::MAX, u32::MAX, |_| {
            count += 1
        });
        assert_eq!(
            count, expected,
            "post-flag-kill: visit count must equal the number of presets the \
             entry's constraints accept; one visit per preset, no profile multiplier",
        );
    }

    #[test]
    fn for_each_gauntlet_variant_monotonic_in_host_capacity() {
        // Comparative-baseline: giving the function MORE host capacity
        // can only let MORE presets pass the cap-size filter, never
        // fewer. The upper-bound assertion in
        // `for_each_gauntlet_variant_skips_presets_exceeding_host_capacity`
        // and the lower-bound assertion in
        // `..._visits_every_fitting_preset` both check one extreme;
        // this test anchors the monotonic relationship between them.
        // A regression that inverted the host-cap comparison (e.g.
        // `host_cpus < preset_cpus` → accept) would pass both
        // endpoint tests but fail here.
        let presets = crate::vm::gauntlet_presets();
        if presets.is_empty() {
            return;
        }
        let entry = find_test("__unit_test_dummy__").unwrap();
        let count_for = |cpus: u32, llcs: u32| {
            let mut n = 0;
            for_each_gauntlet_variant(entry, &presets, cpus, llcs, u32::MAX, |_| n += 1);
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

    /// `KernelEntry.label` preserves the producer-side label
    /// string verbatim. Pinned because the
    /// `sched_kernel_filter_accepts` range-membership branch reads
    /// the raw label to feed into `decompose_version_for_compare`
    /// (the sanitized form has lost dot separators required for
    /// version parsing).
    #[test]
    fn parse_kernel_list_preserves_label() {
        let entries = parse_kernel_list("6.14.2=/a;git_tj_sched_ext_main=/b;6.15-rc3=/c");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].label, "6.14.2");
        assert_eq!(entries[1].label, "git_tj_sched_ext_main");
        assert_eq!(entries[2].label, "6.15-rc3");
    }

    // ---------------------------------------------------------------
    // sched_kernel_filter_accepts + entry_matches_spec
    // (coverage for the per-scheduler kernel filter that gates
    // verifier cell emission against KTSTR_KERNEL_LIST)
    // ---------------------------------------------------------------

    /// Build a `KernelEntry` for filter testing without round-
    /// tripping through `parse_kernel_list`. Wraps the test-only
    /// `SanitizedKernelLabel::from_pre_sanitized_for_test` so
    /// fixtures can hand-write the exact label strings the
    /// production parser would emit.
    fn mk_entry(raw: &str, sanitized: &str, dir: &str) -> KernelEntry {
        KernelEntry {
            label: raw.to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test(sanitized),
            kernel_dir: PathBuf::from(dir),
        }
    }

    #[test]
    fn filter_accepts_everything_when_declared_empty() {
        // Empty sched.kernels means "no per-scheduler filter" — every
        // KTSTR_KERNEL_LIST entry passes.
        let e = mk_entry("6.14.2", "kernel_6_14_2", "/a");
        assert!(sched_kernel_filter_accepts(&[], &e));
        let weird = mk_entry("anything", "kernel_anything", "/b");
        assert!(sched_kernel_filter_accepts(&[], &weird));
    }

    #[test]
    fn filter_matches_version_by_label() {
        let e = mk_entry("6.14.2", "kernel_6_14_2", "/a");
        // Exact raw-label equality is the primary match path.
        assert!(entry_matches_spec(&e, "6.14.2"));
        // sched.kernels = ["6.14.2"] accepts this entry.
        assert!(sched_kernel_filter_accepts(&["6.14.2"], &e));
    }

    #[test]
    fn filter_matches_version_by_sanitized_label() {
        // Different raw label but the spec sanitizes to the same
        // sanitized form — match via the sanitized-equality fallback.
        // Example: spec "6.14.2" sanitizes to "kernel_6_14_2" and
        // the entry's sanitized label is the same.
        let e = mk_entry("6.14.2-tarball-x86_64-kcabc", "kernel_6_14_2", "/a");
        // label != "6.14.2", but sanitized matches.
        assert!(entry_matches_spec(&e, "6.14.2"));
    }

    #[test]
    fn filter_rejects_version_mismatch() {
        let e = mk_entry("6.15.0", "kernel_6_15_0", "/a");
        // Neither raw nor sanitized matches "6.14.2".
        assert!(!entry_matches_spec(&e, "6.14.2"));
        assert!(!sched_kernel_filter_accepts(&["6.14.2"], &e));
    }

    #[test]
    fn filter_matches_range_membership_inclusive() {
        // Range "6.14..6.16" (both endpoints inclusive). Entries
        // inside the range match; outside reject.
        let inside_low = mk_entry("6.14", "kernel_6_14", "/a");
        let inside_mid = mk_entry("6.15.3", "kernel_6_15_3", "/b");
        let inside_high = mk_entry("6.16", "kernel_6_16", "/c");
        let below = mk_entry("6.13.7", "kernel_6_13_7", "/d");
        let above = mk_entry("6.17.0", "kernel_6_17_0", "/e");

        assert!(entry_matches_spec(&inside_low, "6.14..6.16"));
        assert!(entry_matches_spec(&inside_mid, "6.14..6.16"));
        assert!(entry_matches_spec(&inside_high, "6.14..6.16"));
        assert!(!entry_matches_spec(&below, "6.14..6.16"));
        assert!(!entry_matches_spec(&above, "6.14..6.16"));
    }

    #[test]
    fn filter_matches_range_inclusive_form_too() {
        // `..=` spelling produces the same inclusive range as `..`
        // per KernelId::parse (both inclusive on both endpoints).
        let inside = mk_entry("6.15.0", "kernel_6_15_0", "/a");
        assert!(entry_matches_spec(&inside, "6.14..=6.16"));
        let above = mk_entry("6.17.0", "kernel_6_17_0", "/b");
        assert!(!entry_matches_spec(&above, "6.14..=6.16"));
    }

    #[test]
    fn filter_handles_unparseable_entry_label_in_range() {
        // Entry whose label isn't version-shaped (e.g. a Git
        // label) can't be in a version range — reject.
        let git_entry = mk_entry("git_tj_sched_ext_main", "kernel_git_tj_sched_ext_main", "/a");
        assert!(!entry_matches_spec(&git_entry, "6.14..6.16"));
    }

    #[test]
    fn filter_matches_path_spec_by_sanitized() {
        // Path specs match by sanitized-label equality.
        let e = mk_entry("path_linux_a3f2b1", "kernel_path_linux_a3f2b1", "/some/dir");
        // The matching spec for a Path uses the path-derived sanitized
        // form. A user-supplied "../linux" sanitizes differently from
        // the producer's path_kernel_label output, so a Path spec in
        // sched.kernels typically wouldn't be useful — but pin the
        // sanitized-equality path anyway.
        let same_path_spec = "/some/dir";
        // KernelId::parse("/some/dir") → Path. sanitize_kernel_label
        // turns it into "kernel_some_dir" — not equal to the entry's
        // "kernel_path_linux_a3f2b1". Reject.
        assert!(!entry_matches_spec(&e, same_path_spec));
    }

    #[test]
    fn filter_matches_cache_key_spec_by_sanitized() {
        // CacheKey spec matches when sanitized labels align.
        let e = mk_entry(
            "6.14.2-tarball-x86_64-kcabc",
            "kernel_6_14_2_tarball_x86_64_kcabc",
            "/cache/foo",
        );
        // The spec parsed as CacheKey sanitizes to the same form.
        assert!(entry_matches_spec(
            &e,
            "6.14.2-tarball-x86_64-kcabc",
        ));
    }

    #[test]
    fn filter_accepts_when_any_declared_spec_matches() {
        // Multiple declared specs; entry matches one of them.
        let e = mk_entry("6.15.3", "kernel_6_15_3", "/a");
        assert!(sched_kernel_filter_accepts(
            &["6.14.2", "6.14..6.16", "git+https://example.com/r#main"],
            &e,
        ));
    }

    #[test]
    fn filter_rejects_when_no_declared_spec_matches() {
        let e = mk_entry("7.0.0", "kernel_7_0_0", "/a");
        assert!(!sched_kernel_filter_accepts(
            &["6.14.2", "6.14..6.16"],
            &e,
        ));
    }

    // ---------------------------------------------------------------
    // Pin the exact diagnostic strings emitted by run_verifier_cell
    // when the kernel-list lookup fails. Tests exercise the formatter
    // helpers directly — no need to spawn a separate test binary
    // because the eprintln! call sites now route through these pure
    // formatters.
    // ---------------------------------------------------------------

    #[test]
    fn format_empty_kernel_list_error_names_cell_and_dispatcher() {
        let s = format_empty_kernel_list_error(
            "verifier/sched_foo/kernel_6_14_2/tiny-1llc",
        );
        // Cell name appears verbatim so the operator can grep their
        // own invocation for the failing cell.
        assert!(
            s.contains("verifier/sched_foo/kernel_6_14_2/tiny-1llc"),
            "missing cell name in: {s}",
        );
        // Root cause is named explicitly.
        assert!(s.contains("KTSTR_KERNEL_LIST is empty"), "missing cause: {s}");
        // Actionable hint points back at the dispatcher subcommand
        // (the only supported entry point).
        assert!(
            s.contains("cargo ktstr verifier"),
            "missing actionable hint: {s}",
        );
    }

    #[test]
    fn format_unknown_kernel_label_error_lists_present_labels_and_both_fix_paths() {
        let present = vec!["kernel_6_14_2", "kernel_6_15_0"];
        let s = format_unknown_kernel_label_error(
            "verifier/sched_foo/kernel_7_0_0/tiny-1llc",
            "kernel_7_0_0",
            "sched_foo",
            &present,
        );
        // Cell name + missing label appear so operators see exactly
        // which lookup failed.
        assert!(
            s.contains("verifier/sched_foo/kernel_7_0_0/tiny-1llc"),
            "missing cell name: {s}",
        );
        // Debug-formatted missing label (`{kernel_label:?}` produces
        // double-quoted output).
        assert!(s.contains("\"kernel_7_0_0\""), "missing debug label: {s}");
        // Present-labels enumeration: every entry must appear so the
        // operator can see what IS available.
        assert!(s.contains("kernel_6_14_2"), "missing present[0]: {s}");
        assert!(s.contains("kernel_6_15_0"), "missing present[1]: {s}");
        // Scheduler name surfaces in the declaration-side fix hint.
        assert!(s.contains("sched_foo"), "missing scheduler name: {s}");
        // Both fix paths are documented: add a kernel to the
        // dispatcher OR drop the matching entry from the declaration.
        assert!(s.contains("add --kernel"), "missing dispatcher-side fix: {s}");
        assert!(
            s.contains("declare_scheduler!"),
            "missing declaration-side fix: {s}",
        );
    }

    #[test]
    fn format_unknown_kernel_label_error_empty_present_renders_empty_brackets() {
        // Edge case: kernel_list has entries that fail the find()
        // (string equality drifted) but the present slice the caller
        // assembles is empty — still surfaces the bracket pair so the
        // diagnostic format is uniform with the non-empty case.
        let s = format_unknown_kernel_label_error(
            "verifier/foo/kernel_x/tiny",
            "kernel_x",
            "foo",
            &[],
        );
        assert!(s.contains("Present labels: []"), "missing empty brackets: {s}");
    }

    #[test]
    fn format_unknown_kernel_label_error_joins_present_with_comma_space() {
        // Three-entry present slice must render comma-space separated
        // to match the `present.join(", ")` contract.
        let present = vec!["a", "b", "c"];
        let s = format_unknown_kernel_label_error(
            "verifier/foo/kernel_x/tiny",
            "kernel_x",
            "foo",
            &present,
        );
        assert!(
            s.contains("Present labels: [a, b, c]"),
            "wrong join delimiter: {s}",
        );
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

    // ---------------------------------------------------------------
    // SanitizedKernelLabel — newtype invariants
    // ---------------------------------------------------------------
    //
    // `SanitizedKernelLabel::new(raw)` is the only production
    // path that yields a value of the type, and it always routes
    // through `sanitize_kernel_label`. The tests below pin each
    // surface (constructor, accessors, `PartialEq` impls) directly
    // so a regression that bypassed the sanitizer (e.g. a future
    // `From<String>` impl that wraps verbatim) or dropped one of
    // the comparison-ergonomics impls would surface as a unit-test
    // failure rather than as a downstream filter mismatch.

    /// `SanitizedKernelLabel::new(raw)` yields a value whose
    /// `as_str()` equals `sanitize_kernel_label(raw)` byte-for-
    /// byte. Pins the constructor's "always sanitize" contract:
    /// the only path that builds a `SanitizedKernelLabel` MUST run
    /// `sanitize_kernel_label`, otherwise a future caller could
    /// stuff a raw label into the field via a regression that
    /// forgot to invoke the sanitizer. Multiple inputs covered to
    /// distinguish "happens to match" from "truly routes through
    /// the sanitizer" — version, RC suffix, mixed case, embedded
    /// dots-vs-dashes.
    #[test]
    fn sanitized_kernel_label_new_runs_sanitizer() {
        for raw in [
            "6.14.2",
            "6.15-rc3",
            "ABC-DEF",
            "git_tj_sched_ext_for-next",
            "",
        ] {
            let label = SanitizedKernelLabel::new(raw);
            assert_eq!(
                label.as_str(),
                sanitize_kernel_label(raw),
                "SanitizedKernelLabel::new({raw:?}).as_str() must equal \
                 sanitize_kernel_label({raw:?}); a regression that wrapped \
                 raw input verbatim would surface here",
            );
        }
    }

    /// `as_str()` returns the sanitized inner string, NOT the raw
    /// input. Round-trip through `as_str()` matches what the
    /// sanitizer produced — distinct from
    /// `sanitized_kernel_label_new_runs_sanitizer` which checks
    /// the constructor wires through; this checks that read-side
    /// access exposes the SAME bytes the constructor wrote.
    #[test]
    fn sanitized_kernel_label_as_str_returns_sanitized_form() {
        let label = SanitizedKernelLabel::new("6.14.2");
        assert_eq!(label.as_str(), "kernel_6_14_2");
        // A regression that returned the raw input from `as_str()`
        // would surface here because the raw input contains `.`
        // which is `_` after sanitization.
        assert_ne!(label.as_str(), "6.14.2");
    }

    /// `PartialEq<&str>` lets `assert_eq!(label, "kernel_6_14_2")`
    /// stay readable at every consumer (and is what
    /// `parse_kernel_list_*` tests already use). A regression that
    /// dropped or narrowed the impl (e.g. switched to `PartialEq<
    /// String>` only) would force every consumer to chain
    /// `.as_str()` and break the existing test suite.
    #[test]
    fn sanitized_kernel_label_partial_eq_with_str_ref() {
        let label = SanitizedKernelLabel::new("6.14.2");
        let want: &str = "kernel_6_14_2";
        assert_eq!(label, want);
        // Symmetric inequality: distinct sanitization output must
        // NOT compare equal to a different `&str`.
        let other: &str = "kernel_6_15_0";
        assert_ne!(label, other);
    }

    /// `PartialEq<str>` covers the unsized-`str` comparison path
    /// (e.g. dereferenced `String` slice) distinct from the
    /// `&str` path above. Both impls are explicit because Rust's
    /// auto-deref to `&str` does not bridge the `PartialEq<str>`
    /// case for some assert macros / generic comparators. A
    /// regression that dropped just the `PartialEq<str>` impl
    /// would compile most consumer sites but break callers that
    /// land on the unsized form.
    #[test]
    fn sanitized_kernel_label_partial_eq_with_str_unsized() {
        let label = SanitizedKernelLabel::new("6.14.2");
        let owned: String = "kernel_6_14_2".to_string();
        // `*owned` is `str` (unsized) — exercises `PartialEq<str>`
        // rather than `PartialEq<&str>`. Wrap with `&` to satisfy
        // `PartialEq::eq`'s `&Self`-vs-`&Other` shape; the impl
        // body still operates on the unsized `str`.
        assert!(
            label == *owned.as_str(),
            "PartialEq<str> impl missing — assert against unsized str failed",
        );
        // Symmetric inequality for the unsized path.
        let other: String = "kernel_6_15_0".to_string();
        assert!(label != *other.as_str());
    }

    /// `strip_kernel_suffix` is a no-op for single-kernel mode (0 or
    /// 1 entries) — returns the input verbatim and signals "no
    /// kernel override needed."
    #[test]
    fn strip_kernel_suffix_single_kernel_passthrough() {
        let kernel_list = vec![KernelEntry {
            label: "6.14.2".to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
            kernel_dir: PathBuf::from("/a"),
        }];
        let (stripped, entry) = strip_kernel_suffix("gauntlet/eevdf/2llc", &kernel_list).unwrap();
        assert_eq!(stripped, "gauntlet/eevdf/2llc");
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
                label: "6.14.2".to_string(),
                sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
                kernel_dir: PathBuf::from("/a"),
            },
            KernelEntry {
                label: "6.15.0".to_string(),
                sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_15_0"),
                kernel_dir: PathBuf::from("/b"),
            },
        ];
        let (stripped, entry) =
            strip_kernel_suffix("gauntlet/eevdf/2llc/kernel_6_14_2", &kernel_list).unwrap();
        assert_eq!(stripped, "gauntlet/eevdf/2llc");
        assert_eq!(entry.unwrap().kernel_dir, PathBuf::from("/a"));

        let (stripped, entry) =
            strip_kernel_suffix("gauntlet/eevdf/2llc/kernel_6_15_0", &kernel_list).unwrap();
        assert_eq!(stripped, "gauntlet/eevdf/2llc");
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
                label: "6.14.2".to_string(),
                sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
                kernel_dir: PathBuf::from("/a"),
            },
            KernelEntry {
                label: "6.15.0".to_string(),
                sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_15_0"),
                kernel_dir: PathBuf::from("/b"),
            },
        ];
        let err = strip_kernel_suffix("gauntlet/eevdf/2llc", &kernel_list)
            .expect_err("missing suffix in multi-kernel mode must error");
        assert!(
            err.contains("no recognised kernel suffix"),
            "error must mention missing suffix, got: {err}",
        );
    }

    /// Suffix peeling is anchored at the end of the test name —
    /// gauntlet variants whose body contains `/` (the test / preset
    /// separator) are not accidentally peeled. A naive
    /// `rsplit_once('/')` would peel the preset segment instead.
    #[test]
    fn strip_kernel_suffix_does_not_peel_preset_segment() {
        let kernel_list = vec![
            KernelEntry {
                label: "6.14.2".to_string(),
                sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
                kernel_dir: PathBuf::from("/a"),
            },
            KernelEntry {
                label: "6.15.0".to_string(),
                sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_15_0"),
                kernel_dir: PathBuf::from("/b"),
            },
        ];
        // The preset name is `2llc`, NOT `kernel_6_14_2` — the
        // peeler must require an EXACT match against a known
        // sanitized label, not just any `/<word>` ending.
        let (stripped, entry) =
            strip_kernel_suffix("gauntlet/eevdf/2llc/kernel_6_14_2", &kernel_list).unwrap();
        // Stripped name still contains both of the original path
        // segments (eevdf, 2llc).
        assert_eq!(stripped, "gauntlet/eevdf/2llc");
        assert!(entry.is_some());
    }

    // ---------------------------------------------------------------
    // host_only kernel-suffix skip — multi-kernel listing
    // ---------------------------------------------------------------
    //
    // `list_tests_all` and `list_tests_budget` short-circuit
    // `host_only` entries: a host_only test never boots a VM, so the
    // kernel never affects what runs. Both listers emit ONE entry per
    // host_only test regardless of `KTSTR_KERNEL_LIST` cardinality —
    // otherwise N identical copies of the same host-side function
    // would land in nextest's plan.
    //
    // The tests below register a dedicated `host_only=true` entry in
    // `KTSTR_TESTS` via `linkme::distributed_slice`, set
    // `KTSTR_KERNEL_LIST` to a 2-entry payload, and capture stdout
    // while invoking each lister. The capture asserts the host_only
    // entry name appears EXACTLY once and never with a `/kernel_…`
    // suffix.

    /// Process-wide mutex serializing every stdout-capture call in
    /// this module. `fd 1 → tempfile` redirection is a non-reentrant
    /// process-global mutation — two concurrent callers would see
    /// each other's output land in their sink. Mirrors the
    /// `STDERR_CAPTURE_LOCK` pattern in `test_support::test_helpers`
    /// (see `capture_stderr_serializes_concurrent_callers` for the
    /// rationale and the failure mode without serialization).
    static STDOUT_CAPTURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that restores the saved stdout fd on Drop, even if
    /// the captured closure panics under the `panic = "unwind"` test
    /// profile. Without this guard a panicking closure would leak the
    /// fd-1 swap and every subsequent stdout write in the test
    /// process would land in the orphaned tempfile. `saved` is
    /// `Option` so Drop can `take()` and consume it without an `&mut`
    /// borrow fight. Mirrors `StderrRestoreGuard` in
    /// `test_support::test_helpers`.
    struct StdoutRestoreGuard {
        saved: Option<std::os::fd::OwnedFd>,
    }
    impl Drop for StdoutRestoreGuard {
        fn drop(&mut self) {
            if let Some(saved) = self.saved.take() {
                let _ = nix::unistd::dup2_stdout(&saved);
            }
        }
    }

    /// Run `f` with stdout redirected to an in-memory tempfile;
    /// return both `f`'s value and the captured bytes. Uses
    /// [`STDOUT_CAPTURE_LOCK`] to serialize against every other
    /// stdout-capture call in this module. The RAII
    /// [`StdoutRestoreGuard`] restores fd 1 even if `f` panics
    /// under `panic = "unwind"`. Mirrors `capture_stderr` in
    /// `test_support::test_helpers`.
    fn capture_stdout<R>(f: impl FnOnce() -> R) -> (R, Vec<u8>) {
        use std::io::{Read, Seek, SeekFrom, Write};
        let _lock = STDOUT_CAPTURE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut sink = tempfile::tempfile().expect("create stdout-capture tempfile");
        // Flush before redirect: println! is line-buffered behind
        // the Stdout lock; pre-call bytes need to reach the
        // ORIGINAL fd 1 or they leak into the captured tempfile.
        std::io::stdout().flush().ok();
        let saved = nix::unistd::dup(std::io::stdout()).expect("dup(stdout)");
        nix::unistd::dup2_stdout(&sink).expect("dup2_stdout(sink)");
        let guard = StdoutRestoreGuard { saved: Some(saved) };
        let result = f();
        std::io::stdout().flush().ok();
        drop(guard);
        sink.seek(SeekFrom::Start(0)).expect("rewind sink");
        let mut bytes = Vec::new();
        sink.read_to_end(&mut bytes).expect("read sink");
        (result, bytes)
    }

    /// Stub func for the host_only listing-test entry. The listers
    /// never invoke `func` — they only iterate `KTSTR_TESTS` to
    /// emit names — so an Err-returning stub is sufficient and
    /// matches the pattern `default_test_func` uses in
    /// `entry::DEFAULT`. If a future regression accidentally drove
    /// dispatch through this entry, the bail message would surface
    /// the misroute rather than silently passing.
    fn host_only_listing_stub(
        _ctx: &crate::scenario::Ctx,
    ) -> anyhow::Result<crate::assert::AssertResult> {
        anyhow::bail!(
            "host_only_listing_test_entry::func called — entry exists \
             only to drive the host_only kernel-suffix skip tests in \
             list_tests_all / list_tests_budget; func should never run"
        )
    }

    /// Distinct sentinel name so the listing-output filters in the
    /// tests below match this entry and not the `__unit_test_dummy__`
    /// (also registered in `KTSTR_TESTS` from the
    /// `test_support::tests` module) or any other entry that may be
    /// added later. The `__unit_test_…__` shape collides with
    /// `is_test_sentinel` (see the predicate at the top of this
    /// module) so the `cargo test` harness still classifies it as a
    /// sentinel and the early-dispatch warning logic does not
    /// double-fire.
    const HOST_ONLY_LISTING_NAME: &str = "__unit_test_host_only_listing__";

    #[linkme::distributed_slice(KTSTR_TESTS)]
    static __HOST_ONLY_LISTING_ENTRY: KtstrTestEntry = KtstrTestEntry {
        name: HOST_ONLY_LISTING_NAME,
        func: host_only_listing_stub,
        host_only: true,
        ..KtstrTestEntry::DEFAULT
    };

    /// Two-kernel KTSTR_KERNEL_LIST payload reused by the listing
    /// tests below. Both labels sanitize to distinct nextest
    /// suffixes (`kernel_6_14_2`, `kernel_6_15_0`), so a regression
    /// that started emitting `/kernel_…` suffixes for the host_only
    /// entry would surface as either `2` matches (one per kernel)
    /// rather than the expected `1`, or as suffix substrings on the
    /// emitted line.
    const TWO_KERNEL_LIST: &str = "6.14.2=/cache/a;6.15.0=/cache/b";

    /// Filter the captured listing output to only the lines that
    /// reference `HOST_ONLY_LISTING_NAME`. Other lines from the
    /// `__unit_test_dummy__` entry (and from any future entries
    /// registered in this binary's `KTSTR_TESTS`) are intentionally
    /// dropped so the assertions key on this fixture's behaviour
    /// alone.
    fn host_only_listing_lines(captured: &[u8]) -> Vec<String> {
        std::str::from_utf8(captured)
            .expect("capture must be UTF-8")
            .lines()
            .filter(|l| l.contains(HOST_ONLY_LISTING_NAME))
            .map(str::to_owned)
            .collect()
    }

    /// `list_tests_all` in multi-kernel mode emits exactly ONE line
    /// for a `host_only` entry, with NO `/kernel_…` suffix. Pins the
    /// `if entry.host_only { println!("ktstr/{}: test", entry.name); }`
    /// branch at the top of `list_tests_all` against a regression
    /// that fell through into the kernel-suffix loop. A regression
    /// would yield 2 matches (one per kernel) and at least one line
    /// would carry a `/kernel_6_14_2` or `/kernel_6_15_0` suffix.
    ///
    /// Holds [`crate::test_support::test_helpers::lock_env`] for the
    /// full save/mutate/restore window — `KTSTR_KERNEL_LIST` is
    /// process-wide, and the budget-test sibling below also rewrites
    /// env vars. Without the lock, a concurrent test mutating a
    /// different env key could observe a transiently-corrupt
    /// `KTSTR_KERNEL_LIST` value.
    #[test]
    fn list_tests_all_host_only_skips_kernel_suffix_under_multi_kernel() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _env_lock = lock_env();
        let _kernel_list = EnvVarGuard::set(crate::KTSTR_KERNEL_LIST_ENV, TWO_KERNEL_LIST);
        // Suppress the budget-mode branch — `KTSTR_BUDGET_SECS` would
        // route the dispatcher through `list_tests_budget` instead of
        // `list_tests_all`, but we are calling the lister directly so
        // the dispatcher path is irrelevant. Removing the env var here
        // is defensive against a parallel test that set it without
        // restoring (would not affect this call's output, but keeps
        // the test's runtime hypothesis explicit).
        let _budget_guard = EnvVarGuard::remove("KTSTR_BUDGET_SECS");

        let (_, captured) = capture_stdout(|| list_tests_all(false));
        let lines = host_only_listing_lines(&captured);

        assert_eq!(
            lines.len(),
            1,
            "list_tests_all must emit exactly 1 line for a host_only entry \
             under multi-kernel mode (saw {n}): {lines:?}",
            n = lines.len(),
        );
        let line = &lines[0];
        // Expected exact form (mirrors the `println!("ktstr/{}: test", entry.name)`
        // in the host_only branch of `list_tests_all`).
        assert_eq!(
            line,
            &format!("ktstr/{HOST_ONLY_LISTING_NAME}: test"),
            "host_only line must be `ktstr/<name>: test` with no kernel suffix",
        );
        // Belt-and-suspenders: neither sanitized kernel label appears
        // anywhere on the line, even as a substring.
        assert!(
            !line.contains("kernel_6_14_2") && !line.contains("kernel_6_15_0"),
            "host_only line must carry NO sanitized kernel suffix — \
             a regression that emitted `/kernel_…` would surface here. line: {line:?}",
        );
    }

    /// `list_tests_budget` mirror: in multi-kernel mode, the
    /// budget-selecting lister emits exactly ONE candidate for a
    /// `host_only` entry without a kernel suffix. Pins the second
    /// `if entry.host_only { … } else { … }` branch in
    /// `list_tests_budget` against the same regression class as the
    /// `list_tests_all` sibling.
    ///
    /// Budget is set generously (10000 secs) so the greedy selector
    /// in `crate::budget::select` picks every distinct-feature
    /// candidate including this fixture (the `HOST_ONLY_SHIFT` bit
    /// in `extract_features` makes the host_only entry's feature set
    /// uniquely contributory — see `budget::extract_features`).
    /// The selector prints to stdout AND `eprintln!`s a summary
    /// line to stderr; only stdout is captured here, so the stderr
    /// summary lands on the test runner's normal stderr.
    #[test]
    fn list_tests_budget_host_only_skips_kernel_suffix_under_multi_kernel() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _env_lock = lock_env();
        let _kernel_list = EnvVarGuard::set(crate::KTSTR_KERNEL_LIST_ENV, TWO_KERNEL_LIST);

        let (_, captured) = capture_stdout(|| list_tests_budget(false, 10_000.0));
        let lines = host_only_listing_lines(&captured);

        assert_eq!(
            lines.len(),
            1,
            "list_tests_budget must emit exactly 1 candidate line for a \
             host_only entry under multi-kernel mode (saw {n}): {lines:?}",
            n = lines.len(),
        );
        let line = &lines[0];
        assert_eq!(
            line,
            &format!("ktstr/{HOST_ONLY_LISTING_NAME}: test"),
            "host_only candidate name must be `ktstr/<name>: test` with no kernel suffix",
        );
        assert!(
            !line.contains("kernel_6_14_2") && !line.contains("kernel_6_15_0"),
            "host_only candidate must carry NO sanitized kernel suffix — \
             a regression that emitted `/kernel_…` would surface here. line: {line:?}",
        );
    }

    // ---------------------------------------------------------------
    // KTSTR_CARGO_TEST_MODE listing behavior
    // ---------------------------------------------------------------
    //
    // `list_tests_all` and `list_tests_budget` skip gauntlet
    // emission when `KTSTR_CARGO_TEST_MODE` is active. Pins the
    // dispatch contract: bare `cargo test` runs each test once
    // with its declared topology, no per-preset fan-out.
    // Multi-kernel suffix emission is also suppressed because the
    // cargo-ktstr resolver that produces `KTSTR_KERNEL_LIST` is
    // not on the cargo-test path.

    /// Under `KTSTR_CARGO_TEST_MODE=1`, `list_tests_all` emits
    /// exactly one `ktstr/{name}: test` line per registered entry
    /// — no `gauntlet/...` lines. Pins the gauntlet-skip branch.
    #[test]
    fn list_tests_all_cargo_test_mode_skips_gauntlet() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _env_lock = lock_env();
        let _cargo = EnvVarGuard::set("KTSTR_CARGO_TEST_MODE", "1");
        let _no_kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
        let _budget_guard = EnvVarGuard::remove("KTSTR_BUDGET_SECS");

        let (_, captured) = capture_stdout(|| list_tests_all(false));
        let stdout = std::str::from_utf8(&captured).expect("utf-8");
        let gauntlet_lines: Vec<&str> = stdout
            .lines()
            .filter(|l| l.starts_with("gauntlet/"))
            .collect();
        assert!(
            gauntlet_lines.is_empty(),
            "cargo-test-mode must suppress every `gauntlet/...` line; \
             got {} lines: {gauntlet_lines:?}",
            gauntlet_lines.len(),
        );
    }

    /// Multi-kernel suffix emission is suppressed in
    /// cargo-test mode even when `KTSTR_KERNEL_LIST` is set —
    /// the bare `cargo test` path doesn't drive the cargo-ktstr
    /// resolver, so any `KTSTR_KERNEL_LIST` is a stale leftover
    /// from a prior session and must not influence listing.
    #[test]
    fn list_tests_all_cargo_test_mode_ignores_kernel_list() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _env_lock = lock_env();
        let _cargo = EnvVarGuard::set("KTSTR_CARGO_TEST_MODE", "1");
        let _kernel_list = EnvVarGuard::set(crate::KTSTR_KERNEL_LIST_ENV, TWO_KERNEL_LIST);
        let _budget_guard = EnvVarGuard::remove("KTSTR_BUDGET_SECS");

        let (_, captured) = capture_stdout(|| list_tests_all(false));
        let stdout = std::str::from_utf8(&captured).expect("utf-8");
        assert!(
            !stdout.contains("kernel_6_14_2") && !stdout.contains("kernel_6_15_0"),
            "cargo-test-mode must suppress multi-kernel suffix emission \
             even when KTSTR_KERNEL_LIST is set; got stdout containing a \
             sanitized kernel label:\n{stdout}",
        );
    }
}
