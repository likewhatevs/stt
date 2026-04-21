//! Shared fixtures for per-module `#[cfg(test)]` blocks.
//!
//! This file is compiled only under `#[cfg(test)]`. Each fixture
//! lives here because it is used by two or more submodule test
//! blocks; single-use helpers stay co-located with their sole
//! consumer.
//!
//! Nothing in this module is exposed outside the crate — all items
//! are `pub(crate)` and the module itself is `#[cfg(test)]` so they
//! disappear from non-test builds entirely.

#![cfg(test)]

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use anyhow::Result;
use tempfile::TempDir;

use crate::assert::AssertResult;
use crate::scenario::Ctx;
use crate::scenario::flags::FlagDecl;

use super::entry::{KtstrTestEntry, Scheduler, SchedulerSpec, TopologyConstraints};
use crate::vmm::topology::Topology;

/// Serializes tests that mutate env vars. Shared across every
/// `#[cfg(test)]` module in the crate: nextest runs tests in
/// parallel within a binary, and `std::env::set_var` is process-wide,
/// so any test that mutates an env var must hold this mutex for its
/// full save/mutate/restore window.
pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Lock [`ENV_LOCK`] for the lifetime of a test, recovering from
/// poisoning. If a previous test panicked while holding the lock we
/// still want the current test to run: env-touching tests establish
/// no shared invariant beyond their own save/restore pair, so the
/// poisoned inner guard is safe to take.
pub(crate) fn lock_env() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Tempdir bound as the process-wide `KTSTR_CACHE_DIR`. While the
/// returned value is live:
///   * the temp directory exists on disk and is pointed at by
///     `KTSTR_CACHE_DIR`;
///   * on drop, the env var's previous value is restored and the
///     directory is removed.
///
/// Field order is load-bearing: Rust drops fields in declaration
/// order, so `_guard` drops first (restoring `KTSTR_CACHE_DIR` to
/// whatever the test process saw before construction) and `tmp`
/// drops second (removing the directory). Reversing the order
/// would leave `KTSTR_CACHE_DIR` pointing at a deleted tempdir
/// during the window between `tmp`'s and `_guard`'s drop calls —
/// a dangling-env-ref hazard in any code that races with the
/// tail of the drop sequence.
///
/// Callers that also mutate other env vars must hold [`lock_env`]
/// for the guard's full lifetime so the save/restore pair does not
/// race with another test in the same binary.
pub(crate) struct IsolatedCacheDir {
    _guard: EnvVarGuard,
    tmp: TempDir,
}

impl IsolatedCacheDir {
    /// The temp directory's root path.
    pub(crate) fn path(&self) -> &std::path::Path {
        self.tmp.path()
    }
}

/// Create a fresh tempdir and point `KTSTR_CACHE_DIR` at it. See
/// [`IsolatedCacheDir`] for drop semantics.
pub(crate) fn isolated_cache_dir() -> IsolatedCacheDir {
    let tmp = TempDir::new().expect("tempdir for isolated cache root");
    let guard = EnvVarGuard::set(
        "KTSTR_CACHE_DIR",
        tmp.path()
            .to_str()
            .expect("tempdir path is UTF-8 on every supported target"),
    );
    IsolatedCacheDir { _guard: guard, tmp }
}

/// Shared `Topology` used by `evaluate_vm_result` tests: the
/// 1-numa-1-llc-2-core-1-thread topology is unremarkable on every
/// dimension monitor checks touch, so tests that exercise other
/// fields don't also have to reason about topology effects.
pub(crate) const EVAL_TOPO: Topology = Topology::new(1, 1, 2, 1);

/// Placeholder test function used by `eevdf_entry` / `sched_entry`
/// fixtures. Always returns pass — tests that care about the
/// function's behaviour construct their own `KtstrTestEntry.func`.
pub(crate) fn dummy_test_fn(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// RAII guard that mutates an environment variable and restores the
/// original value on drop. Not thread-safe: hold [`ENV_LOCK`] for the
/// guard's full lifetime when another test in the same binary might
/// mutate the same key.
///
/// nextest runs each test in its own process, so simple single-variable
/// tests don't need the lock — the lock is only needed when multiple
/// tests in a single process mutate overlapping keys.
///
/// Shared across cache / remote_cache / anywhere else that needs to
/// rewrite HOME / XDG_CACHE_HOME / KTSTR_CACHE_DIR / ACTIONS_CACHE_URL
/// and friends during a test run. Previously duplicated in
/// `cache::tests` and `remote_cache::tests`.
pub(crate) struct EnvVarGuard {
    key: String,
    original: Option<String>,
}

impl EnvVarGuard {
    pub(crate) fn set(key: &str, value: &str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: nextest runs each test in its own process; callers
        // that share a key across concurrent tests in the same process
        // must take `ENV_LOCK` before constructing the guard.
        unsafe { std::env::set_var(key, value) };
        EnvVarGuard {
            key: key.to_string(),
            original,
        }
    }

    pub(crate) fn remove(key: &str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: same rationale as `set`.
        unsafe { std::env::remove_var(key) };
        EnvVarGuard {
            key: key.to_string(),
            original,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            // SAFETY: nextest runs each test in its own process.
            Some(val) => unsafe { std::env::set_var(&self.key, val) },
            None => unsafe { std::env::remove_var(&self.key) },
        }
    }
}

/// Build a minimal `KtstrTestEntry` bound to the EEVDF scheduler.
/// Intended for `evaluate_vm_result` tests that exercise the no-scx
/// code path — see the `sched_entry` sibling for the scx variant.
pub(crate) fn eevdf_entry(name: &'static str) -> KtstrTestEntry {
    KtstrTestEntry {
        name,
        func: dummy_test_fn,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    }
}

/// Scheduler used by `sched_entry` to represent a generic scx-style
/// scheduler for evaluate_vm_result tests that need the
/// `has_active_scheduling == true` path.
pub(crate) static SCHED_TEST: Scheduler = Scheduler {
    name: "test_sched",
    binary: SchedulerSpec::Discover("test_sched_bin"),
    flags: &[],
    sysctls: &[],
    kargs: &[],
    assert: crate::assert::Assert::NO_OVERRIDES,
    cgroup_parent: None,
    sched_args: &[],
    topology: Topology {
        llcs: 1,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
    },
    constraints: TopologyConstraints::DEFAULT,
    config_file: None,
};

/// Payload wrapper around [`SCHED_TEST`] so tests can plug it into
/// the `scheduler: &'static Payload` slot without re-wrapping at
/// every call site.
pub(crate) static SCHED_TEST_PAYLOAD: crate::test_support::Payload =
    crate::test_support::Payload::from_scheduler(&SCHED_TEST);

/// Build a minimal `KtstrTestEntry` bound to the scx-style
/// `SCHED_TEST` fixture. Pair to `eevdf_entry` for the scheduler
/// path in evaluate_vm_result tests.
pub(crate) fn sched_entry(name: &'static str) -> KtstrTestEntry {
    KtstrTestEntry {
        name,
        func: dummy_test_fn,
        scheduler: &SCHED_TEST_PAYLOAD,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    }
}

/// No-op repro probe: returns `None` so evaluate_vm_result tests
/// don't attempt to spawn a second VM while running outside a kernel
/// context.
pub(crate) fn no_repro(_output: &str) -> Option<String> {
    None
}

/// Construct a `VmResult` with explicit output/stderr/exit and
/// default monitor/stimulus/verifier/kvm/crash fields. Used by
/// evaluate_vm_result tests to drive specific error paths without
/// touching an actual VM.
pub(crate) fn make_vm_result(
    output: &str,
    stderr: &str,
    exit_code: i32,
    timed_out: bool,
) -> crate::vmm::VmResult {
    crate::vmm::VmResult {
        success: !timed_out && exit_code == 0,
        exit_code,
        duration: std::time::Duration::from_secs(1),
        timed_out,
        output: output.to_string(),
        stderr: stderr.to_string(),
        monitor: None,
        shm_data: None,
        stimulus_events: Vec::new(),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        crash_message: None,
    }
}

/// Build a `KtstrTestEntry` with overridden memory/duration/workers
/// for `KtstrTestEntry::validate` tests. Everything else inherits
/// from `DEFAULT`.
pub(crate) fn validate_entry(
    name: &'static str,
    memory_mb: u32,
    duration: Duration,
    workers_per_cgroup: u32,
) -> KtstrTestEntry {
    KtstrTestEntry {
        name,
        memory_mb,
        duration,
        workers_per_cgroup,
        ..KtstrTestEntry::DEFAULT
    }
}

// ---------------------------------------------------------------------------
// Shared FlagDecl fixtures
// ---------------------------------------------------------------------------
//
// These `&'static FlagDecl` values are referenced by Scheduler flag
// lists in multiple test blocks (scheduler_generate_profiles_*,
// validate_entry_flags_*). Declaring them once here keeps the raw
// flag definitions out of individual tests so a name/arg tweak only
// requires touching one place.

pub(crate) static FLAG_A: FlagDecl = FlagDecl {
    name: "flag_a",
    args: &["--flag-a"],
    requires: &[],
};
pub(crate) static BORROW: FlagDecl = FlagDecl {
    name: "borrow",
    args: &["--borrow"],
    requires: &[],
};
pub(crate) static REBAL: FlagDecl = FlagDecl {
    name: "rebal",
    args: &["--rebal"],
    requires: &[],
};
pub(crate) static TEST_LLC: FlagDecl = FlagDecl {
    name: "llc",
    args: &["--llc"],
    requires: &[],
};
pub(crate) static TEST_STEAL: FlagDecl = FlagDecl {
    name: "steal",
    args: &["--steal"],
    requires: &[&TEST_LLC],
};
pub(crate) static BORROW_LONG: FlagDecl = FlagDecl {
    name: "borrow",
    args: &["--enable-borrow"],
    requires: &[],
};
pub(crate) static TEST_A: FlagDecl = FlagDecl {
    name: "a",
    args: &["-a"],
    requires: &[],
};
pub(crate) static TEST_B: FlagDecl = FlagDecl {
    name: "b",
    args: &["-b"],
    requires: &[],
};

pub(crate) static FLAGS_A: &[&FlagDecl] = &[&FLAG_A];
pub(crate) static FLAGS_BORROW_REBAL: &[&FlagDecl] = &[&BORROW, &REBAL];
pub(crate) static FLAGS_STEAL_LLC: &[&FlagDecl] = &[&TEST_STEAL, &TEST_LLC];
pub(crate) static FLAGS_BORROW_LONG: &[&FlagDecl] = &[&BORROW_LONG];
pub(crate) static FLAGS_AB: &[&FlagDecl] = &[&TEST_A, &TEST_B];
pub(crate) static FLAGS_LLC_STEAL: &[&FlagDecl] = &[&TEST_LLC, &TEST_STEAL];

// ---------------------------------------------------------------------------
// EnvVarGuard drop-time restoration tests
// ---------------------------------------------------------------------------
//
// Each test uses a distinct env var name so even when `lock_env()` is
// not held across the full set, the tests don't clobber each other's
// pre-existing values. `lock_env()` still serializes against any other
// env-touching test in the crate so the save/mutate/restore window
// does not race with `HOME` / `XDG_CACHE_HOME` / `KTSTR_CACHE_DIR`
// rewrites in cache / model / sidecar tests.

/// `EnvVarGuard::set` restores the PRE-EXISTING value of the key
/// when the guard drops. Pre-seed the key to a known sentinel,
/// construct a guard that overwrites it, confirm the overwrite is
/// visible, drop the guard, and re-read: the sentinel must be back.
#[test]
fn env_var_guard_set_restores_original_value_on_drop() {
    const KEY: &str = "KTSTR_TEST_ENV_VAR_GUARD_SET_RESTORES_ORIGINAL";
    let _lock = lock_env();
    // SAFETY: process-wide env mutation; we hold `lock_env()` so no
    // other test in this binary is racing a key mutation.
    unsafe { std::env::set_var(KEY, "pre-existing") };
    {
        let _g = EnvVarGuard::set(KEY, "overwritten");
        assert_eq!(std::env::var(KEY).ok().as_deref(), Some("overwritten"));
    }
    assert_eq!(
        std::env::var(KEY).ok().as_deref(),
        Some("pre-existing"),
        "EnvVarGuard::set must restore the pre-existing value on drop"
    );
    // Cleanup so a subsequent test run doesn't inherit our sentinel.
    unsafe { std::env::remove_var(KEY) };
}

/// `EnvVarGuard::set` on a key that is currently absent restores it
/// to absent on drop — the guard remembered `None`, so dropping must
/// `remove_var` rather than re-`set_var` the string `"None"` or a
/// stale previous value.
#[test]
fn env_var_guard_set_restores_absent_key_on_drop() {
    const KEY: &str = "KTSTR_TEST_ENV_VAR_GUARD_SET_RESTORES_ABSENT";
    let _lock = lock_env();
    // Guarantee the key is absent before the guard fires.
    unsafe { std::env::remove_var(KEY) };
    assert!(std::env::var(KEY).is_err(), "setup: key must start absent");
    {
        let _g = EnvVarGuard::set(KEY, "transient");
        assert_eq!(std::env::var(KEY).ok().as_deref(), Some("transient"));
    }
    assert!(
        std::env::var(KEY).is_err(),
        "EnvVarGuard::set on an absent key must restore absence, not leave the value behind"
    );
}

/// `EnvVarGuard::remove` preserves the PRE-EXISTING value across its
/// lifetime: the guard takes the key out of the environment while
/// alive, then puts the original back on drop.
#[test]
fn env_var_guard_remove_restores_original_value_on_drop() {
    const KEY: &str = "KTSTR_TEST_ENV_VAR_GUARD_REMOVE_RESTORES_ORIGINAL";
    let _lock = lock_env();
    unsafe { std::env::set_var(KEY, "pre-existing") };
    {
        let _g = EnvVarGuard::remove(KEY);
        assert!(
            std::env::var(KEY).is_err(),
            "EnvVarGuard::remove must remove the key for the guard's lifetime"
        );
    }
    assert_eq!(
        std::env::var(KEY).ok().as_deref(),
        Some("pre-existing"),
        "EnvVarGuard::remove must restore the pre-existing value on drop"
    );
    unsafe { std::env::remove_var(KEY) };
}
