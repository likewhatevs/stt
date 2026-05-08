//! End-to-end proof that `Scheduler::config_file_def` paired with a
//! `#[ktstr_test(config = ktstr::json!({...}))]` attribute reaches the
//! guest filesystem at the declared path with the exact bytes the
//! macro emitted.
//!
//! What this test pins:
//!
//! 1. **`ktstr::json!()` macro** lowers a JSON-shaped Rust token tree
//!    to a `&'static str` at compile time. The emitted string is
//!    deterministic — the scenario body compares the guest-side bytes
//!    byte-for-byte against the same `const`.
//! 2. **Macro-side pairing** between `config = ...` and
//!    `Scheduler::config_file_def(arg_template, guest_path)` is a
//!    compile-time gate (`const _: () = assert!(...)` emitted by
//!    `#[ktstr_test]`). Compilation of this file alone proves the gate
//!    accepted the matching pair; the runtime body proves the runtime
//!    side does the right thing with that pair.
//! 3. **`runtime::config_content_parts`** writes the inline content to
//!    a host-side temp file, then **the initramfs packing pipeline**
//!    (`build_initramfs_base`) places that file at the declared
//!    `guest_path` (`/include-files/...`). The scenario reads back
//!    the same path inside the guest and compares against the
//!    `&'static str` the macro emitted. Any wiring break — wrong
//!    archive path, missing include, lost bytes, encoding skew — fails
//!    the comparison.
//! 4. **CLI-arg substitution** from `arg_template`. The template
//!    `"--config {file}"` expands to `--config /include-files/...`
//!    and is appended to the scheduler's argv. `scx-ktstr` accepts
//!    arbitrary unknown flags via `has_flag` / `parse_delay_flag` —
//!    the binary inspects only the flags it cares about and ignores
//!    the rest, so a non-zero argv survives intact through
//!    `scx_ops_open`. The proof that the scheduler boots is
//!    `execute_steps` returning a passing `AssertResult` — the
//!    scenario hold runs to completion against a live scheduler.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};

/// `scx-ktstr` is the workspace's fixture scheduler. It boots a real
/// sched_ext BPF program, attaches, and runs a dispatch loop — i.e.
/// "scheduler boots" is observable end-to-end. Its CLI is
/// permissive: unknown flags are ignored (only `--stall-after`,
/// `--degrade-after`, `--degrade`, `--fail-verify`, `--scattershot`,
/// `--slow`, `--verify-loop` are recognised), so the
/// framework-injected `--config /include-files/config_file_def_e2e.json`
/// passes through harmlessly.
const CFG_E2E_SCHED: Scheduler = Scheduler::new("config_file_def_e2e_sched")
    .binary(SchedulerSpec::Discover("scx-ktstr"))
    .config_file_def("--config {file}", "/include-files/config_file_def_e2e.json");

const CFG_E2E_PAYLOAD: Payload = Payload::from_scheduler(&CFG_E2E_SCHED);

/// Inline scheduler config built via the `ktstr::json!` proc-macro.
/// The macro lowers Rust tokens to a `&'static str` at compile time,
/// so the string can be referenced from the `#[ktstr_test]`
/// attribute (which requires a `const`-expressible value).
///
/// The shape mirrors the kind of layered-style config a real
/// scheduler binary might consume: a `layers` array with a single
/// `Confine` layer, plus a `knobs` object. The exact fields don't
/// matter for the framework wiring — only that the bytes survive
/// the host→initramfs→guest path unchanged.
const CFG_E2E_CONTENT: &str = ktstr::json!({
    "layers": [
        { "name": "default", "kind": { "Confine": {} } }
    ],
    "knobs": { "max_dispatch": 256 }
});

/// Guest-side path where the framework writes the config file. Pinned
/// here so the scenario reads exactly what `config_file_def`'s second
/// argument declares — any divergence between the two values would
/// surface as a `read_to_string` ENOENT.
const GUEST_CFG_PATH: &str = "/include-files/config_file_def_e2e.json";

#[ktstr_test(
    scheduler = CFG_E2E_PAYLOAD,
    config = CFG_E2E_CONTENT,
    duration_s = 3,
    watchdog_timeout_s = 15,
    workers_per_cgroup = 2,
    auto_repro = false,
)]
fn config_file_def_e2e_pipeline(ctx: &Ctx) -> Result<AssertResult> {
    // Step 1: prove the file landed at the declared guest path with
    // the exact bytes the macro emitted. Reads first so a packing
    // failure surfaces before the workload runs and obscures the
    // root cause behind a scheduler error.
    let observed = match std::fs::read_to_string(GUEST_CFG_PATH) {
        Ok(s) => s,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "guest read of {GUEST_CFG_PATH} failed ({e}). The \
                     framework's `config_file_def` + `config = \
                     ktstr::json!({{..}})` pipeline did not place the \
                     inline content at the declared guest path. \
                     Likely break sites: \
                     `runtime::config_content_parts` (host temp file \
                     + arg template), the `include_files` builder \
                     wire-up in `eval.rs`, or the initramfs cpio \
                     archive layout in `build_initramfs_base`."
                ),
            )));
        }
    };
    if observed != CFG_E2E_CONTENT {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "guest config bytes diverge from the macro-emitted \
                 string. Expected len={}: {CFG_E2E_CONTENT:?}; \
                 observed len={}: {observed:?}. The pipeline mutated \
                 the bytes between host write and guest read — \
                 inspect the temp-file write in \
                 `runtime::config_content_parts` and the cpio entry \
                 emitted by `build_initramfs_base`.",
                CFG_E2E_CONTENT.len(),
                observed.len(),
            ),
        )));
    }

    // Step 2: prove the scheduler boots with the framework-injected
    // `--config <guest_path>` arg in argv. `execute_steps` driving a
    // hold against a real `scx-ktstr` scheduler that successfully
    // attached its sched_ext ops is the observable signal —
    // `scx-ktstr`'s permissive CLI swallows the unknown `--config`
    // flag without error, so a verifier-rejected scheduler or a
    // failed attach would surface here as a scenario failure.
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
