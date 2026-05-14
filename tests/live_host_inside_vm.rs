//! VM-backed integration test for the live-host introspection
//! pipeline.
//!
//! Boots a minimal KVM guest via the `#[ktstr_test]` harness with
//! the scx-ktstr scheduler attached, then runs the live-host
//! pipeline INSIDE the guest:
//!
//! 1. [`BpfSyscallAccessor::from_running_kernel_filtered`] enumerates
//!    every BPF map the guest kernel currently knows about, pinning
//!    fds for the ones that match the scx-ktstr scheduler's name
//!    suffix.
//! 2. [`LiveHostKernelEnv::discover`] resolves
//!    `/sys/kernel/btf/vmlinux` (always present with sched_ext) plus
//!    `/proc/kallsyms` for symbol lookups.
//! 3. [`KallsymsTable::load_from`] parses the symbol table — root
//!    only, but ktstr always runs as root in the test environment.
//!
//! From the live-host pipeline's perspective the guest IS a real
//! host: it runs a kernel, a loaded scx scheduler, BPF maps with
//! data, and the relevant procfs/sysfs surfaces. The same code
//! that a real-host capture-mode binary would call works here.
//!
//! The guest is a real host from the pipeline's perspective. No
//! bare-metal hardware needed for CI. The VM provides the complete
//! kernel environment the live-host pipeline requires.
//!
//! This test asserts the live-host pipeline's outputs have the
//! expected SHAPE — non-zero map enumeration, BTF reachable,
//! kallsyms readable. It deliberately does NOT compare against the
//! frozen-VM path's output byte-for-byte: the two backends produce
//! identical [`BpfMapInfo`] / byte buffers but at slightly different
//! moments, so a strict equality check would race against
//! mid-flight scheduler activity. The shape-equivalence assertion
//! is the right contract: same fields populated, same surface
//! types.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::live_host::{BpfMapAccessor, BpfSyscallAccessor, KallsymsTable, LiveHostKernelEnv};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};

const KTSTR_SCHED: ktstr::prelude::Scheduler = ktstr::prelude::Scheduler::new("ktstr_sched")
    .binary(ktstr::prelude::SchedulerSpec::Discover("scx-ktstr"));

/// Discover the running kernel + every BPF map visible to it,
/// then assert the pipeline produced the expected shape:
///
/// - At least one BPF map enumerated (scx-ktstr loads several).
/// - At least one `scx_*`-named map reachable (proves the scheduler
///   was attached when the test ran).
/// - BTF source resolvable (`/sys/kernel/btf/vmlinux` present —
///   sched_ext-capable kernels mandate `CONFIG_DEBUG_INFO_BTF`).
/// - kallsyms readable (root + `kptr_restrict=0` test environment).
///
/// Topology: 1 LLC / 2 cores / 1 thread — the live-host pipeline's
/// behavior doesn't depend on topology size; a larger setup just
/// lengthens the boot. Duration short — the assertions run on the
/// captured pipeline state, not on scheduler-level behavior over
/// time.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, duration_s = 10, watchdog_timeout_s = 60, scheduler = KTSTR_SCHED)]
fn live_host_pipeline_inside_guest_produces_expected_shape(ctx: &Ctx) -> Result<AssertResult> {
    // Run a brief workload so the scheduler is exercised and any
    // lazily-populated BPF maps (event counters, struct_ops state)
    // have entries before we enumerate.
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::Fixed(std::time::Duration::from_secs(2)),
    }];
    let _ = execute_steps(ctx, steps)?;

    // 1. Discover the guest kernel environment — same code path a
    //    real-host capture-mode binary would call.
    let env = match LiveHostKernelEnv::discover() {
        Ok(env) => env,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "LiveHostKernelEnv::discover() failed inside the \
                     guest: {e}. The live-host pipeline cannot run \
                     without BTF (sched_ext-capable kernels mandate \
                     CONFIG_DEBUG_INFO_BTF, so /sys/kernel/btf/vmlinux \
                     should be present)."
                ),
            )));
        }
    };

    if env.release.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "LiveHostKernelEnv::discover() returned an empty kernel \
             release string — uname(2) wrapper produced an empty \
             utsname.release, indicating a libc / kernel ABI \
             regression",
        )));
    }
    if !env.btf_path.exists() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "LiveHostKernelEnv resolved btf_path={} but the file \
                 does not exist — the discovery walk found a stale \
                 candidate path",
                env.btf_path.display()
            ),
        )));
    }

    // 2. Pin every BPF map visible to the guest kernel. ktstr runs
    //    as root in the test environment so CAP_SYS_ADMIN is in
    //    place; map enumeration via BPF_MAP_GET_NEXT_ID +
    //    BPF_MAP_GET_FD_BY_ID will succeed.
    let accessor = match BpfSyscallAccessor::from_running_kernel() {
        Ok(a) => a,
        Err(e) => {
            // Diagnostic: dump /proc/self/status to understand
            // why BPF_MAP_GET_NEXT_ID returned EPERM despite running
            // as PID 1.
            let status = std::fs::read_to_string("/proc/self/status")
                .unwrap_or_else(|e| format!("(read /proc/self/status: {e})"));
            let pid = unsafe { libc::getpid() };
            let euid = unsafe { libc::geteuid() };
            let mut diagnostic =
                format!("pid={pid} euid={euid}\n--- /proc/self/status ---\n{status}\n",);
            // Also try to read /proc/sys/kernel/unprivileged_bpf_disabled
            if let Ok(v) = std::fs::read_to_string("/proc/sys/kernel/unprivileged_bpf_disabled") {
                diagnostic.push_str(&format!("--- unprivileged_bpf_disabled ---\n{v}"));
            }
            // Check if /sys/kernel/security/lockdown reports lockdown state
            if let Ok(v) = std::fs::read_to_string("/sys/kernel/security/lockdown") {
                diagnostic.push_str(&format!("--- /sys/kernel/security/lockdown ---\n{v}"));
            }
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "BpfSyscallAccessor::from_running_kernel() failed \
                     inside the guest: {e}. ktstr runs as root and \
                     CAP_SYS_ADMIN should be available — a failure \
                     here means BPF_MAP_GET_NEXT_ID returned a kernel \
                     error other than ENOENT (the end-of-iteration \
                     sentinel), which would indicate a deeper BPF \
                     subsystem problem.\n\nDIAGNOSTIC:\n{diagnostic}"
                ),
            )));
        }
    };

    let maps = accessor.maps();
    if maps.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "BpfSyscallAccessor::maps() returned an empty list — the \
             scx-ktstr scheduler should have at least one BPF map \
             loaded by the time this test runs. Either the scheduler \
             didn't attach (would have shown earlier in the test \
             pipeline) or BPF_MAP_GET_NEXT_ID enumerated nothing, \
             which contradicts the kernel's id space",
        )));
    }

    // Look for a scx-related map. scx-ktstr's BSS section maps are
    // named after the scheduler's source object — a substring match
    // on "scx" or "ktstr" survives renames in the BPF object's data
    // section name.
    let any_scx = maps
        .iter()
        .any(|m| m.name().contains("scx") || m.name().contains("ktstr"));
    if !any_scx {
        let names: Vec<std::borrow::Cow<'_, str>> = maps.iter().map(|m| m.name()).collect();
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "BpfSyscallAccessor enumerated {} maps but none \
                 contained 'scx' or 'ktstr' in their name. Map \
                 names observed: {:?}. Either the scheduler isn't \
                 attached or its map names diverged from the \
                 scx-ktstr / scx_* convention",
                maps.len(),
                names
            ),
        )));
    }

    // 3. kallsyms — root-readable in the ktstr test environment.
    //    Empty tables are valid (per KallsymsTable::is_empty's
    //    contract: parsed-but-empty when CAP_SYSLOG is missing and
    //    every address is zero-redacted), but we expect a
    //    non-empty table inside ktstr's root-privileged guest.
    let kallsyms = match KallsymsTable::load_from(&env) {
        Ok(t) => t,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "KallsymsTable::load_from(env) failed inside the \
                     guest: {e}. /proc/kallsyms is root-readable; \
                     ktstr runs as root, so a load failure here \
                     means /proc/kallsyms wasn't mounted or returned \
                     EIO"
                ),
            )));
        }
    };
    if kallsyms.is_empty() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            "KallsymsTable parsed /proc/kallsyms inside the guest \
             but the resulting table is empty — every line had a \
             zero address, which is the CAP_SYSLOG-redacted view. \
             ktstr's guest runs with kptr_restrict=0 and the \
             process has CAP_SYSLOG, so this indicates a sysctl / \
             capability regression",
        )));
    }

    // Spot-check that a couple of well-known sched_ext symbols are
    // resolvable. Their presence is a tighter equivalence check
    // against the frozen-VM path: both backends should be able to
    // reach `ext_sched_class` (the sched_ext sched_class symbol)
    // and `scx_disable_workfn` (the unload path).
    let sched_class = kallsyms.resolve("ext_sched_class");
    let disable_fn = kallsyms.resolve("scx_disable_workfn");
    if sched_class.is_none() && disable_fn.is_none() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "Neither ext_sched_class nor scx_disable_workfn \
                 resolved via the live-host kallsyms table. \
                 Total symbols parsed: {}. Sample names from the \
                 table won't help here without dumping the whole \
                 table; the failure indicates either a symbol \
                 rename in the running kernel or a parse \
                 regression",
                kallsyms.len()
            ),
        )));
    }

    Ok(AssertResult::pass())
}
