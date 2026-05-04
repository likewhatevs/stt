//! Guest PMU end-to-end tests.
//!
//! Boots a real KVM VM via `#[ktstr_test]` and runs guest-side
//! Rust that exercises the synthesized PMU surface:
//!   - On x86_64: reads CPUID leaf 0xA via the `cpuid` instruction
//!     and verifies the synthesized PMU-v2 fields surface to the
//!     guest. The handler in `src/vmm/x86_64/topology.rs` writes
//!     version=2, num_gp=4, gp_width=48, mask_length=7, num_fixed=3,
//!     fixed_width=48 when the host advertises a non-zero PMU
//!     version. A regression in the synthesizer (or in KVM's
//!     `intel_pmu_refresh` clamping it) would surface here as the
//!     guest seeing version=0.
//!   - Opens `perf_event_open(PERF_TYPE_HARDWARE,
//!     PERF_COUNT_HW_INSTRUCTIONS)` against the current thread,
//!     spins for ~10M iterations, reads the counter, and asserts
//!     the value advances. This pins the end-to-end pipeline:
//!     guest kernel `intel_pmu_init` accepting the PMU surface,
//!     `perf_event_open(2)` succeeding, the kernel's perf core
//!     wiring to the underlying counter, and the counter
//!     advancing on guest CPU instructions.
//!
//! Why this matters:
//!   - `scx_layered --membw-tracking` and `scx_cosmos -e/-y`
//!     read PMU counters via BPF kfuncs. A broken PMU surface
//!     means those schedulers attach but every counter reads
//!     zero — no kernel error, just silent miscapture.
//!   - The synthesized PMU-v2 surface is unconditional on
//!     x86 when the host advertises a non-zero version
//!     (gated in `topology.rs`); the unit tests in that file
//!     verify the synthesizer in isolation, but cannot pin
//!     the guest-side observability across the full
//!     CPUID → kernel-init → perf-event-open chain.
//!
//! aarch64 PMUv3 coverage: the analogous guest-side
//! perf_event_open against PMUv3 lives in a separate
//! arch-gated body; the FDT and KVM_ARM_VCPU_PMU_V3
//! initialization are unit-tested in `src/vmm/aarch64/`
//! kvm.rs and fdt.rs.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;

/// CPUID leaf 0xA EAX bit-field decode helper. See
/// `arch/x86/include/asm/perf_event.h::union cpuid10_eax`:
///   - bits[7:0]: version_id
///   - bits[15:8]: num_counters
///   - bits[23:16]: bit_width
///   - bits[31:24]: mask_length
struct CpuidLeafA {
    version: u32,
    num_gp: u32,
    gp_width: u32,
    mask_length: u32,
    num_fixed: u32,
    fixed_width: u32,
}

/// Read CPUID leaf 0xA on the current CPU. Uses `core::arch::x86_64::__cpuid_count`
/// — guest-side userspace code can issue cpuid directly because
/// VMX (Intel) and SVM (AMD) both intercept it and KVM writes
/// the synthesized values into the guest's GP registers. The
/// `__cpuid_count` intrinsic is a safe wrapper around an internal
/// `asm!` block (see `core_arch/src/x86/cpuid.rs`), so no `unsafe`
/// block is needed at the call site.
///
/// On non-x86_64 architectures this is unreachable (the test
/// scenario short-circuits before calling this helper).
#[cfg(target_arch = "x86_64")]
fn read_cpuid_leaf_a() -> CpuidLeafA {
    let r = core::arch::x86_64::__cpuid_count(0xa, 0);
    CpuidLeafA {
        version: r.eax & 0xff,
        num_gp: (r.eax >> 8) & 0xff,
        gp_width: (r.eax >> 16) & 0xff,
        mask_length: (r.eax >> 24) & 0xff,
        num_fixed: r.edx & 0x1f,
        fixed_width: (r.edx >> 5) & 0xff,
    }
}

/// Stub for non-x86_64 hosts. The `guest_pmu_cpuid_leaf_a_synthesized`
/// scenario short-circuits before calling this on aarch64 because
/// CPUID does not exist outside x86; the helper is here so the test
/// crate compiles on every supported target.
#[cfg(not(target_arch = "x86_64"))]
fn read_cpuid_leaf_a() -> CpuidLeafA {
    CpuidLeafA {
        version: 0,
        num_gp: 0,
        gp_width: 0,
        mask_length: 0,
        num_fixed: 0,
        fixed_width: 0,
    }
}

/// Best-effort host-PMU presence probe accessible from inside the
/// guest. Returns `true` when independent CPUID signals indicate the
/// underlying host CPU has a PMU with PEBS support, distinct from
/// leaf 0xA itself.
///
/// Why this matters: when leaf 0xA reports version=0, we cannot
/// trivially distinguish "host has no PMU / kvm.enable_pmu=0" (the
/// synthesizer correctly preserves the zero) from "synthesizer
/// regressed silently" (it should have written v2 but didn't).
/// Without an independent signal, a regression in
/// `src/vmm/x86_64/topology.rs::leaf 0xa` would produce a silent
/// pass — the test would report version=0, classify it as "no PMU",
/// and never fail.
///
/// The probe combines two signals KVM passes through independently
/// of leaf 0xA's vendor-plus-version gate:
///   - CPUID leaf 0 vendor (EBX:EDX:ECX). KVM's `do_host_cpuid`
///     in `arch/x86/kvm/cpuid.c` calls `cpuid_count` directly on
///     the host CPU for leaf 0 and the leaf-0 case in
///     `__do_cpuid_func` only updates the max-leaf field, leaving
///     the vendor string unchanged.
///   - CPUID leaf 1 EDX bit 21 (DS, "dts" in /proc/cpuinfo).
///     Filtered by KVM through `cpuid_entry_override(entry,
///     CPUID_1_EDX)` against `kvm_cpu_caps`. DS is set in
///     `kvm_cpu_caps` only when `vmx_pebs_supported()` is true at
///     `arch/x86/kvm/vmx/vmx.c::vmx_set_cpu_caps`
///     (`kvm_cpu_cap_check_and_set(X86_FEATURE_DS)` inside the
///     `if (vmx_pebs_supported())` arm). `vmx_pebs_supported()` in
///     `arch/x86/kvm/vmx/capabilities.h` checks
///     `boot_cpu_has(X86_FEATURE_PEBS) && kvm_pmu_cap.pebs_ept &&
///     !enable_mediated_pmu`, and `kvm_pmu_cap` is zeroed under
///     `!enable_pmu` (see `arch/x86/kvm/pmu.c::kvm_init_pmu_capability`,
///     `if (!enable_pmu) { memset(&kvm_pmu_cap, 0, ...); return; }`).
///     So DS clears under `!enable_pmu` AND under `!PEBS`.
///
/// Returns `true` iff vendor=GenuineIntel AND DS=1. The probe is an
/// Intel + PEBS + PMU positive signal: when it returns `true`, the
/// host has PMU enabled with PEBS hardware and KVM passing it
/// through, so leaf 0xA version=0 in that environment must be a
/// synthesizer regression (no other path produces this combination
/// inside the guest). PMU-without-PEBS hosts (e.g. older Intel
/// without `boot_cpu_has(X86_FEATURE_PEBS)`, or hosts without
/// `pebs_ept`), `kvm.enable_pmu=0`, hosts with mediated vPMU
/// (`enable_mediated_pmu=1`, which short-circuits
/// `vmx_pebs_supported()` to `false`), and AMD all produce a false
/// negative — the probe returns `false` and the test falls back to
/// the conservative non-failing skip. This means the probe makes
/// the test stricter than the prior silent pass on Intel+PEBS
/// hosts, while staying silent on the configurations where it
/// can't disambiguate.
#[cfg(target_arch = "x86_64")]
fn host_likely_has_pmu() -> bool {
    // `__cpuid` is a safe wrapper around an internal `asm!` block
    // (see `core_arch/src/x86/cpuid.rs`). cpuid is unprivileged on
    // x86_64; KVM intercepts and writes the synthesized result
    // into the guest GPRs.
    let leaf0 = core::arch::x86_64::__cpuid(0);
    let leaf1 = core::arch::x86_64::__cpuid(1);
    host_likely_has_pmu_decode(leaf0.ebx, leaf0.edx, leaf0.ecx, leaf1.edx)
}

/// Stub for non-x86_64. Always returns `false` — aarch64 PMUv3
/// presence is verified by `src/vmm/aarch64/{kvm,fdt}.rs` unit
/// tests; the leaf-0xA test short-circuits on non-x86 hosts before
/// calling this helper.
#[cfg(not(target_arch = "x86_64"))]
fn host_likely_has_pmu() -> bool {
    false
}

/// Pure decode helper extracted from [`host_likely_has_pmu`] so the
/// vendor + DS-bit logic is unit-testable without mocking CPUID.
/// Returns `true` iff the leaf-0 vendor is "GenuineIntel" AND leaf 1
/// EDX bit 21 (DS) is set.
fn host_likely_has_pmu_decode(
    leaf0_ebx: u32,
    leaf0_edx: u32,
    leaf0_ecx: u32,
    leaf1_edx: u32,
) -> bool {
    // "GenuineIntel" = EBX:0x756e6547 EDX:0x49656e69 ECX:0x6c65746e —
    // matches `detect_vendor` in src/vmm/x86_64/topology.rs.
    let intel = (leaf0_ebx, leaf0_edx, leaf0_ecx)
        == (0x756e_6547, 0x4965_6e69, 0x6c65_746e);
    if !intel {
        return false;
    }
    // CPUID leaf 1 EDX bit 21 = DS (Debug Store). Set when the host
    // has the DS feature and KVM exposes it. Strongly correlated
    // with PMU presence on Intel CPUs.
    (leaf1_edx >> 21) & 1 == 1
}

/// Outcome of classifying a `perf_event_open(2)` errno on the
/// PMU-pipeline test. Either a non-failing skip (host-config errors
/// the test cannot fix) or a fail (errnos that point at the
/// synthesized PMU surface).
#[derive(Debug, PartialEq, Eq)]
enum PerfOpenResult {
    /// Host-policy / kernel-config error. The errno does not
    /// indicate a synthesizer bug; the test reports a non-failing
    /// `AssertDetail` and skips the counter assertion. Carries a
    /// short reason describing the cause.
    Skip(&'static str),
    /// Synthesizer-regression error. The errno indicates the
    /// synthesized PMU surface failed to bind to a backend; the
    /// test fails. Carries a short reason describing what the
    /// errno typically maps to.
    Fail(&'static str),
}

/// Classify a raw errno from `perf_event_open(2)` into a
/// [`PerfOpenResult`]. Extracted so each errno path is unit-testable
/// without booting a guest.
///
/// Mapping:
///   - `EACCES` / `EPERM`: host-policy denial
///     (`kernel.perf_event_paranoid > 2` or missing `CAP_PERFMON`).
///     Skip.
///   - `ENOSYS`: kernel built without `CONFIG_PERF_EVENTS`. Skip.
///   - everything else (notably `EINVAL`, `ENODEV`, `EOPNOTSUPP`):
///     the synthesized PMU surface failed to bind to a backend.
///     `EINVAL` fires when the kernel rejects the requested
///     event/type pair against an empty backend (e.g.
///     `intel_pmu_init` returning `-ENODEV` upstream of
///     `perf_event_open` surfaces as `EINVAL` on the syscall —
///     see `kernel/events/core.c::perf_event_open`'s `return
///     -EINVAL` arms). A regression in
///     `src/vmm/x86_64/topology.rs::leaf 0xa` that drops the
///     synthesized v2 surface would surface as exactly that.
///     Fail.
fn classify_perf_open_errno(raw: i32) -> PerfOpenResult {
    match raw {
        libc::EACCES | libc::EPERM => PerfOpenResult::Skip(
            "EACCES/EPERM = perf_event_paranoid > 2 or missing CAP_PERFMON",
        ),
        libc::ENOSYS => PerfOpenResult::Skip(
            "ENOSYS = kernel without CONFIG_PERF_EVENTS",
        ),
        _ => PerfOpenResult::Fail(
            "EINVAL/ENODEV/EOPNOTSUPP indicate the synthesized PMU surface \
             did not bind to a backend — check src/vmm/x86_64/topology.rs::leaf 0xa \
             for x86 or src/vmm/aarch64/kvm.rs::init_pmuv3 for aarch64",
        ),
    }
}

/// Asserts the synthesized PMU-v2 surface is visible to the guest
/// via CPUID leaf 0xA. When the host has no PMU
/// (kvm.enable_pmu=0 or PMU-less hardware), the synthesizer in
/// `src/vmm/x86_64/topology.rs` leaves the leaf zeroed; under that
/// condition the test consults [`host_likely_has_pmu`] (an
/// independent CPUID probe via vendor + DS feature bit) to
/// distinguish "host has no PMU — expected" from "synthesizer
/// regressed silently — fail." Hosts with a PMU advertise
/// version >= 1; the synthesizer overwrites with v2.
///
/// On non-x86 hosts the test short-circuits with a non-failing
/// AssertDetail — CPUID is x86-only; the analogous aarch64
/// PMUv3 surface is verified via FDT and KVM unit tests in
/// `src/vmm/aarch64/`.
#[ktstr_test(llcs = 1, cores = 1, threads = 1, memory_mb = 512)]
fn guest_pmu_cpuid_leaf_a_synthesized(_ctx: &Ctx) -> Result<AssertResult> {
    if !cfg!(target_arch = "x86_64") {
        let mut result = AssertResult::pass();
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "CPUID leaf 0xA is x86-only; skipping on non-x86_64 \
             target. PMUv3 surface coverage on aarch64 lives in \
             src/vmm/aarch64/{kvm,fdt}.rs unit tests."
                .to_string(),
        ));
        return Ok(result);
    }
    let leaf = read_cpuid_leaf_a();
    let mut result = AssertResult::pass();
    if leaf.version == 0 {
        // The synthesizer in src/vmm/x86_64/topology.rs gates leaf
        // 0xA synthesis on `entry.eax & 0xff != 0` — version=0 in
        // the guest means KVM passed through 0 (host has no PMU,
        // kvm.enable_pmu=0, or vendor=AMD which doesn't define
        // leaf 0xA), OR the synthesizer regressed.
        //
        // Probe an independent host-PMU signal (vendor + DS bit;
        // see `host_likely_has_pmu`). DS=1 only when KVM ran the
        // `vmx_pebs_supported()` arm in vmx_set_cpu_caps, which
        // requires enable_pmu=1 (kvm_pmu_cap is zeroed under
        // !enable_pmu) AND host PEBS hardware AND
        // kvm_pmu_cap.pebs_ept AND !enable_mediated_pmu. So when
        // the probe returns true, the only path that produces
        // version=0 in the guest is a synthesizer regression —
        // fail the test instead of silently passing.
        if host_likely_has_pmu() {
            result.passed = false;
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                "CPUID leaf 0xA reports version=0 but the host \
                 appears PMU-capable (vendor=GenuineIntel and \
                 leaf 1 EDX bit 21 (DS) is set, which on a \
                 KVM guest requires enable_pmu=1 + host PEBS). \
                 src/vmm/x86_64/topology.rs::leaf 0xa regressed: \
                 the synthesizer should have written PMU v2 but \
                 the guest sees version=0, which breaks \
                 scx_layered/scx_cosmos perf-counter reads."
                    .to_string(),
            ));
            return Ok(result);
        }
        // Host appears to have no PMU; the leaf is zeroed by
        // design — the synthesizer skips it on a zero-version
        // base. This is the expected behavior on a no-PMU host
        // (AMD, ancient Intel without DS, or PMU-less hardware);
        // report it without failing the test.
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "CPUID leaf 0xA reports version=0 and the host probe \
             (vendor + DS bit) does not indicate a PMU; \
             synthesizer correctly preserved the zeroed leaf"
                .to_string(),
        ));
        return Ok(result);
    }

    // PMU present — assert the synthesized v2 shape.
    if leaf.version != 2 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "CPUID leaf 0xA reports version={}, expected 2 \
                 (synthesized PMU-v2). The handler in \
                 src/vmm/x86_64/topology.rs may have regressed.",
                leaf.version,
            ),
        ));
        return Ok(result);
    }
    if leaf.num_gp != 4 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "CPUID leaf 0xA reports num_gp={}, expected 4 \
                 (synthesized PMU-v2 GP counter count)",
                leaf.num_gp,
            ),
        ));
        return Ok(result);
    }
    if leaf.gp_width != 48 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "CPUID leaf 0xA reports gp_width={}, expected 48 \
                 (synthesized PMU-v2 GP counter width)",
                leaf.gp_width,
            ),
        ));
        return Ok(result);
    }
    if leaf.mask_length != 7 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "CPUID leaf 0xA reports mask_length={}, expected 7 \
                 (ARCH_PERFMON_EVENTS_COUNT). intel_pmu_init in \
                 arch/x86/events/intel/core.c returns -ENODEV on \
                 mask_length < 7.",
                leaf.mask_length,
            ),
        ));
        return Ok(result);
    }
    if leaf.num_fixed != 3 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "CPUID leaf 0xA reports num_fixed={}, expected 3 \
                 (synthesized PMU-v2 fixed counter count)",
                leaf.num_fixed,
            ),
        ));
        return Ok(result);
    }
    if leaf.fixed_width != 48 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "CPUID leaf 0xA reports fixed_width={}, expected 48 \
                 (synthesized PMU-v2 fixed counter width)",
                leaf.fixed_width,
            ),
        ));
        return Ok(result);
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "CPUID leaf 0xA synthesized PMU-v2 visible: version={}, \
             num_gp={}, gp_width={}, mask_length={}, num_fixed={}, \
             fixed_width={}",
            leaf.version,
            leaf.num_gp,
            leaf.gp_width,
            leaf.mask_length,
            leaf.num_fixed,
            leaf.fixed_width,
        ),
    ));
    Ok(result)
}

/// Open a `perf_event_open` HW counter against the current thread,
/// burn ~10M iterations of CPU work, read the counter, and assert
/// the value is non-zero. Pins the end-to-end pipeline:
///   1. Guest kernel's `intel_pmu_init` (x86) /
///      `armv8_pmuv3_init` (aarch64) accepted the surface.
///   2. `perf_event_open(2)` succeeds for PERF_TYPE_HARDWARE.
///   3. The kernel's perf core wires to a real counter.
///   4. The counter advances under guest CPU work.
///
/// The body uses `core::hint::black_box` to prevent the compiler
/// from optimizing the loop away. 10M iterations is well above
/// the noise floor for HW_INSTRUCTIONS / HW_CPU_CYCLES — even an
/// extreme PMU multiplexing window (10:1) would still yield ~1M
/// counted events. Asserting >= 1 is the lower bound that pins
/// "a real counter is wired" without over-fitting to specific
/// micro-arch IPC.
///
/// Errno classification on `perf_event_open` failure:
///   - EACCES / EPERM: host-policy denial (`kernel.perf_event_paranoid >
///     2` or missing `CAP_PERFMON`); skip with non-failing detail.
///   - ENOSYS: the kernel was built without `CONFIG_PERF_EVENTS`; skip
///     with non-failing detail (pure host-config issue).
///   - everything else (notably EINVAL, ENODEV, EOPNOTSUPP): the
///     synthesized PMU surface is broken or the kernel could not bind
///     the requested event to a PMU backend. Fail the test so a
///     synthesizer regression is visible. EINVAL fires when the kernel
///     rejects the requested event/type pair against an empty backend
///     (intel_pmu_init returning -ENODEV upstream of perf_event_open
///     surfaces here as EINVAL on the syscall — see
///     kernel/events/core.c::perf_event_open's `return -EINVAL` arms);
///     a regression in `src/vmm/x86_64/topology.rs::leaf_0xa` that
///     drops the synthesized v2 surface would surface as exactly that.
#[ktstr_test(llcs = 1, cores = 1, threads = 1, memory_mb = 512)]
fn guest_pmu_perf_event_open_counts_instructions(_ctx: &Ctx) -> Result<AssertResult> {
    use perf_event_open_sys as pes;
    use perf_event_open_sys::bindings::{PERF_COUNT_HW_INSTRUCTIONS, PERF_TYPE_HARDWARE};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let mut attr = pes::bindings::perf_event_attr::default();
    attr.size = std::mem::size_of::<pes::bindings::perf_event_attr>() as u32;
    attr.type_ = PERF_TYPE_HARDWARE;
    attr.config = PERF_COUNT_HW_INSTRUCTIONS as u64;
    // disabled=1: don't count until we explicitly enable.
    // exclude_kernel=1: only count guest userspace work, not
    // kernel-side overhead (interrupt handlers, syscall
    // entry/exit). The test work is a tight userspace loop —
    // including kernel time would inflate the count for noise.
    attr.set_disabled(1);
    attr.set_exclude_kernel(1);
    attr.set_exclude_hv(1);

    // pid=0, cpu=-1: count this thread on whatever CPU it runs on.
    // group_fd=-1: standalone leader. flags=0: default.
    // SAFETY: `attr` is a valid initialized perf_event_attr; the
    // syscall reads it as a `*const`. The fd is checked below.
    let fd = unsafe { pes::perf_event_open(&mut attr, 0, -1, -1, 0) };
    if fd < 0 {
        let errno = std::io::Error::last_os_error();
        // raw_os_error returns Some for OS errors raised via
        // last_os_error; unwrap_or(0) maps the unreachable None
        // arm to 0, which falls into the catch-all Fail branch
        // and surfaces the original `errno` Display in the
        // diagnostic.
        let raw = errno.raw_os_error().unwrap_or(0);
        match classify_perf_open_errno(raw) {
            PerfOpenResult::Skip(reason) => {
                let mut result = AssertResult::pass();
                result.details.push(AssertDetail::new(
                    DetailKind::Other,
                    format!(
                        "perf_event_open failed with host-config errno \
                         ({errno}, raw={raw}); skipping counter assertion. \
                         {reason}.",
                    ),
                ));
                return Ok(result);
            }
            PerfOpenResult::Fail(reason) => {
                let mut result = AssertResult::pass();
                result.passed = false;
                result.details.push(AssertDetail::new(
                    DetailKind::Other,
                    format!(
                        "perf_event_open(PERF_TYPE_HARDWARE, \
                         PERF_COUNT_HW_INSTRUCTIONS) failed with errno {errno} \
                         (raw={raw}). {reason}.",
                    ),
                ));
                return Ok(result);
            }
        }
    }
    // SAFETY: the kernel returned a non-negative fd; we own it.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };

    // Typed ioctl wrappers from perf-event-open-sys — verified to expose
    // pes::ioctls::{RESET, ENABLE, DISABLE} taking c_uint arg via
    // perf-event-open-sys-6.0.0/src/functions.rs. Each returns -1 with
    // errno set on failure; the wrappers internally call the same
    // PERF_EVENT_IOC_* numbers (9219/9216/9217 on most arches) so the
    // semantics match the prior raw-number form, with the binding
    // crate as the source of truth instead of locally-typed constants.
    // SAFETY: the fd is valid and owned; ioctls touch only the fd's
    // perf event state and read no userspace memory beyond the fd.
    unsafe {
        if pes::ioctls::RESET(fd.as_raw_fd(), 0) != 0 {
            anyhow::bail!(
                "pes::ioctls::RESET failed: {}",
                std::io::Error::last_os_error(),
            );
        }
        if pes::ioctls::ENABLE(fd.as_raw_fd(), 0) != 0 {
            anyhow::bail!(
                "pes::ioctls::ENABLE failed: {}",
                std::io::Error::last_os_error(),
            );
        }
    }

    // Tight loop with black_box to defeat dead-code elimination.
    // 10M iterations of integer add: well above the noise floor
    // for HW_INSTRUCTIONS even under heavy PMU multiplexing.
    let mut acc: u64 = 0;
    for i in 0..10_000_000u64 {
        acc = std::hint::black_box(acc.wrapping_add(i));
    }
    std::hint::black_box(&acc);

    // SAFETY: same as enable.
    unsafe {
        if pes::ioctls::DISABLE(fd.as_raw_fd(), 0) != 0 {
            anyhow::bail!(
                "pes::ioctls::DISABLE failed: {}",
                std::io::Error::last_os_error(),
            );
        }
    }

    let mut buf = [0u64; 1];
    // SAFETY: buf is a valid 8-byte writable region; the kernel
    // writes a single u64 (default read_format) into it.
    let n = unsafe {
        libc::read(
            fd.as_raw_fd(),
            buf.as_mut_ptr() as *mut libc::c_void,
            std::mem::size_of_val(&buf),
        )
    };
    if n != std::mem::size_of_val(&buf) as isize {
        let errno = std::io::Error::last_os_error();
        anyhow::bail!(
            "short read from perf fd: returned {n} bytes, expected {} ({errno})",
            std::mem::size_of_val(&buf),
        );
    }
    let count = buf[0];

    let mut result = AssertResult::pass();
    if count == 0 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "perf_event_open(PERF_COUNT_HW_INSTRUCTIONS) reported \
                 count=0 after 10M black_box iterations. The PMU surface \
                 is visible (open succeeded) but the underlying counter \
                 is not advancing. Either KVM's intel_pmu_refresh \
                 clamped the v2 surface to 0 counters silently, or the \
                 guest kernel's perf core wired the event to a no-op \
                 backend. acc={acc}",
            ),
        ));
        return Ok(result);
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "perf_event_open(PERF_COUNT_HW_INSTRUCTIONS) advanced: \
             count={count} after 10M black_box iterations (acc={acc}); \
             guest PMU pipeline is live end-to-end",
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// Unit tests for the pure helpers extracted above.
//
// Plain `#[test]` functions in a `ktstr_test` integration-test
// binary are visible to nextest because the early-dispatch ctor's
// `--list` path falls through to libtest after printing gauntlet
// names (see `list_plain_tests` in
// `src/test_support/dispatch.rs`). The `parse_stall_duration_test.rs`
// file uses the same top-level pattern.
// ----------------------------------------------------------------------------

#[test]
fn classify_perf_open_errno_eacces_skips() {
    match classify_perf_open_errno(libc::EACCES) {
        PerfOpenResult::Skip(reason) => {
            assert!(
                reason.contains("EACCES") || reason.contains("perf_event_paranoid"),
                "EACCES skip reason should mention EACCES/EPERM or paranoid: got {reason:?}",
            );
        }
        other => panic!("EACCES should be Skip, got {other:?}"),
    }
}

#[test]
fn classify_perf_open_errno_eperm_skips() {
    match classify_perf_open_errno(libc::EPERM) {
        PerfOpenResult::Skip(reason) => {
            assert!(
                reason.contains("EPERM") || reason.contains("CAP_PERFMON"),
                "EPERM skip reason should mention EPERM or CAP_PERFMON: got {reason:?}",
            );
        }
        other => panic!("EPERM should be Skip, got {other:?}"),
    }
}

#[test]
fn classify_perf_open_errno_enosys_skips() {
    match classify_perf_open_errno(libc::ENOSYS) {
        PerfOpenResult::Skip(reason) => {
            assert!(
                reason.contains("ENOSYS") || reason.contains("CONFIG_PERF_EVENTS"),
                "ENOSYS skip reason should mention ENOSYS or CONFIG_PERF_EVENTS: got {reason:?}",
            );
        }
        other => panic!("ENOSYS should be Skip, got {other:?}"),
    }
}

#[test]
fn classify_perf_open_errno_einval_fails() {
    match classify_perf_open_errno(libc::EINVAL) {
        PerfOpenResult::Fail(reason) => {
            assert!(
                reason.contains("synthesized PMU surface"),
                "EINVAL fail reason should reference the synthesized PMU surface: got {reason:?}",
            );
        }
        other => panic!("EINVAL should be Fail, got {other:?}"),
    }
}

#[test]
fn classify_perf_open_errno_enodev_fails() {
    match classify_perf_open_errno(libc::ENODEV) {
        PerfOpenResult::Fail(reason) => {
            assert!(
                reason.contains("synthesized PMU surface"),
                "ENODEV fail reason should reference the synthesized PMU surface: got {reason:?}",
            );
        }
        other => panic!("ENODEV should be Fail, got {other:?}"),
    }
}

#[test]
fn classify_perf_open_errno_eopnotsupp_fails() {
    match classify_perf_open_errno(libc::EOPNOTSUPP) {
        PerfOpenResult::Fail(reason) => {
            assert!(
                reason.contains("synthesized PMU surface"),
                "EOPNOTSUPP fail reason should reference the synthesized PMU surface: got {reason:?}",
            );
        }
        other => panic!("EOPNOTSUPP should be Fail, got {other:?}"),
    }
}

#[test]
fn classify_perf_open_errno_unknown_falls_into_fail() {
    // Catch-all branch: any unmapped errno should land in Fail so a
    // novel kernel/KVM divergence surfaces as a real test failure
    // instead of being silently mapped to Skip. 9999 is well past
    // any defined errno on Linux (max errno < 200).
    match classify_perf_open_errno(9999) {
        PerfOpenResult::Fail(reason) => {
            assert!(
                reason.contains("synthesized PMU surface"),
                "unknown errno fail reason should reference the synthesized PMU surface: \
                 got {reason:?}",
            );
        }
        other => panic!("unknown errno (9999) should be Fail, got {other:?}"),
    }
}

#[test]
fn classify_perf_open_errno_zero_falls_into_fail() {
    // raw_os_error returning None (mapped to 0 at the call site)
    // hits the catch-all Fail branch.
    match classify_perf_open_errno(0) {
        PerfOpenResult::Fail(_) => {}
        other => panic!("errno=0 should be Fail (catch-all), got {other:?}"),
    }
}

#[test]
fn host_likely_has_pmu_decode_intel_with_ds_returns_true() {
    // "GenuineIntel" = EBX:0x756e6547 EDX:0x49656e69 ECX:0x6c65746e.
    // Leaf 1 EDX bit 21 set = DS feature present.
    let leaf1_edx_with_ds = 1u32 << 21;
    assert!(host_likely_has_pmu_decode(
        0x756e_6547,
        0x4965_6e69,
        0x6c65_746e,
        leaf1_edx_with_ds,
    ));
}

#[test]
fn host_likely_has_pmu_decode_intel_without_ds_returns_false() {
    // GenuineIntel vendor but DS bit clear → not enough signal to
    // call the host PMU-capable. Bit 20 set, bit 21 clear.
    let leaf1_edx_no_ds = 1u32 << 20;
    assert!(!host_likely_has_pmu_decode(
        0x756e_6547,
        0x4965_6e69,
        0x6c65_746e,
        leaf1_edx_no_ds,
    ));
}

#[test]
fn host_likely_has_pmu_decode_amd_with_ds_returns_false() {
    // "AuthenticAMD" = EBX:0x68747541 EDX:0x69746e65 ECX:0x444d4163.
    // AMD does not define leaf 0xA; treat version=0 as expected
    // regardless of leaf 1 EDX bit 21.
    let leaf1_edx_with_ds = 1u32 << 21;
    assert!(!host_likely_has_pmu_decode(
        0x6874_7541,
        0x6974_6e65,
        0x444d_4163,
        leaf1_edx_with_ds,
    ));
}

#[test]
fn host_likely_has_pmu_decode_unknown_vendor_returns_false() {
    // Unknown vendor (zeroed EBX/EDX/ECX) → no Intel-specific PMU
    // claim, so false even with DS asserted.
    let leaf1_edx_with_ds = 1u32 << 21;
    assert!(!host_likely_has_pmu_decode(0, 0, 0, leaf1_edx_with_ds));
}

#[test]
fn host_likely_has_pmu_decode_intel_all_other_bits_set_but_ds_clear() {
    // Stress: every leaf-1-EDX bit set EXCEPT bit 21. Verifies the
    // shift+mask isolates bit 21 and does not accidentally accept
    // adjacent bits as a positive signal.
    let leaf1_edx_no_ds = !(1u32 << 21);
    assert!(!host_likely_has_pmu_decode(
        0x756e_6547,
        0x4965_6e69,
        0x6c65_746e,
        leaf1_edx_no_ds,
    ));
}

#[test]
fn host_likely_has_pmu_decode_intel_only_ds_bit_set() {
    // Inverse stress: only bit 21 set. Verifies the function reads
    // bit 21 (not bit 20 or bit 22) — a one-off in the shift would
    // make this case return false.
    let leaf1_edx_only_ds = 1u32 << 21;
    assert!(host_likely_has_pmu_decode(
        0x756e_6547,
        0x4965_6e69,
        0x6c65_746e,
        leaf1_edx_only_ds,
    ));
}

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn host_likely_has_pmu_stub_returns_false_off_x86() {
    // Non-x86_64 stub. Confirms the leaf-0xA test will hit the
    // existing non-x86 short-circuit branch first; this test only
    // pins the stub's value when the short-circuit ever drops.
    assert!(!host_likely_has_pmu());
}
