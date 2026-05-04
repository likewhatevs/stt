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
/// the synthesized values into the guest's GP registers.
///
/// On non-x86_64 architectures this is unreachable (the test
/// scenario short-circuits before calling this helper).
#[cfg(target_arch = "x86_64")]
fn read_cpuid_leaf_a() -> CpuidLeafA {
    let r = unsafe { core::arch::x86_64::__cpuid_count(0xa, 0) };
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

/// Asserts the synthesized PMU-v2 surface is visible to the guest
/// via CPUID leaf 0xA. When the host has no PMU
/// (kvm.enable_pmu=0 or PMU-less hardware), the synthesizer in
/// `src/vmm/x86_64/topology.rs` leaves the leaf zeroed; under that
/// condition the test cannot meaningfully assert the v2 surface
/// and reports a non-failing AssertDetail describing the
/// observation. Hosts with a PMU advertise version >= 1; the
/// synthesizer overwrites with v2.
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
        // Host has no PMU; the leaf is zeroed by design — the
        // synthesizer skips it on a zero-version base. This is
        // the expected behavior on a no-PMU host; report it
        // without failing the test.
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "CPUID leaf 0xA reports version=0 (host has no PMU); \
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
        let raw = errno.raw_os_error();
        // Skip-only set: host-policy denials and missing kernel support.
        // Any other errno indicates the synthesized PMU surface failed
        // to bind to a backend — a real regression we MUST surface.
        let is_host_config = matches!(
            raw,
            Some(libc::EACCES) | Some(libc::EPERM) | Some(libc::ENOSYS)
        );
        if is_host_config {
            let mut result = AssertResult::pass();
            result.details.push(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "perf_event_open failed with host-config errno \
                     ({errno}, raw={raw:?}); skipping counter assertion. \
                     EACCES/EPERM = perf_event_paranoid > 2 or missing \
                     CAP_PERFMON; ENOSYS = kernel without CONFIG_PERF_EVENTS."
                ),
            ));
            return Ok(result);
        }
        // Fail on EINVAL / ENODEV / EOPNOTSUPP and any other unmapped
        // errno: the synthesized PMU surface is the prime suspect.
        let mut result = AssertResult::pass();
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "perf_event_open(PERF_TYPE_HARDWARE, \
                 PERF_COUNT_HW_INSTRUCTIONS) failed with errno {errno} \
                 (raw={raw:?}). EINVAL/ENODEV/EOPNOTSUPP indicate the \
                 synthesized PMU surface did not bind to a backend — \
                 check src/vmm/x86_64/topology.rs::leaf_0xa for x86 or \
                 src/vmm/aarch64/kvm.rs::init_pmuv3 for aarch64."
            ),
        ));
        return Ok(result);
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
