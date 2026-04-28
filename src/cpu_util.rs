//! CPU-affinity utilities shared across the crate.
//!
//! Two helpers for reading and parsing per-task CPU affinity:
//!
//! - [`parse_cpu_list`] decodes the kernel cpulist string format
//!   (`"0-3,5,7-9"`) emitted by `/proc/<pid>/status:Cpus_allowed_list`
//!   and `/sys/devices/system/cpu/online`.
//! - [`read_affinity`] calls `sched_getaffinity(2)` with a
//!   dynamically-sized buffer so `CONFIG_NR_CPUS > 1024` hosts are
//!   handled correctly (libc's fixed `cpu_set_t` is only 1024 bits).
//!
//! Both produce sorted-deduped `Vec<u32>` of CPU ids and route
//! garbled / over-cap input to `None`. Used by the per-thread
//! profiler (host_state) AND the VM topology planner
//! (vmm::host_topology) — the function shape is generic enough that
//! either subsystem could have owned it; keeping the impls here so
//! neither has to depend on the other for a CPU-list helper.

use libc;

/// Parse a cpulist string of the form `"0-3,5,7-9"` into a
/// sorted deduped vec of CPU ids. `None` on empty input or any
/// malformed token (partial results are rejected so the caller
/// can distinguish "no data" from "data but garbled").
///
/// # Range expansion cap
///
/// A single `lo-hi` token that would expand to more than
/// 65 536 CPUs is rejected as malformed. Without this gate a
/// hostile or corrupted `Cpus_allowed_list:` value like
/// `0-4294967295` would allocate 16 GiB for the expansion vec
/// and crash the capture (or OOM the process). The cap is far
/// above any realistic `CONFIG_NR_CPUS` (current Linux defaults
/// top out at a few thousand; even `NR_CPUS=8192` builds stay
/// inside this bound), so legitimate input is never rejected.
pub fn parse_cpu_list(s: &str) -> Option<Vec<u32>> {
    /// Upper bound on the number of CPUs a single `lo-hi` token
    /// can expand to. 64 Ki — orders of magnitude above any
    /// in-production `NR_CPUS` — leaves headroom for future
    /// large-NUMA hosts while capping the worst-case allocation
    /// at 256 KiB (64 Ki × u32 = 256 KiB).
    const MAX_CPU_RANGE_EXPANSION: u64 = 65_536;

    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut out: Vec<u32> = Vec::new();
    for token in s.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = token.split_once('-') {
            let lo: u32 = lo.parse().ok()?;
            let hi: u32 = hi.parse().ok()?;
            if hi < lo {
                return None;
            }
            // Guard against hostile / corrupt range expansions.
            // Use u64 arithmetic so the `hi - lo + 1` compute
            // cannot overflow even at u32::MAX. Reject rather
            // than clamp so the caller's "no data vs data but
            // garbled" distinction stays intact.
            let span = (hi as u64) - (lo as u64) + 1;
            if span > MAX_CPU_RANGE_EXPANSION {
                return None;
            }
            for c in lo..=hi {
                out.push(c);
            }
        } else {
            out.push(token.parse::<u32>().ok()?);
        }
    }
    out.sort_unstable();
    out.dedup();
    Some(out)
}

/// Read the effective CPU affinity for a task via the
/// `sched_getaffinity(2)` syscall. Kernel accepts any pid/tid in
/// the caller's namespace; root or same-uid required per the
/// kernel's ptrace-access check. Returns sorted CPU ids.
/// `None` on syscall failure (EPERM, ESRCH) or when the kernel's
/// mask exceeds [`AFFINITY_MAX_BITS`] (hosts beyond 262144 CPUs).
///
/// # Dynamic buffer sizing
///
/// The kernel's `SYSCALL_DEFINE3(sched_getaffinity)`
/// (`kernel/sched/syscalls.c`) rejects a caller buffer shorter
/// than `nr_cpu_ids / BITS_PER_BYTE` with `EINVAL`. The kernel
/// supports `CONFIG_NR_CPUS` values up to 8192 on x86_64 default
/// and higher on custom builds (large NUMA / partitioning
/// hardware). libc's fixed [`libc::cpu_set_t`] is only 1024 bits
/// wide, so calling `sched_getaffinity` with
/// `size_of::<cpu_set_t>()` against a `CONFIG_NR_CPUS > 1024`
/// kernel fails EINVAL even when the caller has legitimate
/// access.
///
/// This helper avoids the cap by allocating a dynamically-sized
/// `Vec<c_ulong>` (an array of kernel `unsigned long` — the
/// wire format the syscall expects, aligned and byte-length a
/// multiple of `sizeof(unsigned long)` per the kernel's second
/// validation). On EINVAL the buffer doubles and the call
/// retries, capped at [`AFFINITY_MAX_BITS`] = 262144 (32 KiB of
/// mask data — covers every real-world `CONFIG_NR_CPUS` setting
/// and bounds the worst-case allocation).
///
/// # Error-class handling
///
/// - `EINVAL` → buffer too small. Double and retry until the
///   ceiling is reached, then surface None.
/// - `EPERM` / `ESRCH` → real access / process-identity failures.
///   Return None so the caller falls back to the procfs
///   `Cpus_allowed_list:` path, which bypasses the permission
///   check (reading `/proc/<tid>/status` only requires directory
///   traversal permission, not `PTRACE_MODE_READ`).
/// - Any other error → return None. The procfs fallback will
///   produce the correct value or its own None.
///
/// Without this split, the previous implementation collapsed
/// every error to None indistinguishably — EINVAL on a
/// \>1024-CPU host was treated the same as EPERM, and every
/// caller had to rely on the procfs fallback for correctness,
/// making the syscall path effectively useless on the very
/// hosts where affinity data matters most (1000-plus-CPU NUMA
/// boxes).
pub fn read_affinity(tid: i32) -> Option<Vec<u32>> {
    let mut bits = AFFINITY_INITIAL_BITS;
    loop {
        let mut buffer = affinity_zeroed_buffer(bits);
        let bytes = std::mem::size_of_val(buffer.as_slice());
        // SAFETY: `buffer.as_mut_ptr()` produces a live pointer
        // valid for `bytes` writes; the kernel writes at most
        // `min(bytes, cpumask_size)` and returns the actual byte
        // count. `bits` is always a multiple of
        // `c_ulong::BITS`, so `bytes` satisfies the kernel's
        // alignment validation (`len & (sizeof(unsigned long)-1)
        // == 0`).
        let ret = unsafe {
            libc::syscall(
                libc::SYS_sched_getaffinity,
                tid as libc::pid_t,
                bytes,
                buffer.as_mut_ptr(),
            )
        };
        if ret >= 0 {
            // ret carries the actual byte count the kernel
            // wrote. Bits beyond `ret * 8` were not touched and
            // stay at the zero-init value above — safe to
            // iterate the full buffer, but tightening the bound
            // avoids wasted work on small masks inside a large
            // buffer.
            let written_bytes = ret as usize;
            return extract_cpus_from_mask(&buffer, written_bytes);
        }
        // Error path: classify via errno.
        let errno = std::io::Error::last_os_error().raw_os_error();
        // Only EINVAL warrants a retry — it signals "buffer too
        // small" under the kernel's
        // `(len * BITS_PER_BYTE) < nr_cpu_ids` check. Every other
        // error (EPERM permission denied, ESRCH process gone,
        // EFAULT bad pointer, etc.) is terminal.
        if errno != Some(libc::EINVAL) {
            return None;
        }
        let Some(next) = affinity_next_bits(bits) else {
            // Ceiling reached without success — the host claims
            // more CPUs than the helper is willing to allocate
            // for. Surface None so the caller falls back to the
            // procfs string form, which has no bit-count cap.
            return None;
        };
        bits = next;
    }
}

/// Initial number of CPU bits the affinity buffer starts at.
/// 8192 matches the x86_64 default `CONFIG_NR_CPUS`, so the
/// overwhelming majority of hosts resolve on the first syscall.
pub const AFFINITY_INITIAL_BITS: usize = 8192;

/// Maximum number of CPU bits [`read_affinity`] is willing to
/// allocate for. 262144 bits = 32 KiB of mask data, well above
/// the largest in-production `CONFIG_NR_CPUS` this project
/// targets. Capping bounds the worst-case allocation and
/// bounds the retry loop's iteration count
/// (`log2(AFFINITY_MAX_BITS / AFFINITY_INITIAL_BITS)` = 5
/// doublings).
pub const AFFINITY_MAX_BITS: usize = 262144;

/// Given the current buffer size in bits, return the size for
/// the next retry attempt — double the current size, rejecting
/// any step that would exceed [`AFFINITY_MAX_BITS`]. Returns
/// `None` when the ceiling has been reached and no further
/// retry is allowed.
///
/// Extracted from [`read_affinity`] so the loop-termination
/// policy is unit-testable without syscall dispatch.
pub(crate) fn affinity_next_bits(current_bits: usize) -> Option<usize> {
    let doubled = current_bits.checked_mul(2)?;
    if doubled > AFFINITY_MAX_BITS {
        None
    } else {
        Some(doubled)
    }
}

/// Allocate a zeroed buffer of `c_ulong` words sized to hold
/// `bits` CPU-mask bits. The kernel's
/// `sys_sched_getaffinity` rejects any `len & (sizeof(unsigned
/// long)-1) != 0`, so the buffer is allocated in whole-word
/// units.
///
/// Extracted so [`read_affinity`]'s reset-on-retry contract is
/// visible (a fresh zeroed buffer per attempt prevents stale
/// bits from a truncated earlier read leaking into the current
/// attempt's iteration).
fn affinity_zeroed_buffer(bits: usize) -> Vec<libc::c_ulong> {
    let word_bits = libc::c_ulong::BITS as usize;
    let words = bits.div_ceil(word_bits);
    vec![0 as libc::c_ulong; words]
}

/// Walk a successfully-filled cpu-mask buffer and return the
/// sorted list of set CPU ids, or `None` when no bits were set
/// (the kernel writes a mask with at least one bit for any
/// task that was dispatchable at all; an all-zero mask would
/// imply the task has been taken off every CPU, which the
/// kernel does not expose as a valid affinity — surface None
/// rather than `Some(vec![])` so downstream callers can tell
/// "no data" from "legitimately empty mask").
///
/// `written_bytes` is the byte count the syscall reported; we
/// iterate only that range so a small mask inside a large
/// buffer does not scan past what the kernel actually wrote.
fn extract_cpus_from_mask(buffer: &[libc::c_ulong], written_bytes: usize) -> Option<Vec<u32>> {
    let word_bytes = std::mem::size_of::<libc::c_ulong>();
    let word_bits = libc::c_ulong::BITS as usize;
    let written_words = written_bytes / word_bytes;
    let mut cpus: Vec<u32> = Vec::new();
    for (word_idx, &word) in buffer.iter().take(written_words).enumerate() {
        if word == 0 {
            continue;
        }
        for bit in 0..word_bits {
            if word & (1 as libc::c_ulong) << bit != 0 {
                let cpu = word_idx * word_bits + bit;
                cpus.push(cpu as u32);
            }
        }
    }
    if cpus.is_empty() { None } else { Some(cpus) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_accepts_ranges_singletons_and_mixtures() {
        assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("5").unwrap(), vec![5]);
        assert_eq!(parse_cpu_list("0,2,4").unwrap(), vec![0, 2, 4]);
        assert_eq!(parse_cpu_list("0-2,4,6-7").unwrap(), vec![0, 1, 2, 4, 6, 7]);
    }

    #[test]
    fn parse_cpu_list_rejects_malformed_input() {
        assert!(parse_cpu_list("").is_none());
        assert!(parse_cpu_list("5-3").is_none());
        assert!(parse_cpu_list("abc").is_none());
        assert!(parse_cpu_list("0-").is_none());
        assert!(parse_cpu_list("-3").is_none());
    }

    #[test]
    fn parse_cpu_list_dedups_and_sorts() {
        assert_eq!(parse_cpu_list("3,0-2,1,2").unwrap(), vec![0, 1, 2, 3]);
    }

    /// A range whose expansion would exceed 64 Ki CPUs is
    /// rejected as malformed rather than allocating
    /// gigabytes. Without the `span > MAX_CPU_RANGE_EXPANSION`
    /// gate, a hostile or corrupt `Cpus_allowed_list:` value
    /// like `0-4294967295` would try to push 4 billion u32s
    /// into a Vec and either OOM the process or crash the
    /// capture. The cap sits orders of magnitude above any
    /// realistic `CONFIG_NR_CPUS` so legitimate inputs are
    /// never rejected.
    #[test]
    fn parse_cpu_list_rejects_huge_range() {
        // Malicious u32::MAX range — cap bites.
        assert_eq!(parse_cpu_list("0-4294967295"), None);
        // Just above the 64 Ki cap — still rejected.
        assert_eq!(parse_cpu_list("0-65536"), None);
        // At the cap — accepted (65_536 elements, the inclusive
        // `lo..=hi` boundary: 0 through 65_535).
        let at_cap = parse_cpu_list("0-65535").unwrap();
        assert_eq!(at_cap.len(), 65_536);
        // A realistic large-CPU range (e.g. 8192-way host) is
        // well under the cap and passes.
        let realistic = parse_cpu_list("0-8191").unwrap();
        assert_eq!(realistic.len(), 8192);
    }

    /// parse_cpu_list on a single-CPU range (`"5-5"`) must return
    /// a 1-element vec. `lo == hi` is the boundary of the inclusive
    /// range expansion — a regression that skipped the `lo == hi`
    /// case (e.g. `lo < hi` instead of `lo <= hi` in the loop)
    /// would drop the single element.
    #[test]
    fn parse_cpu_list_single_element_range_lo_equals_hi() {
        assert_eq!(parse_cpu_list("5-5").unwrap(), vec![5]);
        // Also pin at the cap boundary and bottom edge.
        assert_eq!(parse_cpu_list("0-0").unwrap(), vec![0]);
    }

    /// parse_cpu_list with a trailing comma (`"0,1,"`) must succeed
    /// and drop the empty token — the tokenizer has a dedicated
    /// `if token.is_empty() { continue }` arm precisely for this
    /// case. A user-pasted cpulist sometimes carries a stray comma
    /// from copy+paste; rejecting it would be a usability
    /// regression.
    #[test]
    fn parse_cpu_list_trailing_comma_accepted() {
        assert_eq!(parse_cpu_list("0,1,").unwrap(), vec![0, 1]);
        // Also the leading-comma case — same codepath.
        assert_eq!(parse_cpu_list(",0,1").unwrap(), vec![0, 1]);
    }

    /// `affinity_next_bits` doubles the buffer until the
    /// [`AFFINITY_MAX_BITS`] ceiling bites, then returns `None`
    /// to signal "give up". Pins the exact sequence 8192 →
    /// 16384 → 32768 → 65536 → 131072 → 262144 → None so a
    /// regression that replaced `checked_mul(2)` with `+= step`
    /// (or otherwise changed the growth curve) surfaces here.
    #[test]
    fn affinity_next_bits_doubles_until_ceiling() {
        assert_eq!(AFFINITY_INITIAL_BITS, 8192);
        assert_eq!(AFFINITY_MAX_BITS, 262144);
        // Full doubling chain from the initial size to the cap.
        let mut cur = AFFINITY_INITIAL_BITS;
        let expected = [16384usize, 32768, 65536, 131072, 262144];
        for &want in &expected {
            let next = affinity_next_bits(cur).expect("doubling must succeed below ceiling");
            assert_eq!(next, want, "expected {want}, got {next}");
            cur = next;
        }
        // At the cap, the next step would be 524288 > 262144 — return None.
        assert_eq!(
            affinity_next_bits(AFFINITY_MAX_BITS),
            None,
            "at the ceiling, no further retry must be allowed",
        );
    }

    /// A single-set-bit mask in the first word must be extracted
    /// to exactly that CPU id. Pins the word_idx*word_bits +
    /// bit offset arithmetic against off-by-one drift.
    #[test]
    fn extract_cpus_from_mask_single_bit_in_first_word() {
        let mut buf = vec![0 as libc::c_ulong; 4];
        // Set CPU 5 in word 0.
        buf[0] = (1 as libc::c_ulong) << 5;
        let bytes = std::mem::size_of_val(buf.as_slice());
        let cpus = extract_cpus_from_mask(&buf, bytes).expect("non-empty mask");
        assert_eq!(cpus, vec![5]);
    }

    /// A bit set in a NON-first word must be offset by
    /// word_bits * word_idx. Guards against a regression that
    /// dropped the `word_idx * word_bits` term and reported the
    /// bit position within the word instead of the absolute CPU
    /// id.
    #[test]
    fn extract_cpus_from_mask_offset_bit_in_later_word() {
        let word_bits = libc::c_ulong::BITS as usize;
        let mut buf = vec![0 as libc::c_ulong; 4];
        // Set CPU (2 * word_bits + 3) in word 2, bit 3.
        buf[2] = (1 as libc::c_ulong) << 3;
        let bytes = std::mem::size_of_val(buf.as_slice());
        let cpus = extract_cpus_from_mask(&buf, bytes).expect("non-empty mask");
        let expected = (2 * word_bits + 3) as u32;
        assert_eq!(cpus, vec![expected]);
    }

    /// `written_bytes` tighter than the buffer size must stop
    /// iteration at that byte count — bits beyond it belong to
    /// caller-zeroed padding and a kernel that returned a
    /// smaller mask than our buffer doesn't promise their shape.
    /// Pins that a stale bit planted past `written_bytes` is
    /// NOT harvested.
    #[test]
    fn extract_cpus_from_mask_respects_written_bytes() {
        let mut buf = vec![0 as libc::c_ulong; 4];
        // Plant CPU bits in word 0 AND word 3; tell the
        // extractor only word 0 was written by the kernel.
        buf[0] = (1 as libc::c_ulong) << 7; // CPU 7
        buf[3] = 1 as libc::c_ulong; // would-be CPU 3*word_bits
        let one_word_bytes = std::mem::size_of::<libc::c_ulong>();
        let cpus = extract_cpus_from_mask(&buf, one_word_bytes).expect("non-empty mask");
        // Only the bit in the first (kernel-written) word comes back.
        assert_eq!(cpus, vec![7]);
    }

    /// Empty mask (every word zero) → `None`. Pins the
    /// "Some(vec![]) is NOT a valid return" invariant — any
    /// caller that dispatches on `.is_some()` must be able to
    /// trust that a Some carries at least one CPU.
    #[test]
    fn extract_cpus_from_mask_empty_buffer_returns_none() {
        let buf = vec![0 as libc::c_ulong; 4];
        let bytes = std::mem::size_of_val(buf.as_slice());
        assert_eq!(extract_cpus_from_mask(&buf, bytes), None);
    }

    /// `affinity_zeroed_buffer` rounds UP to whole words so the
    /// byte length satisfies the kernel's
    /// `len & (sizeof(unsigned long)-1) == 0` alignment check.
    /// An off-by-one in the `div_ceil` would produce a
    /// non-multiple-of-word-size buffer and the syscall would
    /// reject with EINVAL forever (retry loop would churn but
    /// never succeed).
    #[test]
    fn affinity_zeroed_buffer_rounds_up_and_is_zeroed() {
        let word_bits = libc::c_ulong::BITS as usize;
        // Ask for exactly one word — get exactly one word.
        let exact = affinity_zeroed_buffer(word_bits);
        assert_eq!(exact.len(), 1);
        // Ask for one bit more than a word — get two words.
        let over = affinity_zeroed_buffer(word_bits + 1);
        assert_eq!(over.len(), 2);
        // Initial bits → 8192 / word_bits words.
        let init = affinity_zeroed_buffer(AFFINITY_INITIAL_BITS);
        assert_eq!(init.len(), AFFINITY_INITIAL_BITS / word_bits);
        // Every slot must be zeroed.
        assert!(init.iter().all(|&w| w == 0));
    }

    /// Smoke test against the real syscall for the current
    /// process — `read_affinity(getpid())` must succeed and
    /// return at least one CPU. The test process always has an
    /// affinity set (the kernel never runs a task off all
    /// CPUs), so None here signals a regression in the retry
    /// loop / errno classification.
    ///
    /// Distinct from the per-thread capture-path test in
    /// host_state — this test focuses on `read_affinity` in
    /// isolation so a failure localizes to the fn's own logic
    /// rather than a capture-path wiring issue.
    #[test]
    fn read_affinity_for_self_returns_at_least_one_cpu() {
        let pid = std::process::id() as i32;
        let cpus = read_affinity(pid).expect("own affinity must resolve");
        assert!(
            !cpus.is_empty(),
            "self affinity must carry at least one CPU"
        );
        // CPUs come out sorted.
        let mut sorted = cpus.clone();
        sorted.sort_unstable();
        assert_eq!(cpus, sorted, "cpus must be returned sorted ascending");
    }
}
