// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "intf.h"

char _license[] SEC("license") = "GPL";

/* Per-CPU counter infrastructure. Each hot counter is a slot in a
 * cacheline-aligned per-CPU array in `.bss`, indexed by
 * `bpf_get_smp_processor_id() & CPU_MASK`. Replaces N independent
 * `__sync_fetch_and_add(&ktstr_<name>, 1)` against shared globals,
 * which caused full cacheline bounces on every fire from per-CPU
 * tracepoint handlers (preempt_disable / preempt_enable in
 * particular). The struct is forced to 128-byte alignment so each
 * slot occupies its own cacheline (every common host arch ktstr
 * targets has cachelines <= 128 bytes); the array shape
 * `[MAX_CPUS][KTSTR_PCPU_NR]` keeps each CPU's counters
 * contiguous, which the host-side reader sums over by walking the
 * `.bss` Datasec via BTF.
 *
 * MAX_CPUS = 256 covers every realistic ktstr VM (host topology
 * far exceeds guest vCPU counts; ktstr's own kconfig caps the
 * guest CPU count well below 256). The CPU_MASK & operation is a
 * cheap saturating fold for the impossible case where
 * `bpf_get_smp_processor_id()` returns >= MAX_CPUS — the slot
 * still hits a valid array entry; cross-CPU collisions on the
 * folded slot are benign because the atomic add still preserves
 * counter monotonicity. */
#define CPU_MASK 255
#define MAX_CPUS (CPU_MASK + 1)

struct pcpu_counter {
	long value;
} __attribute__((aligned(128)));

enum ktstr_pcpu_idx {
	KTSTR_PCPU_PROBE_COUNT = 0,
	KTSTR_PCPU_KPROBE_RETURNS,
	KTSTR_PCPU_META_MISS,
	KTSTR_PCPU_RINGBUF_DROPS,
	KTSTR_PCPU_TIMELINE_COUNT,
	KTSTR_PCPU_TIMELINE_DROPS,
	KTSTR_PCPU_PI_COUNT,
	KTSTR_PCPU_PI_ORPHAN_FEXITS,
	KTSTR_PCPU_PI_CLASS_CHANGE_COUNT,
	KTSTR_PCPU_PI_DROPS,
	KTSTR_PCPU_LOCK_CONTEND_COUNT,
	KTSTR_PCPU_LOCK_CONTEND_DROPS,
	KTSTR_PCPU_PREEMPT_DISABLE_COUNT,
	KTSTR_PCPU_PREEMPT_ENABLE_COUNT,
	KTSTR_PCPU_TRIGGER_COUNT,
	KTSTR_PCPU_NR
};

struct pcpu_counter ktstr_pcpu_counters[MAX_CPUS][KTSTR_PCPU_NR];

static __always_inline void ktstr_pcpu_inc(u32 idx)
{
	u32 cpu = bpf_get_smp_processor_id() & CPU_MASK;
	__sync_add_and_fetch(&ktstr_pcpu_counters[cpu][idx].value, 1);
}

/* Userspace-populated: maps func_ip -> func_meta. */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, u64);
	__type(value, struct func_meta);
	__uint(max_entries, MAX_FUNCS);
} func_meta_map SEC(".maps");

/* Per-probe-hit data: (func_ip, task_ptr) -> probe_entry. */
struct probe_key {
	u64 func_ip;
	u64 task_ptr;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, struct probe_key);
	__type(value, struct probe_entry);
	__uint(max_entries, MAX_FUNCS * 1024);
} probe_data SEC(".maps");

/* Per-CPU scratch buffer for probe_entry construction. Avoids
 * stack-allocating ~395 bytes (probe_entry with exit fields)
 * which would exceed the 512-byte BPF stack limit. */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct probe_entry);
	__uint(max_entries, 1);
} probe_scratch SEC(".maps");

/* Ring buffer for events to userspace. Prefixed `ktstr_` so the
 * failure-dump renderer's bare-name skip list can drop the
 * framework's own ringbuf without colliding with a user
 * scheduler's map literally named `events`. */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} ktstr_events SEC(".maps");

/* Dedicated timeline ringbuf for the sched_switch /
 * sched_migrate_task / sched_wakeup tracepoint handlers (#27). Sized
 * for the "drained only on test failure" contract: 1 MiB / 40 B per
 * record = ~26k events of headroom (~a few seconds of full-tilt
 * scheduler activity on a small VM). On overflow, the producer's
 * `bpf_ringbuf_reserve` returns NULL, the new event is dropped, and
 * the `KTSTR_PCPU_TIMELINE_DROPS` slot in `ktstr_pcpu_counters` is
 * incremented (host-side reader sums across CPUs). The host-side
 * consumer polls this ringbuf only after the error-exit latch fires
 * (see `ktstr_err_exit_detected`) — zero syscall traffic / consumer
 * wakeups during a passing test. */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 * 1024 * 1024);
} timeline_events SEC(".maps");

/* Global enable flag. Set by userspace after all probes attached.
 *
 * Gates kprobe execution only — the tp_btf/sched_ext_exit trigger
 * fires regardless of this flag.
 */
volatile const bool ktstr_enabled = false;

/*
 * Sticky error-exit latch. Set to non-zero by the tp_btf/sched_ext_exit
 * handler when an error-class exit (kind >= SCX_EXIT_ERROR) fires.
 * Lives in writable .bss so an external observer with read access
 * to guest memory can detect the transition. Sticky: re-firing the
 * tracepoint does not unset it. volatile so the BPF verifier does
 * not optimize the store away.
 *
 * u32 width (not bool) because the BPF backend rejects atomic ops on
 * 8-bit slots ("unsupported atomic operation, please use 32/64 bit
 * version"). The publishing site uses __sync_val_compare_and_swap
 * for cross-core-ordered publication on weakly-ordered architectures.
 *
 * Byte offset within .bss is resolved via BTF Datasec lookup at
 * freeze-coordinator startup (`vmm::load_probe_bss_offset` ->
 * `monitor::btf_offsets::resolve_var_offset_in_section` walks the
 * probe's BTF for the VarSecinfo named "ktstr_err_exit_detected").
 * Falls back to 0 during early boot before the program BTF is
 * loadable. This declaration's position relative to other globals
 * therefore no longer matters; reordering or adding more writable
 * globals is safe.
 *
 * Lifecycle (one-shot per VM run):
 *  - Initial value: `0` at probe load. libbpf zeroes .bss when the
 *    BPF program is loaded; the freeze coordinator sees `0` until
 *    the latch fires.
 *  - Set: the tp_btf handler above CAS's `0 -> 1` on the first
 *    error-class exit. Sticky: subsequent fires no-op.
 *  - Read: the freeze coordinator polls this value via host-side
 *    guest-memory access (`vmm::mod.rs` lazy `BpfMapAccessor`
 *    discovery + `mem.read_u32`), then triggers a single freeze on
 *    `!= 0`.
 *  - Clear: the freeze coordinator NEVER clears this byte. The
 *    latch is intentionally one-shot per VM run — the
 *    coordinator triggers at most one failure dump, and a re-armed
 *    latch would only matter if the VM kept running past the
 *    first error, which it does not (the dump is followed by VM
 *    teardown).
 *  - Reload-within-run contract: the probe BPF program stays
 *    loaded for the VM's lifetime; only the *scheduler under test*
 *    reloads when a test exercises multiple schedulers in one VM
 *    run. Because the latch is sticky and the freeze coordinator
 *    never resets it, a second scheduler's error-class exit
 *    cannot trigger a second freeze on its own — the first
 *    scheduler's transition already drove `0 -> 1`, and the
 *    second sched_ext_exit's CAS no-ops. To get a per-reload
 *    dump, the host MUST zero this `.bss` byte (at the BTF-
 *    resolved offset above) BEFORE the new scheduler is
 *    permitted to attach. Two distinct call paths, with
 *    different scopes:
 *      * Guest-context (libbpf API, INSIDE the VM) —
 *        `bpf_map__update_elem(probe_bp__bss_map, &zero_key,
 *        &zero_val, BPF_ANY)` issues a kernel-side update via
 *        the bpf() syscall, lowering to the same .bss page the
 *        BPF program reads. Available only to code running
 *        inside the guest with a libbpf handle on the probe
 *        skeleton.
 *      * Host-side (direct guest-memory write, OUTSIDE the VM) —
 *        translate the .bss map's `value_kva` plus the BTF-
 *        resolved field offset to a guest physical address (the
 *        same translation the freeze coordinator does at
 *        vmm/mod.rs:`load_probe_bss_offset` +
 *        `translate_any_kva`), then zero the byte at that PA in
 *        the host-mapped `GuestMem`. The libbpf API is NOT
 *        available from host code outside the guest — only the
 *        direct PA write works there.
 *    Skipping the clear leaves the latch at `1`; the very first
 *    poll iteration after reload would observe the flipped flag
 *    and trigger a failure dump for state belonging to the
 *    *previous* scheduler — a stale and misleading dump.
 *  - Reset: across VM runs, the BPF program is reloaded; libbpf
 *    re-zeroes .bss. There is no "clear and resume" path inside
 *    the framework. If a future caller reuses the same BPF
 *    program object across multiple VM runs without reload, that
 *    caller MUST zero this `.bss` byte before reuse (otherwise
 *    the second run would see a pre-set latch and trigger a
 *    spurious freeze immediately). For guest-context callers
 *    `bpf_map__update_elem` against the `.bss` map at the
 *    resolved offset with `value=0` works on libbpf master; for
 *    host-side reset use the same translated-PA write described
 *    in the Reload-within-run contract above.
 */
volatile u32 ktstr_err_exit_detected = 0;

/* Diagnostic counters live in the per-CPU `ktstr_pcpu_counters`
 * array above; see the `enum ktstr_pcpu_idx` declaration for the
 * slot-to-name mapping. The host-side reader sums each slot across
 * CPUs to recover the cumulative count. The previous global
 * `__sync_fetch_and_add(&ktstr_<name>, 1)` pattern is replaced by
 * `ktstr_pcpu_inc(KTSTR_PCPU_<NAME>)` at every fire site. */

/* Nanosecond timestamp (bpf_ktime_get_ns) of the first error-class
 * sched_ext_exit fire — written exactly once when the latch flips
 * 0 -> 1. Lets the timeline render "first error visible at T+X ms"
 * and lets a host-side observer correlate the latch transition with
 * the rest of the sample series. Sticky: stays at the first value. */
u64 ktstr_last_trigger_ts = 0;

/* System-wide SCX_EV_* event counter snapshot captured at the
 * first error-class `sched_ext_exit` fire via `scx_bpf_events`
 * (kernel/sched/ext.c:9417). Mirrors `struct scx_event_stats` from
 * `kernel/sched/ext_internal.h:867` (13 s64 counters in declaration
 * order). The Datasec walker on the host side renders this struct
 * by name in the failure-dump's `.bss` map output, so an operator
 * sees the system-wide counter values exactly when the scheduler
 * errored. Cross-CPU aggregation happens kernel-side
 * (`scx_read_events`); this BPF program just stores the
 * aggregated snapshot.
 *
 * Sticky: written exactly once when the error latch flips
 * 0 -> 1, so a host-side observer that polls
 * `ktstr_err_exit_detected` and sees `1` is guaranteed to see a
 * matching populated `ktstr_exit_event_stats`. Subsequent fires
 * (which might come from racing `scx_sched` instances) skip the
 * write to keep the snapshot causally tied to the first error. */
struct scx_event_stats ktstr_exit_event_stats = {};

/* `KTSTR_PCPU_TIMELINE_COUNT` / `KTSTR_PCPU_TIMELINE_DROPS` are
 * per-CPU slots in the array above. The timeline producers
 * (sched_switch, sched_migrate_task, sched_wakeup) fire on every
 * scheduler decision per CPU — turning the previous shared-global
 * counter into a per-CPU slot eliminates the cacheline bounce that
 * was the steady-state cost of having the timeline ringbuf
 * attached. */

/* PI fentry/fexit counters live in the per-CPU array as
 * `KTSTR_PCPU_PI_COUNT`, `KTSTR_PCPU_PI_ORPHAN_FEXITS`,
 * `KTSTR_PCPU_PI_CLASS_CHANGE_COUNT`, and `KTSTR_PCPU_PI_DROPS`.
 * `rt_mutex_setprio` is a sparse kernel path so the steady-state
 * fire rate is low, but moving it into the per-CPU array keeps
 * the hot-path counter pattern uniform across every event class
 * — a future tracepoint addition just appends another slot to
 * `enum ktstr_pcpu_idx` instead of reintroducing a shared
 * global. */

/* `KTSTR_PCPU_LOCK_CONTEND_COUNT` /
 * `KTSTR_PCPU_LOCK_CONTEND_DROPS` are per-CPU slots in the array
 * above. `tp_btf/contention_begin` fires from every contended-
 * lock waiter path on every CPU, so per-CPU storage is critical:
 * a lock-storm test on a CONFIG_LOCK_STAT-enabled kernel can
 * generate hundreds of millions of fires across the run. */

/* Sticky scx_sched-state snapshot taken at the same atomic moment as
 * the `ktstr_err_exit_detected` latch (BEFORE the publishing CAS, so
 * a host observer that polls the latch and sees `1` is guaranteed to
 * also see populated snapshot fields). The host-side dump renderer
 * resolves these vars by name via the probe's BTF Datasec walk and
 * uses them as a fallback for `read_scx_sched_state` when the live
 * `*scx_root` deref returns NULL — which happens during the narrow
 * teardown window where `scx_unregister` has already nulled the
 * root pointer but the failure dump is still in flight.
 *
 * The kernel writes `*scx_root` to NULL during scheduler teardown
 * (kernel/sched/ext.c::scx_unregister); a freeze that fires AFTER
 * the err exit but BEFORE the kernel reaches the next idle would
 * still see the populated `*scx_root` and read live state. A freeze
 * delayed past `scx_unregister` (slow guest, contended lock, etc.)
 * would observe `*scx_root == 0` and lose every scheduler scalar —
 * `aborting`, `bypass_depth`, `exit_kind`, `watchdog_timeout`. The
 * snapshot below is captured BEFORE the scheduler reaches the
 * teardown path because the BPF tp_btf handler fires from inside
 * `scx_claim_exit` (kernel/sched/ext.c:9210) — well before
 * `scx_unregister` runs. So the values written here represent the
 * scheduler at the instant it errored out, even if `*scx_root` has
 * been nulled by the time the host reads guest memory.
 *
 * All five fields are sticky: written exactly once when the latch
 * flips 0 -> 1. Subsequent error-class fires (racing scx_sched
 * instances) skip the writes to keep the snapshot causally tied to
 * the first error — same rule as `ktstr_exit_event_stats` /
 * `ktstr_last_trigger_ts`.
 */

/* `scx_sched.aborting` at the moment the first error-class exit
 * fired. Mirrors the 1-byte bool in `struct scx_sched`; written via
 * `BPF_CORE_READ` so a kernel build with the bit at a different
 * offset (debug vs release) still resolves correctly. */
bool ktstr_exit_aborting = false;

/* `scx_sched.bypass_depth` at the same instant. Non-zero indicates
 * the kernel was in bypass mode (dispatching tasks without the BPF
 * scheduler) when the error fired. */
s32 ktstr_exit_bypass_depth = 0;

/* The `kind` argument the tp_btf handler received. Stored even when
 * `*scx_root` is NULL (no BPF_CORE_READ chain needed) so the
 * fallback path always has the SCX_EXIT_* class even when every
 * scheduler-scalar read fails. */
u32 ktstr_exit_kind_snap = 0;

/* The kernel virtual address of the `scx_sched` instance the kernel
 * read `*scx_root` to at the snapshot instant. Zero when
 * `*scx_root == 0` already (the BPF program reads `&scx_root` via
 * `bpf_probe_read_kernel`, then dereferences). The host renderer
 * uses this to confirm the snapshot's scope when multiple scheds
 * are loaded. */
u64 ktstr_exit_sched_kva = 0;

/* `scx_sched.watchdog_timeout` (jiffies) at the same instant. Lets
 * the dump report the kernel's observed timeout setting independent
 * of the host-side `KtstrTestEntry.watchdog_timeout` plumbing — a
 * scheduler that runtime-overrode the timeout (e.g. via
 * `scx_sched.watchdog_timeout` write in init) is captured as it
 * was. */
u64 ktstr_exit_watchdog_timeout = 0;

/* Per-task scratch map for `rt_mutex_setprio` fentry/fexit
 * pairing (#61). Keyed by `p` (the boosted task's `task_struct *`),
 * storing the entry-side snapshot the fexit handler needs to
 * detect a class transition and emit a complete prio-pair record.
 * Sized at 1024 entries — at most `num_online_cpus`
 * `rt_mutex_setprio` calls can be in flight simultaneously (the
 * function holds `p->pi_lock`), but mutex chains can boost many
 * distinct tasks; 1024 gives ample headroom for any realistic
 * ktstr scenario.
 *
 * BPF_MAP_TYPE_HASH (not LRU) so an orphan entry that fexit never
 * paired stays around and surfaces via `KTSTR_PCPU_PI_ORPHAN_FEXITS`
 * on the next fentry that reuses the slot — LRU silent-eviction
 * would mask the producer bug. The fexit handler always deletes
 * the entry after a successful pair, so steady-state map
 * occupancy stays at the in-flight count. */
struct ktstr_pi_entry {
	unsigned long long ts;
	int oldprio;
	unsigned long long prev_class;  /* `p->sched_class` kva at entry */
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, u64);
	__type(value, struct ktstr_pi_entry);
	__uint(max_entries, 1024);
} pi_scratch SEC(".maps");

/* Log of IPs that missed func_meta_map lookup, for diagnosis. */
u64 ktstr_miss_log[MAX_MISS_LOG] = {};
u32 ktstr_miss_log_idx = 0;

/* `scx_bpf_events` kfunc declaration. Kernel definition lives at
 * `kernel/sched/ext.c:9417`; the kfunc takes a writable pointer
 * to a `struct scx_event_stats` plus its size (the kernel uses
 * `min(events__sz, sizeof(*events))` so passing a smaller-or-equal
 * size is always safe — same vmlinux.h here as the running kernel
 * means the size match is exact, but the `__sz` suffix is required
 * by the BPF verifier convention for size-paired kfunc params).
 *
 * Declared `extern` so the BPF loader resolves it via kfunc symbol
 * lookup at attach time; the static assert below catches a vmlinux.h
 * desync that would land bytes in the wrong fields. */
extern void scx_bpf_events(struct scx_event_stats *events,
			   __u64 events__sz) __ksym;

/* `scx_root` data-symbol extern. The kernel definition is a global
 * `struct scx_sched __rcu *scx_root` (kernel/sched/ext.c:22). Taking
 * `&scx_root` gives the kernel virtual address of the pointer
 * variable; the BPF tp_btf handler reads through that with
 * `bpf_probe_read_kernel` to get the live `*scx_root` (the actual
 * scx_sched* the kernel currently has attached).
 *
 * Declared `__weak` so a kernel image without `scx_root` exported
 * (pre-6.16, stripped vmlinux, sched_ext-disabled config) still
 * loads the probe — the loader resolves &scx_root to NULL and the
 * tp_btf handler skips the snapshot capture rather than failing the
 * BPF program load. The host-side `read_scx_sched_state` path stays
 * as fallback in those cases. The snapshot is the strict subset of
 * scheduler state the host renderer needs when `*scx_root == 0` at
 * dump time — a scenario impossible to recover from purely
 * host-side. */
extern struct scx_sched *scx_root __ksym __weak;

/* CO-RE forward-compat shadow of `struct scx_sched`. The three fields
 * captured into the error-exit snapshot (`aborting`, `bypass_depth`,
 * `watchdog_timeout`) were added to `struct scx_sched` after 6.14/7.0:
 * `aborting` and `bypass_depth` migrated from globals; `watchdog_timeout`
 * was made sub-sched-aware in 2026 (Tejun Heo, "sched_ext: Make
 * watchdog sub-sched aware"). On older kernels whose vmlinux.h still
 * predates those moves, `BPF_CORE_READ(sched, aborting)` fails to
 * compile because the C struct emitted by `vmlinux_gen` lacks the
 * member entirely.
 *
 * The `___fwd` suffix is the libbpf CO-RE convention for "match this
 * shadow against the un-suffixed type in the target kernel BTF":
 * `___fwd` is stripped on relocation, so `BPF_CORE_READ` calls cast
 * through this shadow get rewritten against `struct scx_sched`'s real
 * layout at load time. `preserve_access_index` annotates every member
 * access so CO-RE knows to relocate the offset.
 *
 * Each access site is gated with `bpf_core_field_exists(struct
 * scx_sched___fwd, <field>)` — on a kernel BTF lacking the field, the
 * built-in returns 0, the gate skips the read, and the host-side
 * snapshot field stays at its 0/false default (the host renderer
 * treats those as "snapshot unavailable, fall back to live read"). */
struct scx_sched___fwd {
	bool aborting;
	s32 bypass_depth;
	unsigned long watchdog_timeout;
} __attribute__((preserve_access_index));

#define EVENT_NAME_MAX 32

/*
 * Generic kprobe handler. Attached at runtime to each target function
 * via attach_kprobe(). Uses bpf_get_func_ip() to identify which
 * function fired, then captures args and BTF-resolved fields.
 */
SEC("kprobe/ktstr_probe")
int ktstr_probe(struct pt_regs *ctx)
{
	if (!ktstr_enabled)
		return 0;

	ktstr_pcpu_inc(KTSTR_PCPU_PROBE_COUNT);

	u64 ip = bpf_get_func_ip(ctx);
	u64 task_ptr = (u64)bpf_get_current_task();

	struct func_meta *meta = bpf_map_lookup_elem(&func_meta_map, &ip);
	if (!meta) {
		ktstr_pcpu_inc(KTSTR_PCPU_META_MISS);
		u32 idx = __sync_fetch_and_add(&ktstr_miss_log_idx, 1);
		if (idx < MAX_MISS_LOG)
			ktstr_miss_log[idx] = ip;
		return 0;
	}

	u32 zero = 0;
	struct probe_entry *entry = bpf_map_lookup_elem(&probe_scratch, &zero);
	if (!entry)
		return 0;
	__builtin_memset(entry, 0, sizeof(*entry));

	entry->ts = bpf_ktime_get_ns();

	/* Capture raw args (up to 6). */
	entry->args[0] = PT_REGS_PARM1_CORE(ctx);
	entry->args[1] = PT_REGS_PARM2_CORE(ctx);
	entry->args[2] = PT_REGS_PARM3_CORE(ctx);
	entry->args[3] = PT_REGS_PARM4_CORE(ctx);
	entry->args[4] = PT_REGS_PARM5_CORE(ctx);
	entry->args[5] = PT_REGS_PARM6_CORE(ctx);

	/* Dereference struct fields via BTF-resolved offsets. */
	entry->nr_fields = meta->nr_field_specs;
	for (int i = 0; i < MAX_FIELDS && i < meta->nr_field_specs; i++) {
		struct field_spec *spec = &meta->specs[i];
		u32 pidx = spec->param_idx;
		u32 fidx = spec->field_idx;

		if (pidx >= MAX_ARGS || fidx >= MAX_FIELDS || !spec->size)
			continue;

		u64 base = entry->args[pidx];
		if (!base)
			continue;

		/* Chained pointer dereference: read intermediate pointer
		 * first, then read through it (e.g. ->cpus_ptr->bits[0]). */
		if (spec->ptr_offset) {
			u64 ptr = 0;
			int r = bpf_probe_read_kernel(&ptr, sizeof(ptr),
						(void *)(base + spec->ptr_offset));
			if (r != 0 || !ptr)
				continue;
			base = ptr;
		}

		u64 val = 0;
		u32 sz = spec->size;
		if (sz > sizeof(val))
			sz = sizeof(val);
		int ret = bpf_probe_read_kernel(&val, sz,
						(void *)(base + spec->offset));
		if (ret == 0)
			entry->fields[fidx] = val;
	}

	/* Read string arg if func_meta specifies one. */
	if (meta->str_param_idx < MAX_ARGS) {
		u64 str_ptr = entry->args[meta->str_param_idx];
		if (str_ptr) {
			bpf_probe_read_kernel_str(entry->str_val,
						  sizeof(entry->str_val),
						  (void *)str_ptr);
			entry->has_str = 1;
			entry->str_param_idx = meta->str_param_idx;
		}
	}

	struct probe_key key = { .func_ip = ip, .task_ptr = task_ptr };
	bpf_map_update_elem(&probe_data, &key, entry, BPF_ANY);

	ktstr_pcpu_inc(KTSTR_PCPU_KPROBE_RETURNS);
	return 0;
}

/*
 * Tracepoint trigger. Fires from inside scx_claim_exit() after the
 * per-scx_sched atomic cmpxchg succeeds. Each scx_sched (top-level
 * scheduler and any sub-scheds reached via PARENT propagation) fires
 * its own tracepoint instance, in the context of the current task at
 * exit time.
 *
 * Typed arg gives the exit kind directly.
 */
SEC("tp_btf/sched_ext_exit")
int BPF_PROG(ktstr_trigger_tp, unsigned int kind)
{
	ktstr_pcpu_inc(KTSTR_PCPU_TRIGGER_COUNT);

	/*
	 * Skip non-error exits (kind < SCX_EXIT_ERROR). The error-exit
	 * latch and auto-repro both trigger only on error-class exits.
	 */
	if (kind < SCX_EXIT_ERROR)
		return 0;

	/*
	 * Latch the error-exit flag for any error-class exit
	 * (SCX_EXIT_ERROR, SCX_EXIT_ERROR_BPF, SCX_EXIT_ERROR_STALL).
	 * Sticky: re-firing the tracepoint does not unset it.
	 *
	 * Capture the timestamp BEFORE the latch CAS so a host-side
	 * observer that polls `ktstr_err_exit_detected` and sees `1` is
	 * guaranteed to also see a non-zero `ktstr_last_trigger_ts`.
	 * The previous order (CAS first, ts after) opened a window where
	 * the host could observe `latch=1` while `ts` was still the
	 * initial 0 — surfacing a "first error visible at T+0 ms"
	 * artifact in the timeline. Storing ts first and then publishing
	 * the latch transition closes that window: the CAS provides
	 * release semantics so the ts store happens-before the latch
	 * write any other CPU sees.
	 *
	 * Use __sync_val_compare_and_swap() rather than a plain store so
	 * the publication has full-barrier semantics: the BPF backend
	 * lowers it to a BPF atomic compare-exchange which carries an
	 * implicit memory barrier. A plain store would not provide the
	 * cross-core ordering an external observer needs on
	 * weakly-ordered architectures (aarch64). __sync_synchronize()
	 * cannot be used because the BPF LLVM backend cannot select an
	 * AtomicFence node.
	 *
	 * Concurrent error-class fires across multiple scx_sched
	 * instances can race on the ts store — every fire writes its
	 * own bpf_ktime_get_ns() result before attempting the CAS, so
	 * the persisted ts is one of the racing fires' timestamps
	 * (always non-zero by the time any reader sees latch=1). This
	 * relaxes the older "first writer's ts wins" sticky semantic
	 * to "any racing fire's ts wins" — the deviations between
	 * concurrent racing fires are sub-microsecond on modern x86
	 * (see `bpf_ktime_get_ns` -> `ktime_get_mono_fast_ns`) and
	 * irrelevant to the timeline-correlation use case the field
	 * exists for.
	 */
	ktstr_last_trigger_ts = bpf_ktime_get_ns();
	/*
	 * Snapshot the system-wide SCX_EV_* counters BEFORE the latch
	 * CAS publishes the error. Same happens-before ordering as the
	 * timestamp store above: a host-side observer that polls
	 * `ktstr_err_exit_detected` and sees `1` is then guaranteed to
	 * see populated `ktstr_exit_event_stats` because the CAS below
	 * provides release semantics over the prior plain stores.
	 *
	 * Concurrent racing fires (multiple `scx_sched` instances
	 * exiting in parallel) may overwrite the snapshot with their
	 * own read; the kernel-side aggregation in `scx_bpf_events`
	 * folds across the active sched_ext root anyway, so the
	 * "last writer's view of the system" semantic is what we
	 * want — every racing fire's snapshot is a valid system-wide
	 * view at its own ktime.
	 */
	scx_bpf_events(&ktstr_exit_event_stats,
		       sizeof(ktstr_exit_event_stats));

	/*
	 * Snapshot scheduler scalars BEFORE the latch CAS so a host-side
	 * observer that polls `ktstr_err_exit_detected` and sees `1` is
	 * guaranteed to also see populated snapshot fields. Same
	 * happens-before edge as the `ts` and `event_stats` stores
	 * above: the CAS below provides release semantics over the prior
	 * plain stores.
	 *
	 * `kind` is always recorded — it's the tracepoint argument and
	 * does not depend on `*scx_root` being non-NULL. The four
	 * scheduler-state fields require a successful `*scx_root`
	 * dereference; on a kernel image where the `__weak` resolution
	 * left `&scx_root == NULL` (no scx_root symbol exported), the
	 * pointer-read short-circuits and the four fields stay at their
	 * 0/false defaults — the host renderer treats those as "snapshot
	 * unavailable, fall back to live read".
	 *
	 * Use `bpf_probe_read_kernel` to read the live `*scx_root`
	 * pointer rather than a direct dereference — the kernel pointer
	 * could be racing with `scx_unregister`'s NULL store. The probe
	 * read returns the pointer value at the read instant; the
	 * subsequent BPF_CORE_READ chain on `sched` then reads the
	 * scheduler scalars via the same atomicity guarantee.
	 *
	 * Sticky: each store is a plain assignment so the natural
	 * "last-writer wins" semantic on racing fires applies. The
	 * happens-before contract relies on the CAS below — every
	 * snapshot field is published before any consumer can observe
	 * `latch == 1`, so the consumer reads a coherent snapshot
	 * regardless of which racing fire's values landed.
	 */
	ktstr_exit_kind_snap = kind;
	if (&scx_root != NULL) {
		struct scx_sched *sched = NULL;
		int r = bpf_probe_read_kernel(&sched, sizeof(sched),
					      &scx_root);
		if (r == 0 && sched != NULL) {
			ktstr_exit_sched_kva = (u64)sched;
			/* Forward-compat reads via shadow struct:
			 * `aborting` / `bypass_depth` / `watchdog_timeout`
			 * may be absent in pre-2026 kernel BTF. The
			 * `bpf_core_field_exists` gate evaluates the
			 * relocation at BPF load time against the running
			 * kernel — when the field is missing the gate
			 * skips the read and the snapshot field stays at
			 * its 0/false default (host renderer treats those
			 * as "snapshot unavailable"). */
			struct scx_sched___fwd *sched_fwd =
				(struct scx_sched___fwd *)sched;
			if (bpf_core_field_exists(sched_fwd->aborting))
				ktstr_exit_aborting =
					BPF_CORE_READ(sched_fwd, aborting);
			if (bpf_core_field_exists(sched_fwd->bypass_depth))
				ktstr_exit_bypass_depth =
					BPF_CORE_READ(sched_fwd, bypass_depth);
			if (bpf_core_field_exists(sched_fwd->watchdog_timeout))
				ktstr_exit_watchdog_timeout =
					BPF_CORE_READ(sched_fwd, watchdog_timeout);
		}
	}

	__sync_val_compare_and_swap(&ktstr_err_exit_detected, 0u, 1u);

	/*
	 * Skip the auto-repro ringbuf path for SCX_EXIT_ERROR_STALL: the
	 * watchdog kthread or scheduler tick fires the tracepoint, so
	 * bpf_get_current_task() is unrelated to the cause and would
	 * produce misleading probe output. The error-exit latch above
	 * still records the exit, so the stall is still observable.
	 * Other error-class kinds (current and any future additions)
	 * default to getting auto-repro data unless their causal-task
	 * semantics turn out to be misleading.
	 */
	if (kind == SCX_EXIT_ERROR_STALL)
		return 0;

	u32 tid = (u32)bpf_get_current_pid_tgid();

	struct probe_event *event = bpf_ringbuf_reserve(&ktstr_events,
							sizeof(*event), 0);
	if (!event) {
		ktstr_pcpu_inc(KTSTR_PCPU_RINGBUF_DROPS);
		return 0;
	}

	event->type = EVENT_TRIGGER;
	event->tid = tid;
	event->func_idx = 0;
	event->ts = bpf_ktime_get_ns();
	event->nr_fields = 0;
	/*
	 * args[0] = causal task pointer. Only SCX_EXIT_ERROR_BPF is
	 * unambiguously caused by the currently-running task (a BPF
	 * scheduler callback faulted in the task's context, so
	 * `current` IS the task that hit the bug). SCX_EXIT_ERROR can
	 * fire from kworker context — e.g. async unregistration or
	 * sysrq — where `current` is the worker thread, not the task
	 * that triggered the exit; emitting that as args[0] would
	 * splatter the probe output with unstitched kworker frames.
	 * The target_tptr filter in run_probe_skeleton drops events
	 * with args[0] == 0, so emitting 0 here suppresses the probe
	 * output for these non-causal kinds. The error-exit latch
	 * above still records the exit, so the failure remains
	 * observable in the dump.
	 */
	event->args[0] = (kind == SCX_EXIT_ERROR_BPF)
		? (u64)bpf_get_current_task()
		: 0;

	/* Capture kernel stack. */
	int stack_sz = bpf_get_stack(ctx, event->kstack,
				     sizeof(event->kstack), 0);
	event->kstack_sz = stack_sz > 0 ? stack_sz / sizeof(u64) : 0;

	/* Store exit kind in args[1] for diagnostics. */
	event->args[1] = (u64)kind;

	bpf_ringbuf_submit(event, 0);

	return 0;
}

/*
 * Tracepoint timeline buffer (#27).
 *
 * Three tp_btf handlers — sched_switch, sched_migrate_task,
 * sched_wakeup — write a `struct timeline_event` into the dedicated
 * `timeline_events` ringbuf. The host-side consumer drains this
 * ringbuf only after the error-exit latch fires
 * (`ktstr_err_exit_detected`), so the success path pays only the
 * tracepoint hit + `bpf_ringbuf_reserve` + 40-byte memcpy + submit
 * — no syscalls, no consumer wakeups.
 *
 * All three are gated on `ktstr_enabled` so timeline recording does
 * not start until userspace has finished probe attach. The kernel
 * tp_btf prototypes used here are pinned by
 * `include/trace/events/sched.h`:
 *   - sched_switch:        (preempt, prev, next, prev_state)
 *   - sched_migrate_task:  (p, dest_cpu)
 *   - sched_wakeup:        (p)  [DECLARE_EVENT_CLASS sched_wakeup_template]
 *
 * The handlers do BTF reads (`BPF_CORE_READ`) for `prev->pid`,
 * `next->pid`, `task_cpu(p)` so a future kernel-internal layout
 * change rebuilds correctly.
 *
 * sched_stat_wait/blocked are deliberately NOT used — the schedstat
 * tracepoints do not fire for sched_ext tasks. The (sched_switch,
 * sched_wakeup) pair lets userspace reconstruct per-task wait time
 * post-hoc by diffing wake-time and on-cpu time.
 */

SEC("tp_btf/sched_switch")
int BPF_PROG(ktstr_tl_switch, bool preempt, struct task_struct *prev,
	     struct task_struct *next, unsigned int prev_state)
{
	if (!ktstr_enabled)
		return 0;

	struct timeline_event *e = bpf_ringbuf_reserve(&timeline_events,
						       sizeof(*e), 0);
	if (!e) {
		ktstr_pcpu_inc(KTSTR_PCPU_TIMELINE_DROPS);
		return 0;
	}

	e->type     = TL_EVT_SWITCH;
	e->cpu      = bpf_get_smp_processor_id();
	e->ts       = bpf_ktime_get_ns();
	e->prev_pid = (unsigned int)BPF_CORE_READ(prev, pid);
	e->next_pid = (unsigned int)BPF_CORE_READ(next, pid);
	e->a        = (u64)prev_state;
	e->b        = (u64)preempt;

	bpf_ringbuf_submit(e, 0);
	ktstr_pcpu_inc(KTSTR_PCPU_TIMELINE_COUNT);
	return 0;
}

SEC("tp_btf/sched_migrate_task")
int BPF_PROG(ktstr_tl_migrate, struct task_struct *p, int dest_cpu)
{
	if (!ktstr_enabled)
		return 0;

	struct timeline_event *e = bpf_ringbuf_reserve(&timeline_events,
						       sizeof(*e), 0);
	if (!e) {
		ktstr_pcpu_inc(KTSTR_PCPU_TIMELINE_DROPS);
		return 0;
	}

	/* `task_cpu(p)` is `p->thread_info.cpu` on x86 / `p->cpu` on
	 * older arches, so use BPF_CORE_READ on the wrapper field
	 * `wake_cpu` which the kernel keeps in lockstep with the
	 * scheduler's last-CPU view (see kernel/sched/core.c
	 * `set_task_cpu`). `wake_cpu` is on `task_struct` directly,
	 * so the read is a single dereference regardless of arch. */
	e->type     = TL_EVT_MIGRATE;
	e->cpu      = bpf_get_smp_processor_id();
	e->ts       = bpf_ktime_get_ns();
	e->prev_pid = (unsigned int)BPF_CORE_READ(p, pid);
	e->next_pid = 0;
	e->a        = (u64)(unsigned int)dest_cpu;
	e->b        = (u64)BPF_CORE_READ(p, wake_cpu);

	bpf_ringbuf_submit(e, 0);
	ktstr_pcpu_inc(KTSTR_PCPU_TIMELINE_COUNT);
	return 0;
}

SEC("tp_btf/sched_wakeup")
int BPF_PROG(ktstr_tl_wakeup, struct task_struct *p)
{
	if (!ktstr_enabled)
		return 0;

	struct timeline_event *e = bpf_ringbuf_reserve(&timeline_events,
						       sizeof(*e), 0);
	if (!e) {
		ktstr_pcpu_inc(KTSTR_PCPU_TIMELINE_DROPS);
		return 0;
	}

	e->type     = TL_EVT_WAKEUP;
	e->cpu      = bpf_get_smp_processor_id();
	e->ts       = bpf_ktime_get_ns();
	e->prev_pid = (unsigned int)BPF_CORE_READ(p, pid);
	e->next_pid = 0;
	/* Target CPU at wakeup time — the scheduler's chosen CPU for
	 * `p` (set by `try_to_wake_up` -> `select_task_rq` ->
	 * `set_task_cpu`). For sched_ext tasks this is the CPU the
	 * scheduler's `ops.select_cpu` returned. */
	e->a        = (u64)BPF_CORE_READ(p, wake_cpu);
	e->b        = 0;

	bpf_ringbuf_submit(e, 0);
	ktstr_pcpu_inc(KTSTR_PCPU_TIMELINE_COUNT);
	return 0;
}

/*
 * Priority-inheritance fentry/fexit on `rt_mutex_setprio` (#61).
 *
 * `rt_mutex_setprio(struct task_struct *p, struct task_struct *pi_task)`
 * (kernel/sched/core.c) is the canonical entry point for PI-driven
 * priority changes. The function:
 *   - reads the boosted task's old priority (`p->prio`);
 *   - if `pi_task != NULL`, sets `p->prio = pi_task->prio` (boost);
 *   - otherwise resets `p->prio` from `p->normal_prio` (deboost);
 *   - calls `__setscheduler_class` to flip `p->sched_class` if the
 *     new prio crosses the RT boundary (e.g. CFS -> RT under boost).
 *
 * The fentry/fexit pair captures (oldprio, prev_class) at entry and
 * (newprio, next_class) at exit, stitched via the `pi_scratch` map
 * keyed by `p`. The fexit handler emits a TL_EVT_PI_BOOST timeline
 * record carrying the prio pair; class flips bump
 * `KTSTR_PCPU_PI_CLASS_CHANGE_COUNT` separately so the wire shape
 * stays compatible with the existing `struct timeline_event`.
 *
 * Both probes gate on `ktstr_enabled` so PI events only land once
 * userspace has finished probe attach — fentry/fexit are
 * registered before tests start, but rt_mutex_setprio can fire
 * during early kernel boot (e.g. systemd's PI-using mutexes).
 *
 * Sparse by design: `rt_mutex_setprio` is only invoked from the
 * rt_mutex chain-walk path (kernel/locking/rtmutex.c
 * `task_blocks_on_rt_mutex` -> `rt_mutex_adjust_prio_chain` ->
 * `rt_mutex_setprio`) plus a single call from `do_set_cpus_allowed`
 * for affinity changes, so steady-state fire count is zero on a
 * test that does not exercise rt_mutex contention. The 1024-entry
 * `pi_scratch` map is amply sized for realistic concurrency.
 */
SEC("fentry/rt_mutex_setprio")
int BPF_PROG(ktstr_pi_fentry, struct task_struct *p,
	     struct task_struct *pi_task)
{
	if (!ktstr_enabled)
		return 0;

	struct ktstr_pi_entry entry = {};
	entry.ts = bpf_ktime_get_ns();
	entry.oldprio = BPF_CORE_READ(p, prio);
	entry.prev_class = (u64)BPF_CORE_READ(p, sched_class);

	u64 key = (u64)p;
	bpf_map_update_elem(&pi_scratch, &key, &entry, BPF_ANY);
	return 0;
}

SEC("fexit/rt_mutex_setprio")
int BPF_PROG(ktstr_pi_fexit, struct task_struct *p,
	     struct task_struct *pi_task)
{
	if (!ktstr_enabled)
		return 0;

	u64 key = (u64)p;
	struct ktstr_pi_entry *entry = bpf_map_lookup_elem(&pi_scratch, &key);
	if (!entry) {
		ktstr_pcpu_inc(KTSTR_PCPU_PI_ORPHAN_FEXITS);
		return 0;
	}

	int newprio = BPF_CORE_READ(p, prio);
	u64 next_class = (u64)BPF_CORE_READ(p, sched_class);

	/* Class flip count bumps BEFORE the ringbuf reserve so a
	 * drop on the wire still surfaces the structural class-
	 * transition fact via the counter. */
	if (next_class != entry->prev_class) {
		ktstr_pcpu_inc(KTSTR_PCPU_PI_CLASS_CHANGE_COUNT);
	}

	struct timeline_event *e = bpf_ringbuf_reserve(&timeline_events,
						       sizeof(*e), 0);
	if (!e) {
		ktstr_pcpu_inc(KTSTR_PCPU_PI_DROPS);
		bpf_map_delete_elem(&pi_scratch, &key);
		return 0;
	}

	e->type     = TL_EVT_PI_BOOST;
	e->cpu      = bpf_get_smp_processor_id();
	e->ts       = bpf_ktime_get_ns();
	e->prev_pid = (unsigned int)bpf_get_current_pid_tgid();
	e->next_pid = (unsigned int)BPF_CORE_READ(p, pid);
	/* `prio` is `int` in the kernel (signed -20..139 range plus
	 * sentinel). Widen to u64 via the s32 conversion so a negative
	 * value sign-extends predictably; userspace re-narrows to i32
	 * for display. */
	e->a        = (u64)(s64)entry->oldprio;
	e->b        = (u64)(s64)newprio;

	bpf_ringbuf_submit(e, 0);
	ktstr_pcpu_inc(KTSTR_PCPU_PI_COUNT);

	bpf_map_delete_elem(&pi_scratch, &key);
	return 0;
}

/*
 * Lock contention begin tracepoint (#63).
 *
 * `tp_btf/contention_begin` fires from `kernel/locking/lockdep.c`
 * (`lock_contended` -> `__lock_contended` -> `trace_contention_begin`)
 * whenever a waiter blocks on a contended lock. The tracepoint is
 * unconditionally available in mainline — `CONFIG_LOCK_STAT` is NOT
 * a gate (only the trace_pipe / debugfs surface depends on it; the
 * tp_btf attach point is always present per
 * `include/trace/events/lock.h::DECLARE_EVENT_CLASS(contention_begin)`).
 *
 * Tracepoint signature: `(void *lock, unsigned int flags)`. The
 * `flags` field carries `LCB_*` class bits — `F_SPIN`, `F_READ`,
 * `F_WRITE`, `F_RT`, `F_PERCPU`, `F_MUTEX` — which userspace can
 * decode to attribute the contention to spinlock vs rwsem vs mutex
 * vs RT-mutex contention.
 *
 * Gated on `ktstr_enabled` so the timeline only records once
 * userspace has finished probe attach.
 */
SEC("tp_btf/contention_begin")
int BPF_PROG(ktstr_lock_contend, void *lock, unsigned int flags)
{
	if (!ktstr_enabled)
		return 0;

	struct timeline_event *e = bpf_ringbuf_reserve(&timeline_events,
						       sizeof(*e), 0);
	if (!e) {
		ktstr_pcpu_inc(KTSTR_PCPU_LOCK_CONTEND_DROPS);
		return 0;
	}

	e->type     = TL_EVT_LOCK_CONTEND;
	e->cpu      = bpf_get_smp_processor_id();
	e->ts       = bpf_ktime_get_ns();
	e->prev_pid = (unsigned int)bpf_get_current_pid_tgid();
	e->next_pid = 0;
	e->a        = (u64)(unsigned long)lock;
	e->b        = (u64)flags;

	bpf_ringbuf_submit(e, 0);
	ktstr_pcpu_inc(KTSTR_PCPU_LOCK_CONTEND_COUNT);
	return 0;
}

/*
 * Per-CPU preempt-disabled duration tracking (#64).
 *
 * Two tp_btf handlers — preempt_disable / preempt_enable — track
 * the outermost preempt-disable transitions per CPU. The kernel
 * tracepoints (declared in include/trace/events/preemptirq.h,
 * implemented in kernel/trace/trace_preemptirq.c) fire only on
 * preempt_count transitions FROM 0 (outermost disable) and TO 0
 * (outermost enable) — nested preempt_disable calls do NOT fire
 * the tracepoint, so the (disable, enable) ts pairing tracks the
 * full window the CPU was in preempt-disabled context.
 *
 * Storage: a per-CPU array map carrying `(enter_ts, max_ns)`. On
 * disable, write enter_ts. On enable, compute `now - enter_ts`,
 * update max_ns if greater. The host-side dumper reads each
 * CPU's max_ns via the existing per-CPU array reader.
 *
 * CONFIG dependency: tp_btf/preempt_disable and tp_btf/preempt_enable
 * are emitted only when CONFIG_TRACE_PREEMPT_TOGGLE is set
 * (kernel/trace/trace_preemptirq.c). When the option is absent,
 * libbpf attach gracefully fails for the tp_btf — same pattern as
 * other optional tp_btf attaches in this probe. ktstr.kconfig
 * enables CONFIG_TRACE_PREEMPT_TOGGLE so the standard ktstr-built
 * kernel always carries the tracepoints; out-of-tree kernels that
 * lack the option drop the metric without breaking probe load.
 *
 * Why per-CPU array instead of timeline ringbuf: preempt-disable
 * fires on every spinlock acquisition — emitting a ringbuf
 * record per fire would saturate the dedicated `timeline_events`
 * ring within milliseconds of a busy test. The aggregate "max
 * duration over the run" is the operationally useful metric;
 * shipping per-event records would only add noise. The wire
 * format here mirrors the per-CPU CPU-time stats surfaced via
 * `kernel_cpustat` reads — one summary-per-CPU aggregate.
 */
struct preempt_disabled_state {
	unsigned long long enter_ts;  /* ktime when the outermost
				       * preempt_disable fired; 0 when
				       * the CPU is currently in
				       * preempt-enabled context. */
	unsigned long long max_ns;    /* longest observed
				       * disable->enable interval since
				       * probe attach. Sticky-monotonic
				       * over the run; updated only when
				       * the latest interval exceeds the
				       * prior max. */
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct preempt_disabled_state);
	__uint(max_entries, 1);
} preempt_disabled_per_cpu SEC(".maps");

/* `KTSTR_PCPU_PREEMPT_DISABLE_COUNT` /
 * `KTSTR_PCPU_PREEMPT_ENABLE_COUNT` are per-CPU slots in the
 * array above. The preempt_disable / preempt_enable tracepoints
 * fire on every spinlock acquisition outermost transition, so
 * the per-CPU storage is mandatory — the prior shared-global
 * counter generated a multi-million-per-second cacheline bounce
 * across every busy CPU on a contended lock test. */

SEC("tp_btf/preempt_disable")
int BPF_PROG(ktstr_preempt_disable_tp, unsigned long ip,
	     unsigned long parent_ip)
{
	if (!ktstr_enabled)
		return 0;

	u32 zero = 0;
	struct preempt_disabled_state *st =
		bpf_map_lookup_elem(&preempt_disabled_per_cpu, &zero);
	if (!st)
		return 0;

	st->enter_ts = bpf_ktime_get_ns();
	ktstr_pcpu_inc(KTSTR_PCPU_PREEMPT_DISABLE_COUNT);
	return 0;
}

SEC("tp_btf/preempt_enable")
int BPF_PROG(ktstr_preempt_enable_tp, unsigned long ip,
	     unsigned long parent_ip)
{
	if (!ktstr_enabled)
		return 0;

	u32 zero = 0;
	struct preempt_disabled_state *st =
		bpf_map_lookup_elem(&preempt_disabled_per_cpu, &zero);
	if (!st)
		return 0;

	/* Skip if no paired enter_ts was recorded — CONFIG races at
	 * boot can deliver an enable before its enter on the same
	 * CPU (e.g. probe attached mid-section). Without a matching
	 * enter, the duration computation is invalid. */
	if (st->enter_ts == 0)
		return 0;

	u64 now = bpf_ktime_get_ns();
	u64 dur = now - st->enter_ts;
	st->enter_ts = 0;
	if (dur > st->max_ns)
		st->max_ns = dur;
	ktstr_pcpu_inc(KTSTR_PCPU_PREEMPT_ENABLE_COUNT);
	return 0;
}
