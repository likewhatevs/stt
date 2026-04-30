// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "intf.h"

char _license[] SEC("license") = "GPL";

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

/* Ring buffer for events to userspace. */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} events SEC(".maps");

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
 */
volatile u32 ktstr_err_exit_detected = 0;

/* Diagnostic counters — readable from userspace after drain.
 * ktstr_trigger_count counts ALL sched_ext_exit fires (including
 * non-error kinds like DONE/UNREG), not just error-class exits. */
u64 ktstr_trigger_count = 0;
u64 ktstr_probe_count = 0;
u64 ktstr_meta_miss = 0;

/* Log of IPs that missed func_meta_map lookup, for diagnosis. */
u64 ktstr_miss_log[MAX_MISS_LOG] = {};
u32 ktstr_miss_log_idx = 0;

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

	__sync_fetch_and_add(&ktstr_probe_count, 1);

	u64 ip = bpf_get_func_ip(ctx);
	u64 task_ptr = (u64)bpf_get_current_task();

	struct func_meta *meta = bpf_map_lookup_elem(&func_meta_map, &ip);
	if (!meta) {
		__sync_fetch_and_add(&ktstr_meta_miss, 1);
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
	__sync_fetch_and_add(&ktstr_trigger_count, 1);

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
	 * Use __sync_val_compare_and_swap() rather than a plain store so
	 * the publication has full-barrier semantics: the BPF backend
	 * lowers it to a BPF atomic compare-exchange which carries an
	 * implicit memory barrier. A plain store would not provide the
	 * cross-core ordering an external observer needs on
	 * weakly-ordered architectures (aarch64). __sync_synchronize()
	 * cannot be used because the BPF LLVM backend cannot select an
	 * AtomicFence node.
	 */
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

	struct probe_event *event = bpf_ringbuf_reserve(&events,
							sizeof(*event), 0);
	if (!event)
		return 0;

	event->type = EVENT_TRIGGER;
	event->tid = tid;
	event->func_idx = 0;
	event->ts = bpf_ktime_get_ns();
	event->nr_fields = 0;
	event->args[0] = (u64)bpf_get_current_task();

	/* Capture kernel stack. */
	int stack_sz = bpf_get_stack(ctx, event->kstack,
				     sizeof(event->kstack), 0);
	event->kstack_sz = stack_sz > 0 ? stack_sz / sizeof(u64) : 0;

	/* Store exit kind in args[1] for diagnostics. */
	event->args[1] = (u64)kind;

	bpf_ringbuf_submit(event, 0);

	return 0;
}
