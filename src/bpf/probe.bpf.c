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

/* Per-probe-hit data: (func_ip, tid) -> probe_entry. */
struct probe_key {
	u64 func_ip;
	u32 tid;
	u32 _pad;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, struct probe_key);
	__type(value, struct probe_entry);
	__uint(max_entries, MAX_FUNCS * 1024);
} probe_data SEC(".maps");

/* Ring buffer for events to userspace. */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} events SEC(".maps");

/* Global enable flag. Set by userspace after all probes attached. */
volatile const bool stt_enabled = false;

/*
 * Generic kprobe handler. Attached at runtime to each target function
 * via attach_kprobe(). Uses bpf_get_func_ip() to identify which
 * function fired, then captures args and BTF-resolved fields.
 */
SEC("kprobe/stt_probe")
int stt_probe(struct pt_regs *ctx)
{
	if (!stt_enabled)
		return 0;

	u64 ip = bpf_get_func_ip(ctx);
	u32 tid = (u32)bpf_get_current_pid_tgid();

	struct func_meta *meta = bpf_map_lookup_elem(&func_meta_map, &ip);
	if (!meta)
		return 0;

	struct probe_entry entry = {};
	entry.ts = bpf_ktime_get_ns();

	/* Capture raw args (up to 6). */
	entry.args[0] = PT_REGS_PARM1_CORE(ctx);
	entry.args[1] = PT_REGS_PARM2_CORE(ctx);
	entry.args[2] = PT_REGS_PARM3_CORE(ctx);
	entry.args[3] = PT_REGS_PARM4_CORE(ctx);
	entry.args[4] = PT_REGS_PARM5_CORE(ctx);
	entry.args[5] = PT_REGS_PARM6_CORE(ctx);

	/* Dereference struct fields via BTF-resolved offsets. */
	entry.nr_fields = meta->nr_field_specs;
	for (int i = 0; i < MAX_FIELDS && i < meta->nr_field_specs; i++) {
		struct field_spec *spec = &meta->specs[i];
		if (spec->param_idx >= MAX_ARGS)
			continue;

		u64 base = entry.args[spec->param_idx];
		if (!base)
			continue;

		u64 val = 0;
		int ret = bpf_probe_read_kernel(&val, sizeof(val),
						(void *)(base + spec->offset));
		if (ret == 0)
			entry.fields[spec->field_idx] = val;
	}

	struct probe_key key = { .func_ip = ip, .tid = tid };
	bpf_map_update_elem(&probe_data, &key, &entry, BPF_ANY);

	return 0;
}

/*
 * Trigger kprobe. Attached to scx_exit (or user-specified function).
 * Reads all captured probe data for the current tid and sends it to
 * userspace via ring buffer.
 */
SEC("kprobe/stt_trigger")
int stt_trigger(struct pt_regs *ctx)
{
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

	/* Capture kernel stack. */
	int stack_sz = bpf_get_stack(ctx, event->kstack,
				     sizeof(event->kstack), 0);
	event->kstack_sz = stack_sz > 0 ? stack_sz / sizeof(u64) : 0;

	bpf_ringbuf_submit(event, 0);

	return 0;
}
