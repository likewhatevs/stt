// SPDX-License-Identifier: GPL-2.0
//
// Fentry/fexit probe skeleton for BPF struct_ops callbacks and kernel functions.
//
// Separate from the kprobe skeleton (probe.bpf.c) because fentry
// programs require set_attach_target() per target, which must happen
// before load. Loading in batches of FENTRY_BATCH (4) programs per
// skeleton instance reduces verifier passes.
//
// Each ktstr_fentry_N handler receives the raw fentry context. For
// struct_ops BPF programs, ctx[0] is a void *ctx that points to the
// real callback arguments packed contiguously. The handler dereferences
// through ctx[0] to read up to 6 args, then applies BTF-resolved field
// specs from func_meta_map to capture struct fields.
//
// Maps are shared with the kprobe skeleton via reuse_fd(): probe_data
// and func_meta_map declarations must match probe.bpf.c exactly. The
// userspace side (run_probe_skeleton in process.rs) calls reuse_fd()
// before load so both skeletons write to the same maps.
//
// Per-slot func_idx values are set via rodata before load. The sentinel
// IP (func_idx | (1<<63)) keys func_meta_map lookups.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "intf.h"

char _license[] SEC("license") = "GPL";

/* Maps shared with the kprobe skeleton via reuse_fd(). Declarations
 * must match probe.bpf.c exactly for compatible map FD reuse. */

struct probe_key {
	u64 func_ip;
	u64 task_ptr;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, u64);
	__type(value, struct func_meta);
	__uint(max_entries, MAX_FUNCS);
} func_meta_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, struct probe_key);
	__type(value, struct probe_entry);
	__uint(max_entries, MAX_FUNCS * 1024);
} probe_data SEC(".maps");

/* Per-CPU scratch buffer for probe_entry construction. */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct probe_entry);
	__uint(max_entries, 1);
} probe_scratch SEC(".maps");

volatile const bool ktstr_enabled = false;

/* Per-slot func_idx, set via rodata before load. */
volatile const u32 ktstr_fentry_func_idx_0 = 0;
volatile const u32 ktstr_fentry_func_idx_1 = 0;
volatile const u32 ktstr_fentry_func_idx_2 = 0;
volatile const u32 ktstr_fentry_func_idx_3 = 0;

/* Per-slot kernel flag. 0 = BPF callback (ctx[0] deref for args),
 * 1 = kernel function (args directly in ctx[0..N]). Controls how
 * fentry/fexit handlers read function arguments. */
volatile const u8 ktstr_fentry_is_kernel_0 = 0;
volatile const u8 ktstr_fentry_is_kernel_1 = 0;
volatile const u8 ktstr_fentry_is_kernel_2 = 0;
volatile const u8 ktstr_fentry_is_kernel_3 = 0;

u64 ktstr_fentry_probe_count = 0;

/* Read function args into entry->args based on ctx mode.
 * Kernel fentry/fexit: args are directly in ctx[0..5].
 * BPF struct_ops: args are behind ctx[0] pointer dereference. */
static __always_inline void read_ctx_args(unsigned long long *ctx, u64 *args, u8 is_kernel)
{
	if (is_kernel) {
		args[0] = ctx[0];
		args[1] = ctx[1];
		args[2] = ctx[2];
		args[3] = ctx[3];
		args[4] = ctx[4];
		args[5] = ctx[5];
	} else {
		u64 real_ctx = ctx[0];
		if (real_ctx) {
			bpf_probe_read_kernel(&args[0], sizeof(u64), (void *)real_ctx);
			bpf_probe_read_kernel(&args[1], sizeof(u64), (void *)(real_ctx + 8));
			bpf_probe_read_kernel(&args[2], sizeof(u64), (void *)(real_ctx + 16));
			bpf_probe_read_kernel(&args[3], sizeof(u64), (void *)(real_ctx + 24));
			bpf_probe_read_kernel(&args[4], sizeof(u64), (void *)(real_ctx + 32));
			bpf_probe_read_kernel(&args[5], sizeof(u64), (void *)(real_ctx + 40));
		}
	}
}

static __always_inline int ktstr_fentry_common(unsigned long long *ctx, u32 func_idx, u8 is_kernel)
{
	if (!ktstr_enabled)
		return 0;

	__sync_fetch_and_add(&ktstr_fentry_probe_count, 1);

	u64 task_ptr = (u64)bpf_get_current_task();

	u32 zero = 0;
	struct probe_entry *entry = bpf_map_lookup_elem(&probe_scratch, &zero);
	if (!entry)
		return 0;
	__builtin_memset(entry, 0, sizeof(*entry));

	entry->ts = bpf_ktime_get_ns();

	read_ctx_args(ctx, entry->args, is_kernel);

	/* Probe data key: kernel functions use real IP from bpf_get_func_ip,
	 * BPF callbacks use sentinel IP (func_idx | (1<<63)). */
	u64 map_ip = is_kernel ? bpf_get_func_ip(ctx)
			       : ((u64)func_idx | (1ULL << 63));

	struct func_meta *meta = bpf_map_lookup_elem(&func_meta_map, &map_ip);
	if (meta) {
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
	}

	/* Read string arg if func_meta specifies one. */
	if (meta && meta->str_param_idx < MAX_ARGS) {
		u64 str_ptr = entry->args[meta->str_param_idx];
		if (str_ptr) {
			bpf_probe_read_kernel_str(entry->str_val,
						  sizeof(entry->str_val),
						  (void *)str_ptr);
			entry->has_str = 1;
			entry->str_param_idx = meta->str_param_idx;
		}
	}

	struct probe_key key = { .func_ip = map_ip, .task_ptr = task_ptr };
	bpf_map_update_elem(&probe_data, &key, entry, BPF_ANY);

	return 0;
}

/*
 * Fexit common handler. Looks up the probe_data entry written by
 * fentry (BPF callbacks) or kprobe (kernel functions) and re-reads
 * struct fields into exit_fields. Uses read_ctx_args for arg access
 * (direct ctx for kernel, ctx[0] deref for BPF).
 */
static __always_inline int ktstr_fexit_common(unsigned long long *ctx, u32 func_idx, u8 is_kernel)
{
	if (!ktstr_enabled)
		return 0;

	u64 task_ptr = (u64)bpf_get_current_task();
	u64 map_ip = is_kernel ? bpf_get_func_ip(ctx)
			       : ((u64)func_idx | (1ULL << 63));

	struct probe_key key = { .func_ip = map_ip, .task_ptr = task_ptr };
	struct probe_entry *entry = bpf_map_lookup_elem(&probe_data, &key);
	if (!entry)
		return 0;

	struct func_meta *meta = bpf_map_lookup_elem(&func_meta_map, &map_ip);
	if (!meta)
		return 0;

	u64 args[MAX_ARGS] = {};
	read_ctx_args(ctx, args, is_kernel);

	entry->exit_ts = bpf_ktime_get_ns();
	entry->has_exit = 1;
	entry->nr_exit_fields = meta->nr_field_specs;

	for (int i = 0; i < MAX_FIELDS && i < meta->nr_field_specs; i++) {
		struct field_spec *spec = &meta->specs[i];
		u32 pidx = spec->param_idx;
		u32 fidx = spec->field_idx;

		if (pidx >= MAX_ARGS || fidx >= MAX_FIELDS || !spec->size)
			continue;

		u64 base = args[pidx];
		if (!base)
			continue;

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
			entry->exit_fields[fidx] = val;
	}

	return 0;
}

/* Fentry handlers. set_attach_target() retargets each slot at runtime.
 * is_kernel controls ctx access mode (direct vs ctx[0] deref). */
SEC("fentry")
int ktstr_fentry_0(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_0, ktstr_fentry_is_kernel_0); }

SEC("fentry")
int ktstr_fentry_1(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_1, ktstr_fentry_is_kernel_1); }

SEC("fentry")
int ktstr_fentry_2(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_2, ktstr_fentry_is_kernel_2); }

SEC("fentry")
int ktstr_fentry_3(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_3, ktstr_fentry_is_kernel_3); }

/* Fexit handlers: same slot mapping, writes exit_fields in-place. */
SEC("fexit")
int ktstr_fexit_0(unsigned long long *ctx) { return ktstr_fexit_common(ctx, ktstr_fentry_func_idx_0, ktstr_fentry_is_kernel_0); }

SEC("fexit")
int ktstr_fexit_1(unsigned long long *ctx) { return ktstr_fexit_common(ctx, ktstr_fentry_func_idx_1, ktstr_fentry_is_kernel_1); }

SEC("fexit")
int ktstr_fexit_2(unsigned long long *ctx) { return ktstr_fexit_common(ctx, ktstr_fentry_func_idx_2, ktstr_fentry_is_kernel_2); }

SEC("fexit")
int ktstr_fexit_3(unsigned long long *ctx) { return ktstr_fexit_common(ctx, ktstr_fentry_func_idx_3, ktstr_fentry_is_kernel_3); }
