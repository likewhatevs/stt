// SPDX-License-Identifier: GPL-2.0
//
// Fentry probe skeleton for BPF struct_ops scheduler callbacks.
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
	u32 tid;
	u32 _pad;
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

volatile const bool ktstr_enabled = false;

/* Per-slot func_idx, set via rodata before load. */
volatile const u32 ktstr_fentry_func_idx_0 = 0;
volatile const u32 ktstr_fentry_func_idx_1 = 0;
volatile const u32 ktstr_fentry_func_idx_2 = 0;
volatile const u32 ktstr_fentry_func_idx_3 = 0;

u64 ktstr_fentry_probe_count = 0;

static __always_inline int ktstr_fentry_common(unsigned long long *ctx, u32 func_idx)
{
	if (!ktstr_enabled)
		return 0;

	__sync_fetch_and_add(&ktstr_fentry_probe_count, 1);

	u32 tid = (u32)bpf_get_current_pid_tgid();

	struct probe_entry entry = {};
	entry.ts = bpf_ktime_get_ns();

	/* For struct_ops BPF programs, fentry receives ONE arg: ctx[0]
	 * is a void *ctx pointer. The real callback arguments are stored
	 * in the array that ctx points to. Dereference through it. */
	u64 real_ctx = ctx[0];
	if (real_ctx) {
		bpf_probe_read_kernel(&entry.args[0], sizeof(u64),
				      (void *)real_ctx);
		bpf_probe_read_kernel(&entry.args[1], sizeof(u64),
				      (void *)(real_ctx + 8));
		bpf_probe_read_kernel(&entry.args[2], sizeof(u64),
				      (void *)(real_ctx + 16));
		bpf_probe_read_kernel(&entry.args[3], sizeof(u64),
				      (void *)(real_ctx + 24));
		bpf_probe_read_kernel(&entry.args[4], sizeof(u64),
				      (void *)(real_ctx + 32));
		bpf_probe_read_kernel(&entry.args[5], sizeof(u64),
				      (void *)(real_ctx + 40));
	}

	u64 fake_ip = (u64)func_idx | (1ULL << 63);

	struct func_meta *meta = bpf_map_lookup_elem(&func_meta_map, &fake_ip);
	if (meta) {
		entry.nr_fields = meta->nr_field_specs;
		for (int i = 0; i < MAX_FIELDS && i < meta->nr_field_specs; i++) {
			struct field_spec *spec = &meta->specs[i];
			u32 pidx = spec->param_idx;
			u32 fidx = spec->field_idx;

			if (pidx >= MAX_ARGS || fidx >= MAX_FIELDS || !spec->size)
				continue;

			u64 base = entry.args[pidx];
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
				entry.fields[fidx] = val;
		}
	}

	/* Read string arg if func_meta specifies one. */
	if (meta && meta->str_param_idx < MAX_ARGS) {
		u64 str_ptr = entry.args[meta->str_param_idx];
		if (str_ptr) {
			bpf_probe_read_kernel_str(entry.str_val,
						  sizeof(entry.str_val),
						  (void *)str_ptr);
			entry.has_str = 1;
			entry.str_param_idx = meta->str_param_idx;
		}
	}

	struct probe_key key = { .func_ip = fake_ip, .tid = tid };
	bpf_map_update_elem(&probe_data, &key, &entry, BPF_ANY);

	return 0;
}

/* Fentry handlers use unsigned long long * to access the raw fentry
 * context. set_attach_target() retargets each slot at runtime. */
SEC("fentry")
int ktstr_fentry_0(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_0); }

SEC("fentry")
int ktstr_fentry_1(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_1); }

SEC("fentry")
int ktstr_fentry_2(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_2); }

SEC("fentry")
int ktstr_fentry_3(unsigned long long *ctx) { return ktstr_fentry_common(ctx, ktstr_fentry_func_idx_3); }
