/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __KTSTR_INTF_H
#define __KTSTR_INTF_H

#define MAX_ARGS 6
#define MAX_FIELDS 16
#define MAX_STACK_DEPTH 32
#define MAX_FUNCS 64
#define MAX_MISS_LOG 16
#define FENTRY_BATCH 4
#define MAX_STR_LEN 64

/* sched_ext exit-kind values mirrored from kernel/sched/ext_internal.h
 * enum scx_exit_kind. The error-class kinds (>= SCX_EXIT_ERROR) are the
 * values the probe filters on; mirrored here so probe.bpf.c can use
 * named constants instead of magic numbers and so userspace can match
 * the same wire values when consuming the .bss latch and ringbuf
 * events. Values must stay in sync with the kernel enum.
 */
#define SCX_EXIT_ERROR       1024
#define SCX_EXIT_ERROR_BPF   1025
#define SCX_EXIT_ERROR_STALL 1026

/* Per-probe-hit captured data, stored in hash map keyed by (func_ip, task_ptr).
 * Entry fields are written by fentry/kprobe at function entry.
 * Exit fields are written in-place by fexit at function exit
 * via bpf_map_lookup_elem on the same key. */
struct probe_entry {
	unsigned long long ts;
	unsigned long long args[MAX_ARGS];
	unsigned long long fields[MAX_FIELDS];
	unsigned int nr_fields;
	char str_val[MAX_STR_LEN];   /* string arg (char *), NUL-terminated */
	unsigned char has_str;        /* nonzero if str_val is populated */
	unsigned char str_param_idx;  /* which arg was the string source */
	/* Exit-side capture (written by fexit). */
	unsigned long long exit_ts;
	unsigned long long exit_fields[MAX_FIELDS];
	unsigned int nr_exit_fields;
	unsigned char has_exit;       /* nonzero if fexit fired */
};

/* Field dereference spec: for a pointer param, read at base + offset.
 * For chained pointer dereferences (e.g. ->cpus_ptr->bits[0]):
 *   ptr_offset != 0: first read a pointer at base + ptr_offset,
 *                     then read size bytes at pointer + offset.
 *   ptr_offset == 0: single-level read at base + offset.
 */
struct field_spec {
	unsigned int param_idx;      /* which arg (0..5) is the base pointer */
	unsigned int offset;         /* byte offset from base (or from deref'd ptr) */
	unsigned int size;           /* bytes to read (1/2/4/8) */
	unsigned int field_idx;      /* index into probe_entry.fields[] */
	unsigned int ptr_offset;     /* if nonzero: byte offset to intermediate ptr */
};

/* Per-function metadata written by userspace before attachment. */
struct func_meta {
	unsigned int func_idx;       /* index in the userspace function list */
	unsigned int nr_field_specs; /* how many field_spec entries for this func */
	struct field_spec specs[MAX_FIELDS];
	unsigned char str_param_idx; /* param index for char * string (0xff = none) */
};

/* Event type for ring buffer. */
enum event_type {
	EVENT_TRIGGER   = 2,
};

/* Ring buffer event sent from BPF to userspace on trigger.
 *
 * For EVENT_TRIGGER: args[0] = bpf_get_current_task() ptr,
 * args[1] = exit kind (scx_exit_kind enum value).
 */
struct probe_event {
	unsigned int type;
	unsigned int tid;
	unsigned int func_idx;
	unsigned long long ts;
	unsigned long long args[MAX_ARGS];
	unsigned long long fields[MAX_FIELDS];
	unsigned int nr_fields;
	unsigned long long kstack[MAX_STACK_DEPTH];
	unsigned int kstack_sz;
};

#endif /* __KTSTR_INTF_H */
