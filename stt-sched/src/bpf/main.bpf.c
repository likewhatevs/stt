/* SPDX-License-Identifier: GPL-2.0 */
#include <scx/common.bpf.h>

enum {
	SHARED_DSQ = 0,
};

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

/* When non-zero, stt_dispatch stops moving tasks from the shared DSQ,
 * causing a deliberate stall that triggers the scx watchdog. */
volatile int stall;

/* When non-zero, stt_dispatch calls scx_bpf_error() to trigger an
 * immediate scheduler abort with a stack trace. Set from the host
 * via BPF map write to the .bss section. */
volatile int crash;

void BPF_STRUCT_OPS(stt_enqueue, struct task_struct *p, u64 enq_flags)
{
	scx_bpf_dsq_insert(p, SHARED_DSQ, SCX_SLICE_DFL, enq_flags);
}

void BPF_STRUCT_OPS(stt_dispatch, s32 cpu, struct task_struct *prev)
{
	if (crash)
		scx_bpf_error("stt: host-triggered crash");
	if (stall)
		return;
	scx_bpf_dsq_move_to_local(SHARED_DSQ);
}

s32 BPF_STRUCT_OPS_SLEEPABLE(stt_init)
{
	return scx_bpf_create_dsq(SHARED_DSQ, -1);
}

void BPF_STRUCT_OPS(stt_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(stt_ops,
	       .enqueue		= (void *)stt_enqueue,
	       .dispatch	= (void *)stt_dispatch,
	       .init		= (void *)stt_init,
	       .exit		= (void *)stt_exit,
	       .timeout_ms	= 5000,
	       .name		= "stt_sched");
