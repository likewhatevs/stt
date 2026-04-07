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

/* When non-zero, stt_dispatch skips 63 out of 64 calls and
 * burns ~5ms via bpf_loop on the 64th before dispatching.
 * Uses SCX_SLICE_DFL so the scheduler process stays alive
 * (1us timeslice causes ~1M dispatch/s/CPU which starves
 * userspace). The combination of skipped dispatches plus
 * the 5ms delay on every 64th call produces scheduling gaps
 * above the test's max_gap_ms(50) threshold.
 * const volatile (.rodata) so the verifier prunes the
 * bpf_loop path when degrade=0. Set via rodata before load. */
const volatile int degrade = 0;

/* When non-zero, stt_dispatch performs an out-of-bounds map
 * access that the BPF verifier rejects. const volatile (.rodata)
 * so the verifier prunes the path when fail_verify=0. */
const volatile int fail_verify = 0;

/* When non-zero, stt_enqueue inserts tasks onto the local DSQ of a
 * random online CPU (via SCX_DSQ_LOCAL_ON | cpu) instead of the
 * shared DSQ. Cross-LLC placement causes migration storms.
 * Mutually exclusive with slow/degrade: scattershot bypasses
 * SHARED_DSQ, so dispatch-side skip logic has no effect. */
const volatile int scattershot = 0;

/* When non-zero, stt_dispatch skips approximately 3 out of every 4
 * dispatch calls. Creates throughput degradation without the bpf_loop
 * spin of --degrade. Mutually exclusive with scattershot (see above). */
const volatile int slow = 0;

/* Skip 3 out of 4 dispatches (mask 0x3 = skip when any of low 2
 * bits set). Not configurable from CLI — fixed ratio. */
#define SLOW_SKIP_MASK 0x3

u32 degrade_cnt;
u32 slow_cnt;

static int degrade_spin_cb(u32 idx, void *ctx)
{
	u64 *deadline = ctx;
	return bpf_ktime_get_ns() >= *deadline ? 1 : 0;
}

void BPF_STRUCT_OPS(stt_enqueue, struct task_struct *p, u64 enq_flags)
{
	if (scattershot) {
		const struct cpumask *online;
		u32 nr = scx_bpf_nr_cpu_ids();
		u32 cpu;

		online = scx_bpf_get_online_cpumask();
		cpu = bpf_get_prandom_u32() % (nr ?: 1);
		if (!bpf_cpumask_test_cpu(cpu, online))
			cpu = bpf_cpumask_first(online);
		scx_bpf_put_cpumask(online);

		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu,
				    SCX_SLICE_DFL, enq_flags);
		return;
	}
	scx_bpf_dsq_insert(p, SHARED_DSQ, SCX_SLICE_DFL, enq_flags);
}

void BPF_STRUCT_OPS(stt_dispatch, s32 cpu, struct task_struct *prev)
{
	if (crash)
		scx_bpf_error("stt: host-triggered crash");
	if (stall)
		return;
	if (fail_verify) {
		/* Unbounded pointer arithmetic — verifier rejects this. */
		volatile int *p = (volatile int *)0;
		*p = 1;
	}
	if (degrade) {
		if (++degrade_cnt & 0x3F)
			return;
		{
			u64 deadline = bpf_ktime_get_ns() + 5000000ULL;
			bpf_loop(1 << 20, degrade_spin_cb, &deadline, 0);
		}
	}
	if (slow) {
		if (++slow_cnt & SLOW_SKIP_MASK)
			return;
	}
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
