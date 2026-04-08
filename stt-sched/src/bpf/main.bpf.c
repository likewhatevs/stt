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

/* When non-zero, stt_enqueue inserts tasks onto a random online
 * CPU's local DSQ and stt_dispatch skips every other call.
 * Random placement drives up migrations; skipped dispatches
 * reduce throughput. Slows scheduling without stalling.
 * const volatile (.rodata) so the verifier prunes the path
 * when degrade=0. Set via rodata before load. */
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

/* When non-zero, stt_dispatch contains an infinite loop that the
 * BPF verifier rejects. The verifier traces the back-edge at the
 * same instruction addresses each iteration, producing repetitive
 * output that collapse_cycles() compresses. libbpf prints the
 * verifier log to stderr on load failure.
 * const volatile (.rodata) so the verifier prunes the path when
 * verify_loop=0. Set via rodata before load. */
const volatile int verify_loop = 0;

/* Runtime-mutable degrade flag. Set from userspace via .bss map write,
 * --degrade-after timer, or /tmp/stt_degrade sentinel. Same behavior
 * as const volatile degrade: random enqueue + skip 1/2 dispatches.
 * volatile (.bss) so the verifier always verifies the path. */
volatile int degrade_rt;

/* Skip 3 out of 4 dispatches (mask 0x3 = skip when any of low 2
 * bits set). Not configurable from CLI — fixed ratio. */
#define SLOW_SKIP_MASK 0x3

u32 degrade_cnt;
u32 slow_cnt;


void BPF_STRUCT_OPS(stt_enqueue, struct task_struct *p, u64 enq_flags)
{
	if (scattershot || degrade || degrade_rt) {
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
	if (degrade || degrade_rt) {
		/* Skip half of dispatches. Tasks stay in the shared DSQ
		 * longer, slowing throughput without fully stalling. */
		if (++degrade_cnt & 1)
			return;
	}
	if (verify_loop) {
		volatile u32 acc = 0;
		while (1)
			acc += bpf_ktime_get_ns();
	}
	if (fail_verify) {
		/* Null pointer dereference — verifier rejects this. */
		volatile int *p = (volatile int *)0;
		*p = 1;
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
