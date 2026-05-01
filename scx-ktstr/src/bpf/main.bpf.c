/* SPDX-License-Identifier: GPL-2.0 */
#include <scx/common.bpf.h>
#include <lib/sdt_task.h>

/* The BPF arena map itself is defined `__weak` by `<lib/arena_map.h>`
 * (pulled in by the fetched `lib/sdt_alloc.bpf.c`, which then defines
 * the map at file scope). Linking that .bpf.o into the final
 * `bpf.bpf.o` brings exactly one map definition into scope, so
 * main.bpf.c does not redeclare it here. The `__arena` qualifier this
 * file uses on `struct ktstr_arena_ctx __arena *` is provided by
 * `<bpf_arena_common.bpf.h>`, which `<lib/sdt_task.h>` includes. */

enum {
	SHARED_DSQ = 0,
};

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

/* Per-task arena context allocated in `ktstr_init_task` and freed in
 * `ktstr_exit_task`. Each instance lives in BPF arena memory; the
 * payload is reached from userspace via `scx_task_data(p)` (see
 * `lib/sdt_task.bpf.c::scx_task_data`).
 *
 * The struct is laid out so the failure-dump renderer can recognize a
 * captured page in the arena snapshot:
 *   - `magic` carries `KTSTR_ARENA_MAGIC`, an 8-byte recognizable
 *     constant the host-side BTF Datasec walker looks for to confirm
 *     a page belongs to this scheduler.
 *   - `counter` carries `KTSTR_TASK_COUNTER`, a small u32 set
 *     unconditionally on every alloc so the renderer sees a non-zero
 *     value in the captured page contents.
 *   - `_pad` keeps the struct 16-byte aligned to match
 *     `SDT_TASK_MIN_ELEM_PER_ALLOC`'s round-up assumption in
 *     `lib/sdt_alloc.bpf.c::scx_alloc_init` (data_size is rounded
 *     up to 8 there; 16-byte alignment removes any pool-fragmentation
 *     question for downstream readers).
 */
struct ktstr_arena_ctx {
	__u64 magic;
	__u32 counter;
	__u32 _pad;
};

/* Sentinel written into `ktstr_arena_ctx::magic` at every alloc.
 * Recognizable as an 8-byte little-endian pattern in arena page
 * captures so the host-side renderer can confirm a page belongs to
 * this scheduler. */
#define KTSTR_ARENA_MAGIC 0xDEADBEEFCAFEBABEULL

/* Recognizable counter value stamped into `ktstr_arena_ctx::counter`
 * on every alloc. */
#define KTSTR_TASK_COUNTER 42U

/* Cumulative count of `ktstr_init_task` allocations that succeeded.
 * Lives in .bss so the BTF Datasec walker surfaces it under the
 * .bss section alongside the stall/crash flags. Updated with
 * `__sync_fetch_and_add` because `ktstr_init_task` is called per
 * task across all CPUs concurrently. */
__u64 ktstr_alloc_count;

/* When non-zero, ktstr_dispatch stops moving tasks from the shared DSQ,
 * causing a deliberate stall that triggers the scx watchdog. */
volatile int stall;

/* When non-zero, ktstr_dispatch calls scx_bpf_error() to trigger an
 * immediate scheduler abort with a stack trace. Set from the host
 * via BPF map write to the .bss section. */
volatile int crash;

/* When non-zero, ktstr_enqueue inserts tasks onto a random online
 * CPU's local DSQ and ktstr_dispatch skips every other call.
 * Random placement drives up migrations; skipped dispatches
 * reduce throughput. Slows scheduling without stalling.
 * const volatile (.rodata) so the verifier prunes the path
 * when degrade=0. Set via rodata before load. */
const volatile int degrade = 0;

/* When non-zero, ktstr_dispatch performs an out-of-bounds map
 * access that the BPF verifier rejects. const volatile (.rodata)
 * so the verifier prunes the path when fail_verify=0. */
const volatile int fail_verify = 0;

/* When non-zero, ktstr_enqueue inserts tasks onto the local DSQ of a
 * random online CPU (via SCX_DSQ_LOCAL_ON | cpu) instead of the
 * shared DSQ. Cross-LLC placement causes migration storms.
 * Mutually exclusive with slow/degrade: scattershot bypasses
 * SHARED_DSQ, so dispatch-side skip logic has no effect. */
const volatile int scattershot = 0;

/* When non-zero, ktstr_dispatch skips approximately 3 out of every 4
 * dispatch calls. Creates throughput degradation without the bpf_loop
 * spin of --degrade. Mutually exclusive with scattershot (see above). */
const volatile int slow = 0;

/* When non-zero, ktstr_dispatch contains a #pragma unroll loop
 * followed by while(1). The compiler unrolls the loop into
 * sequential copies of the same instruction block. The trailing
 * while(1) forces verifier rejection so libbpf prints the full
 * trace to stderr. collapse_cycles() compresses the repetitive
 * unrolled output. const volatile (.rodata) so the verifier
 * prunes the path when verify_loop=0. */
const volatile int verify_loop = 0;

/* Runtime-mutable degrade flag. Set from userspace via .bss map write,
 * --degrade-after timer, or /tmp/ktstr_degrade sentinel. Same behavior
 * as const volatile degrade: random enqueue + skip 1/2 dispatches.
 * volatile (.bss) so the verifier always verifies the path. */
volatile int degrade_rt;

/* Skip 3 out of 4 dispatches (mask 0x3 = skip when any of low 2
 * bits set). Not configurable from CLI — fixed ratio. */
#define SLOW_SKIP_MASK 0x3

u32 degrade_cnt;
u32 slow_cnt;


void BPF_STRUCT_OPS(ktstr_enqueue, struct task_struct *p, u64 enq_flags)
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

void BPF_STRUCT_OPS(ktstr_dispatch, s32 cpu, struct task_struct *prev)
{
	if (crash)
		scx_bpf_error("ktstr: host-triggered crash");
	if (stall)
		return;
	if (degrade || degrade_rt) {
		/* Skip half of dispatches. Under degrade, ktstr_enqueue
		 * inserts to random LOCAL DSQs so this skip is effectively
		 * dead for those tasks, but slows any tasks that reached
		 * the shared DSQ via the normal path. */
		if (++degrade_cnt & 1)
			return;
	}
	if (verify_loop) {
		/* Unrolled loop produces 8 copies of the same instruction
		 * block at sequential addresses. The verifier traces all
		 * copies linearly (no back-edges to prune). After
		 * normalize_for_cycle_detection strips instruction addresses,
		 * all copies look identical, so detect_cycle finds the
		 * pattern. The trailing while(1) forces a verifier rejection
		 * so libbpf prints the full trace to stderr. */
		volatile u32 acc = 0;
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			u64 t = bpf_ktime_get_ns();
			acc += (u32)t;
			acc ^= (u32)(t >> 16);
			acc += (u32)(t >> 32);
			acc *= 7;
			acc += 1;
		}
		while (1)
			acc += 1;
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

s32 BPF_STRUCT_OPS_SLEEPABLE(ktstr_init)
{
	int ret;

	ret = scx_bpf_create_dsq(SHARED_DSQ, -1);
	if (ret)
		return ret;

	/* Bring the sdt_alloc per-task allocator online so subsequent
	 * `scx_task_alloc(p)` calls in `ktstr_init_task` succeed. The
	 * data_size argument is the per-task payload size that the
	 * allocator's pool will hand out; passing
	 * `sizeof(struct ktstr_arena_ctx)` matches the struct that
	 * `ktstr_init_task` writes into. `scx_task_init` rounds this up
	 * to 8 bytes inside `lib/sdt_alloc.bpf.c::pool_set_size`, so
	 * the actual pool element size is at least 16 bytes. */
	return scx_task_init(sizeof(struct ktstr_arena_ctx));
}

s32 BPF_STRUCT_OPS_SLEEPABLE(ktstr_init_task, struct task_struct *p,
			     struct scx_init_task_args *args)
{
	struct ktstr_arena_ctx __arena *taskc;

	/* `scx_task_alloc` takes the allocator spinlock and calls
	 * `bpf_arena_alloc_pages` from a sleepable context to back-fill
	 * the chunk pool when needed; init_task is sleepable so this
	 * is allowed. */
	taskc = scx_task_alloc(p);
	if (!taskc)
		return -ENOMEM;

	/* Stamp the recognizable values directly through the `__arena`
	 * pointer. LLVM lowers each store via the BPF arena
	 * address-space cast emitted from the `__arena` attribute, so
	 * the verifier sees a normal arena store and the JIT writes
	 * into the per-page kernel-VM mapping. */
	taskc->magic = KTSTR_ARENA_MAGIC;
	taskc->counter = KTSTR_TASK_COUNTER;
	taskc->_pad = 0;

	__sync_fetch_and_add(&ktstr_alloc_count, 1);
	return 0;
}

void BPF_STRUCT_OPS(ktstr_exit_task, struct task_struct *p,
		    struct scx_exit_task_args *args)
{
	scx_task_free(p);
}

void BPF_STRUCT_OPS(ktstr_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(ktstr_ops,
	       .enqueue		= (void *)ktstr_enqueue,
	       .dispatch	= (void *)ktstr_dispatch,
	       .init_task	= (void *)ktstr_init_task,
	       .exit_task	= (void *)ktstr_exit_task,
	       .init		= (void *)ktstr_init,
	       .exit		= (void *)ktstr_exit,
	       .timeout_ms	= 20000,
	       .name		= "ktstr");
