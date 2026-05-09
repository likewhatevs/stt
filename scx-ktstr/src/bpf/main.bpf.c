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
 * Struct layout exercises the host-side BPF cast analysis pipeline
 * (`src/monitor/cast_analysis.rs`) end-to-end. The analyzer parses
 * scx-ktstr's `.bpf.objs` ELF blob at VM-builder time and produces a
 * `CastMap` keyed on `(struct_btf_id, field_byte_offset) → (target_btf_id,
 * AddrSpace)`. The dump renderer then promotes flagged `u64` fields into
 * typed-pointer renders that chase through `read_arena` (AddrSpace::Arena)
 * or `read_kva` (AddrSpace::Kernel).
 *
 * Field roles (offsets are visible to the host analyzer via the BPF
 * object's BTF):
 *   - `magic` (offset 0): a u64 sentinel field. Loaded but never
 *     dereferenced as a pointer base. The analyzer must NOT promote
 *     this to a typed pointer — the BPF code reads it only to stamp
 *     a recognizable sentinel into the page, never as a pointer
 *     base. Verified by the cast E2E test's negative assertion.
 *   - `counter` (offset 8): a u32 counter. Sub-u64 fields are gated
 *     out of the cast intercept by the renderer's `int.size() != 8`
 *     check; this field exists so the size gate is exercised on a
 *     real captured arena page.
 *   - `task_kptr` (offset 16): u64 holding a kernel `task_struct *`.
 *     Written via plain STX from the `task_struct *p` parameter that
 *     the analyzer typed via the FuncProto seeding path; the STX-side
 *     detection in `cast_analysis::handle_stx` produces a CastMap entry
 *     `(ktstr_arena_ctx, 16) -> (task_struct, AddrSpace::Kernel)`. The
 *     renderer then chases via `MemReader::read_kva` and recurses into
 *     the task_struct bytes, so the captured field renders as
 *     `Ptr{value, deref: Some(Struct{type_name: "task_struct", ...})}`
 *     instead of a raw u64 counter. This is the cross-domain
 *     "arena-source -> kernel-target" chase the E2E test asserts on.
 *
 * The 24-byte size keeps the struct padded to a multiple of 8 (matches
 * `SDT_TASK_MIN_ELEM_PER_ALLOC`'s round-up in
 * `lib/sdt_alloc.bpf.c::scx_alloc_init`).
 */
struct ktstr_arena_ctx {
	__u64 magic;
	__u32 counter;
	__u32 _pad;
	__u64 task_kptr;
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

/* Test fixture: BSS-resident struct whose `arena_target` u64 carries
 * the user-side address of the most recent ktstr_arena_ctx allocation.
 * Renders inside the dump's `.bss` map and exercises the cast intercept
 * on the `arena_target` member.
 *
 * `ktstr_train_bss_to_arena` (below) loads `arena_target` and
 * dereferences the resulting u64 as a `struct ktstr_arena_ctx __arena *`,
 * reading enough fields (`magic` u64@0, `counter` u32@8, `task_kptr`
 * u64@16) that the host-side cast analyzer
 * (`src/monitor/cast_analysis.rs::analyze_casts`) can intersect the
 * observed access pattern against the program BTF and uniquely resolve
 * the target struct. The result is a `CastMap` entry
 * `(ktstr_bss_arena_holder, 0) -> (ktstr_arena_ctx, AddrSpace::Arena)`.
 *
 * Field roles:
 *   - `arena_target` (offset 0): u64 holding the arena VA. Loaded as
 *     a typed pointer base inside the helper so the analyzer's LDX
 *     path produces a `LoadedU64Field` register state. The observed
 *     access pattern on the loaded value (`magic` + `counter` +
 *     `task_kptr` reads, plus the addr_space_cast that the `__arena`
 *     attribute lowers to) trains the analyzer to flag this offset
 *     as a `(ktstr_arena_ctx, AddrSpace::Arena)` cast.
 *   - `bss_plain_counter` (offset 8): u64 counter. Loaded but never
 *     used as a pointer base. The analyzer must NOT promote it; the
 *     E2E test asserts it renders as plain Uint as a negative control
 *     (mirrors the `magic`/`counter` negative assertions on the
 *     arena-side fixture).
 *
 * Volatile keeps the writes that publish the arena VA from being
 * optimized away — even though the helper is `__noinline`, the global
 * write in `ktstr_init_task` happens against a non-tracked register
 * base, and a non-volatile field can be elided by the compiler when
 * it observes no in-program reader. .bss-resident (no initializer)
 * so the dump's BTF Datasec walker surfaces the struct under the same
 * `.bss` map as the existing scheduler globals.
 */
struct ktstr_bss_arena_holder {
	__u64 arena_target;
	__u64 bss_plain_counter;
};

volatile struct ktstr_bss_arena_holder ktstr_bss_arena_holder;

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

/* Cumulative counters surfaced via the scx_stats userspace protocol
 * (KtstrStats in scx-ktstr/src/stats.rs). Updated with
 * `__sync_fetch_and_add` because each ops callback runs concurrently
 * across all CPUs. .bss-resident so the userspace
 * `bss_data.nr_*` accessor reads them atomically with respect to the
 * BPF-side increments. */
volatile u64 nr_dispatched;
volatile u64 nr_enqueued;
volatile u64 nr_select_cpu;


s32 BPF_STRUCT_OPS(ktstr_select_cpu, struct task_struct *p,
		   s32 prev_cpu, u64 wake_flags)
{
	bool is_idle = false;
	s32 cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
	if (is_idle)
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, SCX_SLICE_DFL, 0);
	__sync_fetch_and_add(&nr_select_cpu, 1);
	return cpu;
}

void BPF_STRUCT_OPS(ktstr_enqueue, struct task_struct *p, u64 enq_flags)
{
	__sync_fetch_and_add(&nr_enqueued, 1);
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
	__sync_fetch_and_add(&nr_dispatched, 1);
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

/*
 * Helper that stamps the task_struct kernel address into a u64 field
 * of the per-task arena context. Defined as a static BPF-to-BPF helper
 * (not inlined) so the host-side cast analyzer
 * (`src/monitor/cast_analysis.rs::analyze_casts`) can seed both
 * parameters as typed kernel pointers at function entry via the
 * `.BTF.ext` `func_info` table:
 *   - R1 ← `Pointer{ktstr_arena_ctx}` (the FuncProto's first param peels
 *     `__arena *` -> `Ptr` -> `Struct(ktstr_arena_ctx)`).
 *   - R2 ← `Pointer{task_struct}` (the FuncProto's second param peels
 *     `*` -> `Ptr` -> `Struct(task_struct)`).
 * The body's only meaningful instruction is the STX of R2 into
 * `*(R1 + offsetof(task_kptr))`, which `handle_stx` records as the
 * cast finding `(ktstr_arena_ctx, 16) -> (task_struct, AddrSpace::Kernel)`.
 *
 * `static __noinline` keeps the helper as a real BPF-to-BPF call
 * with its own `.BTF.ext` `func_info` entry — the analyzer only
 * seeds parameters at function entries reported by `.BTF.ext`, so
 * an inlined version would lose the seeding context and the STX
 * would happen against an Unknown base register (`taskc` returned
 * by `scx_task_alloc` is not typed by the analyzer because the
 * analyzer does not currently model BPF-to-BPF return types from
 * non-kfunc helpers).
 */
static __noinline
int ktstr_stash_task_kptr(struct ktstr_arena_ctx __arena *taskc,
			  struct task_struct *p)
{
	if (!taskc)
		return -EINVAL;
	taskc->task_kptr = (u64)p;
	return 0;
}

/*
 * BSS→arena cast trainer. Loads `holder->arena_target` (a u64 in .bss)
 * and dereferences the resulting value as a `struct ktstr_arena_ctx
 * __arena *`. Static BPF-to-BPF helper so the host-side cast analyzer
 * (`src/monitor/cast_analysis.rs::analyze_casts`) seeds R1 with
 * `Pointer{ktstr_bss_arena_holder}` at function entry via the
 * `.BTF.ext` `func_info` table.
 *
 * What the analyzer sees on this body:
 *   - LDX r2, [r1 + 0]      → `LoadedU64Field { source: ktstr_bss_arena_holder, offset: 0 }`
 *   - addr_space_cast       → propagates state, marks (ktstr_bss_arena_holder, 0)
 *                              as `arena_confirmed`
 *   - LDX r3, [r2 + 0]  (8) → records access {0, 8} under
 *                              (ktstr_bss_arena_holder, 0)
 *   - LDX r4, [r2 + 8]  (4) → records access {8, 4}
 *   - LDX r5, [r2 + 16] (8) → records access {16, 8}
 *
 * After the forward walk, `Analyzer::finalize` intersects the recorded
 * `(offset, size)` pattern against every BTF struct in scx-ktstr's
 * program BTF. The pattern {(0,8), (8,4), (16,8)} matches exactly
 * one struct: `ktstr_arena_ctx` (other 24-byte structs in the BTF do
 * not have this field-width layout). The resulting `CastMap` entry is
 * `(ktstr_bss_arena_holder, 0) -> (ktstr_arena_ctx, AddrSpace::Arena)`.
 *
 * The body publishes its computed value back into a sibling BSS
 * counter so the BPF compiler keeps every load and the address-space
 * cast as observable side effects — without an externally-visible
 * effect, LLVM's interprocedural pass would have collapsed the
 * function into a constant return at the call site, dropping the
 * `func_info` entry the analyzer relies on. The
 * `__attribute__((used))` annotation pins the symbol even if a
 * future caller is dropped.
 *
 * The verifier needs the addr_space_cast (forced by the `__arena`
 * attribute) so the JIT inserts the kernel-VM translation; without
 * it, the dereference would attempt to read a user-virtual address
 * from a non-arena context and the verifier would reject the
 * program.
 *
 * `static __noinline __attribute__((used))` keeps the helper as a
 * real BPF-to-BPF call with its own `.BTF.ext` `func_info` entry —
 * the analyzer only seeds parameters at function entries that
 * `.BTF.ext` reports, so an inlined version would collapse the
 * parameter into the caller's frame and the analyzer would lose
 * the typed-parent context.
 */
static __noinline __attribute__((used))
int ktstr_train_bss_to_arena(struct ktstr_bss_arena_holder *holder)
{
	struct ktstr_arena_ctx __arena *p;
	__u64 raw;
	__u64 acc = 0;

	if (!holder)
		return -EINVAL;
	raw = holder->arena_target;
	if (!raw)
		return -ENOENT;
	/* The cast-through-(unsigned long) is the idiom the lavd
	 * scheduler uses (see scx/scheds/rust/scx_lavd's `taskc =
	 * (task_ctx __arena *)(unsigned long)cpuc->cached_taskc_raw`
	 * pattern). LLVM lowers the conversion to a
	 * `BPF_ADDR_SPACE_CAST` instruction, which the analyzer's
	 * arena-confirmed path records. */
	p = (struct ktstr_arena_ctx __arena *)(unsigned long)raw;
	if (!p)
		return -EINVAL;
	/* Read three discriminating fields so the analyzer's shape
	 * intersection ({(0,8), (8,4), (16,8)}) uniquely resolves
	 * `ktstr_arena_ctx` against the program BTF. Accumulate every
	 * load into `acc` so LLVM cannot elide reads it considers
	 * unused — a comparison-only pattern in an earlier draft was
	 * hoisted out by the optimizer because the compares decide
	 * the return value, not any externally-visible effect. */
	acc ^= p->magic;
	acc ^= (__u64)p->counter;
	acc ^= p->task_kptr;
	/* Publish through the sibling BSS counter so LLVM cannot
	 * collapse the helper's body into a tail-call ABI placeholder.
	 * `holder` is typed `Pointer{ktstr_bss_arena_holder}` per
	 * FuncProto seeding, so this STX has both a typed base
	 * register and a non-typed (integer) source — `handle_stx`
	 * does NOT record a kptr finding here (the source isn't a
	 * `Pointer{T}`), so the store is purely a code-keep-alive
	 * effect. */
	holder->bss_plain_counter += acc & 1;
	return 0;
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

	/* Stash the task_struct kernel pointer in the task_kptr u64 field.
	 * The store happens inside the helper so the host-side cast analyzer
	 * sees a typed STX through typed parameter registers, producing an
	 * AddrSpace::Kernel cast finding the renderer chases at dump time.
	 * Failing the helper does not abort init_task — the field stays
	 * zero, which the renderer surfaces as a null pointer. */
	(void)ktstr_stash_task_kptr(taskc, p);

	/* Publish the arena VA into the BSS-resident holder so the
	 * trainer below has a concrete pointer to dereference, AND so
	 * the dump-time renderer reads a valid arena VA from
	 * `ktstr_bss_arena_holder.arena_target` and chases through the
	 * cast intercept. Cast through (unsigned long) so the verifier
	 * lowers to a kernel→arena address-space conversion that
	 * extracts the user-side VA without keeping the `__arena`
	 * tag on the result; storing into a plain `volatile u64` field
	 * cannot accept the qualified pointer otherwise.
	 *
	 * The store goes through `BPF_LD_IMM64 r6, .bss_map_value_fd;
	 * STX [r6 + 0] = r_taskc_as_u64`. The analyzer's r6 is Unknown
	 * because BPF_LD_IMM64 of a map value clears RegState; the STX
	 * therefore records nothing on the .bss side. The cast detection
	 * runs through the LDX path inside `ktstr_train_bss_to_arena`,
	 * not this STX. */
	ktstr_bss_arena_holder.arena_target = (__u64)(unsigned long)taskc;
	__sync_fetch_and_add(&ktstr_bss_arena_holder.bss_plain_counter, 1);

	/* Run the BSS→arena trainer with the freshly-stamped arena VA.
	 * The helper's body teaches the analyzer to recognize the
	 * `arena_target` u64 as a `ktstr_arena_ctx __arena *` cast.
	 * Failing the helper does not abort init_task — the trainer is
	 * a static analysis affordance, not a load-bearing scheduler
	 * operation. */
	(void)ktstr_train_bss_to_arena((struct ktstr_bss_arena_holder *)&ktstr_bss_arena_holder);

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

/*
 * ops.dump callback. Invoked by `scx_dump_state()` (kernel/sched/ext.c)
 * inside the SCX_KF_UNLOCKED region. Output goes to the per-`scx_sched`
 * dump buffer that is rendered into kernel printk on error-class exit
 * and copied into `scx_exit_info::dump` for userspace consumers.
 *
 * Surfaces every runtime-mutable knob the framework injects so an
 * operator scanning a dump sees exactly what state the test fixture
 * was driving at the freeze instant: stall/crash/degrade_rt are
 * ".bss" volatiles flipped from the host; degrade/slow/scattershot/
 * verify_loop/fail_verify are const_volatile rodata pinned at load
 * time and surfacing them confirms which test-mode the scheduler was
 * built for. The cumulative counters distinguish a "configured but
 * never fired" path (e.g. degrade_rt set, but no enqueue/dispatch
 * reached the slow path because the test exited first) from one that
 * was actively shaping behaviour right up to the failure.
 */
void BPF_STRUCT_OPS(ktstr_dump, struct scx_dump_ctx *dctx)
{
	scx_bpf_dump("ktstr scheduler state:\n");
	scx_bpf_dump("  stall=%d crash=%d degrade_rt=%d\n",
		     stall, crash, degrade_rt);
	scx_bpf_dump("  rodata: degrade=%d slow=%d scattershot=%d verify_loop=%d fail_verify=%d\n",
		     degrade, slow, scattershot, verify_loop, fail_verify);
	scx_bpf_dump("  ktstr_alloc_count=%llu degrade_cnt=%u slow_cnt=%u\n",
		     ktstr_alloc_count, degrade_cnt, slow_cnt);
}

/*
 * ops.dump_cpu callback. Invoked by `scx_dump_state()` for every
 * possible CPU between `ops.dump` and the per-task pass. ktstr-fixture
 * carries no per-CPU scheduler state (no cpuctx struct, no per-CPU
 * scratch maps), so on a non-idle CPU we emit a one-line marker
 * confirming the callback fired, and on idle CPUs we emit nothing.
 *
 * Skipping idle CPUs piggybacks on `scx_dump_state`'s
 * "if (idle && used == seq_buf_used(&ns)) goto next;" gate
 * (kernel/sched/ext.c:6127-6283): when ops.dump_cpu writes zero
 * bytes for an idle CPU, the kernel suppresses the entire per-CPU
 * section for that CPU. Emitting a marker on every idle CPU instead
 * defeats that gate and floods the failure dump with N copies of the
 * same "no per-cpu state" line on otherwise-idle systems. The line
 * keeps the same prefix scheme as `ops.dump`/`ops.dump_task` so the
 * three layers render uniformly when surfaced.
 */
void BPF_STRUCT_OPS(ktstr_dump_cpu, struct scx_dump_ctx *dctx,
		    s32 cpu, bool idle)
{
	if (idle)
		return;
	scx_bpf_dump("ktstr cpu %d: no per-cpu state\n", cpu);
}

/*
 * ops.dump_task callback. Invoked by `scx_dump_state()` for every
 * runnable task on every CPU after `ops.dump_cpu`.
 *
 * `scx_task_data(p)` returns the per-task arena context allocated in
 * `ktstr_init_task`. NULL on tasks the allocator never touched (e.g.
 * pre-existing kthreads enqueued before `init_task` ran). Surface the
 * magic and counter so an operator can confirm the per-task arena
 * write actually landed for tasks that were running at freeze time.
 */
void BPF_STRUCT_OPS(ktstr_dump_task, struct scx_dump_ctx *dctx,
		    struct task_struct *p)
{
	struct ktstr_arena_ctx __arena *taskc;

	taskc = scx_task_data(p);
	if (!taskc) {
		scx_bpf_dump("  ktstr task: <no arena ctx>\n");
		return;
	}
	scx_bpf_dump("  ktstr task: magic=0x%llx counter=%u task_kptr=0x%llx\n",
		     taskc->magic, taskc->counter, taskc->task_kptr);
}

SCX_OPS_DEFINE(ktstr_ops,
	       .select_cpu	= (void *)ktstr_select_cpu,
	       .enqueue		= (void *)ktstr_enqueue,
	       .dispatch	= (void *)ktstr_dispatch,
	       .init_task	= (void *)ktstr_init_task,
	       .exit_task	= (void *)ktstr_exit_task,
	       .dump		= (void *)ktstr_dump,
	       .dump_cpu	= (void *)ktstr_dump_cpu,
	       .dump_task	= (void *)ktstr_dump_task,
	       .init		= (void *)ktstr_init,
	       .exit		= (void *)ktstr_exit,
	       .timeout_ms	= 20000,
	       .name		= "ktstr");
