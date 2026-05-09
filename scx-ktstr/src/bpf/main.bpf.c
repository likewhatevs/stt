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
 *   - `stashed_arena_ptr` (offset 24): u64 holding the arena VA of a
 *     `struct ktstr_cross_btf_target` allocated via `scx_static_alloc`
 *     and cached in `ktstr_cross_btf_map`. Written by the chase
 *     helper `ktstr_cross_btf_chase` via the STX-flow alias-tracking
 *     path: the publish helper STXs the allocator-return arena VA
 *     into the hash map value's `cached_ptr`, after which an LDX
 *     through the same hash-value typed base inherits
 *     `RegState::ArenaU64FromAlloc` and the subsequent STX into this
 *     field records `(ktstr_arena_ctx, 24) -> AddrSpace::Arena` with
 *     `target_type_id == 0`. The follow-up E2E layer wires payload-
 *     type recovery for `scx_static_alloc` payloads (the existing
 *     `MemReader::resolve_arena_type` bridge in
 *     `src/monitor/dump/render_map.rs` only indexes sdt_alloc
 *     slots; scx_static_alloc has no per-slot header, so the bridge
 *     extension that consumes this fixture is the matching test
 *     scenario's responsibility). Mirrors the cgx_raw chain in
 *     lavd's cgroup_bw library — generic-named so the fixture
 *     doesn't bind to that scheduler.
 *
 * The 32-byte size keeps the struct padded to a multiple of 8 (matches
 * `SDT_TASK_MIN_ELEM_PER_ALLOC`'s round-up in
 * `lib/sdt_alloc.bpf.c::scx_alloc_init`).
 */
struct ktstr_arena_ctx {
	__u64 magic;
	__u32 counter;
	__u32 _pad;
	__u64 task_kptr;
	__u64 stashed_arena_ptr;
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

/* Target struct allocated via `scx_static_alloc` and cached through
 * `ktstr_cross_btf_map`. Mirrors the lavd `scx_cgroup_ctx` shape: a
 * small arena-resident struct whose pointer is stashed in a u64 slot
 * via the allocator-return -> STX-flow path.
 *
 * Field roles:
 *   - `magic` (offset 0): u64 sentinel stamped at every alloc so the
 *     E2E test can verify the chase landed on a real allocation
 *     (not a stale arena page or a same-shape decoy).
 *   - `marker` (offset 8): u64 second sentinel that disambiguates
 *     this struct from any other 16-byte u64-pair struct in the
 *     program BTF. Two distinct constants make a {(0,8), (8,8)}
 *     read pattern uniquely fingerprint this struct shape.
 */
struct ktstr_cross_btf_target {
	__u64 magic;
	__u64 marker;
};

/* Sentinels written into `ktstr_cross_btf_target` at every alloc.
 * The two values are distinct so a successful chase displaying both
 * proves the renderer descended into the chased struct correctly,
 * not just that bytes happened to match a single sentinel. */
#define KTSTR_CROSS_BTF_TARGET_MAGIC 0xC0FFEEFEEDFACE01ULL
#define KTSTR_CROSS_BTF_TARGET_MARKER 0x1234567890ABCDEFULL

/* Hash map value carrying the cached arena VA of the most recent
 * `ktstr_cross_btf_target` allocation. The publish helper STXs the
 * `scx_static_alloc` return into `cached_ptr`; the chase helper LDXs
 * `cached_ptr` through a typed `Pointer{ktstr_cross_btf_value}` base
 * so the cast analyzer's STX-flow alias-tracking path inherits the
 * `RegState::ArenaU64FromAlloc` tag onto the loaded register.
 *
 * Single-field struct keeps the access pattern simple — the only
 * load through a typed base is at offset 0, so the analyzer's
 * `(ktstr_cross_btf_value, 0) -> AddrSpace::Arena` finding records
 * cleanly. */
struct ktstr_cross_btf_value {
	__u64 cached_ptr;
};

/* Hash map keyed by a fixed u32 sentinel. The fixture only uses key
 * `KTSTR_CROSS_BTF_KEY` so a single map entry round-trips the arena
 * VA between the publish and chase helpers. `BPF_F_NO_PREALLOC`
 * matches the lavd `cbw_cgrp_map` shape, but the choice is
 * incidental here — the cast analyzer only cares about the BTF
 * value type and the hash-map kind, not the allocation policy.
 *
 * `__type(value, struct ktstr_cross_btf_value)` makes libbpf publish
 * the struct as the map's value BTF; the kernel verifier types
 * `bpf_map_lookup_elem(&ktstr_cross_btf_map, &key)` returns as
 * `Pointer{ktstr_cross_btf_value}` so the body's typed access
 * pattern survives verification. The host-side cast analyzer
 * (`src/monitor/cast_analysis.rs`) does not currently parse the
 * map's value BTF off `bpf_map_lookup_elem` returns; the chase
 * detection on this fixture relies on a follow-up enhancement that
 * mirrors `FuncEntry`'s parameter-typing path for the helper
 * return. */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, __u32);
	__type(value, struct ktstr_cross_btf_value);
	__uint(max_entries, 1);
} ktstr_cross_btf_map SEC(".maps");

/* Single sentinel key for `ktstr_cross_btf_map`. The fixture only
 * uses one entry — distinct keys would diverge the publish and
 * chase helpers' lookups, producing a useless empty entry. */
#define KTSTR_CROSS_BTF_KEY 0u

/* Page count for `scx_static_init` in `ktstr_init`. One page (4 KiB)
 * holds many `ktstr_cross_btf_target` slots (each 16 bytes); the
 * fixture only ever allocates one because the hash map's
 * `BPF_F_NO_PREALLOC` lookup-or-insert in
 * `ktstr_cross_btf_publish` short-circuits subsequent calls. The
 * allocator's bump pointer never advances past the first slot, so a
 * single page is comfortably sufficient. */
#define KTSTR_CROSS_BTF_STATIC_PAGES 1

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

	/* Bring the static allocator online so `scx_static_alloc` calls
	 * in `ktstr_cross_btf_publish` succeed. `scx_static_init` calls
	 * `bpf_arena_alloc_pages` to back-fill the static pool from the
	 * BPF arena (sleepable). One page is sufficient for the fixture
	 * — see `KTSTR_CROSS_BTF_STATIC_PAGES`. Failure here is
	 * recoverable downstream: `scx_static_alloc` returns NULL, the
	 * publish helper's NULL check bails, and the chase fixture
	 * surfaces an empty hash entry rather than corrupted state. */
	ret = scx_static_init(KTSTR_CROSS_BTF_STATIC_PAGES);
	if (ret)
		return ret;

	/* Bring the sdt_alloc per-task allocator online so subsequent
	 * `scx_task_alloc(p)` calls in `ktstr_init_task` succeed. The
	 * data_size argument is the per-task payload size that the
	 * allocator's pool will hand out; passing
	 * `sizeof(struct ktstr_arena_ctx)` matches the struct that
	 * `ktstr_init_task` writes into. `scx_task_init` rounds this up
	 * to 8 bytes inside `lib/sdt_alloc.bpf.c::pool_set_size`. */
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
 * one struct: `ktstr_arena_ctx` (no other struct in the BTF carries a
 * u64 at offset 0 plus u32 at offset 8 plus u64 at offset 16, so the
 * shape is uniquely fingerprinted regardless of fields beyond offset
 * 24). The resulting `CastMap` entry is
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

/*
 * Cross-BTF-class arena chase trainer (publish side).
 *
 * Allocates a `struct ktstr_cross_btf_target` via `scx_static_alloc`
 * and stashes the returned arena VA into the `cached_ptr` field of a
 * hash map value. Mirrors the lavd `cbw_alloc_cgx` -> hash-map-update
 * idiom: the allocator returns a u64 that BTF declares as a generic
 * integer, but the host-side cast analyzer's allocator-return seed
 * tags R0 as `RegState::ArenaU64FromAlloc` after the
 * `BPF_PSEUDO_CALL` to `scx_static_alloc_internal`. The subsequent
 * STX into the hash value's `cached_ptr` field through a typed
 * `Pointer{ktstr_cross_btf_value}` base records
 * `(ktstr_cross_btf_value, 0) -> AddrSpace::Arena` in the cast
 * map's STX-flow findings.
 *
 * The lookup-or-insert dance primes the hash entry on first call and
 * short-circuits subsequent calls when `cached_ptr` is already
 * populated, so the static allocator pool advances by exactly one
 * `sizeof(ktstr_cross_btf_target)` slot across the lifetime of the
 * scheduler — preventing the bump pointer from racing past the
 * `KTSTR_CROSS_BTF_STATIC_PAGES`-page reservation under concurrent
 * `init_task` callbacks.
 *
 * `static __noinline __attribute__((used))` follows the same shape as
 * `ktstr_train_bss_to_arena` and `ktstr_stash_task_kptr`: keeps the
 * helper as a real BPF-to-BPF call with its own `.BTF.ext`
 * `func_info` entry so the cast analyzer's function-entry seeding
 * applies to the helper's first instruction.
 *
 * Returns 0 on a successful publish OR when the entry was already
 * populated by a prior call; negative on first-time failures
 * (allocator empty, map insert failed, post-insert lookup raced).
 * The caller treats failures as best-effort — the chase fixture is
 * a static-analysis affordance, not a load-bearing scheduler op.
 */
static __noinline __attribute__((used))
int ktstr_cross_btf_publish(void)
{
	struct ktstr_cross_btf_target __arena *target;
	struct ktstr_cross_btf_value *entry;
	struct ktstr_cross_btf_value initial = {};
	__u32 key = KTSTR_CROSS_BTF_KEY;

	/* Fast path: the entry already carries a non-zero `cached_ptr`
	 * from a prior call, so re-allocating would just leak static
	 * pool space. The publish work was completed; the chase helper
	 * has everything it needs. */
	entry = bpf_map_lookup_elem(&ktstr_cross_btf_map, &key);
	if (entry && entry->cached_ptr)
		return 0;

	/* Static-pool allocate. The bump allocator backs onto BPF arena
	 * pages reserved by `scx_static_init` in `ktstr_init`, so the
	 * returned arena VA is in the same arena window as the per-task
	 * sdt_alloc allocations — both routes funnel into the same
	 * `MemReader::is_arena_addr` window at chase time. */
	target = scx_static_alloc(sizeof(*target), 8);
	if (!target)
		return -ENOMEM;
	target->magic = KTSTR_CROSS_BTF_TARGET_MAGIC;
	target->marker = KTSTR_CROSS_BTF_TARGET_MARKER;

	/* Insert an empty entry if missing, then update its cached_ptr.
	 * The two-step pattern matches lavd's `cbw_alloc_cgx`-then-
	 * `bpf_map_update_elem` flow and keeps the STX of the arena VA
	 * separate from the create — STXing into a freshly-zeroed
	 * lookup result lets the analyzer see the typed `Pointer{ktstr_cross_btf_value}`
	 * base register at the store site (the lookup return is what
	 * the cast analyzer's helper-return type tracking relies on). */
	if (!entry) {
		bpf_map_update_elem(&ktstr_cross_btf_map, &key, &initial,
				    BPF_NOEXIST);
		entry = bpf_map_lookup_elem(&ktstr_cross_btf_map, &key);
		if (!entry)
			return -ENOENT;
	}

	/* The cast through (unsigned long) is the same arena-tag-stripping
	 * idiom `ktstr_init_task` uses on the BSS holder side: the verifier
	 * lowers it to a `BPF_ADDR_SPACE_CAST` (kernel→arena), keeping the
	 * arena VA value but dropping the `__arena` qualifier so the u64
	 * field accepts the assignment. Once the cast analyzer types
	 * `bpf_map_lookup_elem` returns from the map's BTF value type,
	 * its STX-flow path records
	 * `(ktstr_cross_btf_value, 0) -> AddrSpace::Arena` here:
	 *   - R0 carries `RegState::ArenaU64FromAlloc` from the
	 *     `scx_static_alloc_internal` subprog-return seed (already
	 *     wired in `cast_analysis.rs::SubprogReturn`).
	 *   - The store target is `Pointer{ktstr_cross_btf_value} + 0`,
	 *     typed by the prospective helper-return tracking pass over
	 *     `bpf_map_lookup_elem` (mirrors `FuncEntry`'s parameter-
	 *     typing path, follow-up to this fixture).
	 *   - `handle_stx`'s `StxValueKind::Arena` arm records the
	 *     finding without shape inference. */
	entry->cached_ptr = (__u64)(unsigned long)target;
	return 0;
}

/*
 * Cross-BTF-class arena chase trainer (chase side).
 *
 * Receives the per-task arena context as a `__u64 task_ctx_raw`
 * parameter — the lavd cgroup_bw library uses the same shape (e.g.
 * `cbw_cgroup_bw_throttled(u64 cgrp_id, u64 taskc_raw)`) so a
 * cast-analyzer enhancement that types u64-cast-to-pointer
 * parameters at the call boundary can apply uniformly to both
 * scx-ktstr and lavd. Looks up the `ktstr_cross_btf_map` entry
 * keyed by `KTSTR_CROSS_BTF_KEY`, reads the cached arena VA out of
 * `entry->cached_ptr`, and stashes that u64 into the per-task
 * `taskc->stashed_arena_ptr` slot.
 *
 * The expected analyzer trace through the body:
 *   1. The `task_ctx_raw` u64 parameter is cast to
 *      `struct ktstr_arena_ctx __arena *taskc`. Once the analyzer's
 *      cross-call FuncProto inference types this parameter as a
 *      `Pointer{ktstr_arena_ctx}`, the cast lowers to a
 *      `BPF_ADDR_SPACE_CAST` and downstream STX through `taskc`
 *      records into the `(ktstr_arena_ctx, …)` keyspace.
 *   2. `bpf_map_lookup_elem(&ktstr_cross_btf_map, &key)` returns
 *      `Pointer{ktstr_cross_btf_value}` once the analyzer's helper-
 *      return type tracking consumes the map's BTF value type.
 *   3. `LDX raw, [entry + 0]` (offset 0 = `cached_ptr`) inherits
 *      `RegState::ArenaU64FromAlloc` via the alias-set tracking
 *      path: the publish helper's STX previously recorded
 *      `(ktstr_cross_btf_value, 0) -> Arena` in
 *      `arena_stx_findings`, so `handle_ldx`'s
 *      `arena_stx_findings.contains_key` check sets the loaded
 *      register to `ArenaU64FromAlloc` instead of the generic
 *      `LoadedU64Field`.
 *   4. `STX [taskc + offsetof(stashed_arena_ptr)] = raw` records
 *      `(ktstr_arena_ctx, 24) -> AddrSpace::Arena` with
 *      `target_type_id == 0`. The fixture's matching test layer
 *      extends `MemReader::resolve_arena_type` (currently
 *      sdt_alloc-only) to recover the `ktstr_cross_btf_target`
 *      payload type from `scx_static_alloc` slot metadata at chase
 *      time.
 *
 * `static __noinline __attribute__((used))` keeps the helper as a
 * real BPF-to-BPF call so the analyzer's function-entry seeding
 * applies. The `used` attribute pins the symbol against
 * interprocedural-elimination passes that might collapse the body
 * into the caller when `task_ctx_raw` is the only parameter.
 *
 * Returns 0 on success, negative on every short-circuit (NULL
 * input, missing map entry, zero `cached_ptr`). Failure does not
 * abort the caller — the chase fixture is a static-analysis
 * affordance, not a load-bearing scheduler op.
 */
static __noinline __attribute__((used))
int ktstr_cross_btf_chase(__u64 task_ctx_raw)
{
	struct ktstr_arena_ctx __arena *taskc;
	struct ktstr_cross_btf_value *entry;
	__u32 key = KTSTR_CROSS_BTF_KEY;
	__u64 raw;

	taskc = (struct ktstr_arena_ctx __arena *)(unsigned long)task_ctx_raw;
	if (!taskc)
		return -EINVAL;

	entry = bpf_map_lookup_elem(&ktstr_cross_btf_map, &key);
	if (!entry)
		return -ENOENT;

	raw = entry->cached_ptr;
	if (!raw)
		return -ENOENT;

	taskc->stashed_arena_ptr = raw;
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
	/* Zero `stashed_arena_ptr` ahead of the chase helper. The
	 * sdt_alloc layer hands out memory whose contents are
	 * undefined (no implicit zero-fill — see
	 * `lib/sdt_alloc.bpf.c::scx_alloc_internal`), so a chase that
	 * fails before reaching the STX site (publish helper raced and
	 * left an empty entry, hash lookup miss, etc.) would otherwise
	 * surface stale bytes from a prior allocation as a phantom
	 * arena pointer. */
	taskc->stashed_arena_ptr = 0;

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

	/* Cross-BTF-class chase fixture (publish + chase). Run the
	 * publish first so the hash entry's `cached_ptr` is populated
	 * before the chase helper looks it up. Both helpers
	 * short-circuit cleanly on missing state — the chase fixture is
	 * a static-analysis affordance, not a load-bearing scheduler
	 * operation. The chase helper's u64 parameter shape mirrors
	 * lavd's cgroup_bw library so a future cast-analyzer
	 * cross-call FuncProto inference enhancement applies uniformly
	 * across both fixtures. */
	(void)ktstr_cross_btf_publish();
	(void)ktstr_cross_btf_chase((__u64)(unsigned long)taskc);

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
 * magic, counter, kernel kptr, and the cross-BTF-class chase target
 * so an operator can confirm the per-task arena writes (including
 * the publish/chase sequence) actually landed for tasks that were
 * running at freeze time.
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
	scx_bpf_dump("  ktstr task: magic=0x%llx counter=%u task_kptr=0x%llx stashed_arena_ptr=0x%llx\n",
		     taskc->magic, taskc->counter, taskc->task_kptr,
		     taskc->stashed_arena_ptr);
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
