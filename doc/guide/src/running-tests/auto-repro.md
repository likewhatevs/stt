# Auto-Repro

When a test fails because the scheduler crashes or exits, auto-repro
boots a second VM with BPF probes attached to capture function arguments
and struct fields along the scheduling path. Stack functions extracted
from the crash output seed the probe list; when no crash stack is
available (e.g. a BPF text error or verifier failure with no backtrace),
auto-repro falls back to dynamic BPF program discovery in the repro VM.

## How it works

1. **First VM** -- the test runs normally. If the scheduler crashes or
   exits (BPF error, verifier failure, stall), ktstr captures any stack
   trace from the scheduler log (COM2) or kernel console (COM1).

2. **Stack extraction** -- function names are parsed from the crash
   trace when available. BPF program symbols (`bpf_prog_*`) are
   recognized and their short names extracted. Generic functions
   (scheduler entry points, spinlocks, syscall handlers, sched_ext
   exit machinery, BPF trampolines, stack dump helpers) are filtered
   out. When no stack functions are found, the pipeline continues with
   an empty probe list.

3. **BPF discovery** -- in the repro VM, ktstr discovers loaded
   struct_ops programs via libbpf-rs and adds them to the probe list
   alongside any stack-extracted functions. Their kernel-side callers
   are added (e.g. `enqueue` -> `do_enqueue_task`) for bridge kprobes.
   This step ensures probes can capture variable states across the
   scheduler exit call chain even when the crash produced no
   extractable stack.

4. **BTF resolution** -- function signatures are resolved from vmlinux
   BTF (kernel functions) and program BTF (BPF callbacks). Known struct
   types (task_struct, rq, scx_dispatch_q, etc.) have their fields
   resolved to byte offsets. Unknown BPF-local types (e.g. task_ctx)
   have scalar and cpumask pointer fields auto-discovered.

5. **Second VM** -- ktstr boots a new VM and reruns the scenario with two
   BPF skeletons:
   - Kprobe skeleton for kernel functions (uses `bpf_get_func_ip`)
   - Fentry skeleton for BPF callbacks (batched in groups of 4,
     shares maps via `reuse_fd`)

6. **Stitching** -- the task_struct pointer is found from the last
   event (closest to trigger time) that has a non-zero task_struct
   argument. Events with a task_struct parameter are filtered to that
   pointer; events without a task_struct parameter are kept
   unconditionally for context. Events are sorted by timestamp and
   formatted with decoded field values (cpumask ranges, DSQ names,
   enqueue flags, etc.) and source locations (DWARF for kernel,
   line_info for BPF).

## Enabling auto-repro

In `#[ktstr_test]`:

```rust,ignore
#[ktstr_test(auto_repro = true)]
fn my_test(ctx: &Ctx) -> Result<AssertResult> { ... }
```

`auto_repro` defaults to `true` in `#[ktstr_test]`.

## Repro mode

During the second VM run, ktstr sets "repro mode" which disables the
work-conservation watchdog. Workers normally send SIGUSR2 to the
scheduler when stuck > 2 seconds. In repro mode, the scheduler stays
alive so BPF assertion probes can fire.

## Example output

The `demo_host_crash_auto_repro` test triggers a host-initiated crash
via BPF map write and captures the scheduling path. Probe output shows
each function with decoded struct fields and source locations:

```text
ktstr_test 'demo_host_crash_auto_repro' [sched=scx-ktstr] failed:
  scheduler died

--- auto-repro ---
=== AUTO-PROBE: scx_exit fired ===

  ktstr_enqueue                                                   main.bpf.c:21
    task_struct *p
      pid         97
      cpus_ptr    0xf(0-3)
      dsq_id      SCX_DSQ_INVALID
      enq_flags   NONE
      slice       0
      vtime       0
      weight      100
      sticky_cpu  -1
      scx_flags   QUEUED|ENABLED
  ktstr_dispatch                                                  main.bpf.c:28
      cpu         0
    task_struct *p
      pid         97
      cpus_ptr    0xf(0-3)
      dsq_id      SCX_DSQ_INVALID
      enq_flags   NONE
      slice       19911318
      vtime       0
      weight      100
      sticky_cpu  -1
      scx_flags   RESET_RUNNABLE_AT|DEQD_FOR_SLEEP|ENABLED
  do_enqueue_task                                               kernel/sched/ext.c:1344
    rq *rq
      cpu         1
    task_struct *p
      pid         97
      cpus_ptr    0xf(0-3)
      dsq_id      LOCAL
      enq_flags   NONE
      slice       20000000
      vtime       0
      weight      100
      sticky_cpu  -1
      scx_flags   QUEUED|DEQD_FOR_SLEEP|ENABLED
  pick_task_scx                                                 kernel/sched/ext.c:2511
    rq *rq
      cpu         1
```

## Demo test

The `demo_host_crash_auto_repro` test:

```rust,ignore
use ktstr::prelude::*;
use ktstr::test_support::{BpfMapWrite, KtstrTestEntry, run_ktstr_test};

fn scenario(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("demo_workers")
                .work_type(WorkType::YieldHeavy)
                .workers(4),
        ],
        HoldSpec::Fixed(Duration::from_secs(8)),
    )];
    execute_steps(ctx, steps)
}
```

Run manually to see full output:

```sh
cargo nextest run --run-ignored ignored-only -E 'test(demo_host_crash_auto_repro)'
```
