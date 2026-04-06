# Auto-Repro

When a test crashes, auto-repro extracts function names from the crash
stack and reruns in a second VM with BPF probes attached to those
functions. The probes capture function arguments and struct fields at
each point in the crash path.

## How it works

1. **First VM** -- the test runs normally. If the scheduler crashes, stt
   captures the stack trace from the scenario output.

2. **Stack extraction** -- function names are parsed from the crash
   trace. BPF program symbols (`bpf_prog_*`) are recognized and their
   short names extracted. Generic functions (scheduler entry points,
   spinlocks, syscall handlers) are filtered out.

3. **BPF discovery** -- for BPF functions, stt discovers loaded
   struct_ops programs via libbpf-rs and adds their kernel-side callers
   (e.g. `enqueue` -> `do_enqueue_task`) for bridge kprobes.

4. **BTF resolution** -- function signatures are resolved from vmlinux
   BTF (kernel functions) and program BTF (BPF callbacks). Known struct
   types (task_struct, rq, scx_dispatch_q, etc.) have their fields
   resolved to byte offsets. Unknown BPF-local types (e.g. task_ctx)
   have scalar and cpumask pointer fields auto-discovered.

5. **Second VM** -- stt boots a new VM and reruns the scenario with two
   BPF skeletons:
   - Kprobe skeleton for kernel functions (uses `bpf_get_func_ip`)
   - Fentry skeleton for BPF callbacks (batched in groups of 4,
     shares maps via `reuse_fd`)

6. **Stitching** -- captured events are filtered by task_struct pointer
   or tid, sorted by timestamp, and formatted with decoded field values
   (cpumask ranges, DSQ names, enqueue flags, etc.) and source locations
   (DWARF for kernel, line_info for BPF).

## Enabling auto-repro

From the CLI:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --auto-repro
```

In `#[stt_test]`:

```rust,ignore
#[stt_test(auto_repro = true)]
fn my_test(ctx: &Ctx) -> Result<VerifyResult> { ... }
```

`auto_repro` defaults to `true` in `#[stt_test]` but `false` on the
CLI (`--auto-repro` is a flag, not set by default).

## Repro mode

During the second VM run, stt sets "repro mode" which disables the
work-conservation watchdog. Workers normally send SIGUSR2 to the
scheduler when stuck > 2 seconds. In repro mode, the scheduler stays
alive so BPF assertion probes can fire.

## Example output

The `demo_host_crash_auto_repro` test triggers a host-initiated crash
via BPF map write and captures the scheduling path. Probe output shows
each function with decoded struct fields and source locations:

```text
stt_test 'demo_host_crash_auto_repro' [sched=stt-sched] failed:
  scheduler died

--- auto-repro ---
=== AUTO-PROBE: scx_exit fired ===

  stt_enqueue                                                   main.bpf.c:21
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
  stt_dispatch                                                  main.bpf.c:28
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
use stt::prelude::*;
use stt::test_support::{BpfMapWrite, SttTestEntry, run_stt_test};

fn scenario(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("demo_workers")
                .work_type(WorkType::YieldHeavy)
                .workers(4),
        ].into(),
        ops: vec![],
        hold: HoldSpec::Fixed(Duration::from_secs(8)),
    }];
    execute_steps(ctx, steps)
}
```

Run manually to see full output:

```sh
cargo test demo_host_crash_auto_repro -- --ignored --nocapture
```
