# Investigate a Crash

When a scheduler crashes during a test, stt provides auto-repro
for investigation.

## Auto-repro

Enable `auto_repro` in your `#[stt_test]` (it defaults to `true`):

```rust,ignore
#[stt_test(auto_repro = true, scheduler = MY_SCHED)]
fn my_crash_test(ctx: &Ctx) -> Result<AssertResult> { ... }
```

If the scheduler crashes, stt automatically:

1. Captures the crash stack trace from the scenario output.
2. Boots a second VM with BPF kprobes on each function in the crash
   chain.
3. Reruns the scenario to capture function arguments at each crash
   point.

The second run reruns the same workload with BPF probes attached.

## Reading failure output

A test failure message contains up to eight sections, each present
only when relevant:

| Section | Content |
|---|---|
| Error line | Test name, scheduler, failure reason. |
| `--- stats ---` | Per-cgroup worker count, CPU count, spread, gap, migrations, iterations. |
| `--- diagnostics ---` | Init stage classification, VM exit code, last 20 lines of kernel console. |
| `--- timeline ---` | Kernel version, topology, scheduler, scenario duration, phase breakdown with monitor samples. |
| `--- scheduler log ---` | Scheduler process stdout+stderr (cycle-collapsed). |
| `--- monitor ---` | Host-side monitor: sample count, max imbalance, max DSQ depth, stall flag, threshold verdict. |
| `--- sched_ext dump ---` | `sched_ext_dump` trace lines from the guest kernel. |
| `--- auto-repro ---` | BPF kprobe data from a second VM run (when `auto_repro = true`). |

`--- diagnostics ---` appears automatically when the scheduler died
or when `RUST_BACKTRACE=1` is set.

## Reading auto-repro output

The probe output shows each function in the crash chain with:

- Function signature and argument values during execution of the same workload
- Source file and line number
- Call chain context

See [Auto-Repro](../running-tests/auto-repro.md) for details on how
the two-VM repro cycle works.
