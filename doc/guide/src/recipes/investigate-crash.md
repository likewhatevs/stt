# Investigate a Crash

When a scheduler crashes during a test, the failure output and
auto-repro pipeline help identify the cause.

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
| `--- auto-repro ---` | BPF probe data (kprobes, fentry, tp_btf trigger) from a second VM run. |

`--- diagnostics ---` appears automatically when the scheduler died
or when `RUST_BACKTRACE=1` is set.

## Auto-repro

`auto_repro` defaults to `true` in `#[ktstr_test]`. When the scheduler
crashes, ktstr automatically:

1. Captures the crash stack trace from the scenario output.
2. Boots a second VM with BPF kprobes (kernel functions) and fentry
   probes (BPF callbacks) on each function in the crash chain, plus
   a `tp_btf/sched_ext_exit` tracepoint trigger.
3. Reruns the scenario to capture function arguments at each crash
   point.

## Reading auto-repro output

The probe output shows each function in the crash chain with:

- Function signature and argument values during execution of the same workload
- Source file and line number
- Call chain context

See [Auto-Repro](../running-tests/auto-repro.md) for details on how
the two-VM repro cycle works.
