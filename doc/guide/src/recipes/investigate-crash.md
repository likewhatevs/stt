# Investigate a Crash

When a scheduler crashes during a test, the failure output and
auto-repro pipeline help identify the cause.

## First step: enable full diagnostics

Rerun the failing test with `RUST_BACKTRACE=1` before digging into
individual sections:

```sh
RUST_BACKTRACE=1 cargo nextest run -E 'test(my_test)'
```

Setting `RUST_BACKTRACE=1` unconditionally appends the
`--- diagnostics ---` section (init stage, VM exit code, last lines
of kernel console) to every failure, not only when the scheduler
self-dies. It also enables verbose VM console output (equivalent to
`KTSTR_VERBOSE=1`).

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
| `--- monitor ---` | Host-side monitor: sample count, max imbalance ratio, max local-DSQ depth, sustained-violation flag, SCX event counters (select_cpu_fallback, keep_last, skip_exiting, skip_migration_disabled), per-sched_domain load-balance rates, per-BPF-program `verified_insns`, and the merged threshold verdict. |
| `--- sched_ext dump ---` | `sched_ext_dump` trace lines from the guest kernel. |
| `--- auto-repro ---` | BPF probe data from a second VM run, plus repro VM duration, scheduler log, sched_ext dump, and dmesg tails. |

`--- diagnostics ---` appears automatically when the scheduler died
or crashed, or when `RUST_BACKTRACE` is set to `1` or `full`.

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

After the probe data, the auto-repro section includes the repro VM
duration and the last 40 lines of the repro VM's scheduler log
(cycle-collapsed), sched_ext dump, and kernel console (dmesg). These
supplement probe data when the crash produces sparse or no probe events.
When probe data is absent, a crash reproduction status line replaces it.

See [Auto-Repro](../running-tests/auto-repro.md) for details on how
the two-VM repro cycle works.
