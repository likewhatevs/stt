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

The second run uses "repro mode" -- the work-conservation watchdog is
disabled so the scheduler stays alive for BPF probes.

## Reading the output

The probe output shows each function in the crash chain with:

- Function signature and argument values at the time of the crash
- Source file and line number
- Call chain context

See [Auto-Repro](../running-tests/auto-repro.md) for details on how
the two-VM repro cycle works.
