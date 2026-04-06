# Investigate a Crash

When a scheduler crashes during a test, stt provides two tools for
investigation: auto-repro and manual probe.

## Auto-repro

Run a scenario with `--auto-repro`:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  --scheduler-bin ./target/release/scx_my_scheduler \
  -- cgroup_steady --flags=llc,borrow --auto-repro
```

If the scheduler crashes, stt automatically:

1. Captures the crash stack trace from the scenario output.
2. Boots a second VM with BPF kprobes on each function in the crash
   chain.
3. Reruns the scenario to capture function arguments at each crash
   point.

The second run uses "repro mode" -- the work-conservation watchdog is
disabled so the scheduler stays alive for BPF probes.

## Manual probe

If you have a crash stack from dmesg:

```sh
cargo stt probe --dmesg
```

Or from a file:

```sh
cargo stt probe crash_stack.txt
```

Specify functions directly:

```sh
cargo stt probe --functions "scx_bpf_dispatch,put_prev_task_scx"
```

Add `--kernel-dir` for source-level symbolization:

```sh
cargo stt probe --dmesg --kernel-dir ../linux
```

## Reading the output

The probe output shows each function in the crash chain with:

- Function signature and argument values at the time of the crash
- Source file and line number (when `--kernel-dir` is provided)
- Call chain context

See [Auto-Repro](../running-tests/auto-repro.md) for details on how
the two-VM repro cycle works.
