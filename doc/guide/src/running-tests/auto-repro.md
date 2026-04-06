# Auto-Repro

When a test crashes, auto-repro extracts function names from the crash
stack and reruns in a second VM with BPF kprobes attached to those
functions.

## How it works

1. **First VM** -- the test runs normally. If the scheduler crashes, stt
   captures the stack trace from the scenario output.

2. **Stack extraction** -- function names are parsed from the crash
   trace.

3. **Second VM** -- stt boots a new VM and reruns the scenario with BPF
   kprobes attached to each function in the crash chain. The probes
   capture function arguments at each point in the crash path.

## Enabling auto-repro

From the CLI:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --auto-repro
```

In `#[stt_test]`:

```rust
#[stt_test(auto_repro = true)]
fn my_test(ctx: &Ctx) -> anyhow::Result<VerifyResult> { ... }
```

`auto_repro` defaults to `true` in `#[stt_test]` but `false` on the
CLI (`--auto-repro` is a flag, not set by default).

## Repro mode

During the second VM run, stt sets "repro mode" which disables the
work-conservation watchdog. Workers normally send SIGUSR2 to the
scheduler when stuck > 2 seconds. In repro mode, the scheduler stays
alive so BPF assertion probes can fire.
