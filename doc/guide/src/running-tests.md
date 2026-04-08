# Running Tests

All test execution goes through `stt vm`, which boots a KVM
virtual machine and runs scenarios inside it.

## Quick reference

```sh
# Run one scenario
stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30

# Run all scenarios
stt vm --sockets 2 --cores 4 --threads 2

# Run with flags
stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --flags=borrow,rebal --duration-s 30
```

## Flags

Flags enable scheduler features. Pass them with `--flags`:

```sh
stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --flags=borrow,rebal
```

Available flags: `llc`, `borrow`, `steal`, `rebal`, `reject-pin`,
`no-ctrl`. `steal` requires `llc` -- this is enforced automatically.

`--all-flags` runs every valid flag combination. See [Flags](concepts/flags.md)
for details on flag declarations and profile generation.

## Custom scheduler

Use `--scheduler-bin` to inject a pre-built scheduler binary:

```sh
stt vm --sockets 2 --cores 4 --threads 2 \
  --scheduler-bin ./target/release/scx_mitosis \
  -- cgroup_steady --flags=borrow,rebal
```

The binary is injected into the VM's initramfs and started before
scenarios run. See [Test a New Scheduler](recipes/test-new-scheduler.md)
for the full end-to-end workflow.
