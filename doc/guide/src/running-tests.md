# Running Tests

All test execution goes through `cargo stt vm`, which boots a KVM
virtual machine and runs scenarios inside it.

## Quick reference

```sh
# Run one scenario
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30

# Run all scenarios
cargo stt vm --sockets 2 --cores 4 --threads 2

# Run with flags
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --flags=borrow,rebal --duration-s 30

# Run all scenarios x all topologies
cargo stt vm --gauntlet

# Run integration tests
cargo stt test
```

## Flags

Flags enable scheduler features. Pass them with `--flags`:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --flags=borrow,rebal
```

Available flags: `llc`, `borrow`, `steal`, `rebal`, `reject-pin`,
`no-ctrl`. `steal` requires `llc` -- this is enforced automatically.

`--all-flags` runs every valid flag combination. See [Flags](concepts/flags.md)
for details on flag declarations and profile generation.

## Custom scheduler

Use `-p` to build a scheduler from its cargo package and inject it:

```sh
cargo stt vm -p scx_mitosis --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --flags=borrow,rebal
```

For a pre-built binary, use `--scheduler-bin` instead:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  --scheduler-bin ./target/release/scx_mitosis \
  -- cgroup_steady --flags=borrow,rebal
```

The binary is injected into the VM's initramfs and started before
scenarios run. See [Test a New Scheduler](recipes/test-new-scheduler.md)
for the full end-to-end workflow.
