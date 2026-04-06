# Getting Started

## Prerequisites

- Linux host with KVM access (`/dev/kvm`)
- Rust toolchain (stable)

## Install

```sh
cargo install --path cargo-stt
```

## Run a scenario

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30
```

`vm` boots a KVM virtual machine with the specified CPU topology.
Arguments after `--` are passed to `stt run` inside the VM.

To test with a scheduler, use `-p` to build and inject it:

```sh
cargo stt vm -p scx_mitosis --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30
```

Expected output:

```text
[stt] booting VM: 2s4c2t (16 cpus), 4096 MB
[stt] running: cgroup_steady/default
[stt]   PASS  cgroup_steady/default (30.1s)
```

Omit the scenario name to run all scenarios:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2
```

## List scenarios

List catalog scenarios (data-driven):

```sh
stt list
```

List `#[stt_test]` integration tests:

```sh
cargo stt list
```

## View topology

```sh
cargo stt topo
```

Prints the host CPU topology (LLCs, NUMA nodes, CPU IDs).

## Next steps

To run existing tests with different flags, topologies, or schedulers:
[Running Tests](running-tests.md).

To understand scenarios, flags, and verification:
[Core Concepts](concepts.md).

To write new tests: [Writing Tests](writing-tests.md).
