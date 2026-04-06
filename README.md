# stt

[![CI](https://github.com/likewhatevs/stt/actions/workflows/ci.yml/badge.svg)](https://github.com/likewhatevs/stt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/likewhatevs/stt/graph/badge.svg)](https://codecov.io/gh/likewhatevs/stt)

Test harness for Linux process schedulers, with a focus on
[sched_ext](https://github.com/sched-ext/scx). Boots kernels in KVM
VMs with synthetic CPU topologies, runs workloads, and verifies
scheduling correctness. Also tests under the kernel's default EEVDF
scheduler.

- **Real isolation** -- each test boots its own kernel. No host interference, no shared state.
- **Any topology** -- 1 to 252 CPUs with arbitrary LLC structure via synthetic ACPI tables.
- **Data-driven** -- scenarios declare cgroups, cpusets, workloads, and verification as data.
- **Gauntlet** -- all scenarios across 13 topology presets in parallel VMs. Baseline save/compare for A/B testing.
- **`#[stt_test]`** -- proc macro for integration tests that boot their own VMs.
- **Auto-repro** -- reruns failures with BPF kprobes on the crash call chain.

## Quick start

```sh
cargo install --path cargo-stt

# single scenario
cargo stt vm --sockets 2 --cores 4 --threads 2 -- cgroup_steady

# with a scheduler
cargo stt vm -p scx_mitosis --sockets 2 --cores 4 --threads 2 -- cgroup_steady

# gauntlet
cargo stt vm --gauntlet --parallel 4
```

## Documentation

**[Guide](doc/guide/src/SUMMARY.md)** -- getting started, concepts,
writing tests, recipes, architecture.

## License

GPL-2.0-only
