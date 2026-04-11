# ktstr

`ktstr` runs ktstr scenarios directly on the host against a
scheduler. Unlike `#[ktstr_test]` (which boots KVM VMs),
`ktstr` operates on the host's real topology and cgroups.

Build with the `cli` feature:

```sh
cargo build --features cli --bin ktstr
```

## Subcommands

### run

Run scenarios against a scheduler binary:

```sh
ktstr run --scheduler scx_my_sched
ktstr run --scheduler scx_my_sched --flags llc,borrow --duration 30
ktstr run --scheduler scx_my_sched --filter cpuset --json
ktstr run --scheduler scx_my_sched --work-type YieldHeavy
```

Without `--scheduler`, scenarios run under the kernel's default
scheduler (EEVDF).

Without `--flags`, all valid flag profiles are generated for each
scenario. With `--flags`, only the specified flags are active.

`--filter` selects scenarios whose name contains the given substring.

**Flags:** llc, borrow, steal, rebal, reject-pin, no-ctrl.

**Work types:** CpuSpin, YieldHeavy, Mixed, IoSync, Bursty, PipeIo,
FutexPingPong, CachePressure, CacheYield, CachePipe, FutexFanOut.

### list

List available scenarios:

```sh
ktstr list
ktstr list --filter dynamic
ktstr list --json
```

### topo

Show the host CPU topology (CPUs, LLCs, NUMA nodes):

```sh
ktstr topo
```

### cleanup

Remove leftover cgroups from a previous run:

```sh
ktstr cleanup
ktstr cleanup --parent-cgroup /sys/fs/cgroup/ktstr
```
