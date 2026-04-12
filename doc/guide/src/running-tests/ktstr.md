# ktstr

`ktstr` runs ktstr scenarios directly on the host under whatever
scheduler is already active. Unlike `#[ktstr_test]` (which boots
KVM VMs), `ktstr` operates on the host's real topology and cgroups.
It does not manage scheduler lifecycle -- start your scheduler
externally before running.

Build from the workspace:

```sh
cargo build --bin ktstr
```

## Subcommands

### run

Run scenarios on the host:

```sh
ktstr run
ktstr run --flags llc,borrow --duration 30
ktstr run --filter cpuset --json
ktstr run --work-type YieldHeavy
```

Scenarios run under whatever scheduler is currently active on the
host. Start your scheduler before invoking `ktstr run`.

Without `--flags`, all valid flag profiles are generated for each
scenario. With `--flags`, only the specified profile is run. Flags
select which test profiles to run -- they do not configure the
scheduler. Start the scheduler with the desired features before
running ktstr.

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

### kernel list

List cached kernel images:

```sh
ktstr kernel list
ktstr kernel list --json
```

### kernel build

Download, build, and cache a kernel image:

```sh
ktstr kernel build 6.14.2
ktstr kernel build 6.15-rc3
ktstr kernel build                                    # latest stable
ktstr kernel build --source ../linux
ktstr kernel build --git https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git --ref v6.14
ktstr kernel build --force 6.14.2
ktstr kernel build --source ../linux --clean
```

Three source modes: positional VERSION (tarball download), `--source PATH`
(local tree), `--git URL --ref REF` (shallow clone). Without arguments,
downloads the latest stable release.

`--force` rebuilds even if a cached image exists. `--clean` runs
`make mrproper` before configuring (local source only). Dirty trees
(uncommitted changes) are built but not cached.

### kernel clean

Remove cached kernel images:

```sh
ktstr kernel clean
ktstr kernel clean --keep 2
ktstr kernel clean --force
```

`--keep N` retains the N most recent entries. Without `--force`,
prompts for confirmation (requires a terminal).

### completions

Generate shell completions:

```sh
ktstr completions bash
ktstr completions zsh
ktstr completions fish
```

Supported shells: bash, zsh, fish, elvish, powershell.
