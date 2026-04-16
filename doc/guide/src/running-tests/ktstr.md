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

### kernel

The `kernel` subcommand manages cached kernel images. Subcommands:
`list`, `build`, `clean`. See
[cargo-ktstr kernel](cargo-ktstr.md#kernel) for full documentation
-- the kernel subcommands are identical in both binaries.

### shell

Boot an interactive shell in a KVM virtual machine. Launches a VM
with busybox and drops into a shell.

```sh
ktstr shell
ktstr shell --kernel ../linux
ktstr shell --kernel 6.14.2
ktstr shell --topology 1,2,4,1
ktstr shell -i /path/to/binary
ktstr shell -i my_tool -i another_tool
```

Files and directories passed via `-i` are available at
`/include-files/<name>` inside the guest. Directories are walked
recursively, preserving structure (e.g. `-i ./release` includes all
files under `release/` at `/include-files/release/...`). Bare names
(without path separators) are resolved via `PATH` lookup.
Dynamically-linked ELF binaries get automatic shared library
resolution via ELF DT_NEEDED parsing. Non-ELF files are copied as-is.

Stdin is a terminal requirement. The host terminal enters raw mode
for bidirectional stdin/stdout forwarding. Terminal state is restored
on all exit paths.

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel ID` | auto | Kernel identifier (path, version, or cache key). |
| `--topology N,L,C,T` | `1,1,1,1` | Virtual CPU topology as `numa_nodes,llcs,cores,threads`. All values must be >= 1. |
| `-i, --include-files PATH` | -- | Files or directories to include in the guest. Repeatable. Directories are walked recursively. |
| `--memory-mb MB` | auto | Guest memory in MB (minimum 128). When absent, estimated from payload and include file sizes. |
| `--dmesg` | off | Forward kernel console (COM1/dmesg) to stderr in real-time. Sets loglevel=7 for verbose kernel output. |
| `--exec CMD` | -- | Run a command in the VM instead of an interactive shell. The VM exits after the command completes. |

The same subcommand is available as `cargo ktstr shell`.

### completions

Generate shell completions for ktstr.

```sh
ktstr completions bash
ktstr completions zsh
ktstr completions fish
```

| Arg | Description |
|------|-------------|
| `SHELL` | Shell to generate completions for (`bash`, `zsh`, `fish`, `elvish`, `powershell`). |

The same subcommand is available as `cargo ktstr completions` (which
also accepts `--binary` to set the binary name for completions).
