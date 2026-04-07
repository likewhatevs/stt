# Single Scenario

## Basic invocation

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --duration-s 30
```

Arguments before `--` configure the VM. Arguments after `--` configure
the test scenarios (names, flags, duration).

## VM arguments

| Argument | Default | Description |
|---|---|---|
| `--sockets N` | 2 | CPU sockets |
| `--cores N` | 2 | Cores per socket |
| `--threads N` | 2 | Threads per core |
| `--memory-mb N` | 4096 | VM memory |
| `--kernel PATH` | -- | Kernel image (falls back to `/boot/vmlinuz` if neither `--kernel` nor `--kernel-dir` is set) |
| `--kernel-dir PATH` | -- | Linux source tree (uses `arch/x86/boot/bzImage`) |
| `-p, --package PKG` | -- | Build scheduler from cargo package and inject |
| `--scheduler-bin PATH` | -- | Scheduler binary to inject (direct path, skips build) |

## Run arguments (after --)

| Argument | Default | Description |
|---|---|---|
| `SCENARIO...` | all | Scenario names to run |
| `--duration-s N` | 15 | Per-scenario duration in seconds |
| `--workers N` | 4 | Workers per cgroup |
| `--flags=X,Y` | none | Flags to enable |
| `--all-flags` | -- | Run all valid flag combinations |
| `--verbose` | -- | Verbose output |
| `--json` | -- | JSON output |
| `--work-type NAME` | -- | Override work type. Valid names: `CpuSpin`, `YieldHeavy`, `Mixed`, `IoSync`, `Bursty`, `PipeIo`, `FutexPingPong`, `CachePressure`, `CacheYield`, `CachePipe`. WorkProgram presets: `cpu_spin`, `mixed`, `bursty`, `yield`, `io`, `pipe`, `cache_l1`, `cache_yield`, `cache_pipe`, `futex`. |

## Investigating failures

Run one scenario with specific flags and a longer duration:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_cpuset_crossllc_race --flags=llc,rebal --duration-s 60
```
