# CLI Reference

stt has two binaries with distinct roles.

## `stt` -- core binary

The `stt` binary handles VM management, guest-side test execution,
kernel builds, and crash investigation. Build it with
`cargo build -p stt` or install with `cargo install --path .`.

| Subcommand | Description |
|---|---|
| `stt topo` | Show host CPU topology (LLCs, NUMA nodes, CPU IDs) |
| `stt probe` | Probe kernel functions from a crash stack |
| `stt kernel build PATH` | Build a kernel with stt's config fragment |
| `stt kernel clean PATH` | Clean a kernel source tree (`make mrproper`) |
| `stt kernel kconfig` | Print stt's kernel config fragment to stdout |

`stt run` exists but is hidden internal plumbing -- it is the
guest-side dispatch that runs inside the VM. Do not call it directly.

## `cargo stt` -- cargo plugin

The `cargo-stt` binary wraps `stt` with test discovery, scheduler
builds, and gauntlet orchestration. Install it with
`cargo install --path cargo-stt`.

| Subcommand | Description |
|---|---|
| `cargo stt vm` | Boot a VM and run data-driven scenarios |
| `cargo stt vm --gauntlet` | Run catalog scenarios across 13 topology presets |
| `cargo stt test` | Run `#[stt_test]` integration tests via nextest |
| `cargo stt gauntlet` | Run `#[stt_test]` tests across topology presets |
| `cargo stt list` | List registered `#[stt_test]` entries |
| `cargo stt topo` | Show host CPU topology |
| `cargo stt probe` | Probe kernel functions from a crash stack |
| `cargo stt verifier` | Boot scheduler in VM and report verifier stats |

### `cargo stt vm`

Boots a KVM VM with the specified topology and runs scenarios inside
it. Arguments before `--` configure the VM; arguments after `--`
configure the test scenarios (names, flags, duration).

Key options: `--sockets`, `--cores`, `--threads`, `--memory-mb`,
`--kernel`, `--kernel-dir`, `-p`/`--package`, `--scheduler-bin`,
`--gauntlet`, `--parallel`, `--retries`, `--save-baseline`,
`--compare`, `--work-types`.

See [Single Scenario](running-tests/single-scenario.md) for the full
argument table.

### `cargo stt gauntlet`

Runs `#[stt_test]` integration tests across topology presets in
parallel VMs.

Key options: `--parallel`, `-p`/`--package`, `--filter`, `--flags`,
`--work-types`, `--save-baseline`, `--compare`.

See [Gauntlet](running-tests/gauntlet.md) for topology presets and
flag profile dimensions.

### `cargo stt test`

Runs `#[stt_test]` integration tests via nextest with sidecar
collection. Tests with `performance_mode = true` are automatically
scheduled with `threads-required` based on host LLC topology
(sum of CPUs in used LLC groups + 1) via a generated nextest tool
config, preventing CPU oversubscription on the host.

Key options: `--filter`, `--kernel`, `--scheduler-bin`,
`--save-baseline`, `--compare`, `--nextest-profile`,
`-p`/`--package`.

### `cargo stt list`

Lists all registered `#[stt_test]` entries by building the test
binary and querying it with `--stt-list`.

Key options: `-p`/`--package`.

### `cargo stt verifier`

Boots a scheduler in a VM and reports per-program verifier statistics.
Default output applies cycle collapse to reduce repetitive loop
unrolling.

Key options: `-p`/`--package` (default: `stt-sched`),
`-v`/`--verbose` (full raw log), `--diff <package>` (A/B instruction
count delta), `--kernel <path>` (kernel image for the VM).

See [BPF Verifier](running-tests/verifier.md) for the wire protocol
and cycle collapse algorithm.

### `cargo stt probe`

Probes kernel functions from a crash stack trace.

Key options: `--dmesg`, `--functions`, `--kernel-dir`, `--bootlin`,
`--trigger`.

See [Investigate a Crash](recipes/investigate-crash.md) for usage.
