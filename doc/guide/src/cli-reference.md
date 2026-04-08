# CLI Reference

Build with `cargo build -p stt` or install with
`cargo install --path .`.

| Subcommand | Description |
|---|---|
| `stt vm` | Boot a VM and run scenarios |
| `stt topo` | Show host CPU topology (LLCs, NUMA nodes, CPU IDs) |
| `stt probe` | Probe kernel functions from a crash stack |
| `stt verifier` | Boot scheduler in VM and report verifier stats |
| `stt kernel build PATH` | Build a kernel with stt's config fragment |
| `stt kernel clean PATH` | Clean a kernel source tree (`make mrproper`) |
| `stt kernel kconfig` | Print stt's kernel config fragment to stdout |

`stt run` exists but is hidden internal plumbing -- it is the
guest-side dispatch that runs inside the VM. Do not call it directly.

## `stt vm`

Boots a KVM VM with the specified topology and runs scenarios inside
it. Arguments before `--` configure the VM; arguments after `--`
configure the test scenarios (names, flags, duration).

Key options: `--sockets`, `--cores`, `--threads`, `--memory-mb`,
`--kernel`, `--kernel-dir`, `--scheduler-bin`.

See [Single Scenario](running-tests/single-scenario.md) for the full
argument table.

## `stt verifier`

Boots a scheduler in a VM and reports per-program verifier statistics.
Default output applies cycle collapse to reduce repetitive loop
unrolling.

Key options: `-p`/`--package` (default: `stt-sched`),
`--diff <package>` (A/B instruction count delta),
`--kernel <path>` (kernel image for the VM).

See [BPF Verifier](running-tests/verifier.md) for the wire protocol
and cycle collapse algorithm.

## `stt probe`

Probes kernel functions from a crash stack trace.

Key options: `--dmesg`, `--functions`, `--kernel-dir`, `--bootlin`,
`--trigger`.

See [Investigate a Crash](recipes/investigate-crash.md) for usage.
