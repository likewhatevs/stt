# Troubleshooting

## /dev/kvm not accessible

```text
/dev/kvm not accessible — KVM is required for stt_test.
Check that KVM is enabled and your user is in the kvm group.
```

stt boots Linux kernels in KVM virtual machines. The host must have
KVM enabled and the user must have read+write access to `/dev/kvm`.

**Fixes:**

- Load the KVM module: `modprobe kvm_intel` or `modprobe kvm_amd`.
- Add your user to the `kvm` group: `sudo usermod -aG kvm $USER`
  (log out and back in for the group change to take effect).
- In CI, ensure the runner has KVM access (e.g. `runs-on: [self-hosted, kvm]`).

## No kernel found

```text
no kernel found. Set STT_TEST_KERNEL or build one at ../linux/
```

stt needs a bootable Linux kernel image (bzImage). See
[Kernel discovery](getting-started.md#kernel-discovery) for the
search order.

**Fixes:**

- Build a kernel using `stt.kconfig` (see
  [Getting Started](getting-started.md#build-a-kernel)).
- Set `STT_TEST_KERNEL` to an explicit path.
- Build a kernel in a sibling `linux/` directory.
- The host's installed kernel works for basic testing.

## Scheduler not found

```text
scheduler 'scx_mitosis' not found. Set STT_SCHEDULER or
place it next to the test binary or in target/{debug,release}/
```

When using `SchedulerSpec::Name`, stt searches for the scheduler
binary in:

1. `STT_SCHEDULER` environment variable.
2. Same directory as the test binary.
3. `target/debug/`.
4. `target/release/`.

**Fixes:**

- Build the scheduler first: `cargo build -p scx_mitosis`.
- Set `STT_SCHEDULER=/path/to/binary`.
- Use `SchedulerSpec::Path` for an explicit path in `#[stt_test]`.

## Scheduler died

```text
scheduler died between step 2 and step 3 (of 5), 12.3s into scenario
```

The scheduler process exited while the scenario was running. This
is usually a crash. The exact message varies by when the death was
detected (between steps, during workload, after completion).

The failure output contains diagnostic sections (each present only
when relevant):

- `--- scheduler log ---`: the scheduler's stdout and stderr,
  cycle-collapsed for readability.
- `--- diagnostics ---`: init stage classification, VM exit code,
  and the last 20 lines of kernel console output.
- `--- sched_ext dump ---`: `sched_ext_dump` trace lines from the
  guest kernel (present when a SysRq-D dump fired).

Set `RUST_BACKTRACE=1` to force `--- diagnostics ---` on all
failures, not just scheduler deaths.

**Next steps:**

- Check the `--- scheduler log ---` for the crash reason.
- Check `--- diagnostics ---` for BPF errors or kernel oops in
  the kernel console.
- Enable `auto_repro` in the test to capture the crash path with
  BPF probes. See [Auto-Repro](running-tests/auto-repro.md).
- Run with a longer duration and specific flags to narrow the
  reproducer.

See [Investigate a Crash](recipes/investigate-crash.md) for the
complete failure output format and auto-repro walkthrough.

## Insufficient hugepages

```text
insufficient free hugepages for N MB guest memory (need M, have K)
```

[Performance mode](concepts/performance-mode.md) requests 2MB
hugepages for guest memory. Without enough free hugepages, the VM
falls back to regular pages (with a warning).

**Fix:**

Allocate hugepages before the run:

```sh
echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

## Monitor threshold failures

```text
worker gap 4500ms exceeds max_gap_ms 3000
worker spread 0.42 exceeds max_spread 0.35
```

The Assert checks (`max_gap_ms`, `max_spread`, etc.) detected a
worker metric outside the configured thresholds.

**Fixes:**

- Check whether the topology has enough CPUs for the scenario. Small
  topologies produce higher contention, larger gaps, and more spread.
- Use `execute_steps_with()` with a custom `Assert` to override
  thresholds for scenarios that need relaxed limits.
- Check the scheduler's behavior under the specific flag profile that
  triggered the failure.

## Cgroup name typos

```text
No such file or directory: /sys/fs/cgroup/.../nonexistent/cgroup.procs
```

A cgroup name passed to `Op::SetCpuset`, `Op::Spawn`, or
`CgroupManager::move_tasks` does not match a previously created
cgroup. Cgroup names are case-sensitive strings.

**Fixes:**

- Verify the cgroup name matches the `name` in `Op::AddCgroup` or
  `CgroupDef::named()`.
- When using dynamic cgroup names (e.g. `format!("cg_{i}")`), ensure
  the same formatting is used in all ops referencing that cgroup.

## CpusetSpec errors

```text
CpusetSpec validation failed: not enough usable CPUs (4) for 8 partitions
CpusetSpec validation failed: index 3 >= partition count 3
```

A `CpusetSpec` cannot produce a valid cpuset for the test topology.
`execute_steps` logs a warning and proceeds (the resolved cpuset may
be empty or unexpected).

**Fixes:**

- Guard with a topology check before creating the step:
  `if ctx.topo.usable_cpus().len() < needed { return Ok(AssertResult::skip(...)); }`
- Use `CpusetSpec::validate(&ctx)` to check before resolve.
- Reduce the partition count or use `CpusetSpec::Llc` instead of
  `Disjoint` on topologies with fewer CPUs than partitions.

## Worker count mismatches

```text
grouped work type PipeIo requires even num_workers, got 3
```

Grouped work types (`PipeIo`, `FutexPingPong`, `CachePipe`,
`FutexFanOut`) require `num_workers` divisible by their group size.
`WorkType::worker_group_size()` returns the divisor.

**Fixes:**

- Set `CgroupDef::workers(n)` to a value divisible by the work
  type's group size (2 for pipe/futex pairs, `fan_out + 1` for
  FutexFanOut).
- Use an ungrouped work type (`CpuSpin`, `Mixed`, `Bursty`,
  `IoSync`, `YieldHeavy`) if worker count flexibility is needed.

## Tests pass locally but fail in CI

Common causes:

- **No KVM**: CI runners need hardware virtualization. Check for
  `/dev/kvm` access.
- **Fewer CPUs**: gauntlet topology presets up to 252 CPUs may
  exceed the runner's capacity. Use smaller topologies.
- **No kernel**: set `STT_TEST_KERNEL` in the CI environment.
- **Debug thresholds**: CI often runs debug builds. Debug builds use
  relaxed thresholds (3000ms gap, 35% spread) but may still hit
  limits on slow runners. See
  [default thresholds](concepts/verification.md#default-thresholds).
