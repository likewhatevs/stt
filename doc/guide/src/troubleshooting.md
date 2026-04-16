# Troubleshooting

## /dev/kvm not accessible

```text
/dev/kvm not accessible — KVM is required for ktstr_test.
Check that KVM is enabled and your user is in the kvm group.
```

ktstr boots Linux kernels in KVM virtual machines. The host must have
KVM enabled and the user must have read+write access to `/dev/kvm`.

**Fixes:**

- Load the KVM module: `modprobe kvm_intel` or `modprobe kvm_amd`.
- Add your user to the `kvm` group: `sudo usermod -aG kvm $USER`
  (log out and back in for the group change to take effect).
- In CI, ensure the runner has KVM access (e.g. `runs-on: [self-hosted, kvm]`).

## No kernel found

```text
no kernel found
  hint: run `cargo ktstr kernel build` to download and build the latest stable kernel
  hint: or set KTSTR_KERNEL=/path/to/linux
  hint: or set KTSTR_TEST_KERNEL=/path/to/bzImage
```

`cargo ktstr shell` and `cargo ktstr verifier` show a similar message:

```text
no kernel found. Provide --kernel or run `cargo ktstr kernel build` to download and cache one.
```

ktstr needs a bootable Linux kernel image (bzImage). See
[Kernel discovery](getting-started.md#kernel-discovery) for the
search order.

**Fixes:**

- Download and cache a kernel: `cargo ktstr kernel build`
- Build from a local tree: `cargo ktstr kernel build --source ../linux`
- Set `KTSTR_TEST_KERNEL` to an explicit image path.
- The host's installed kernel works for basic testing.

## Scheduler not found

```text
scheduler 'scx_mitosis' not found. Set KTSTR_SCHEDULER or
place it next to the test binary or in target/{debug,release}/
```

When using `SchedulerSpec::Name`, ktstr searches for the scheduler
binary in:

1. `KTSTR_SCHEDULER` environment variable.
2. Same directory as the test binary.
3. `target/debug/`.
4. `target/release/`.

**Fixes:**

- Build the scheduler first: `cargo build -p scx_mitosis`.
- Set `KTSTR_SCHEDULER=/path/to/binary`.
- Use `SchedulerSpec::Path` for an explicit path in `#[ktstr_test]`.

## Scheduler crashed

```text
scheduler crashed after completing step 2 of 5 (12.3s into test)
```

The scheduler process exited while the scenario was running. This
is usually a crash. The exact message varies by when the crash was
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
performance_mode: WARNING: not enough hugepages (needed N MB = M pages, available K pages). Using regular pages.
```

[Performance mode](concepts/performance-mode.md) requests 2MB
hugepages for guest memory. Without enough free hugepages, the VM
falls back to regular pages (with a warning).

**Fix:**

Allocate hugepages before the run:

```sh
echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```

## Worker assertion failures

```text
stuck 4500ms on cpu2 at +3200ms (threshold 3000ms)
unfair cgroup: spread=42% (8-50%) 4 workers on 4 cpus (threshold 35%)
```

The Assert checks (`max_gap_ms`, `max_spread_pct`, etc.) detected a
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
PipeIo requires num_workers divisible by 2, got 3
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

## Cache corruption

```text
cached entry 6.14.2-tarball-x86_64-kc... has corrupt metadata
```

A cached kernel entry has missing or unparseable `metadata.json`.
This can happen after a partial write (e.g. disk full, killed process).
`CacheDir::lookup` returns `None` for entries with corrupt metadata
or a missing kernel image file.

**Fixes:**

- Remove the corrupt entry: `cargo ktstr kernel clean --force`
- Rebuild: `cargo ktstr kernel build --force 6.14.2`
- Override the cache directory via `KTSTR_CACHE_DIR` if the default
  location is on a problematic filesystem.

## Cache directory not found

```text
HOME not set; cannot resolve cache directory. Set KTSTR_CACHE_DIR to specify a cache location.
```

The kernel image cache requires a writable directory. ktstr resolves
it as: `KTSTR_CACHE_DIR` > `$XDG_CACHE_HOME/ktstr/kernels/` >
`$HOME/.cache/ktstr/kernels/`.

**Fix:** Set `KTSTR_CACHE_DIR` to an explicit path, or ensure `HOME`
is set.

## Stale kconfig

```text
warning: entries marked (stale kconfig) were built with a different ktstr.kconfig.
Rebuild with: kernel build --force VERSION
```

`cargo ktstr kernel list` marks entries whose stored `ktstr_kconfig_hash`
differs from the current embedded `ktstr.kconfig` fragment. This
happens after updating ktstr (which may change the kconfig fragment).

**Fix:**

Rebuilds happen automatically on the next `cargo ktstr kernel build`
for stale entries. Use `--force` to override the cache for other
reasons.

## Stale ktstr version

```text
warning: entries marked (stale ktstr) were built with a different ktstr version.
Rebuild with: kernel build --force VERSION
```

`cargo ktstr kernel list` marks entries whose stored `ktstr_git_hash`
differs from the running ktstr binary. Cached kernels built by an
older ktstr may have different vmlinux stripping or init behavior.

**Fix:**

Remove stale entries and rebuild:

```sh
cargo ktstr kernel clean
cargo ktstr kernel build --force VERSION
```

## Kernel download failures

```text
download https://cdn.kernel.org/.../linux-6.99.0.tar.xz: HTTP 404
```

The version does not exist on kernel.org. RC releases are removed from
git.kernel.org after the stable version ships.

```text
download ...: server returned HTML instead of tarball (URL may be invalid)
```

Some CDN error pages return HTTP 200 with `text/html` content type.
The download rejects these responses.

**Fixes:**

- Verify the version exists: check
  `https://www.kernel.org/releases.json` for available versions.
- For RC releases, use `--git` with a git.kernel.org URL instead of
  a tarball download.
- Run `cargo ktstr kernel build` without a version to automatically
  fetch the latest stable.

## Shell mode issues

### stdin must be a terminal

```text
stdin must be a terminal for interactive shell mode
```

`cargo ktstr shell` requires a terminal for bidirectional I/O
forwarding. Piped or redirected stdin is rejected.

**Fix:** Run from an interactive terminal session.

### include file not found

```text
-i strace: not found in filesystem or PATH
```

Bare names (without `/`, `.`, or `..`) are searched in `PATH`. If the
binary is not in `PATH`, use an explicit path.

```text
--include-files path not found: ./missing-file
```

Explicit paths (containing `/` or starting with `.`) must exist on
disk.

**Fix:** Verify the file exists and use the correct path.

### include directory contains no files

```text
warning: -i ./empty-dir: directory contains no regular files
```

The directory passed to `--include-files` was walked recursively but
contained no regular files. FIFOs, device nodes, and sockets are
skipped during the walk.

**Fix:** Verify the directory contains the files you expect.

## Tests pass locally but fail in CI

Common causes:

- **No KVM**: CI runners need hardware virtualization. Check for
  `/dev/kvm` access.
- **Fewer CPUs**: gauntlet topology presets up to 252 CPUs may
  exceed the runner's capacity. Use smaller topologies.
- **No kernel**: set `KTSTR_TEST_KERNEL` in the CI environment.
- **Debug thresholds**: CI often runs debug builds. Debug builds use
  relaxed thresholds (3000ms gap, 35% spread) but may still hit
  limits on slow runners. See
  [default thresholds](concepts/verification.md#default-thresholds).
