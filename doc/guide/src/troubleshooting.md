# Troubleshooting

## Build errors

### clang not found

```text
error: failed to run custom build command for `ktstr`
  ...
  clang: No such file or directory
```

The BPF skeleton build (`libbpf-cargo`) invokes clang to compile
`.bpf.c` sources. Install clang:

- Debian/Ubuntu: `sudo apt install clang`
- Fedora: `sudo dnf install clang`

### pkg-config not found

```text
error: failed to run custom build command for `libbpf-sys`
  ...
  pkg-config: command not found
```

libbpf-sys uses pkg-config during its vendored build. Install it:

- Debian/Ubuntu: `sudo apt install pkg-config`
- Fedora: `sudo dnf install pkgconf`

### autotools errors (autoconf, autopoint, aclocal)

```text
autoreconf: command not found
aclocal: command not found
autopoint: command not found
```

The vendored libbpf-sys build compiles bundled libelf and zlib from
source using autotools. These libraries are not system dependencies
-- they ship with libbpf-sys -- but the autotools toolchain is
needed to build them. Install:

- Debian/Ubuntu: `sudo apt install autoconf autopoint flex bison gawk`
- Fedora: `sudo dnf install autoconf gettext-devel flex bison gawk`

### make or gcc not found

```text
busybox build requires 'make' — install build-essential (Debian/Ubuntu) or base-devel (Fedora/Arch)
busybox build requires 'gcc' — install build-essential (Debian/Ubuntu) or base-devel (Fedora/Arch)
```

The build script compiles busybox from source for guest shell mode.
This requires make and gcc.

- Debian/Ubuntu: `sudo apt install make gcc`
- Fedora: `sudo dnf install make gcc`

### BTF errors

```text
no BTF source found. Set KTSTR_KERNEL to a kernel build directory,
or ensure /sys/kernel/btf/vmlinux exists.
```

build.rs generates `vmlinux.h` from kernel BTF data. It searches
the kernel discovery chain (`KTSTR_KERNEL`, `./linux`, `../linux`,
installed kernel) for a `vmlinux` file, falling back to
`/sys/kernel/btf/vmlinux`. Most distros ship
`/sys/kernel/btf/vmlinux` with CONFIG_DEBUG_INFO_BTF enabled.

**Fixes:**

- Verify BTF is available: `ls /sys/kernel/btf/vmlinux`
- If missing, set `KTSTR_KERNEL` to a kernel build directory that
  contains a `vmlinux` with BTF:
  `export KTSTR_KERNEL=/path/to/linux`
- Build a kernel with `CONFIG_DEBUG_INFO_BTF=y`.
- Some minimal/cloud kernels strip BTF. Use a distro kernel or
  build your own.

### busybox download failure

```text
failed to obtain busybox source.
  tarball (https://github.com/mirror/busybox/archive/refs/tags/1_36_1.tar.gz): download: ...
  git clone (https://github.com/mirror/busybox.git): ...
  Check network connectivity. First build requires internet access.
```

build.rs downloads busybox source on first build (tarball first,
git clone fallback). Subsequent builds use the cached binary in
`$OUT_DIR`.

**Fixes:**

- Verify network connectivity to github.com.
- If behind a proxy, set `HTTP_PROXY` / `HTTPS_PROXY`.
- After a successful first build, no network access is needed
  unless `cargo clean` removes the cached binary.

## /dev/kvm not accessible

The host-side pre-flight emits one of the following, depending on
whether the device node is missing or merely unreadable:

```text
/dev/kvm not found. KVM requires:
  - Linux kernel with KVM support (CONFIG_KVM)
  - Access to /dev/kvm (check permissions or add user to 'kvm' group)
  - Hardware virtualization enabled in BIOS (VT-x/AMD-V)
```

```text
/dev/kvm: permission denied. Add your user to the 'kvm' group:
  sudo usermod -aG kvm $USER
  then log out and back in.
```

ktstr boots Linux kernels in KVM virtual machines. The host must have
KVM enabled and the user must have read+write access to `/dev/kvm`.

**Diagnose:**

- Check the device exists and inspect its permissions and owning group:
  `ls -l /dev/kvm`. Typical output: `crw-rw---- 1 root kvm 10, 232 ...`.
- Confirm the `kvm` group exists and see its members:
  `getent group kvm`.

**Fixes:**

- Load the KVM module: `modprobe kvm_intel` or `modprobe kvm_amd`.
- Follow the group-membership hint in the error text above (log out
  and back in afterward for the group change to take effect).
- On cloud VMs (GCP, AWS, Azure) or nested hypervisors, nested
  virtualization is typically off by default. Enable it per the
  provider's instructions (e.g. GCP `--enable-nested-virtualization`,
  AWS metal/`.metal` instance types, Azure Dv3/Ev3+ with nested virt).
- In CI, ensure the runner has KVM access (e.g. `runs-on: [self-hosted, kvm]`).

## No kernel found

```text
no kernel found
  hint: run `cargo ktstr kernel build` to download and build the latest stable kernel
  hint: or set KTSTR_KERNEL=/path/to/linux
  hint: or set KTSTR_TEST_KERNEL=/path/to/bzImage
```

On aarch64 the hint says `Image` instead of `bzImage`.

`ktstr shell` and `cargo ktstr shell` auto-download the latest
stable kernel when no `--kernel` is specified and no kernel is found
via the discovery chain. See
[Kernel auto-download failures](#kernel-auto-download-failures) for
download-specific errors.

ktstr needs a bootable Linux kernel image (`bzImage` on x86_64,
`Image` on aarch64). See
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
`FutexFanOut`, `SchBench`) require `num_workers` divisible by their
group size. `WorkType::worker_group_size()` returns the divisor.

**Fixes:**

- Set `CgroupDef::workers(n)` to a value divisible by the work
  type's group size (2 for pipe/futex pairs, `fan_out + 1` for
  FutexFanOut and SchBench).
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
- See [`cargo ktstr kernel clean`](running-tests/cargo-ktstr.md#kernel-clean)
  for all cleanup options.

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
reasons. See [`cargo ktstr kernel list`](running-tests/cargo-ktstr.md#kernel-list)
for the full listing output.

## Kernel auto-download failures

```text
ktstr: no kernel found, downloading latest stable
fetch https://www.kernel.org/releases.json: <error>
```

ktstr auto-downloads a kernel when no `--kernel` is specified and no
kernel is found via the discovery chain (see
[Kernel discovery](getting-started.md#kernel-discovery)). The same
download path runs when `--kernel` specifies a version (e.g.
`--kernel 6.14.2`) that is not in the cache. The CLI label varies:
`ktstr:` for the standalone binary, `cargo ktstr:` for the cargo
subcommand.

The `<error>` above is the underlying reqwest error (DNS resolution,
connection refused, timeout, TLS handshake failure).

```text
fetch https://www.kernel.org/releases.json: HTTP 503
```

kernel.org returned a non-success status code.

```text
no stable kernel with patch >= 8 found in releases.json
```

ktstr requires a stable or longterm release with patch version >= 8
to avoid brand-new major versions that may have build issues. This
error means releases.json contained no qualifying version.

```text
download https://cdn.kernel.org/.../linux-6.14.10.tar.xz: <error>
```

Network failure during tarball download (same causes as above).

```text
extract tarball: <error>
```

Tarball extraction failed. Common causes: disk full, insufficient
permissions on the temp directory, or a truncated download.

```text
kernel built but cache store failed — cannot return image from temporary directory
```

The kernel built successfully but could not be stored in the cache.
Check disk space and permissions on the cache directory.

For version-specific download errors (HTTP 404, HTML responses), see
[Kernel download failures](#kernel-download-failures).

**Fixes:**

- Verify network connectivity: `curl -sI https://www.kernel.org/releases.json`
- Check DNS resolution for kernel.org and cdn.kernel.org.
- Check disk space — the download, extraction, and build require
  significant disk space.
- If behind a proxy, set `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`
  (reqwest respects these environment variables).
- Override the cache directory via `KTSTR_CACHE_DIR` if the default
  location has insufficient space or permissions.
- Pre-download a kernel explicitly: `cargo ktstr kernel build 6.14.10`
  to isolate whether the failure is in version resolution or download.

## Kernel download failures

These errors occur when `cargo ktstr kernel build` or `--kernel`
specifies an explicit version. For network and extraction errors
during auto-download, see
[Kernel auto-download failures](#kernel-auto-download-failures).

```text
version 6.14.22 not found. latest 6.14.x: 6.14.10
```

The requested version does not exist on kernel.org. When a version in
the same major.minor series is available in releases.json, the error
suggests it.

```text
version 5.4.99 not found
```

When the series is EOL or not in releases.json, only the "not found"
message appears (no suggestion).

```text
RC tarball not found: https://git.kernel.org/torvalds/t/linux-6.15-rc3.tar.gz
  RC releases are removed from git.kernel.org after the stable version ships.
```

RC tarballs are removed from git.kernel.org after the stable version
ships. Use `--git` with a git.kernel.org URL to clone the tag instead.

```text
download ...: server returned HTML instead of tarball (URL may be invalid)
```

Some CDN error pages return HTTP 200 with `text/html` content type.
The download rejects these responses.

**Fixes:**

- Check the suggested version in the error message.
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
- **No CAP_SYS_NICE or rtprio**: performance-mode tests require
  `CAP_SYS_NICE` or an rtprio limit for RT scheduling, and enough
  host CPUs for exclusive LLC reservation. Pass `--no-perf-mode`
  (or set `KTSTR_NO_PERF_MODE=1`) to disable all performance mode
  features. Tests with `performance_mode=true` are skipped entirely
  under `--no-perf-mode`.
- **Debug thresholds**: CI often runs debug builds. Debug builds use
  relaxed thresholds (3000ms gap, 35% spread) but may still hit
  limits on slow runners. See
  [default thresholds](concepts/verification.md#default-thresholds).
