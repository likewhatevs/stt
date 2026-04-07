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

stt needs a bootable Linux kernel image (bzImage). It searches:

1. `STT_TEST_KERNEL` environment variable.
2. `./linux/arch/x86/boot/bzImage` (workspace-local build).
3. `../linux/arch/x86/boot/bzImage` (sibling directory).
4. `/lib/modules/$(uname -r)/vmlinuz` (installed kernel).
5. `/boot/vmlinuz-$(uname -r)` (installed kernel).
6. `/boot/vmlinuz` (unversioned symlink).

**Fixes:**

- Build a kernel with `stt build-kernel ~/linux` (see
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
- Use `SchedulerSpec::Path` for an explicit path.
- Use `-p scx_mitosis` with `cargo stt vm` to build automatically.

## Scheduler died

```text
scheduler died between steps
```

The scheduler process exited while the scenario was running. This
is usually a crash. The error output includes `dmesg` lines from
the VM.

**Next steps:**

- Check the `dmesg` output in the test failure for a BPF error or
  kernel oops.
- Rerun with `--auto-repro` to capture the crash path with BPF
  probes. See [Auto-Repro](running-tests/auto-repro.md).
- Run with a longer duration and specific flags to narrow the
  reproducer.

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
