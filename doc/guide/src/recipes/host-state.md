# Capture and Compare Host State

When a gauntlet run passes on one machine and fails on another —
or passes on Monday and fails on Wednesday — the first thing to
check is whether the host itself changed. `cargo ktstr show-host`
captures a snapshot of the kernel, CPU, memory, scheduler
tunables, and kernel cmdline; `cargo ktstr stats compare`
surfaces the changes between two sidecars in a host-delta
section of its output so you can see what moved.

## Capture: `show-host`

```sh
cargo ktstr show-host
```

Prints a `key: value` report covering:

- CPU model + vendor (first `/proc/cpuinfo` entry).
- Total memory, hugepages total / free, hugepage size (from
  `/proc/meminfo`).
- Transparent hugepage policy (`thp_enabled`, `thp_defrag`) with
  the bracketed selection preserved verbatim.
- Every `/proc/sys/kernel/sched_*` tunable, one entry per line.
- NUMA node count (from CPU→node mapping; memory-only nodes
  without CPUs are not counted).
- `kernel_name` / `kernel_release` / `arch` (from the `uname()`
  syscall).
- `/proc/cmdline` verbatim.

Absent fields render as `(unknown)` — an empty `sched_*` map
renders as `(empty)` and a missing map renders as `(unknown)`.
The distinction matters when you want to know whether a
dimension was inspected but absent, vs failed to populate.

Sidecars written before the `uname_sysname` / `uname_release` /
`uname_machine` → `kernel_name` / `kernel_release` / `arch`
rename render the renamed fields as `(unknown)` in `show-host`
and in `stats compare`'s host-delta section, and re-running the
test against the current binary regenerates the sidecar with
the new field names populated. Mechanically: the old sidecar
still deserializes cleanly (deserialization is forward-compatible
in the "does-not-error" sense), but the renamed fields land as
`None` on the new struct because the old-name data does not
migrate to the new field names.

This output is human-oriented. For programmatic access, read
the `host` field of any sidecar JSON (same schema, identical
values — `show-host` prints the live snapshot the sidecar
writer would attach to a fresh test run).

## Compare: `stats compare`

```sh
cargo ktstr stats compare RUN_A RUN_B
```

Picks the first sidecar with `Some(host)` from each run,
collects every host field that differs, and prints a side-by-side
delta unconditionally as part of the compare output (there is no
opt-in flag — the host-delta section appears whenever the two
sidecars disagree on a host field):

```
host delta ('RUN_A' → 'RUN_B'):
  kernel_release: 6.14.2 → 6.15.0
  thp_enabled: always [madvise] never → always madvise [never]
  sched_tunables.sched_migration_cost_ns: 500000 → 100000
```

Fields that match in both runs are suppressed by design — this
is a diff, not a snapshot. Missing-on-one-side rendering differs
by layer: top-level `Option<T>` host fields (e.g. `kernel_release`,
`thp_enabled`, the whole `sched_tunables` map) render with
`(unknown)` on the None side so a regression in the capture
pipeline surfaces instead of silently hiding. Per-key diffs
inside the `sched_tunables` map use `(absent)` instead, to
distinguish "the map was captured and this key is not in it"
from "the whole map was unknown at capture time".

### CI integration

Gauntlet runs emit the host block automatically in every
sidecar. To diff the host state across two CI runs, point
`stats compare` at the sidecar directories of the two runs —
host delta appears automatically in the compare output when
any host field differs. A CI job can:

1. Run the gauntlet on the candidate commit and the baseline.
2. Run `stats compare` between the two runs and inspect the
   host-delta section of its output.
3. Fail (or annotate the PR) if any host dimension changed —
   an unchanged host set is the precondition for a clean A/B
   of scheduler behavior.

## Typical hits

Each bullet names the `show-host` field that carries the signal so
you can `cargo ktstr show-host | grep <field>` directly, or pluck
the same key out of a sidecar via `jq '.host.<field>'`.

- `thp_enabled` (and its companion `thp_defrag`) changed between
  runs → explains latency-sensitive regressions that vanish when
  you pin THP via `transparent_hugepage=` on the kernel cmdline.
  The bracketed selection inside the value is the active setting;
  compare the bracket position, not just the full string.
- `sched_tunables.sched_migration_cost_ns` differs (look for it
  inside the `sched_*` block printed by `show-host`) → fair
  scheduler migrated the run onto different CPUs, which changes
  the idle-steal pressure on `scx_*` schedulers that depend on
  it. Other `sched_tunables.*` keys
  (`sched_wakeup_granularity_ns`, `sched_min_granularity_ns`,
  `sched_latency_ns`, `sched_rt_runtime_us`, etc.) have the same
  shape — the full set is whatever `/proc/sys/kernel/sched_*`
  lists at capture time. Note: the examples above are CFS-era
  tunables; several of them (`sched_wakeup_granularity_ns`,
  `sched_min_granularity_ns`, `sched_latency_ns`) were dropped
  when CFS was replaced by EEVDF in Linux 6.6+, so a run on an
  EEVDF kernel will simply not have those keys in the map —
  their absence is a kernel-version fact, not a capture failure.
  EEVDF's own latency-floor knob is exposed as
  `sched_tunables.sched_base_slice_ns` on 6.6+ kernels (the
  replacement for the dropped CFS latency / granularity triple);
  check for its presence to confirm an EEVDF-era capture.
  What you get in practice is whatever `/proc/sys/kernel/sched_*`
  exposes on the running kernel.
- `cmdline` diverges → `isolcpus=` / `nohz_full=` / `mitigations=`
  / `transparent_hugepage=` / `numa_balancing=` are all boot-time
  and change the whole scheduling surface. Rebooting the host to
  match is the correct remediation when you need the comparison
  to hold.
- `kernel_release` differs (also check the companion
  `kernel_name` and `arch` fields) → the kernel itself changed;
  every other host dimension is suspect under cross-kernel
  comparison. A `kernel_name` change (`uname -s` reporting a
  different OS family — `Linux` vs `FreeBSD`, say) is a harder
  stop than a same-family version bump and usually means the
  two sidecars were produced on entirely different systems.
- `hugepages_total` / `hugepages_free` / `hugepages_size_kb`
  deltas → benchmark throughput that depends on 2 MiB pages
  (performance_mode tests) flips outcome when the pool shrinks
  or the page size changes. All three are reported by `show-host`
  in the meminfo-derived block.
- `numa_nodes` differs → cpusets and cross-node migration signals
  only make sense within the CPU→node mapping captured at
  sidecar-write time; a host reconfigured to expose or hide
  nodes changes what `cpus_used` and `numa_pages` mean across
  the two runs. See the
  [capture caveat](#capture-show-host) — `numa_nodes` counts
  only nodes that host at least one CPU (memory-only nodes are
  not counted), so a delta here can reflect either a hardware /
  firmware change or a topology reconfiguration that left the
  memory-only nodes untouched.
- CPU-level skew (`cpu_model` / `cpu_vendor`) → microarchitectural
  differences affect cache-sensitive benchmarks. Always inspect
  alongside `cmdline` because a different CPU usually comes with
  a different bootloader.

## Seeing the raw sidecar field

`show-host` reads the live host; the sidecar carries whatever
`show-host` would have captured at sidecar-write time. To see
the sidecar's host block directly:

```sh
jq '.host' path/to/sidecar.ktstr.json
```

The field is emitted on every gauntlet run.
