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
- NUMA node count (from `/sys/devices/system/node`).
- `uname` sysname / release / machine.
- `/proc/cmdline` verbatim.

Absent fields render as `(unknown)` — an empty `sched_*` map
renders as `(empty)` and a missing map renders as `(unknown)`.
The distinction matters when you want to know whether a
dimension was inspected but absent, vs failed to populate.

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
  uname_release: 6.14.2 → 6.15.0
  thp_enabled: always [madvise] never → always madvise [never]
  sched_tunables.sched_migration_cost_ns: 500000 → 100000
```

Fields that match in both runs are suppressed by design — this
is a diff, not a snapshot. Missing-on-one-side rendering differs
by layer: top-level `Option<T>` host fields (e.g. `uname_release`,
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

- `thp_enabled` changed between runs → explains latency-sensitive
  regressions that vanish when you pin THP via `transparent_hugepage=`
  on the kernel cmdline.
- `sched_migration_cost_ns` differs → fair scheduler migrated the
  run onto different CPUs, which changes the idle-steal pressure
  on `scx_*` schedulers that depend on it.
- `cmdline` diverges → `isolcpus=` / `nohz_full=` / `mitigations=`
  are all boot-time and change the whole scheduling surface.
  Rebooting the host to match is the correct remediation when
  you need the comparison to hold.
- `uname_release` differs → the kernel itself changed; every
  other host dimension is suspect under cross-kernel comparison.

## Seeing the raw sidecar field

`show-host` reads the live host; the sidecar carries whatever
`show-host` would have captured at sidecar-write time. To see
the sidecar's host block directly:

```sh
jq '.host' path/to/sidecar.ktstr.json
```

The field is emitted on every gauntlet run (populated by
production paths) and `null` on the `test_fixture` path used by
sidecar unit tests — round-trip tests pin the `null`-on-fixture
vs `Some`-on-production split so a silently-missing host block
in a real run surfaces at commit time.
