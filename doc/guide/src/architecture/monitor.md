# Monitor

The monitor observes scheduler state from the host side by reading
guest VM memory directly. It does not instrument the guest kernel or
the scheduler under test.

## What it reads

The monitor resolves kernel structure offsets via BTF (BPF Type Format)
from the guest kernel. It reads per-CPU runqueue structures to extract:

- `nr_running` -- number of runnable tasks on each CPU
- `scx_nr_running` -- tasks managed by the sched_ext scheduler
- `rq_clock` -- runqueue clock value
- `local_dsq_depth` -- scx local dispatch queue depth
- `scx_flags` -- sched_ext flags for each CPU
- scx event counters (fallback, keep-last, offline dispatch,
  skip-exiting, skip-migration-disabled)

## Sampling

The monitor takes periodic snapshots (`MonitorSample`) of all per-CPU
state. Each sample captures a point-in-time view of every CPU.

`MonitorSummary` aggregates samples into event deltas and overall
statistics.

## Threshold evaluation

`MonitorThresholds` defines pass/fail conditions:

```rust
pub struct MonitorThresholds {
    pub max_imbalance_ratio: f64,
    pub max_local_dsq_depth: u32,
    pub fail_on_stall: bool,
    pub sustained_samples: usize,
    pub max_fallback_rate: f64,
    pub max_keep_last_rate: f64,
}
```

A violation must persist for `sustained_samples` consecutive samples
before triggering a failure. This filters transient spikes from cpuset
transitions and cgroup creation/destruction.

### Stall detection

A stall is detected when a CPU's `rq_clock` does not advance between
consecutive samples. Two exemptions prevent false positives:

- **Idle CPUs**: when `nr_running == 0` in both the current and previous
  sample, the CPU has no runnable tasks. The kernel stops the tick
  (NOHZ) on idle CPUs, so `rq_clock` legitimately does not advance.
  These CPUs are excluded from stall checks.

- **Sustained window**: stall detection uses per-CPU consecutive
  counters and the `sustained_samples` threshold, matching how
  imbalance and DSQ depth checks work. A single stuck sample does
  not trigger failure -- the stall must persist for `sustained_samples`
  consecutive samples on the same CPU.

## Uninitialized memory detection

Before the guest kernel initializes per-CPU structures, monitor reads
return uninitialized data. Two layers handle this:

- **Summary computation** (`MonitorSummary::from_samples`): skips
  individual samples where any CPU's `local_dsq_depth` exceeds
  `DSQ_PLAUSIBILITY_CEILING` (10,000) via `sample_looks_valid()`.

- **Threshold evaluation** (`MonitorThresholds::evaluate`): checks all
  samples globally via `data_looks_valid()`. If all `rq_clock` values
  are identical across every CPU and sample, or any sample exceeds the
  plausibility ceiling, the entire report is passed as "not yet
  initialized" — no per-threshold checks run.

## BPF map introspection

The monitor module also provides host-side BPF map discovery and
read/write access via `bpf_map::BpfMapAccessor`. The host reads and
writes guest BPF maps directly through the physical memory mapping
— no guest cooperation or BPF syscalls are needed.

### GuestMem

`GuestMem` wraps a host pointer to guest physical address 0 and
provides bounds-checked volatile reads and writes for scalar types
(u8/u16/u32/u64). Byte-slice reads (`read_bytes`) use
`copy_nonoverlapping`. It also implements x86-64 page table walks
(`translate_kva`) for both 4-level and 5-level paging.

Scalar accesses use volatile semantics because the guest kernel
modifies memory concurrently.

### GuestKernel

`GuestKernel` builds on `GuestMem` by adding kernel symbol
resolution and address translation. It parses the vmlinux ELF
symbol table at construction and resolves paging configuration
(PAGE_OFFSET, CR3, 4-level vs 5-level) from guest memory.
Subsequent reads use cached state.

Three address translation modes are supported:

- **Text/data/bss**: `kva - __START_KERNEL_map`. For statically-linked
  kernel variables (`read_symbol_*`, `write_symbol_*`).
- **Direct mapping**: `kva - PAGE_OFFSET`. For SLAB allocations,
  per-CPU data, physically contiguous memory (`read_direct_*`).
- **Vmalloc/vmap**: Page table walk via CR3. For BPF maps, vmalloc'd
  memory, module text (`read_kva_*`, `write_kva_*`).

### BpfMapAccessor

`BpfMapAccessor` resolves BTF offsets for BPF map kernel structures
(`struct bpf_map`, `struct bpf_array`, `struct xa_node`, `struct idr`)
and provides map discovery and value read/write. It borrows a
`GuestKernel` for address translation.

`BpfMapAccessorOwned` is a convenience wrapper that owns the
`GuestKernel` internally. Use `BpfMapAccessor::from_guest_kernel`
when you already have a `GuestKernel`; use `BpfMapAccessorOwned::new`
when you want a self-contained accessor.

Map discovery walks the kernel's `map_idr` xarray:

1. Read `map_idr` (BSS symbol, text mapping translation)
2. Walk xa_node tree (SLAB-allocated, direct mapping translation)
3. Read `struct bpf_map` fields (vmalloc'd, page table walk)

`find_map` searches by **name suffix** (e.g. `".bss"` matches
`"mitosis.bss"`). Only `BPF_MAP_TYPE_ARRAY` maps are returned.
Use `maps()` to enumerate all map types without filtering.

Value access for `BPF_MAP_TYPE_ARRAY` maps reads/writes the inline
`bpf_array.value` flex array at the BTF-resolved offset. The value
region is vmalloc'd, so each byte access goes through the page table
walker to handle page boundaries.

### Typed field access

When a map has BTF metadata (`btf_kva != 0`), `resolve_value_layout`
reads the guest's `struct btf` and its `data` blob, parses it with
`btf_rs`, and resolves the value struct's fields. This enables
`read_field` / `write_field` with type-checked `BpfValue` variants.

### Usage example

Find a scheduler's `.bss` map and write a crash variable:

```rust,ignore
let accessor = BpfMapAccessor::from_guest_kernel(&kernel, vmlinux)?;
let bss = accessor.find_map(".bss").expect(".bss map not found");
accessor.write_value_u32(&bss, crash_offset, 1);
```

### Prerequisites

- **vmlinux**: Required for ELF symbols and BTF. Must match the guest
  kernel.
- **nokaslr**: Required on the guest command line. Text mapping
  translation assumes `phys_base = 0`.

## Probe pipeline

The probe pipeline captures function arguments and struct fields during
auto-repro. It operates inside the guest VM (not from the host), using
two BPF skeletons that share maps.

### Architecture

```text
crash stack -> extract functions -> BTF resolve -> load skeletons -> poll
                                                         |
                                    kprobe skeleton      |     fentry skeleton
                                    (kernel funcs)       |     (BPF callbacks)
                                         |               |          |
                                         v               v          v
                                    func_meta_map  <--shared-->  probe_data
                                                         |
                                              trigger fires (ring buffer)
                                                         |
                                              read probe_data entries
                                                         |
                                              stitch by tptr/tid
                                                         |
                                              format with field decoders
```

### Kprobe skeleton (`probe.bpf.c`)

Attaches to kernel functions via `attach_kprobe`. The BPF handler:
1. Gets the function IP via `bpf_get_func_ip`
2. Looks up `func_meta` from `func_meta_map` (keyed by IP)
3. Captures 6 raw args from `pt_regs`
4. Dereferences struct fields via BTF-resolved offsets
5. Reads char * string params if configured
6. Stores result in `probe_data` (keyed by `(func_ip, tid)`)

A separate trigger kprobe fires on `scx_disable_workfn` and sends
an `EVENT_TRIGGER` via ring buffer with the current task pointer
and kernel stack.

### Fentry skeleton (`fentry_probe.bpf.c`)

Attaches to BPF struct_ops callbacks via `set_attach_target`. Loaded
in batches of 4 to reduce verifier passes. Shares `probe_data` and
`func_meta_map` with the kprobe skeleton via `reuse_fd`.

For struct_ops programs, fentry receives one arg: `ctx[0]` is a void
pointer to the real callback arguments packed contiguously. The handler
dereferences through `ctx[0]` to read up to 6 args.

Uses sentinel IPs (`func_idx | (1<<63)`) in `func_meta_map` to
distinguish fentry events from kprobe events.

### BTF resolution

Two BTF sources:

- **vmlinux BTF** (`btf-rs`): resolves kernel struct offsets for types
  in `STRUCT_FIELDS` (task_struct, rq, scx_dispatch_q, scx_exit_info,
  scx_init_task_args, scx_cgroup_init_args). Handles chained pointer
  dereferences (e.g. `->cpus_ptr->bits[0]`).

- **Program BTF** (`libbpf-rs`): resolves BPF-local struct offsets for
  types not in `STRUCT_FIELDS` (e.g. scheduler-defined `task_ctx`).
  Auto-discovers scalar, enum, and cpumask pointer fields.

Callback signatures are resolved by:
1. `____name` inner function in program BTF (typed params)
2. `sched_ext_ops` member in vmlinux BTF (fallback)
3. Wrapper function (void *ctx, no useful params)

### Field decoding

The output formatter decodes field values based on their key name:
- `dsq_id` -> `GLOBAL`, `LOCAL`, `LOCAL_ON|{cpu}`, `DSQ(0x{hex})`
- `cpumask_0..3` -> coalesced `cpus_ptr 0xf(0-3,64)`
- `enq_flags` -> `WAKEUP|HEAD|PREEMPT`
- `exit_kind` -> `ERROR`, `ERROR_BPF`, `ERROR_STALL`, etc.
- `scx_flags` -> `QUEUED|ENABLED`
- `sticky_cpu` -> `-1` for 0xffffffff

### Event stitching

After the trigger fires (or stop is signaled), all `probe_data`
entries are read, matched to functions by IP, then filtered to a
single task's scheduling journey:

1. Find the task_struct pointer from the trigger event's args[0]
2. For functions with a task_struct parameter: keep events where
   `args[param_idx] == tptr`
3. For functions without task_struct: keep events where `tid == trigger_tid`

Events are sorted by timestamp for chronological output.
