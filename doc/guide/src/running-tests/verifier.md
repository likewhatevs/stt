# BPF Verifier

The verifier pipeline boots a scheduler in a KVM VM and captures
per-program verifier statistics from the real kernel verifier.

## Design

The verifier pipeline follows stt's two core principles.

**Fidelity without overhead.** The scheduler binary runs inside a VM
on the same kernel the scheduler will run on in production. The
verifier that runs is the real verifier in the real kernel -- no
host-side BPF loading, no version skew between the host kernel's
verifier and the target kernel's verifier.

**Direct access over tooling layers.** No subprocess to bpftool or
veristat. The host reads per-program `verified_insns` directly from
guest memory via `bpf_prog_aux` introspection and applies cycle
collapse to verifier logs instead of truncating.

## Quick start

```sh
# Run the verifier pipeline test
cargo nextest run -E 'test(verifier_)'
```

## How it works

1. **Build** -- the framework builds the scheduler binary.

2. **Boot VM** -- the framework boots a single-CPU VM. The scheduler
   loads its BPF programs via `scx_ops_load!`; the real kernel
   verifier runs against them.

3. **Collect** -- the host reads per-program `verified_insns` from
   `bpf_prog_aux` via guest physical memory introspection. On load
   failure, libbpf prints the verifier log to stderr, which the VM
   captures between `===SCHED_OUTPUT_START===` /
   `===SCHED_OUTPUT_END===` markers.

4. **Format** -- per-program summary lines, then verifier logs with
   cycle collapse applied (pass `raw: true` to skip collapse).

## Output

### Brief (default)

Per-program summary line:

```text
  stt_enqueue                              verified_insns=500
```

Fields: `verified_insns` is the number of instructions the kernel
verifier processed, read from `bpf_prog_aux` via host-side memory
introspection.

On load failure, the scheduler log section shows libbpf's verifier
output with **cycle collapse** applied -- repeating loop iterations
are reduced to the first iteration, an omission marker, and the last
iteration:

```text
--- 8x of the following 10 lines ---
100: (bf) r0 = r1 ; frame1: R0_w=scalar(id=0,umin=0)
101: (bf) r1 = r2 ; frame1: R1_w=scalar(id=1,umin=1)
...
--- 6 identical iterations omitted ---
100: (bf) r0 = r1 ; frame1: R0_w=scalar(id=70,umin=700)
101: (bf) r1 = r2 ; frame1: R1_w=scalar(id=71,umin=701)
...
--- end repeat ---
```

### Raw (`raw: true`)

Full raw verifier log without cycle collapse. Use for debugging
verification failures where the exact register state at each iteration
matters.

### A/B diff

Boots two VMs -- one for each scheduler binary -- and compares
`verified_insns` per program:

```text
  program                                           A          B      delta
  ------------------------------------------------------------------------
  stt_enqueue                                     500        450        +50
  stt_dispatch                                   1200       1150        +50
```

## Cycle collapse algorithm

The kernel verifier unrolls loops by re-verifying each instruction
with updated register states. A bounded loop of 8 instructions
verified 100 times produces 800 near-identical lines -- differing
only in register-state annotations. Naive truncation loses context.
Cycle collapse preserves structure: the first iteration shows what
the loop does, the last shows the final state, and a count tells you
how many iterations were elided.

The algorithm normalizes lines by stripping variable annotations,
then detects repeating blocks:

1. **Normalize** -- strip `; frame1: R0_w=...` annotations, standalone
   register dumps (`3041: R0=scalar()`), and inline branch-target state
   after `goto pc+N`. Source comments (`; for (int j = 0; ...)`) are
   preserved as cycle anchors.

2. **Detect** -- find the most frequent normalized line (the "anchor"),
   compute gaps between anchor occurrences to determine the cycle
   period, then verify consecutive blocks match after normalization.
   Minimum period: 5 lines. Minimum repetitions: 3.

3. **Collapse** -- replace the cycle with the first iteration, an
   omission count, and the last iteration. Run iteratively (up to 5
   passes) to handle nested loops.

## stt-sched test flags

stt-sched supports these flags to exercise the verifier pipeline:

**`--fail-verify`** -- sets a `.rodata` variable before
`scx_ops_load!`, enabling a code path the BPF verifier rejects.
On failure, libbpf prints the verifier log to stderr.

**`--verify-loop`** -- sets a `.rodata` variable that enables an
unrolled loop followed by `while(1)` in `stt_dispatch`. The verifier
rejects the infinite loop and libbpf prints the full instruction
trace to stderr, exercising cycle collapse.
