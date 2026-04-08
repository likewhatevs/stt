# BPF Verifier

`stt verifier` boots a scheduler in a KVM VM and captures
per-program verifier statistics from the real kernel verifier.

## Design

The verifier subcommand follows stt's two core principles.

**Fidelity without overhead.** The scheduler binary runs inside a VM
on the same kernel the scheduler will run on in production. The
`--dump-verifier` mode uses the same `scx_ops_open!` / `scx_ops_load!`
macros as the normal startup path. When load fails, libbpf prints
the verifier's instruction traces to stderr. The verifier that runs
is the real verifier
in the real kernel -- no host-side BPF loading, no version skew between
the host kernel's verifier and the target kernel's verifier.

**Direct access over tooling layers.** No subprocess to bpftool or
veristat. The scheduler binary emits structured output directly
(`STT_VERIFIER_PROG`, `STT_VERIFIER_LOG`, `STT_VERIFIER_DONE`); the
host parses it. Cycle collapse reduces repetitive loop unrolling
instead of truncating.

## Quick start

```sh
# Run the verifier pipeline test
cargo nextest run -E 'test(verifier_)'
```

## How it works

1. **Build** -- `cargo build -p <package>` produces the scheduler
   binary.

2. **Boot VM** -- a single-CPU VM boots with the scheduler binary
   passed as `--dump-verifier`. The scheduler opens the BPF object,
   records pre-load instruction counts, and calls `scx_ops_load!`.
   On failure, libbpf prints the verifier log to stderr.

3. **Capture** -- the scheduler writes structured lines to stdout,
   which the guest init script forwards to COM2 after exit:
   - `STT_VERIFIER_PROG <name> insn_cnt=<N>` -- start of a program
   - `STT_VERIFIER_LOG <name> <line>` -- one verifier log line
   - `STT_VERIFIER_DONE` -- all programs loaded

4. **Parse** -- the host parses the structured output into per-program
   stats (name, instruction count, verifier log text).

5. **Format** -- per-program summary lines, then verifier logs with
   cycle collapse applied (unless `RUST_BACKTRACE=1` or `full`).

## Output

### Brief (default)

Per-program summary line:

```text
  stt_enqueue                              insns=42    processed=500     states=30/100  time=42us  stack=32+0
```

Fields: program size (`insns`), instructions processed during
verification (`processed`), peak/total verifier states, verification
time, and stack depth per subprogram.

After the summary, the verifier log is printed with **cycle collapse**
applied -- repeating loop iterations are reduced to the first iteration,
an omission marker, and the last iteration:

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

### Raw (`RUST_BACKTRACE=1`)

Full raw verifier log without cycle collapse. Use for debugging
verification failures where the exact register state at each iteration
matters.

### A/B diff (`--diff`)

Boots two VMs -- one for each scheduler package -- and compares
processed instruction counts per program:

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

## `--fail-verify` flag

`stt-sched --fail-verify` sets `fail_verify=1` in .rodata before
`scx_ops_load!`, enabling a code path the BPF verifier rejects.
This produces a real verification failure for negative testing.

## `--dump-verifier` flag

`stt-sched --dump-verifier` opens the BPF object, records pre-load
instruction counts, and calls `scx_ops_load!`. On success, it emits
structured verifier output (`STT_VERIFIER_PROG`, `STT_VERIFIER_DONE`).
On failure (e.g. `--fail-verify` or `--verify-loop`), libbpf prints
the verifier's instruction traces to stderr, which the VM captures
as the scheduler log.
