# Architecture Overview

stt has three execution domains:

1. **Host process** -- the test binary running on the host. Manages
   [VM lifecycle](architecture/vmm.md), monitors guest memory, evaluates
   results.

2. **Guest process** -- the same test binary running inside the VM.
   Creates cgroups, forks [workers](architecture/workers.md), runs
   scenarios, writes results to the serial console.

3. **[Monitor](architecture/monitor.md) thread** -- runs on the host
   while the guest executes. Reads guest VM memory directly to observe
   scheduler state without instrumenting it.

## Execution flow

```text
Host                          Guest
----                          -----
test binary                   
  |                           
  +-- build initramfs         
  |   (test binary + busybox  
  |    + optional scheduler)  
  |                           
  +-- boot KVM VM             
  |                           test binary (ctor dispatch)
  |                             |
  +-- start monitor thread      +-- start scheduler (if any)
  |   (reads guest memory)      +-- create cgroups
  |                             +-- fork workers
  |                             +-- move workers to cgroups
  |                             +-- signal workers to start
  |                             +-- poll scheduler liveness
  |                             +-- stop workers, collect reports
  |                             +-- evaluate results
  |                             +-- write result to serial
  |                           
  +-- read result from serial 
  +-- evaluate monitor data   
  +-- report pass/fail        
```

## Key design decisions

**Same binary, two roles.** The test binary serves as both host
controller and guest test runner. The initramfs embeds the binary.
Guest-side execution is triggered by `ctor` early dispatch (before
`main()`).

**Forked workers, not threads.** Workers are `fork()`ed processes
because cgroups operate on PIDs. Each worker must be a separate
process to be placed in its own cgroup.

**Host-side monitoring.** The monitor reads guest memory via KVM,
avoiding BPF instrumentation of the scheduler under test. This
eliminates observer effects on scheduling decisions.

**Typed flag declarations.** Flags use static references instead of
string matching, enabling compile-time dependency resolution.
