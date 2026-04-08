# Single Scenario

## Running a specific test

```sh
cargo nextest run -E 'test(sched_basic_proportional)'
```

## Running with verbose output

```sh
RUST_BACKTRACE=1 cargo nextest run -E 'test(sched_basic_proportional)'
```

## Investigating failures

Run one test with verbose output to see scheduler logs and kernel
console:

```sh
RUST_BACKTRACE=1 cargo nextest run -E 'test(cover_cgroup_cpuset_crossllc_race)'
```

## VM topology

Each `#[stt_test]` declares its topology via macro attributes:

```rust,ignore
#[stt_test(sockets = 2, cores = 4, threads = 2)]
```

The test framework boots a VM with the specified topology
automatically.
