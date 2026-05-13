// The assert validator's Struct arm recurses into struct-literal
// field values just like the constraints validator. A non-const
// helper call inside `Some(...)` inside a struct literal must be
// rejected at expand time, not at the spread site.
#[allow(unused_imports)]
use ktstr::assert::Assert;
use ktstr::declare_scheduler;

fn build_value() -> u64 {
    50
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    assert = Assert {
        max_gap_ms: Some(build_value()),
        ..Assert::NO_OVERRIDES
    },
});

fn main() {}
