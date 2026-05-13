// Pins assert validator catchall arm: a closure expression
// falls through to the catchall rejection with the base
// const-eligibility message (no call-specific hint, no
// block-specific guidance).
#[allow(unused_imports)]
use ktstr::assert::Assert;
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    assert = || Assert::NO_OVERRIDES,
});

fn main() {}
