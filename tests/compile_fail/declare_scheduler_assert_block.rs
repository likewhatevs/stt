// Pins assert validator block arm: block expressions get the
// tailored drop-braces / const-binding guidance. Field name
// in the diagnostic header is "assert" (not "constraints").
#[allow(unused_imports)]
use ktstr::assert::Assert;
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    assert = { Assert::NO_OVERRIDES },
});

fn main() {}
