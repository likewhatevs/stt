// Pins assert validator: a method chain whose receiver is a
// non-const bare helper call must be rejected. The recursion
// into MethodCall.receiver descends to the inner Call and
// catches the snake_case free-fn pattern.
use ktstr::assert::Assert;
use ktstr::declare_scheduler;

fn build_base() -> Assert {
    Assert::NO_OVERRIDES
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    assert = build_base().max_gap_ms(5000),
});

fn main() {}
