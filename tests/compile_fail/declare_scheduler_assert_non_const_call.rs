// Pins assert validator: bare single-segment lowercase Call
// (`build_assert()`) is the snake_case free-fn pattern and must
// be rejected. Multi-segment paths like `Assert::default_checks()`
// are still accepted.
use ktstr::assert::Assert;
use ktstr::declare_scheduler;

fn build_assert() -> Assert {
    Assert::NO_OVERRIDES
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    assert = build_assert(),
});

fn main() {}
