// The `call_func_is_single_segment_lowercase` heuristic must
// unwrap `Expr::Paren` so a deliberately parenthesized bare ident
// (`(build_assert)()`) cannot bypass the bare-helper rejection.
use ktstr::assert::Assert;
use ktstr::declare_scheduler;

fn build_assert() -> Assert {
    Assert::NO_OVERRIDES
}

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
    binary = "scx_my_sched",
    assert = (build_assert)(),
});

fn main() {}
