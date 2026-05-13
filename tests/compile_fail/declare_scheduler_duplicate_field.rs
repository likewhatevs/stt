use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "first",
    binary = "scx_my_sched",
    name = "second",
});

fn main() {}
