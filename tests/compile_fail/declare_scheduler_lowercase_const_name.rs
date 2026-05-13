use ktstr::declare_scheduler;

declare_scheduler!(my_sched, {
    name = "my_sched",
    binary = "scx_my_sched",
});

fn main() {}
