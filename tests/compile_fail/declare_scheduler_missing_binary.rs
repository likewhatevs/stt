// Pins the `binary` requirement on `declare_scheduler!`. Omitting
// `binary` previously defaulted the emitted Scheduler to
// `SchedulerSpec::Eevdf` (the kernel-default baseline). Now that
// `name = "eevdf"` is reserved, the omit-binary path always silently
// registered a user scheduler as the EEVDF baseline — wrong for
// every user — so the macro must reject the omission at expand time.
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED, {
    name = "my_sched",
});

fn main() {}
