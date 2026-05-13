// A visibility prefix followed by no const_name ident (just a `,`
// separator) is rejected at parse time with an "expected identifier"
// diagnostic. Catches a parser regression that silently accepts an
// empty const-name slot.
use ktstr::declare_scheduler;

declare_scheduler!(pub , {
    name = "my_sched",
    binary = "scx_my_sched",
});

fn main() {}
