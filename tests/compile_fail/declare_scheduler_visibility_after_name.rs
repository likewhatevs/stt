// Visibility prefix must precede the const name. A trailing
// visibility token after the name is rejected at parse time
// (syn::Visibility::parse consumed Inherited for the leading
// position; the unexpected `pub` token between the ident and the
// required `,` fails parsing).
use ktstr::declare_scheduler;

declare_scheduler!(MY_SCHED pub, {
    name = "my_sched",
    binary = "scx_my_sched",
});

fn main() {}
