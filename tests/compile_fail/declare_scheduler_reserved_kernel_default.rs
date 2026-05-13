// Pins the KERNEL_DEFAULT const-ident reservation parallel to
// declare_scheduler_reserved_eevdf.rs. Reserved-name fixtures need
// symmetric coverage: a regression that dropped the
// `"KERNEL_DEFAULT"` arm from the const-name reservation match would
// otherwise slip past while only EEVDF was tested.
use ktstr::declare_scheduler;

declare_scheduler!(KERNEL_DEFAULT, {
    name = "kernel_default_user",
    binary = "scx_kernel_default_user",
});

fn main() {}
