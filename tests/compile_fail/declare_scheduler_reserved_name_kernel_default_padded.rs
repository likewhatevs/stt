// Sibling of declare_scheduler_reserved_name_eevdf_padded.rs.
// Verifies trim-before-reservation also closes the
// kernel_default whitespace-padded bypass — paired coverage per
// keyword per axis matching the established reserved-name fixture
// convention.
use ktstr::declare_scheduler;

declare_scheduler!(RESERVED_NAME_KERNEL_DEFAULT_PADDED, {
    name = "  kernel_default  ",
    binary = "scx_padded_kernel_default",
});

fn main() {}
