// Pins the trim-before-reservation hardening: whitespace-padded
// reserved keywords (`"  eevdf  "`) bypassed the prior to_lowercase()
// + matches!() check because the padding survived the lowercase pass.
// trim().to_lowercase() normalizes both whitespace AND case so the
// reservation's intent ("don't shadow Scheduler::EEVDF") holds
// regardless of how the user formatted the literal.
use ktstr::declare_scheduler;

declare_scheduler!(RESERVED_NAME_EEVDF_PADDED, {
    name = "  eevdf  ",
    binary = "scx_padded_eevdf",
});

fn main() {}
