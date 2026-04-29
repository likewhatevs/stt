//! `DeadCounter` represents a kernel-side dead pointer — the
//! field exists in `task_struct` but no kernel writer ever
//! touches it, so the value is structurally zero. Mode-across
//! on a structural zero is doubly meaningless: every
//! contributor is the same value (0), the mode is trivially
//! 0/N, and surfacing it implies a categorical contrast where
//! none exists. Pin the type-system rejection: a generic site
//! bound on `T: Modeable` must refuse `DeadCounter`.

fn require_modeable<T: ktstr::metric_types::Modeable>() {}

fn main() {
    require_modeable::<ktstr::metric_types::DeadCounter>();
}
