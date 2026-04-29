//! `DeadCounter` represents a kernel-side dead pointer — the
//! field exists in `task_struct` but no kernel writer ever
//! touches it, so the value is structurally zero. Range-across
//! on a structural zero is doubly meaningless: the [min, max]
//! is always [0, 0] and surfacing it in a rendered cell implies
//! "we measured a thing" when in fact nothing was measured. Pin
//! the type-system rejection: a generic site bound on
//! `T: Rangeable` must refuse `DeadCounter`.

fn require_rangeable<T: ktstr::metric_types::Rangeable>() {}

fn main() {
    require_rangeable::<ktstr::metric_types::DeadCounter>();
}
