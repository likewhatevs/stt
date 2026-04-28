//! `PeakNs` is a lifetime high-water mark — summing peaks across
//! threads conflates different tasks' worst single windows into a
//! number with no defensible interpretation. This compile_fail
//! pins the type-system rejection: a generic site bound on
//! `T: Summable` must refuse `PeakNs`.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::PeakNs>();
}
