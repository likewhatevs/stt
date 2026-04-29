//! `OrdinalI32` wraps a bounded ordinal (e.g. `nice` in
//! `[-20, 19]`, `priority` in `[0, 39]` for CFS). Summing
//! ordinals is meaningless — `nice(-5) + nice(-5)` is not a
//! `nice` value, it's a sum that has no place in the bounded
//! domain. The aggregation contract for ordinals is the
//! `[min, max]` range via `Rangeable::range_across`. Pin the
//! type-system rejection: a generic site bound on
//! `T: Summable` must refuse `OrdinalI32`.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::OrdinalI32>();
}
