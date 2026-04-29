//! `OrdinalI32` wraps a bounded ordinal — a numeric scalar
//! with a meaningful min/max but no categorical-mode shape.
//! Mode-across an ordinal would surface "the most-common nice
//! value" which is technically computable but not the operator
//! signal: ordinals want range. `Modeable` is reserved for
//! genuinely-categorical types (`CategoricalString` is the
//! only impl today). Pin the type-system rejection: a generic
//! site bound on `T: Modeable` must refuse `OrdinalI32`.

fn require_modeable<T: ktstr::metric_types::Modeable>() {}

fn main() {
    require_modeable::<ktstr::metric_types::OrdinalI32>();
}
