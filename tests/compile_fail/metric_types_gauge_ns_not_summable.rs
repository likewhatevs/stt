//! `GaugeNs` is an instantaneous gauge — summing N nearly-identical
//! point-in-time samples produces N×gauge with no physical meaning.
//! Pin the type-system rejection.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::GaugeNs>();
}
