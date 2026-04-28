//! `GaugeCount` is an instantaneous unitless count sampled at
//! capture time — summing thread-count gauges across a bucket
//! over-counts shared parent processes N-fold. This compile_fail
//! pins the type-system rejection.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::GaugeCount>();
}
