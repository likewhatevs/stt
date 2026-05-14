//! `Maxable` is not implemented for the four cumulative-counter
//! newtypes (`MonotonicCount`, `MonotonicNs`, `ClockTicks`,
//! `Bytes`) — max-across-snapshots on a lifetime accumulator
//! reduces to "the last snapshot's value," which is mostly noise
//! relative to the lifetime-integrated quantity it reports. This
//! compile_fail pins the negative side of that decision
//! empirically: a generic site bound on `T: Maxable` must refuse
//! `MonotonicCount`.

fn require_maxable<T: ktstr::metric_types::Maxable>() {}

fn main() {
    require_maxable::<ktstr::metric_types::MonotonicCount>();
}
