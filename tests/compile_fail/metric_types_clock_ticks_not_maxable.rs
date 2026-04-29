//! After phase-3 trait hardening, `Maxable` is no longer
//! implemented for the four cumulative-counter newtypes
//! (`MonotonicCount`, `MonotonicNs`, `ClockTicks`, `Bytes`) —
//! max-across-snapshots on a USER_HZ-tick accumulator reduces
//! to "the value of the last snapshot" because the kernel
//! only ever raises a lifetime tick counter. Pin the negative
//! side of the decision empirically: a generic site bound on
//! `T: Maxable` must refuse `ClockTicks`.

fn require_maxable<T: ktstr::metric_types::Maxable>() {}

fn main() {
    require_maxable::<ktstr::metric_types::ClockTicks>();
}
