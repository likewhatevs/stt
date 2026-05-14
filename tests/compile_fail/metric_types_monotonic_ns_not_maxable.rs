//! `Maxable` is not implemented for the four cumulative-counter
//! newtypes (`MonotonicCount`, `MonotonicNs`, `ClockTicks`,
//! `Bytes`) — max-across-snapshots on a thread-lifetime
//! accumulator reduces to "the value of the last snapshot" because
//! each snapshot dominates every prior one by construction (the
//! kernel only ever raises a lifetime counter). That's a sanity
//! bound, not a worst-window signal. Pin the negative side of the
//! decision empirically: a generic site bound on `T: Maxable` must
//! refuse `MonotonicNs`.

fn require_maxable<T: ktstr::metric_types::Maxable>() {}

fn main() {
    require_maxable::<ktstr::metric_types::MonotonicNs>();
}
