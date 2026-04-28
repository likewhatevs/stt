//! Cumulative counters are not bounded ordinals — a `[min, max]`
//! interval over `MonotonicCount` is a category error. Pin the
//! type-system rejection: a generic site bound on
//! `T: Rangeable` must refuse `MonotonicCount`.

fn require_rangeable<T: ktstr::metric_types::Rangeable>() {}

fn main() {
    require_rangeable::<ktstr::metric_types::MonotonicCount>();
}
