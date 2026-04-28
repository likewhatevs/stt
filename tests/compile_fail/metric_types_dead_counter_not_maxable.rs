//! `DeadCounter` does not implement any reduction trait, including
//! `Maxable` — the value is structurally zero, so max-across is
//! trivially zero, but rendering that "zero" through a live
//! reduction implies "we measured zero events" when the truth is
//! "we measured a kernel-side dead pointer." Pin the
//! type-system rejection.

fn require_maxable<T: ktstr::metric_types::Maxable>() {}

fn main() {
    require_maxable::<ktstr::metric_types::DeadCounter>();
}
