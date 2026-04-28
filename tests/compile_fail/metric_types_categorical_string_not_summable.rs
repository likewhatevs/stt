//! Categorical strings have no additive operation — summing
//! `"SCHED_OTHER" + "SCHED_FIFO"` is undefined. Pin the
//! type-system rejection.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::CategoricalString>();
}
