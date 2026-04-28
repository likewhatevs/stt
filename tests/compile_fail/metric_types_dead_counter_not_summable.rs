//! `DeadCounter` is the type-level signal that a kernel counter's
//! update path is dead — the value is structurally zero. Summing
//! dead counters across a group produces a "total" whose meaning
//! is "we counted some zeros." This is the whole point of the
//! newtype: a registry entry that pairs a `DeadCounter` field with
//! a `Summable`-bound `AggRule` variant fails to compile, flagging
//! the dead status at the type level rather than surfacing as a
//! "0 + 0 + 0" rendered cell. Pin the type-system rejection.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::DeadCounter>();
}
