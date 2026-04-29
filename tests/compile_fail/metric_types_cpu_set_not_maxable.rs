//! `CpuSet` is a `Vec<u32>` of CPU IDs — affinity, not an
//! ordinal scalar. "Max of two affinity sets" has no
//! well-defined meaning (lexicographic Vec<u32> ordering would
//! technically work but would render as gibberish). The
//! aggregation contract for affinity is the bespoke
//! `AffinitySummary` reduction in `host_state_compare`, not a
//! generic Maxable trait method. Pin the type-system rejection:
//! a generic site bound on `T: Maxable` must refuse `CpuSet`.

fn require_maxable<T: ktstr::metric_types::Maxable>() {}

fn main() {
    require_maxable::<ktstr::metric_types::CpuSet>();
}
