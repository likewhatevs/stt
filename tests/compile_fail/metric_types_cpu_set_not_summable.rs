//! `CpuSet` is a `Vec<u32>` of CPU IDs — affinity, not a
//! cumulative counter. Summing two affinity sets is undefined
//! (set union is the closest meaningful op, but that's the
//! `AffinitySummary` reduction in `ctprof_compare`, not a
//! Summable trait method). Pin the type-system rejection: a
//! generic site bound on `T: Summable` must refuse `CpuSet`.

fn require_summable<T: ktstr::metric_types::Summable>() {}

fn main() {
    require_summable::<ktstr::metric_types::CpuSet>();
}
