//! `CpuSet` is a `Vec<u32>` of CPU IDs — affinity, not a
//! categorical value. "Mode of three affinity sets" would
//! technically pick the most-frequent literal Vec<u32> but the
//! cell would render as a verbose CPU-id list whose semantic
//! meaning is "the most common cpuset" — useless when the
//! operator wants to see the affinity distribution across a
//! group. The bespoke `AffinitySummary` reduction in
//! `host_state_compare` is the right path. Pin the type-system
//! rejection: a generic site bound on `T: Modeable` must
//! refuse `CpuSet`.

fn require_modeable<T: ktstr::metric_types::Modeable>() {}

fn main() {
    require_modeable::<ktstr::metric_types::CpuSet>();
}
