//! `CpuSet` is a `Vec<u32>` of CPU IDs — affinity, not a
//! bounded scalar. The `[min, max]` range reduction is for
//! ordinals (`OrdinalI32` / `OrdinalU32` / `OrdinalU64`) where
//! the operator wants to see the spread of a per-thread value
//! across a group. Affinity uses the bespoke `AffinitySummary`
//! reduction in `host_state_compare` (num_cpus range +
//! uniform-cpuset flag), not a generic Rangeable trait method.
//! Pin the type-system rejection: a generic site bound on
//! `T: Rangeable` must refuse `CpuSet`.

fn require_rangeable<T: ktstr::metric_types::Rangeable>() {}

fn main() {
    require_rangeable::<ktstr::metric_types::CpuSet>();
}
