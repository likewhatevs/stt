//! Gauntlet topology presets.
//!
//! See the [Gauntlet](https://likewhatevs.github.io/ktstr/guide/running-tests/gauntlet.html)
//! chapter of the guide.

use crate::vmm::Topology;

/// A gauntlet topology preset.
///
/// Each preset defines a specific CPU topology for matrix testing.
/// See [`gauntlet_presets()`] for the full set.
pub struct TopoPreset {
    pub name: &'static str,
    /// Human-readable description; read by preset-audit tests only.
    #[allow(dead_code)]
    pub description: &'static str,
    pub topology: Topology,
    /// Memory budget for this preset's VM; read by preset-audit tests only.
    #[allow(dead_code)]
    pub memory_mb: usize,
}

/// Topology presets used by gauntlet mode.
///
/// Ranges from `tiny-1llc` (4 CPUs) to `max-cpu` (252 CPUs, near
/// the KVM vCPU limit). Includes multi-NUMA presets (`numa2-*`,
/// `numa4-*`) for cross-node scheduling; filtered by default
/// (`max_numa_nodes: Some(1)`). On aarch64, presets with SMT
/// (`threads_per_core > 1`) are excluded because ARM64 CPUs do not
/// have SMT. Non-SMT medium/large/max presets ensure ARM64 still
/// gets full topology scale coverage.
pub fn gauntlet_presets() -> Vec<TopoPreset> {
    let defs: &[(&str, &str, u32, u32, u32, usize)] = &[
        ("tiny-1llc", "4 CPUs, 1 LLC", 1, 4, 1, 2048),
        ("tiny-2llc", "4 CPUs, 2 LLCs", 2, 2, 1, 2048),
        ("odd-3llc", "9 CPUs, 3 LLCs (odd)", 3, 3, 1, 2048),
        ("odd-5llc", "15 CPUs, 5 LLCs (prime)", 5, 3, 1, 2048),
        ("odd-7llc", "14 CPUs, 7 LLCs (prime)", 7, 2, 1, 2048),
        ("smt-2llc", "8 CPUs, 2 LLCs with SMT", 2, 2, 2, 2048),
        ("smt-3llc", "12 CPUs, 3 LLCs with SMT", 3, 2, 2, 2048),
        ("medium-4llc", "32 CPUs, 4 LLCs", 4, 4, 2, 2048),
        ("medium-8llc", "64 CPUs, 8 LLCs", 8, 4, 2, 2048),
        ("large-4llc", "128 CPUs, 4 LLCs", 4, 16, 2, 2048),
        ("large-8llc", "128 CPUs, 8 LLCs", 8, 8, 2, 2048),
        (
            "near-max-llc",
            "240 CPUs, 15 LLCs (near max)",
            15,
            8,
            2,
            2048,
        ),
        (
            "max-cpu",
            "252 CPUs, 14 LLCs (near KVM vCPU limit)",
            14,
            9,
            2,
            4096,
        ),
        // Non-SMT medium/large/max presets for ARM64 coverage.
        // These also run on x86_64 to test non-SMT topologies at scale.
        (
            "medium-4llc-nosmt",
            "32 CPUs, 4 LLCs (no SMT)",
            4,
            8,
            1,
            2048,
        ),
        (
            "medium-8llc-nosmt",
            "64 CPUs, 8 LLCs (no SMT)",
            8,
            8,
            1,
            2048,
        ),
        (
            "large-4llc-nosmt",
            "128 CPUs, 4 LLCs (no SMT)",
            4,
            32,
            1,
            2048,
        ),
        (
            "large-8llc-nosmt",
            "128 CPUs, 8 LLCs (no SMT)",
            8,
            16,
            1,
            2048,
        ),
        (
            "near-max-llc-nosmt",
            "240 CPUs, 15 LLCs (no SMT)",
            15,
            16,
            1,
            2048,
        ),
        (
            "max-cpu-nosmt",
            "252 CPUs, 14 LLCs (no SMT, near KVM vCPU limit)",
            14,
            18,
            1,
            4096,
        ),
    ];
    let numa_defs: &[(&str, &str, u32, u32, u32, u32, usize)] = &[
        (
            "numa2-4llc",
            "16 CPUs, 2 NUMA nodes, 4 LLCs",
            2,
            4,
            4,
            1,
            2048,
        ),
        (
            "numa2-8llc",
            "128 CPUs, 2 NUMA nodes, 8 LLCs",
            2,
            8,
            8,
            2,
            2048,
        ),
        (
            "numa2-8llc-nosmt",
            "128 CPUs, 2 NUMA nodes, 8 LLCs (no SMT)",
            2,
            8,
            16,
            1,
            2048,
        ),
        (
            "numa4-8llc",
            "32 CPUs, 4 NUMA nodes, 8 LLCs",
            4,
            8,
            4,
            1,
            2048,
        ),
        (
            "numa4-12llc",
            "192 CPUs, 4 NUMA nodes, 12 LLCs",
            4,
            12,
            8,
            2,
            4096,
        ),
    ];

    let mut presets: Vec<TopoPreset> = defs
        .iter()
        .map(|&(n, d, s, c, t, m)| TopoPreset {
            name: n,
            description: d,
            topology: Topology {
                llcs: s,
                cores_per_llc: c,
                threads_per_core: t,
                numa_nodes: 1,
                nodes: None,
                distances: None,
            },
            memory_mb: m,
        })
        .chain(numa_defs.iter().map(|&(n, d, nn, s, c, t, m)| TopoPreset {
            name: n,
            description: d,
            topology: Topology {
                llcs: s,
                cores_per_llc: c,
                threads_per_core: t,
                numa_nodes: nn,
                nodes: None,
                distances: None,
            },
            memory_mb: m,
        }))
        .collect();

    // ARM64 has no SMT -- exclude presets with threads_per_core > 1.
    if cfg!(target_arch = "aarch64") {
        presets.retain(|p| p.topology.threads_per_core <= 1);
    }

    presets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauntlet_presets_unique_names() {
        let p = gauntlet_presets();
        let names: Vec<&str> = p.iter().map(|p| p.name).collect();
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(names.len(), unique.len());
    }

    #[test]
    fn gauntlet_presets_total_cpus_match() {
        for p in &gauntlet_presets() {
            let cpus = p.topology.total_cpus();
            assert!(
                p.description.contains(&cpus.to_string()),
                "{}: description '{}' doesn't mention {} CPUs",
                p.name,
                p.description,
                cpus
            );
        }
    }

    #[test]
    fn gauntlet_presets_memory_sane() {
        for p in &gauntlet_presets() {
            assert!(
                p.memory_mb >= 512,
                "{} has too little memory: {}MB",
                p.name,
                p.memory_mb
            );
            let cpus = p.topology.total_cpus() as usize;
            assert!(
                p.memory_mb >= cpus * 8,
                "{} has {}MB for {} CPUs",
                p.name,
                p.memory_mb,
                cpus
            );
        }
    }

    #[test]
    fn gauntlet_presets_topology_pinned() {
        // (name, expected LLCs, expected total CPUs)
        let expected: &[(&str, u32, u32)] = &[
            ("tiny-1llc", 1, 4),
            ("tiny-2llc", 2, 4),
            ("odd-3llc", 3, 9),
            ("odd-5llc", 5, 15),
            ("odd-7llc", 7, 14),
            #[cfg(not(target_arch = "aarch64"))]
            ("smt-2llc", 2, 8),
            #[cfg(not(target_arch = "aarch64"))]
            ("smt-3llc", 3, 12),
            #[cfg(not(target_arch = "aarch64"))]
            ("medium-4llc", 4, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("medium-8llc", 8, 64),
            #[cfg(not(target_arch = "aarch64"))]
            ("large-4llc", 4, 128),
            #[cfg(not(target_arch = "aarch64"))]
            ("large-8llc", 8, 128),
            #[cfg(not(target_arch = "aarch64"))]
            ("near-max-llc", 15, 240),
            #[cfg(not(target_arch = "aarch64"))]
            ("max-cpu", 14, 252),
            ("medium-4llc-nosmt", 4, 32),
            ("medium-8llc-nosmt", 8, 64),
            ("large-4llc-nosmt", 4, 128),
            ("large-8llc-nosmt", 8, 128),
            ("near-max-llc-nosmt", 15, 240),
            ("max-cpu-nosmt", 14, 252),
            ("numa2-4llc", 4, 16),
            #[cfg(not(target_arch = "aarch64"))]
            ("numa2-8llc", 8, 128),
            ("numa2-8llc-nosmt", 8, 128),
            ("numa4-8llc", 8, 32),
            #[cfg(not(target_arch = "aarch64"))]
            ("numa4-12llc", 12, 192),
        ];
        let presets = gauntlet_presets();
        assert_eq!(
            expected.len(),
            presets.len(),
            "pinned list and preset list have different lengths"
        );
        for &(name, llcs, cpus) in expected {
            let p = presets.iter().find(|p| p.name == name).unwrap();
            assert_eq!(
                p.topology.num_llcs(),
                llcs,
                "{}: expected {} LLCs, got {}",
                name,
                llcs,
                p.topology.num_llcs()
            );
            assert_eq!(
                p.topology.total_cpus(),
                cpus,
                "{}: expected {} CPUs, got {}",
                name,
                cpus,
                p.topology.total_cpus()
            );
        }
    }

    #[test]
    fn gauntlet_presets_topology_valid() {
        for p in &gauntlet_presets() {
            p.topology
                .validate()
                .unwrap_or_else(|e| panic!("{}: {e}", p.name));
        }
    }

    #[test]
    fn gauntlet_presets_max_cpu_near_limit() {
        let presets = gauntlet_presets();
        let max_presets: Vec<_> = presets
            .iter()
            .filter(|p| p.name.starts_with("max-cpu"))
            .collect();
        assert!(
            !max_presets.is_empty(),
            "at least one max-cpu preset must exist"
        );
        for p in &max_presets {
            let cpus = p.topology.total_cpus();
            assert!(
                cpus <= 255,
                "{} has {} CPUs, exceeds KVM vCPU limit",
                p.name,
                cpus
            );
            assert!(
                cpus >= 200,
                "{} should be near the limit: {} CPUs",
                p.name,
                cpus
            );
        }
    }

    #[test]
    fn topology_single_cpu() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        assert_eq!(t.total_cpus(), 1);
        assert_eq!(t.num_llcs(), 1);
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn gauntlet_presets_smt_presets_have_threads() {
        let presets = gauntlet_presets();
        for p in &presets {
            if p.name.starts_with("smt-") {
                assert_eq!(
                    p.topology.threads_per_core, 2,
                    "{} should have 2 threads per core",
                    p.name
                );
            }
        }
    }

    #[test]
    fn gauntlet_presets_odd_presets_are_odd() {
        let presets = gauntlet_presets();
        for p in &presets {
            if p.name.starts_with("odd-") {
                assert!(
                    p.topology.llcs % 2 != 0,
                    "{}: odd-* presets must have odd LLC count, got {} LLCs",
                    p.name,
                    p.topology.llcs
                );
            }
        }
    }

    #[test]
    fn gauntlet_presets_numa_presets_have_correct_nodes() {
        for p in &gauntlet_presets() {
            if p.name.starts_with("numa2") {
                assert_eq!(
                    p.topology.numa_nodes, 2,
                    "{}: expected 2 NUMA nodes",
                    p.name
                );
            } else if p.name.starts_with("numa4") {
                assert_eq!(
                    p.topology.numa_nodes, 4,
                    "{}: expected 4 NUMA nodes",
                    p.name
                );
            }
        }
    }

    #[test]
    fn gauntlet_presets_description_non_empty() {
        for p in &gauntlet_presets() {
            assert!(
                !p.description.is_empty(),
                "{} has empty description",
                p.name
            );
        }
    }
}
