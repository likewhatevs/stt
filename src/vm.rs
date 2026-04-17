//! VM launch configuration and gauntlet topology presets.
//!
//! See the [Running Tests](https://likewhatevs.github.io/ktstr/guide/running-tests.html)
//! and [Gauntlet](https://likewhatevs.github.io/ktstr/guide/running-tests/gauntlet.html)
//! chapters of the guide.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;

use crate::vmm::{self, Topology, VmResult};

/// Configuration for launching a test VM.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VmConfig {
    pub kernel: Option<String>,
    pub topology: Topology,
    pub memory_mb: usize,
    pub timeout: Option<Duration>,
    /// Linux source tree with built kernel. Resolves to
    /// `{kernel_dir}/arch/x86/boot/bzImage` on x86_64 or
    /// `{kernel_dir}/arch/arm64/boot/Image` on aarch64.
    pub kernel_dir: Option<String>,
}

impl Default for VmConfig {
    fn default() -> Self {
        Self {
            kernel: None,
            topology: Topology {
                llcs: 2,
                cores_per_llc: 2,
                threads_per_core: 1,
                numa_nodes: 1,
            },
            memory_mb: 4096,
            timeout: None,
            kernel_dir: None,
        }
    }
}

/// Boot a KVM VM with the given config and run ktstr inside it.
///
/// Resolves the kernel image, builds the initramfs (ktstr binary +
/// optional scheduler), and boots the VM. Returns the VM's exit
/// result including serial output.
pub fn run_in_vm(cfg: &VmConfig, ktstr_args: &[String]) -> Result<VmResult> {
    // Resolve kernel
    let kernel = if let Some(ref kd) = cfg.kernel_dir {
        #[cfg(target_arch = "x86_64")]
        let image = PathBuf::from(kd).join("arch/x86/boot/bzImage");
        #[cfg(target_arch = "aarch64")]
        let image = PathBuf::from(kd).join("arch/arm64/boot/Image");
        image
    } else if let Some(ref k) = cfg.kernel {
        PathBuf::from(k)
    } else {
        crate::find_kernel()?.unwrap_or_else(|| PathBuf::from("/boot/vmlinuz"))
    };

    // Find ktstr binary (ourselves)
    let ktstr_bin = crate::resolve_current_exe()?;

    // Build guest args: strip --scheduler-bin (host path) and append
    // the guest-side path (/scheduler) so the inner ktstr finds the binary.
    let mut sched_bin: Option<PathBuf> = None;
    let mut guest_args = Vec::with_capacity(ktstr_args.len() + 2);
    let mut iter = ktstr_args.iter();
    while let Some(a) = iter.next() {
        if a == "--scheduler-bin" {
            if let Some(val) = iter.next() {
                sched_bin = Some(PathBuf::from(val));
            }
        } else if let Some(val) = a.strip_prefix("--scheduler-bin=") {
            sched_bin = Some(PathBuf::from(val));
        } else {
            guest_args.push(a.clone());
        }
    }
    if sched_bin.is_some() {
        guest_args.push("--scheduler-bin".into());
        guest_args.push("/scheduler".into());
    }

    let no_perf_mode = std::env::var("KTSTR_NO_PERF_MODE").is_ok();
    let t = &cfg.topology;
    let mut builder = vmm::KtstrVm::builder()
        .kernel(&kernel)
        .init_binary(&ktstr_bin)
        .topology(t.numa_nodes, t.llcs, t.cores_per_llc, t.threads_per_core)
        .memory_mb(u32::try_from(cfg.memory_mb).context("memory_mb exceeds u32::MAX")?)
        .run_args(&guest_args)
        .no_perf_mode(no_perf_mode);

    if let Some(ref sb) = sched_bin {
        builder = builder.scheduler_binary(sb);
    }
    if let Some(t) = cfg.timeout {
        builder = builder.timeout(t);
    }

    builder.build()?.run()
}

/// Compute a VM timeout based on scenario count, duration, and CPU count.
///
/// Accounts for boot overhead that scales with large CPU counts.
pub fn compute_timeout(num_runs: usize, duration_s: u64, num_cpus: usize) -> Duration {
    // Large VMs boot slower: add 1s per 10 CPUs beyond 16
    let boot_overhead = 10 + (num_cpus.saturating_sub(16) / 10) as u64;
    Duration::from_secs(boot_overhead + num_runs as u64 * (duration_s + 2) * 2)
}

/// A gauntlet topology preset.
///
/// Each preset defines a specific CPU topology for matrix testing.
/// See [`gauntlet_presets()`] for the full set.
#[allow(dead_code)]
pub struct TopoPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub topology: Topology,
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
    fn compute_timeout_basic() {
        // 8 CPUs: no extra boot overhead (below 16 threshold)
        assert_eq!(
            compute_timeout(1, 20, 8),
            Duration::from_secs(10 + (20 + 2) * 2)
        );
    }

    #[test]
    fn compute_timeout_multiple_runs() {
        assert_eq!(
            compute_timeout(5, 15, 8),
            Duration::from_secs(10 + 5 * (15 + 2) * 2)
        );
    }

    #[test]
    fn compute_timeout_large_vm() {
        // 240 CPUs: (240-16)/10 = 22 extra seconds
        assert_eq!(
            compute_timeout(1, 20, 240),
            Duration::from_secs(32 + (20 + 2) * 2)
        );
    }

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
    fn vm_config_default() {
        let c = VmConfig::default();
        assert_eq!(c.topology.total_cpus(), 4);
        assert_eq!(c.memory_mb, 4096);
        assert!(c.kernel.is_none());
        assert!(c.timeout.is_none());
    }

    #[test]
    fn compute_timeout_zero_runs() {
        // num_runs=0: only boot overhead, no per-run time
        assert_eq!(compute_timeout(0, 30, 8), Duration::from_secs(10));
    }

    #[test]
    fn compute_timeout_zero_cpus() {
        // num_cpus=0: saturating_sub(16) clamps to 0, no extra overhead
        assert_eq!(
            compute_timeout(1, 10, 0),
            Duration::from_secs(10 + (10 + 2) * 2)
        );
    }

    #[test]
    fn compute_timeout_large_cpu_count() {
        // num_cpus=1000: (1000-16)/10 = 98 extra seconds
        let t = compute_timeout(1, 10, 1000);
        assert_eq!(t, Duration::from_secs(10 + 98 + (10 + 2) * 2));
        assert!(t.as_secs() < 10_000, "should be finite and reasonable");
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
    fn compute_timeout_16_cpus_boundary() {
        // Exactly 16 CPUs: saturating_sub(16) = 0, no extra overhead
        assert_eq!(
            compute_timeout(1, 10, 16),
            Duration::from_secs(10 + (10 + 2) * 2)
        );
    }

    #[test]
    fn compute_timeout_17_cpus() {
        // 17 CPUs: (17-16)/10 = 0 (integer division), no extra
        assert_eq!(
            compute_timeout(1, 10, 17),
            Duration::from_secs(10 + (10 + 2) * 2)
        );
    }

    #[test]
    fn compute_timeout_26_cpus() {
        // 26 CPUs: (26-16)/10 = 1 extra second
        assert_eq!(
            compute_timeout(1, 10, 26),
            Duration::from_secs(11 + (10 + 2) * 2)
        );
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
    fn vm_config_custom() {
        let c = VmConfig {
            kernel: Some("/boot/custom".into()),
            topology: Topology {
                llcs: 4,
                cores_per_llc: 8,
                threads_per_core: 1,
                numa_nodes: 1,
            },
            memory_mb: 8192,
            timeout: Some(Duration::from_secs(300)),
            kernel_dir: Some("/src/linux".into()),
        };
        assert_eq!(c.topology.total_cpus(), 32);
        assert_eq!(c.memory_mb, 8192);
        assert!(c.timeout.is_some());
        assert_eq!(c.timeout.unwrap(), Duration::from_secs(300));
    }

    #[test]
    fn topology_single_cpu() {
        let t = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
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
