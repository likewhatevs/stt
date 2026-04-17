//! Scenario catalog: registers all scenarios via `all_scenarios()`.

use super::*;

// Static flag slices for Scenario definitions.
static F_BORROW: &[&flags::FlagDecl] = &[&flags::BORROW_DECL];
static F_LLC: &[&flags::FlagDecl] = &[&flags::LLC_DECL];
static F_REBAL: &[&flags::FlagDecl] = &[&flags::REBAL_DECL];
static F_REJECT_PIN: &[&flags::FlagDecl] = &[&flags::REJECT_PIN_DECL];
static F_NO_CTRL: &[&flags::FlagDecl] = &[&flags::NO_CTRL_DECL];
static F_LLC_STEAL: &[&flags::FlagDecl] = &[&flags::LLC_DECL, &flags::STEAL_DECL];
static F_LLC_REBAL: &[&flags::FlagDecl] = &[&flags::LLC_DECL, &flags::REBAL_DECL];
static F_BORROW_REBAL: &[&flags::FlagDecl] = &[&flags::BORROW_DECL, &flags::REBAL_DECL];
static F_NO_CTRL_BORROW: &[&flags::FlagDecl] = &[&flags::NO_CTRL_DECL, &flags::BORROW_DECL];
static F_NO_CTRL_REBAL: &[&flags::FlagDecl] = &[&flags::NO_CTRL_DECL, &flags::REBAL_DECL];

macro_rules! s {
    ($name:expr, $cat:expr, $desc:expr, $cgroups:expr, $cpuset:expr, $works:expr) => {
        Scenario {
            name: $name,
            category: $cat,
            description: $desc,
            required_flags: &[],
            excluded_flags: &[],
            num_cgroups: $cgroups,
            cpuset_mode: $cpuset,
            cgroup_works: $works,
            action: Action::Steady,
        }
    };
}

macro_rules! custom {
    ($name:expr, $cat:expr, $desc:expr, $fn:expr) => {
        Scenario {
            name: $name,
            category: $cat,
            description: $desc,
            required_flags: &[],
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom($fn),
        }
    };
}

/// Returns all registered scenarios: data-driven and custom.
pub fn all_scenarios() -> Vec<Scenario> {
    use super::affinity::*;
    use super::basic::*;
    use super::cpuset::*;
    use super::dynamic::*;
    use super::interaction::*;
    use super::nested::*;
    use super::performance::*;
    use super::stress::*;

    let dfl = || vec![Work::default()];
    let w = |wt, pol| {
        vec![Work {
            work_type: wt,
            sched_policy: pol,
            ..Default::default()
        }]
    };
    let aff = |a| {
        vec![Work {
            affinity: a,
            ..Default::default()
        }]
    };

    vec![
        // Basic steady-state
        s!(
            "cgroup_steady",
            "basic",
            "2 cgroups, no cpusets, equal load",
            2,
            CpusetMode::None,
            dfl()
        ),
        s!(
            "cgroup_cpuset_llc_steady",
            "basic",
            "2 cgroups, LLC-aligned cpusets, equal load",
            2,
            CpusetMode::LlcAligned,
            dfl()
        ),
        s!(
            "cgroup_cpuset_crossllc_steady",
            "basic",
            "2 cgroups, cpusets spanning LLC boundary",
            2,
            CpusetMode::SplitMisaligned,
            dfl()
        ),
        s!(
            "cgroup_cpuset_overlap_steady",
            "cpuset",
            "3 cgroups, 50% overlapping cpusets",
            3,
            CpusetMode::Overlap(0.5),
            dfl()
        ),
        s!(
            "cgroup_partial_cpus_steady",
            "stress",
            "2 cgroups, 1/3 CPUs unused",
            2,
            CpusetMode::Holdback(0.33),
            dfl()
        ),
        s!(
            "cgroup_uneven_steady",
            "stress",
            "2 cgroups, 75/25 CPU split",
            2,
            CpusetMode::Uneven(0.75),
            dfl()
        ),
        s!(
            "cgroup_oversubscribed",
            "stress",
            "2 cgroups, 32 mixed workers >> CPUs",
            2,
            CpusetMode::None,
            vec![Work {
                num_workers: Some(32),
                work_type: WorkType::Mixed,
                ..Default::default()
            }]
        ),
        // Affinity
        s!(
            "cgroup_affinity_inherit",
            "affinity",
            "No explicit affinity, mixed workload",
            2,
            CpusetMode::None,
            w(WorkType::Mixed, SchedPolicy::Normal)
        ),
        s!(
            "cgroup_affinity_random",
            "affinity",
            "Random CPU subset per worker",
            2,
            CpusetMode::SplitHalf,
            aff(AffinityKind::RandomSubset)
        ),
        s!(
            "cgroup_affinity_spanning",
            "affinity",
            "Affinity mask covers all CPUs",
            2,
            CpusetMode::SplitHalf,
            aff(AffinityKind::CrossCgroup)
        ),
        s!(
            "cgroup_affinity_pinned",
            "affinity",
            "One CPU per worker",
            2,
            CpusetMode::SplitHalf,
            aff(AffinityKind::SingleCpu)
        ),
        // Sched classes
        s!(
            "sched_batch",
            "sched_class",
            "SCHED_BATCH workers",
            2,
            CpusetMode::None,
            w(WorkType::CpuSpin, SchedPolicy::Batch)
        ),
        s!(
            "sched_idle",
            "sched_class",
            "SCHED_IDLE workers",
            2,
            CpusetMode::None,
            w(WorkType::CpuSpin, SchedPolicy::Idle)
        ),
        s!(
            "sched_fifo",
            "sched_class",
            "RT SCHED_FIFO + normal workers",
            2,
            CpusetMode::None,
            vec![Work {
                num_workers: Some(1),
                sched_policy: SchedPolicy::Fifo(1),
                work_type: WorkType::Bursty {
                    burst_ms: 500,
                    sleep_ms: 250
                },
                ..Default::default()
            }]
        ),
        s!(
            "sched_rr",
            "sched_class",
            "RT SCHED_RR + normal workers",
            2,
            CpusetMode::None,
            vec![Work {
                num_workers: Some(2),
                sched_policy: SchedPolicy::RoundRobin(1),
                work_type: WorkType::Bursty {
                    burst_ms: 500,
                    sleep_ms: 250
                },
                ..Default::default()
            }]
        ),
        // Work conservation (borrow flag)
        Scenario {
            name: "cgroup_idle_overloaded",
            category: "advanced",
            description: "One cgroup idle, other overloaded",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 2,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![
                Work {
                    num_workers: Some(16),
                    ..Default::default()
                },
                Work {
                    num_workers: Some(0),
                    ..Default::default()
                },
            ],
            action: Action::Steady,
        },
        Scenario {
            name: "cgroup_all_loaded",
            category: "advanced",
            description: "All cgroups loaded, idle CPUs contested",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 2,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![Work {
                num_workers: Some(16),
                ..Default::default()
            }],
            action: Action::Steady,
        },
        // Work stealing (llc + steal)
        Scenario {
            name: "cgroup_llc_overflow",
            category: "advanced",
            description: "Single overloaded cgroup, cross-LLC migration",
            required_flags: F_LLC_STEAL,
            excluded_flags: &[],
            num_cgroups: 1,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![Work {
                num_workers: Some(8),
                ..Default::default()
            }],
            action: Action::Steady,
        },
        // Load rebalancing (rebal flag)
        Scenario {
            name: "cgroup_load_imbalance",
            category: "advanced",
            description: "Heavy + light cgroups",
            required_flags: F_REBAL,
            excluded_flags: &[],
            num_cgroups: 2,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![
                Work {
                    num_workers: Some(16),
                    ..Default::default()
                },
                Work {
                    num_workers: Some(1),
                    work_type: WorkType::YieldHeavy,
                    ..Default::default()
                },
            ],
            action: Action::Steady,
        },
        // Stall
        Scenario {
            name: "stall_detect",
            category: "stall",
            description: "Overloaded single cgroup",
            required_flags: &[],
            excluded_flags: &[],
            num_cgroups: 1,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![Work {
                num_workers: Some(16),
                ..Default::default()
            }],
            action: Action::Steady,
        },
        // Affinity + LLC
        Scenario {
            name: "cgroup_cpuset_llc_affinity",
            category: "affinity",
            description: "LLC-aligned affinity + cpusets",
            required_flags: F_LLC,
            excluded_flags: &[],
            num_cgroups: 2,
            cpuset_mode: CpusetMode::LlcAligned,
            cgroup_works: aff(AffinityKind::LlcAligned),
            action: Action::Steady,
        },
        Scenario {
            name: "cgroup_affinity_change",
            category: "affinity",
            description: "Affinities randomized mid-run",
            required_flags: &[],
            excluded_flags: F_REJECT_PIN,
            num_cgroups: 2,
            cpuset_mode: CpusetMode::None,
            cgroup_works: dfl(),
            action: Action::Custom(custom_cgroup_affinity_change),
        },
        // Cgroup lifecycle
        custom!(
            "cgroup_add_midrun",
            "dynamic",
            "Add cgroups to running system",
            custom_cgroup_add_midrun
        ),
        custom!(
            "cgroup_remove_midrun",
            "dynamic",
            "Remove cgroups mid-run",
            custom_cgroup_remove_midrun
        ),
        custom!(
            "cgroup_rapid_churn",
            "dynamic",
            "Rapid create/destroy cycling",
            custom_cgroup_rapid_churn
        ),
        // Cpuset mutations
        custom!(
            "cgroup_cpuset_apply_midrun",
            "cpuset",
            "Add cpusets to running cgroups",
            custom_cgroup_cpuset_apply_midrun
        ),
        custom!(
            "cgroup_cpuset_clear_midrun",
            "cpuset",
            "Clear cpusets from running cgroups",
            custom_cgroup_cpuset_clear_midrun
        ),
        custom!(
            "cgroup_cpuset_resize",
            "cpuset",
            "Shrink then grow cpusets on live cgroups",
            custom_cgroup_cpuset_resize
        ),
        // NUMA cpusets
        custom!(
            "cgroup_cpuset_numa_swap",
            "cpuset",
            "NUMA-scoped cpusets, swap mid-run",
            custom_cgroup_cpuset_numa_swap
        ),
        // Stress
        custom!(
            "host_cgroup_contention",
            "stress",
            "Host workers competing with cgroup workers",
            custom_host_cgroup_contention
        ),
        custom!(
            "cgroup_per_cpu",
            "stress",
            "One cgroup per CPU",
            custom_cgroup_per_cpu
        ),
        custom!(
            "cgroup_exhaust_reuse",
            "advanced",
            "Fill cgroups, remove half, create replacements",
            custom_cgroup_exhaust_reuse
        ),
        // Sched classes (custom)
        custom!(
            "sched_mixed",
            "sched_class",
            "Mix of all sched classes",
            custom_sched_mixed
        ),
        custom!(
            "cgroup_pipe_io",
            "sched_class",
            "Pipe-paired IO + CPU workers",
            custom_cgroup_pipe_io
        ),
        // Stall recovery
        Scenario {
            name: "stall_recovery",
            category: "stall",
            description: "Scheduler restart recovery",
            required_flags: &[],
            excluded_flags: &[],
            num_cgroups: 2,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![Work {
                num_workers: Some(16),
                ..Default::default()
            }],
            action: Action::Steady,
        },
        // Nested cgroups
        custom!(
            "nested_cgroup_steady",
            "nested",
            "Workers in nested sub-cgroups",
            custom_nested_cgroup_steady
        ),
        custom!(
            "nested_cgroup_task_move",
            "nested",
            "Move tasks between nested cgroups",
            custom_nested_cgroup_task_move
        ),
        custom!(
            "nested_cgroup_rapid_churn",
            "nested",
            "Rapid nested cgroup create/destroy",
            custom_nested_cgroup_rapid_churn
        ),
        custom!(
            "nested_cgroup_cpuset",
            "nested",
            "Nested cgroups with restrictive cpusets",
            custom_nested_cgroup_cpuset
        ),
        // Work conservation + cpusets
        Scenario {
            name: "cgroup_cpuset_workload_imbalance",
            category: "advanced",
            description: "Imbalanced load with cpuset-constrained cgroups",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_workload_imbalance),
        },
        // Load rebalancing + cpusets
        Scenario {
            name: "cgroup_cpuset_load_shift",
            category: "advanced",
            description: "Imbalanced load with cpusets, then add heavy",
            required_flags: F_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_load_shift),
        },
        // Load rebalancing + dynamic cgroup add
        Scenario {
            name: "cgroup_add_load_imbalance",
            category: "advanced",
            description: "Light cgroups then heavy added mid-run",
            required_flags: F_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_add_load_imbalance),
        },
        // Work conservation + rebalancing
        Scenario {
            name: "cgroup_imbalance_mixed_workload",
            category: "advanced",
            description: "Heavy + bursty + IO cgroups",
            required_flags: F_BORROW_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_imbalance_mixed_workload),
        },
        // Affinity rejection
        Scenario {
            name: "cgroup_multicpu_pin",
            category: "advanced",
            description: "Workers pinned to 2 CPUs each",
            required_flags: F_REJECT_PIN,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_multicpu_pin),
        },
        // No controller + rapid cgroup moves
        Scenario {
            name: "cgroup_noctrl_task_migration",
            category: "advanced",
            description: "Rapid per-task moves without controller",
            required_flags: F_NO_CTRL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_noctrl_task_migration),
        },
        // Work conservation + cpuset change
        Scenario {
            name: "cgroup_cpuset_change_imbalance",
            category: "advanced",
            description: "Cpuset resize during imbalanced load",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_change_imbalance),
        },
        // Load oscillation
        Scenario {
            name: "cgroup_load_oscillation",
            category: "advanced",
            description: "Alternating heavy/light cgroups",
            required_flags: F_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_load_oscillation),
        },
        // No controller + nested cgroups
        Scenario {
            name: "nested_cgroup_noctrl",
            category: "advanced",
            description: "Nested cgroups without controller",
            required_flags: F_NO_CTRL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_nested_cgroup_noctrl),
        },
        // Cpuset swap to disjoint range
        Scenario {
            name: "cgroup_cpuset_swap_disjoint",
            category: "cpuset",
            description: "Swap cpusets to non-overlapping ranges",
            required_flags: &[],
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_swap_disjoint),
        },
        // IO + work conservation
        Scenario {
            name: "cgroup_io_compute_imbalance",
            category: "advanced",
            description: "IO cgroup frees CPUs for compute cgroup",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_io_compute_imbalance),
        },
        // 4-way load imbalance
        Scenario {
            name: "cgroup_4way_load_imbalance",
            category: "advanced",
            description: "4 cgroups with asymmetric demand",
            required_flags: F_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_4way_load_imbalance),
        },
        // Work conservation + overlapping cpusets
        Scenario {
            name: "cgroup_cpuset_overlap_imbalance",
            category: "advanced",
            description: "Imbalanced load with overlapping cpusets",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 3,
            cpuset_mode: CpusetMode::Overlap(0.5),
            cgroup_works: vec![
                Work {
                    num_workers: Some(8),
                    ..Default::default()
                },
                Work {
                    num_workers: Some(1),
                    work_type: WorkType::Bursty {
                        burst_ms: 50,
                        sleep_ms: 100,
                    },
                    ..Default::default()
                },
                Work {
                    num_workers: Some(1),
                    work_type: WorkType::YieldHeavy,
                    ..Default::default()
                },
            ],
            action: Action::Steady,
        },
        // Load rebalancing + overlapping cpusets
        Scenario {
            name: "cgroup_cpuset_overlap_load_imbalance",
            category: "advanced",
            description: "Imbalanced load with overlapping cpusets, rebalancing",
            required_flags: F_REBAL,
            excluded_flags: &[],
            num_cgroups: 3,
            cpuset_mode: CpusetMode::Overlap(0.5),
            cgroup_works: vec![
                Work {
                    num_workers: Some(16),
                    ..Default::default()
                },
                Work {
                    num_workers: Some(1),
                    work_type: WorkType::YieldHeavy,
                    ..Default::default()
                },
                Work {
                    num_workers: Some(4),
                    ..Default::default()
                },
            ],
            action: Action::Steady,
        },
        // Work conservation + bursty
        Scenario {
            name: "cgroup_bursty_imbalance",
            category: "advanced",
            description: "Steady + bursty cgroups",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 2,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![
                Work {
                    num_workers: Some(8),
                    ..Default::default()
                },
                Work {
                    num_workers: Some(4),
                    work_type: WorkType::Bursty {
                        burst_ms: 50,
                        sleep_ms: 100,
                    },
                    ..Default::default()
                },
            ],
            action: Action::Steady,
        },
        // Load shift with bursty workload
        Scenario {
            name: "cgroup_bursty_load_shift",
            category: "advanced",
            description: "Bursty heavy + steady light cgroups",
            required_flags: F_REBAL,
            excluded_flags: &[],
            num_cgroups: 2,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![
                Work {
                    num_workers: Some(12),
                    work_type: WorkType::Bursty {
                        burst_ms: 200,
                        sleep_ms: 50,
                    },
                    ..Default::default()
                },
                Work {
                    num_workers: Some(4),
                    ..Default::default()
                },
            ],
            action: Action::Steady,
        },
        // Interactions: work conservation + rebalancing + cpusets
        Scenario {
            name: "cgroup_cpuset_imbalance_combined",
            category: "interaction",
            description: "Imbalanced load with cpusets, work conservation + rebalancing",
            required_flags: F_BORROW_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_imbalance_combined),
        },
        Scenario {
            name: "cgroup_cpuset_overlap_imbalance_combined",
            category: "interaction",
            description: "Imbalanced load with overlapping cpusets, work conservation + rebalancing",
            required_flags: F_BORROW_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_overlap_imbalance_combined),
        },
        // No controller + work conservation
        Scenario {
            name: "cgroup_noctrl_imbalance",
            category: "interaction",
            description: "Per-task moves during imbalanced load without controller",
            required_flags: F_NO_CTRL_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_noctrl_imbalance),
        },
        // No controller + cpusets
        Scenario {
            name: "cgroup_noctrl_cpuset_change",
            category: "interaction",
            description: "Cpuset add/clear without controller",
            required_flags: F_NO_CTRL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_noctrl_cpuset_change),
        },
        // No controller + rebalancing
        Scenario {
            name: "cgroup_noctrl_load_imbalance",
            category: "interaction",
            description: "Heavy + light cgroups without controller",
            required_flags: F_NO_CTRL_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_noctrl_load_imbalance),
        },
        // Affinity rejection + cpusets
        Scenario {
            name: "cgroup_cpuset_multicpu_pin",
            category: "interaction",
            description: "Multi-CPU pin within cpuset-constrained cgroups",
            required_flags: F_REJECT_PIN,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_multicpu_pin),
        },
        // Cgroup add/remove + cpusets
        Scenario {
            name: "cgroup_cpuset_add_remove",
            category: "interaction",
            description: "Add/remove cpuset-constrained cgroups",
            required_flags: &[],
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_add_remove),
        },
        // Cgroup add during imbalance
        Scenario {
            name: "cgroup_add_during_imbalance",
            category: "interaction",
            description: "Add cgroup during imbalanced load",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_add_during_imbalance),
        },
        // Nested + work conservation
        Scenario {
            name: "nested_cgroup_imbalance",
            category: "interaction",
            description: "Nested cgroups with imbalanced load",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_nested_cgroup_imbalance),
        },
        // Dispatch contention
        custom!(
            "cgroup_dsq_contention",
            "stress",
            "Bursty + pinned workers on shared DSQ",
            custom_cgroup_dsq_contention
        ),
        // Workload variety
        custom!(
            "cgroup_workload_variety",
            "stress",
            "5 cgroups, 5 workload types",
            custom_cgroup_workload_variety
        ),
        custom!(
            "cgroup_cpuset_workload_variety",
            "stress",
            "Diverse workloads with cpuset partitioning",
            custom_cgroup_cpuset_workload_variety
        ),
        custom!(
            "cgroup_dynamic_workload_variety",
            "stress",
            "Diverse workloads with cgroup add/remove",
            custom_cgroup_dynamic_workload_variety
        ),
        // Cross-LLC cpuset race
        Scenario {
            name: "cgroup_cpuset_crossllc_race",
            category: "stress",
            description: "Rapid cpuset flips across LLC boundaries",
            required_flags: F_LLC_REBAL,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cgroup_cpuset_crossllc_race),
        },
        // Performance: cache pressure vs compute imbalance
        Scenario {
            name: "cache_pressure_imbalance",
            category: "performance",
            description: "CachePressure vs CpuSpin cgroups",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cache_pressure_imbalance),
        },
        // Performance: cache yield wake-affine placement
        Scenario {
            name: "cache_yield_wake_affine",
            category: "performance",
            description: "CacheYield workers testing wake-affine placement",
            required_flags: F_LLC,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cache_yield_wake_affine),
        },
        // Performance: cache pipe + compute imbalance
        Scenario {
            name: "cache_pipe_io_compute_imbalance",
            category: "performance",
            description: "CachePipe vs CpuSpin cgroups",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_cache_pipe_io_compute_imbalance),
        },
        // Performance: 1:N fan-out wake pattern
        Scenario {
            name: "fanout_wake",
            category: "performance",
            description: "1:N futex fan-out wake vs compute workers",
            required_flags: F_BORROW,
            excluded_flags: &[],
            num_cgroups: 0,
            cpuset_mode: CpusetMode::None,
            cgroup_works: vec![],
            action: Action::Custom(custom_fanout_wake),
        },
        // Stress: task creation/destruction churn
        s!(
            "fork_exit_churn",
            "stress",
            "Rapid fork+exit cycling, task creation pressure",
            2,
            CpusetMode::None,
            w(WorkType::ForkExit, SchedPolicy::Normal)
        ),
        // Stress: dynamic nice level changes
        s!(
            "nice_sweep",
            "stress",
            "Workers cycling nice levels, priority reweighting",
            2,
            CpusetMode::None,
            w(WorkType::NiceSweep, SchedPolicy::Normal)
        ),
        // Stress: rapid affinity changes
        s!(
            "affinity_churn",
            "stress",
            "Workers rapidly changing own CPU affinity",
            2,
            CpusetMode::None,
            w(WorkType::affinity_churn(1024), SchedPolicy::Normal)
        ),
        // Stress: scheduling policy cycling
        s!(
            "policy_churn",
            "stress",
            "Workers cycling scheduling policies mid-run",
            2,
            CpusetMode::None,
            w(WorkType::policy_churn(1024), SchedPolicy::Normal)
        ),
    ]
}
