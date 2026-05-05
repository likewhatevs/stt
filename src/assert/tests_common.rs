//! Shared `WorkerReport` builders for the `assert` test
//! sub-modules. Splitting test files by subject would otherwise
//! duplicate these per-file; centralising them keeps the field
//! list in one place so a `WorkerReport` schema change touches
//! only this fixture.

use super::*;
use crate::workload::WorkerReport;

pub fn rpt(
    tid: i32,
    work: u64,
    wall_ns: u64,
    off_cpu_ns: u64,
    cpus: &[usize],
    gap_ms: u64,
) -> WorkerReport {
    WorkerReport {
        tid,
        work_units: work,
        cpu_time_ns: wall_ns.saturating_sub(off_cpu_ns),
        wall_time_ns: wall_ns,
        off_cpu_ns,
        migration_count: 0,
        cpus_used: cpus.iter().copied().collect(),
        migrations: vec![],
        max_gap_ms: gap_ms,
        max_gap_cpu: cpus.first().copied().unwrap_or(0),
        max_gap_at_ms: 1000,
        resume_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
    }
}

pub fn rpt_with_latencies(
    tid: i32,
    latencies: Vec<u64>,
    iterations: u64,
    wall_ns: u64,
) -> WorkerReport {
    WorkerReport {
        tid,
        work_units: 1000,
        cpu_time_ns: wall_ns / 2,
        wall_time_ns: wall_ns,
        off_cpu_ns: wall_ns / 2,
        migration_count: 0,
        cpus_used: [0].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: 50,
        max_gap_cpu: 0,
        max_gap_at_ms: 1000,
        resume_latencies_ns: latencies,
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
    }
}
