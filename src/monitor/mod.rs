//! Host-side guest memory monitor.
//!
//! Reads per-CPU runqueue structures from guest VM memory via BTF-resolved
//! offsets. Observes scheduler state without instrumenting the guest
//! kernel or the scheduler under test.
//!
//! See the [Monitor](https://sched-ext.github.io/scx/stt/architecture/monitor.html)
//! chapter of the guide.

pub mod btf_offsets;
pub mod reader;
pub mod symbols;

/// DSQ depth above this value indicates uninitialized guest memory.
/// Real kernels never queue this many tasks on a single CPU's local DSQ.
pub const DSQ_PLAUSIBILITY_CEILING: u32 = 10_000;

/// Check whether a single monitor sample contains plausible data.
///
/// Returns false when any CPU's local_dsq_depth exceeds the plausibility
/// ceiling, indicating uninitialized guest memory rather than real
/// scheduler state.
pub fn sample_looks_valid(sample: &MonitorSample) -> bool {
    sample
        .cpus
        .iter()
        .all(|cpu| cpu.local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING)
}

/// Find a vmlinux for tests.
///
/// Resolution order (first match wins):
/// 1. `LINUX_ROOT` env var — joined with `/vmlinux`, or used directly if
///    the path itself is a file named `vmlinux`.
/// 2. `./linux/vmlinux` (workspace-local kernel).
/// 3. `../linux/vmlinux` (sibling directory).
/// 4. `/sys/kernel/btf/vmlinux` (host kernel raw BTF — no ELF symbols).
#[cfg(test)]
pub fn find_test_vmlinux() -> Option<std::path::PathBuf> {
    if let Ok(root) = std::env::var("LINUX_ROOT") {
        let p = std::path::Path::new(&root).join("vmlinux");
        if p.exists() {
            return Some(p);
        }
        let p = std::path::PathBuf::from(&root);
        if p.exists() && p.file_name().is_some_and(|n| n == "vmlinux") {
            return Some(p);
        }
    }
    let p = std::path::Path::new("linux/vmlinux");
    if p.exists() {
        return Some(p.to_path_buf());
    }
    let p = std::path::Path::new("../linux/vmlinux");
    if p.exists() {
        return Some(p.to_path_buf());
    }
    let p = std::path::Path::new("/sys/kernel/btf/vmlinux");
    if p.exists() {
        return Some(p.to_path_buf());
    }
    None
}

/// Collected monitor data from a VM run.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MonitorReport {
    /// Periodic snapshots of per-CPU state.
    pub samples: Vec<MonitorSample>,
    /// Aggregated summary statistics.
    pub summary: MonitorSummary,
}

/// Point-in-time snapshot of all CPUs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MonitorSample {
    /// Milliseconds since VM start.
    pub elapsed_ms: u64,
    /// Per-CPU state at this instant.
    pub cpus: Vec<CpuSnapshot>,
}

impl MonitorSample {
    /// Compute the imbalance ratio for this sample: max(nr_running) / max(1, min(nr_running)).
    /// Returns 1.0 for empty or single-CPU samples.
    pub fn imbalance_ratio(&self) -> f64 {
        if self.cpus.is_empty() {
            return 1.0;
        }
        let mut min_nr = u32::MAX;
        let mut max_nr = 0u32;
        for cpu in &self.cpus {
            min_nr = min_nr.min(cpu.nr_running);
            max_nr = max_nr.max(cpu.nr_running);
        }
        max_nr as f64 / min_nr.max(1) as f64
    }

    /// Sum a field from event counters across all CPUs.
    /// Returns `None` if no CPU has event counters.
    pub fn sum_event_field(&self, f: fn(&ScxEventCounters) -> i64) -> Option<i64> {
        let mut total = 0i64;
        let mut any = false;
        for cpu in &self.cpus {
            if let Some(ev) = &cpu.event_counters {
                total += f(ev);
                any = true;
            }
        }
        any.then_some(total)
    }
}

/// Per-CPU state read from guest VM memory.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CpuSnapshot {
    pub nr_running: u32,
    pub scx_nr_running: u32,
    pub local_dsq_depth: u32,
    pub rq_clock: u64,
    pub scx_flags: u32,
    /// scx event counters (cumulative). None when event counter
    /// offsets are unavailable or scx_root is not set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_counters: Option<ScxEventCounters>,
}

/// Cumulative scx event counter values for a single CPU.
/// These are s64 in the kernel but always non-negative; stored as i64.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScxEventCounters {
    pub select_cpu_fallback: i64,
    pub dispatch_local_dsq_offline: i64,
    pub dispatch_keep_last: i64,
    pub enq_skip_exiting: i64,
    pub enq_skip_migration_disabled: i64,
}

/// Aggregated monitor statistics from a set of samples.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MonitorSummary {
    pub total_samples: usize,
    pub max_imbalance_ratio: f64,
    pub max_local_dsq_depth: u32,
    pub stall_detected: bool,
    /// Aggregate event counter deltas over the monitoring window.
    /// None when event counters are not available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_deltas: Option<ScxEventDeltas>,
}

/// Aggregate event counter statistics computed from first/last samples.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScxEventDeltas {
    /// Total select_cpu_fallback events across all CPUs over the window.
    pub total_fallback: i64,
    /// Fallback events per second (total_fallback / duration_secs).
    pub fallback_rate: f64,
    /// Max single-sample delta of fallback across all CPUs.
    pub max_fallback_burst: i64,
    /// Total dispatch_local_dsq_offline events.
    pub total_dispatch_offline: i64,
    /// Total dispatch_keep_last events.
    pub total_dispatch_keep_last: i64,
    /// Keep-last events per second (total_dispatch_keep_last / duration_secs).
    pub keep_last_rate: f64,
    /// Total enq_skip_exiting events.
    pub total_enq_skip_exiting: i64,
    /// Total enq_skip_migration_disabled events.
    pub total_enq_skip_migration_disabled: i64,
}

impl MonitorSummary {
    pub fn from_samples(samples: &[MonitorSample]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let mut max_imbalance_ratio: f64 = 1.0;
        let mut max_local_dsq_depth: u32 = 0;

        for sample in samples {
            if sample.cpus.is_empty() || !sample_looks_valid(sample) {
                continue;
            }
            for cpu in &sample.cpus {
                max_local_dsq_depth = max_local_dsq_depth.max(cpu.local_dsq_depth);
            }
            let ratio = sample.imbalance_ratio();
            if ratio > max_imbalance_ratio {
                max_imbalance_ratio = ratio;
            }
        }

        // Stall detection: any CPU whose rq_clock did not advance between
        // consecutive samples. Skip invalid samples.
        let mut stall_detected = false;
        let valid_samples: Vec<&MonitorSample> = samples
            .iter()
            .filter(|s| !s.cpus.is_empty() && sample_looks_valid(s))
            .collect();
        for w in valid_samples.windows(2) {
            let prev = w[0];
            let curr = w[1];
            let cpu_count = prev.cpus.len().min(curr.cpus.len());
            for cpu in 0..cpu_count {
                if curr.cpus[cpu].rq_clock != 0
                    && curr.cpus[cpu].rq_clock == prev.cpus[cpu].rq_clock
                {
                    stall_detected = true;
                    break;
                }
            }
            if stall_detected {
                break;
            }
        }

        let event_deltas = Self::compute_event_deltas(samples);

        Self {
            total_samples: samples.len(),
            max_imbalance_ratio,
            max_local_dsq_depth,
            stall_detected,
            event_deltas,
        }
    }

    /// Compute event counter deltas from the sample series.
    /// Returns None if no samples have event counters.
    fn compute_event_deltas(samples: &[MonitorSample]) -> Option<ScxEventDeltas> {
        // Find first and last samples that have event counters on any CPU.
        let has_events = |s: &MonitorSample| s.cpus.iter().any(|c| c.event_counters.is_some());
        let first = samples.iter().find(|s| has_events(s))?;
        let last = samples.iter().rev().find(|s| has_events(s))?;

        let total_fallback = last.sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0)
            - first
                .sum_event_field(|e| e.select_cpu_fallback)
                .unwrap_or(0);
        let total_keep_last = last.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0)
            - first.sum_event_field(|e| e.dispatch_keep_last).unwrap_or(0);

        // Compute rates.
        let duration_ms = last.elapsed_ms.saturating_sub(first.elapsed_ms);
        let duration_secs = duration_ms as f64 / 1000.0;
        let fallback_rate = if duration_secs > 0.0 {
            total_fallback as f64 / duration_secs
        } else {
            0.0
        };
        let keep_last_rate = if duration_secs > 0.0 {
            total_keep_last as f64 / duration_secs
        } else {
            0.0
        };

        // Max per-sample fallback burst: largest delta between consecutive
        // samples, summed across all CPUs.
        let mut max_fallback_burst: i64 = 0;
        for w in samples.windows(2) {
            let prev_sum = w[0].sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0);
            let curr_sum = w[1].sum_event_field(|e| e.select_cpu_fallback).unwrap_or(0);
            let delta = curr_sum - prev_sum;
            if delta > max_fallback_burst {
                max_fallback_burst = delta;
            }
        }

        let delta = |f: fn(&ScxEventCounters) -> i64| -> i64 {
            last.sum_event_field(f).unwrap_or(0) - first.sum_event_field(f).unwrap_or(0)
        };

        Some(ScxEventDeltas {
            total_fallback,
            fallback_rate,
            max_fallback_burst,
            total_dispatch_offline: delta(|e| e.dispatch_local_dsq_offline),
            total_dispatch_keep_last: total_keep_last,
            keep_last_rate,
            total_enq_skip_exiting: delta(|e| e.enq_skip_exiting),
            total_enq_skip_migration_disabled: delta(|e| e.enq_skip_migration_disabled),
        })
    }
}

/// Configurable thresholds for monitor-based pass/fail verdicts.
#[derive(Debug, Clone, Copy)]
pub struct MonitorThresholds {
    /// Max allowed imbalance ratio (max_nr_running / max(1, min_nr_running)).
    pub max_imbalance_ratio: f64,
    /// Max allowed local DSQ depth on any CPU in any sample.
    pub max_local_dsq_depth: u32,
    /// Fail when any CPU's rq_clock does not advance between consecutive samples.
    pub fail_on_stall: bool,
    /// Number of consecutive samples that must violate a threshold before failing.
    pub sustained_samples: usize,
    /// Max sustained select_cpu_fallback events/s across all CPUs.
    pub max_fallback_rate: f64,
    /// Max sustained dispatch_keep_last events/s across all CPUs.
    pub max_keep_last_rate: f64,
}

impl MonitorThresholds {
    /// Default thresholds, usable in const context.
    ///
    /// - imbalance 4.0: a scheduler that can't keep CPUs within 4x
    ///   load for `sustained_samples` consecutive reads has a real
    ///   balancing problem. Lower ratios (2-3) false-positive during
    ///   cpuset transitions when cells are being created/destroyed.
    /// - DSQ depth 50: local DSQ is a per-CPU overflow queue. Sustained
    ///   depth > 50 means the scheduler is not consuming dispatched tasks.
    ///   Transient spikes during cpuset changes are filtered by the
    ///   sustained_samples window.
    /// - fail_on_stall true: rq_clock not advancing means a CPU stopped
    ///   scheduling entirely. Always a bug — no workload makes this normal.
    /// - sustained_samples 5: at ~100ms sample interval, requires ~500ms
    ///   of sustained violation. Filters transient spikes from cpuset
    ///   reconfiguration, cgroup creation, and scheduler restart.
    /// - max_fallback_rate 200.0: select_cpu_fallback fires when the
    ///   scheduler's ops.select_cpu() fails to find a CPU. Sustained
    ///   200/s across all CPUs indicates systematic select_cpu failure.
    /// - max_keep_last_rate 100.0: dispatch_keep_last fires when a CPU
    ///   re-dispatches the previously running task because the scheduler
    ///   provided nothing. Sustained 100/s indicates dispatch starvation.
    pub const DEFAULT: MonitorThresholds = MonitorThresholds {
        max_imbalance_ratio: 4.0,
        max_local_dsq_depth: 50,
        fail_on_stall: true,
        sustained_samples: 5,
        max_fallback_rate: 200.0,
        max_keep_last_rate: 100.0,
    };

    /// Merge per-test overrides on top of these thresholds. Each `Some`
    /// field in `overrides` replaces the corresponding field in `self`.
    pub fn merge(&self, overrides: &ThresholdOverrides) -> MonitorThresholds {
        MonitorThresholds {
            max_imbalance_ratio: overrides
                .max_imbalance_ratio
                .unwrap_or(self.max_imbalance_ratio),
            max_local_dsq_depth: overrides
                .max_local_dsq_depth
                .unwrap_or(self.max_local_dsq_depth),
            fail_on_stall: overrides.fail_on_stall.unwrap_or(self.fail_on_stall),
            sustained_samples: overrides
                .sustained_samples
                .unwrap_or(self.sustained_samples),
            max_fallback_rate: overrides
                .max_fallback_rate
                .unwrap_or(self.max_fallback_rate),
            max_keep_last_rate: overrides
                .max_keep_last_rate
                .unwrap_or(self.max_keep_last_rate),
        }
    }
}

impl Default for MonitorThresholds {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Optional per-field overrides for MonitorThresholds. Used by
/// `SttTestEntry` to carry proc-macro-specified threshold values.
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct ThresholdOverrides {
    pub max_imbalance_ratio: Option<f64>,
    pub max_local_dsq_depth: Option<u32>,
    pub fail_on_stall: Option<bool>,
    pub sustained_samples: Option<usize>,
    pub max_fallback_rate: Option<f64>,
    pub max_keep_last_rate: Option<f64>,
}

impl ThresholdOverrides {
    #[allow(dead_code)]
    pub const NONE: ThresholdOverrides = ThresholdOverrides {
        max_imbalance_ratio: None,
        max_local_dsq_depth: None,
        fail_on_stall: None,
        sustained_samples: None,
        max_fallback_rate: None,
        max_keep_last_rate: None,
    };
}

/// Verdict from evaluating monitor data against thresholds.
#[derive(Debug, Clone)]
pub struct MonitorVerdict {
    pub passed: bool,
    pub details: Vec<String>,
    pub summary: String,
}

impl MonitorThresholds {
    /// Evaluate a MonitorReport against these thresholds.
    ///
    /// Returns a passing verdict when samples are empty or when the monitor
    /// data appears to be uninitialized guest memory (all rq_clocks identical
    /// across every CPU and sample, or DSQ depths above a plausibility
    /// ceiling). The monitor thread reads raw guest memory via BTF offsets;
    /// in short-lived VMs the kernel may not have populated the per-CPU
    /// runqueue structures before the monitor starts sampling.
    pub fn evaluate(&self, report: &MonitorReport) -> MonitorVerdict {
        let mut details = Vec::new();

        if report.samples.is_empty() {
            return MonitorVerdict {
                passed: true,
                details: vec![],
                summary: "no monitor samples".into(),
            };
        }

        // Validity check: detect uninitialized guest memory.
        // If all rq_clock values across every CPU in every sample are
        // identical, the kernel never wrote to these fields — the monitor
        // was reading zeroed or garbage memory.
        if !Self::data_looks_valid(&report.samples) {
            return MonitorVerdict {
                passed: true,
                details: vec![],
                summary: "monitor data not yet initialized".into(),
            };
        }

        // Track consecutive imbalance violations per sample.
        let mut consecutive_imbalance = 0usize;
        let mut worst_imbalance_run = 0usize;
        let mut worst_imbalance_ratio = 0.0f64;
        let mut worst_imbalance_sample_idx = 0usize;

        // Track consecutive DSQ depth violations per sample.
        let mut consecutive_dsq = 0usize;
        let mut worst_dsq_run = 0usize;
        let mut worst_dsq_depth = 0u32;
        let mut worst_dsq_cpu = 0usize;
        let mut worst_dsq_sample_idx = 0usize;

        for (i, sample) in report.samples.iter().enumerate() {
            if sample.cpus.is_empty() {
                consecutive_imbalance = 0;
                consecutive_dsq = 0;
                continue;
            }

            // Imbalance check.
            let ratio = sample.imbalance_ratio();
            if ratio > self.max_imbalance_ratio {
                consecutive_imbalance += 1;
                if consecutive_imbalance > worst_imbalance_run {
                    worst_imbalance_run = consecutive_imbalance;
                    worst_imbalance_ratio = ratio;
                    worst_imbalance_sample_idx = i;
                }
            } else {
                consecutive_imbalance = 0;
            }

            // DSQ depth check.
            let mut dsq_violated = false;
            for (cpu_idx, cpu) in sample.cpus.iter().enumerate() {
                if cpu.local_dsq_depth > self.max_local_dsq_depth {
                    dsq_violated = true;
                    if cpu.local_dsq_depth > worst_dsq_depth
                        || (cpu.local_dsq_depth == worst_dsq_depth
                            && consecutive_dsq + 1 > worst_dsq_run)
                    {
                        worst_dsq_depth = cpu.local_dsq_depth;
                        worst_dsq_cpu = cpu_idx;
                    }
                }
            }
            if dsq_violated {
                consecutive_dsq += 1;
                if consecutive_dsq > worst_dsq_run {
                    worst_dsq_run = consecutive_dsq;
                    worst_dsq_sample_idx = i;
                }
            } else {
                consecutive_dsq = 0;
            }
        }

        let mut failed = false;

        if worst_imbalance_run >= self.sustained_samples {
            failed = true;
            details.push(format!(
                "imbalance ratio {:.1} exceeded threshold {:.1} for {} consecutive samples (ending at sample {})",
                worst_imbalance_ratio,
                self.max_imbalance_ratio,
                worst_imbalance_run,
                worst_imbalance_sample_idx,
            ));
        }

        if worst_dsq_run >= self.sustained_samples {
            failed = true;
            details.push(format!(
                "local DSQ depth {} on cpu{} exceeded threshold {} for {} consecutive samples (ending at sample {})",
                worst_dsq_depth,
                worst_dsq_cpu,
                self.max_local_dsq_depth,
                worst_dsq_run,
                worst_dsq_sample_idx,
            ));
        }

        // Stall detection: any CPU whose rq_clock did not advance between
        // consecutive samples.
        if self.fail_on_stall {
            for i in 1..report.samples.len() {
                let prev = &report.samples[i - 1];
                let curr = &report.samples[i];
                let cpu_count = prev.cpus.len().min(curr.cpus.len());
                for cpu in 0..cpu_count {
                    if curr.cpus[cpu].rq_clock != 0
                        && curr.cpus[cpu].rq_clock == prev.cpus[cpu].rq_clock
                    {
                        failed = true;
                        details.push(format!(
                            "rq_clock stall on cpu{} between samples {} and {} (clock={})",
                            cpu,
                            i - 1,
                            i,
                            curr.cpus[cpu].rq_clock,
                        ));
                    }
                }
            }
        }

        // Event counter rate checks: compute per-sample-interval rates
        // and track sustained violations like imbalance.
        let mut consecutive_fallback_rate = 0usize;
        let mut worst_fallback_rate_run = 0usize;
        let mut worst_fallback_rate_value = 0.0f64;
        let mut worst_fallback_rate_sample_idx = 0usize;

        let mut consecutive_keep_last_rate = 0usize;
        let mut worst_keep_last_rate_run = 0usize;
        let mut worst_keep_last_rate_value = 0.0f64;
        let mut worst_keep_last_rate_sample_idx = 0usize;

        for i in 1..report.samples.len() {
            let prev = &report.samples[i - 1];
            let curr = &report.samples[i];
            let interval_s = curr.elapsed_ms.saturating_sub(prev.elapsed_ms) as f64 / 1000.0;
            if interval_s <= 0.0 {
                consecutive_fallback_rate = 0;
                consecutive_keep_last_rate = 0;
                continue;
            }

            // Fallback rate.
            if let (Some(prev_fb), Some(curr_fb)) = (
                prev.sum_event_field(|e| e.select_cpu_fallback),
                curr.sum_event_field(|e| e.select_cpu_fallback),
            ) {
                let rate = (curr_fb - prev_fb) as f64 / interval_s;
                if rate > self.max_fallback_rate {
                    consecutive_fallback_rate += 1;
                    if consecutive_fallback_rate > worst_fallback_rate_run {
                        worst_fallback_rate_run = consecutive_fallback_rate;
                        worst_fallback_rate_value = rate;
                        worst_fallback_rate_sample_idx = i;
                    }
                } else {
                    consecutive_fallback_rate = 0;
                }
            } else {
                consecutive_fallback_rate = 0;
            }

            // Keep-last rate.
            if let (Some(prev_kl), Some(curr_kl)) = (
                prev.sum_event_field(|e| e.dispatch_keep_last),
                curr.sum_event_field(|e| e.dispatch_keep_last),
            ) {
                let rate = (curr_kl - prev_kl) as f64 / interval_s;
                if rate > self.max_keep_last_rate {
                    consecutive_keep_last_rate += 1;
                    if consecutive_keep_last_rate > worst_keep_last_rate_run {
                        worst_keep_last_rate_run = consecutive_keep_last_rate;
                        worst_keep_last_rate_value = rate;
                        worst_keep_last_rate_sample_idx = i;
                    }
                } else {
                    consecutive_keep_last_rate = 0;
                }
            } else {
                consecutive_keep_last_rate = 0;
            }
        }

        if worst_fallback_rate_run >= self.sustained_samples {
            failed = true;
            details.push(format!(
                "fallback rate {:.1}/s exceeded threshold {:.1}/s for {} consecutive intervals (ending at sample {})",
                worst_fallback_rate_value,
                self.max_fallback_rate,
                worst_fallback_rate_run,
                worst_fallback_rate_sample_idx,
            ));
        }

        if worst_keep_last_rate_run >= self.sustained_samples {
            failed = true;
            details.push(format!(
                "keep_last rate {:.1}/s exceeded threshold {:.1}/s for {} consecutive intervals (ending at sample {})",
                worst_keep_last_rate_value,
                self.max_keep_last_rate,
                worst_keep_last_rate_run,
                worst_keep_last_rate_sample_idx,
            ));
        }

        let summary = if failed {
            format!("monitor FAILED: {} violation(s)", details.len())
        } else {
            "monitor OK".into()
        };

        MonitorVerdict {
            passed: !failed,
            details,
            summary,
        }
    }

    /// Check whether the monitor samples contain plausible data.
    ///
    /// Returns false when the data looks like uninitialized guest memory:
    /// - All rq_clock values across every CPU in every sample are identical
    ///   (the kernel never wrote to these fields).
    /// - Any local_dsq_depth exceeds a plausibility ceiling (real kernels
    ///   never queue millions of tasks on a single CPU's local DSQ).
    fn data_looks_valid(samples: &[MonitorSample]) -> bool {
        let mut first_clock: Option<u64> = None;
        let mut all_clocks_same = true;

        for sample in samples {
            if !sample_looks_valid(sample) {
                return false;
            }
            for cpu in &sample.cpus {
                match first_clock {
                    None => first_clock = Some(cpu.rq_clock),
                    Some(fc) => {
                        if cpu.rq_clock != fc {
                            all_clocks_same = false;
                        }
                    }
                }
            }
        }

        // If we saw at least 2 clock readings and they were all identical,
        // the data is uninitialized.
        if first_clock.is_some() && all_clocks_same {
            // Check we actually had multiple readings to compare.
            let total_readings: usize = samples.iter().map(|s| s.cpus.len()).sum();
            if total_readings > 1 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_samples_default_summary() {
        let summary = MonitorSummary::from_samples(&[]);
        assert_eq!(summary.total_samples, 0);
        assert_eq!(summary.max_imbalance_ratio, 0.0);
        assert_eq!(summary.max_local_dsq_depth, 0);
        assert!(!summary.stall_detected);
    }

    #[test]
    fn single_sample_imbalanced_cpus() {
        let sample = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    local_dsq_depth: 3,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 4,
                    local_dsq_depth: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        assert!((summary.max_imbalance_ratio - 4.0).abs() < f64::EPSILON);
        assert_eq!(summary.max_local_dsq_depth, 3);
        assert!(!summary.stall_detected);
    }

    #[test]
    fn stall_detected_when_clock_stuck() {
        let s1 = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[s1, s2]);
        assert!(summary.stall_detected);
    }

    #[test]
    fn balanced_cpus_ratio_one() {
        let sample = MonitorSample {
            elapsed_ms: 50,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 3,
                    rq_clock: 100,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 3,
                    rq_clock: 200,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert!(!summary.stall_detected);
    }

    #[test]
    fn single_cpu_no_division_by_zero() {
        let sample = MonitorSample {
            elapsed_ms: 10,
            cpus: vec![CpuSnapshot {
                nr_running: 5,
                local_dsq_depth: 2,
                rq_clock: 1000,
                ..Default::default()
            }],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        // Single CPU: min == max, ratio = 1.0
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert_eq!(summary.max_local_dsq_depth, 2);
        assert!(!summary.stall_detected);
    }

    #[test]
    fn all_zero_snapshots() {
        let sample = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        // nr_running=0 for all CPUs: max/max(min,1) = 0/1 = 0.0, but
        // initial max_imbalance_ratio is 1.0 and 0.0 < 1.0, so stays 1.0.
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert_eq!(summary.max_local_dsq_depth, 0);
        // rq_clock=0 is excluded from stall detection
        assert!(!summary.stall_detected);
    }

    #[test]
    fn empty_cpus_in_sample() {
        let sample = MonitorSample {
            elapsed_ms: 10,
            cpus: vec![],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert_eq!(summary.total_samples, 1);
        // Empty cpus slice is skipped via `continue`
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn min_nr_zero_division_guard() {
        // All CPUs have nr_running=0. The code uses min_nr.max(1) as
        // divisor, so ratio = 0/1 = 0.0, which is < initial 1.0.
        let sample = MonitorSample {
            elapsed_ms: 10,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 100,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 200,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        // Should not panic from division by zero.
        // max_imbalance_ratio stays at initial 1.0 since 0/1=0 < 1.0.
        assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn min_nr_zero_max_nr_nonzero() {
        // min_nr=0, max_nr=5: ratio = 5/max(0,1) = 5.0
        let sample = MonitorSample {
            elapsed_ms: 10,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 100,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 200,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[sample]);
        assert!((summary.max_imbalance_ratio - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn advancing_clocks_no_stall() {
        let s1 = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2500,
                    ..Default::default()
                },
            ],
        };
        let s3 = MonitorSample {
            elapsed_ms: 300,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[s1, s2, s3]);
        assert!(!summary.stall_detected);
        assert_eq!(summary.total_samples, 3);
    }

    #[test]
    fn different_length_cpu_vecs() {
        // First sample has 2 CPUs, second has 3. Stall detection uses
        // min(prev.len, curr.len) = 2, so only CPUs 0-1 are compared.
        let s1 = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        };
        let s2 = MonitorSample {
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        };
        let summary = MonitorSummary::from_samples(&[s1, s2]);
        assert!(!summary.stall_detected);
        assert_eq!(summary.total_samples, 2);
        // max_local_dsq_depth comes from all CPUs in all samples.
        assert_eq!(summary.max_local_dsq_depth, 0);
    }

    // -- MonitorThresholds tests --

    fn balanced_sample(elapsed_ms: u64, clock_base: u64) -> MonitorSample {
        MonitorSample {
            elapsed_ms,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base,
                    local_dsq_depth: 3,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base + 100,
                    local_dsq_depth: 2,
                    ..Default::default()
                },
            ],
        }
    }

    #[test]
    fn thresholds_default_values() {
        let t = MonitorThresholds::default();
        assert!((t.max_imbalance_ratio - 4.0).abs() < f64::EPSILON);
        assert_eq!(t.max_local_dsq_depth, 50);
        assert!(t.fail_on_stall);
        assert_eq!(t.sustained_samples, 5);
    }

    #[test]
    fn thresholds_empty_report_passes() {
        let t = MonitorThresholds::default();
        let report = MonitorReport {
            samples: vec![],
            summary: MonitorSummary::default(),
        };
        let v = t.evaluate(&report);
        assert!(v.passed);
        assert!(v.details.is_empty());
    }

    #[test]
    fn thresholds_balanced_samples_pass() {
        let t = MonitorThresholds::default();
        let samples: Vec<_> = (0..10)
            .map(|i| balanced_sample(i * 100, 1000 + i * 500))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(v.passed, "balanced samples should pass: {:?}", v.details);
    }

    #[test]
    fn thresholds_imbalance_below_sustained_passes() {
        let t = MonitorThresholds {
            sustained_samples: 5,
            max_imbalance_ratio: 4.0,
            ..Default::default()
        };
        // 4 consecutive imbalanced samples (below sustained_samples=5).
        let mut samples = Vec::new();
        for i in 0..4 {
            samples.push(MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        // Then a balanced one to break the streak.
        samples.push(balanced_sample(400, 3000));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "4 imbalanced < sustained_samples=5: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_imbalance_at_sustained_fails() {
        let t = MonitorThresholds {
            sustained_samples: 5,
            max_imbalance_ratio: 4.0,
            ..Default::default()
        };
        // 5 consecutive imbalanced samples (ratio=10, threshold=4).
        let mut samples = Vec::new();
        for i in 0..5u64 {
            samples.push(MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("imbalance")));
    }

    #[test]
    fn thresholds_dsq_depth_sustained_fails() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_local_dsq_depth: 10,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples = Vec::new();
        for i in 0..3u64 {
            samples.push(MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 20,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 5,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("DSQ depth")));
    }

    #[test]
    fn thresholds_dsq_depth_below_sustained_passes() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_local_dsq_depth: 10,
            fail_on_stall: false,
            ..Default::default()
        };
        // Only 2 consecutive DSQ violations, then a clean sample.
        let mut samples = Vec::new();
        for i in 0..2u64 {
            samples.push(MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 20,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 2,
                        local_dsq_depth: 5,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        samples.push(balanced_sample(200, 2000));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(v.passed, "2 DSQ violations < sustained=3: {:?}", v.details);
    }

    #[test]
    fn thresholds_stall_detected_fails() {
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 100,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    }, // stuck
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
    }

    #[test]
    fn thresholds_stall_disabled_passes() {
        let t = MonitorThresholds {
            fail_on_stall: false,
            sustained_samples: 100,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                }],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    }, // stuck but stall check disabled
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(v.passed, "stall disabled should pass: {:?}", v.details);
    }

    #[test]
    fn thresholds_imbalance_interrupted_by_balanced_resets() {
        // 3 imbalanced, 1 balanced, 3 imbalanced — never reaches sustained=5.
        let t = MonitorThresholds {
            sustained_samples: 5,
            max_imbalance_ratio: 4.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples = Vec::new();
        for i in 0..3u64 {
            samples.push(MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        samples.push(balanced_sample(300, 2500));
        for i in 4..7u64 {
            samples.push(MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 3000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 3100 + i * 500,
                        ..Default::default()
                    },
                ],
            });
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "interrupted imbalance should pass: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_multiple_violations() {
        // Both imbalance and stall in the same report.
        let t = MonitorThresholds {
            sustained_samples: 2,
            max_imbalance_ratio: 2.0,
            fail_on_stall: true,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 5,
                        rq_clock: 2000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    }, // stall + imbalance
                    CpuSnapshot {
                        nr_running: 5,
                        rq_clock: 3000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("imbalance")));
        assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
    }

    #[test]
    fn thresholds_empty_cpus_samples_pass() {
        let t = MonitorThresholds::default();
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(v.passed);
    }

    #[test]
    fn thresholds_uninitialized_memory_passes() {
        // Simulates what happens when monitor reads guest memory before
        // kernel initialization: all rq_clocks identical, DSQ depths garbage.
        let t = MonitorThresholds::default();
        let garbage_clock = 10314579376562252011u64;
        let samples: Vec<_> = (0..10)
            .map(|i| MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 0,
                        rq_clock: garbage_clock,
                        local_dsq_depth: 1550435906,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 0,
                        rq_clock: garbage_clock,
                        local_dsq_depth: 1550435906,
                        ..Default::default()
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "uninitialized guest memory should be skipped: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_all_same_clocks_passes() {
        // All clocks identical across all CPUs and samples = uninitialized.
        let t = MonitorThresholds {
            fail_on_stall: true,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "all-same clocks should be treated as uninitialized: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_dsq_over_plausibility_ceiling_passes() {
        let t = MonitorThresholds::default();
        let samples = vec![MonitorSample {
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    local_dsq_depth: 50000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 2000,
                    local_dsq_depth: 5,
                    ..Default::default()
                },
            ],
        }];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "implausible DSQ depth should skip evaluation: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_single_cpu_single_sample_valid() {
        // A single reading cannot be compared, so all_clocks_same with
        // total_readings=1 should still be treated as valid.
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };
        let samples = vec![MonitorSample {
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                ..Default::default()
            }],
        }];
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(v.passed, "single reading should be valid: {:?}", v.details);
    }

    // -- Event counter rate threshold tests --

    /// Build a sample with event counters. Each CPU gets the same counter
    /// values so the total across CPUs = ncpus * per_cpu_value.
    fn sample_with_events(
        elapsed_ms: u64,
        clock_base: u64,
        fallback: i64,
        keep_last: i64,
    ) -> MonitorSample {
        MonitorSample {
            elapsed_ms,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: fallback,
                        dispatch_keep_last: keep_last,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: clock_base + 100,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: fallback,
                        dispatch_keep_last: keep_last,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        }
    }

    #[test]
    fn thresholds_fallback_rate_sustained_fails() {
        // sustained_samples=3, max_fallback_rate=10.0.
        // 100ms intervals, 2 CPUs. Each CPU increments fallback by 10
        // per sample -> delta = 20 total per interval / 0.1s = 200/s > 10.
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_fallback_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..4)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("fallback rate")));
    }

    #[test]
    fn thresholds_fallback_rate_below_sustained_passes() {
        // 2 violating intervals then a clean one — below sustained=3.
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_fallback_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples: Vec<_> = (0..3)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
            .collect();
        // 4th sample: same fallback as 3rd -> rate = 0.
        samples.push(sample_with_events(300, 2500, 20, 0));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
    }

    #[test]
    fn thresholds_keep_last_rate_sustained_fails() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_keep_last_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..4)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        assert!(v.details.iter().any(|d| d.contains("keep_last rate")));
    }

    #[test]
    fn thresholds_keep_last_rate_below_sustained_passes() {
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_keep_last_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples: Vec<_> = (0..3)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
            .collect();
        // Reset: same keep_last as previous -> rate = 0.
        samples.push(sample_with_events(300, 2500, 0, 20));
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
    }

    #[test]
    fn thresholds_event_rate_interrupted_resets() {
        // 2 violating intervals, 1 clean, 2 violating — never reaches sustained=3.
        let t = MonitorThresholds {
            sustained_samples: 3,
            max_fallback_rate: 10.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let mut samples = Vec::new();
        // 3 samples = 2 intervals of high fallback rate.
        for i in 0..3u64 {
            samples.push(sample_with_events(
                i * 100,
                1000 + i * 500,
                i as i64 * 10,
                0,
            ));
        }
        // Clean interval: same fallback -> rate = 0.
        samples.push(sample_with_events(300, 2500, 20, 0));
        // 3 more samples = 2 intervals of high fallback rate (not 3).
        // The fallback delta for the first interval covers sample 3->4,
        // which is (30-20)/0.1 = 100/s (violating), then 4->5 is also
        // violating. That's 2 intervals, below sustained=3.
        for i in 0..2u64 {
            samples.push(sample_with_events(
                400 + i * 100,
                3000 + i * 500,
                30 + (i + 1) as i64 * 10,
                0,
            ));
        }
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "interrupted rate violations should pass: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_no_event_counters_skips_rate_check() {
        // Samples without event counters should not trigger rate violations.
        let t = MonitorThresholds {
            sustained_samples: 1,
            max_fallback_rate: 0.0, // any rate would fail
            max_keep_last_rate: 0.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..5)
            .map(|i| balanced_sample(i * 100, 1000 + i * 500))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(
            v.passed,
            "no event counters should skip rate check: {:?}",
            v.details
        );
    }

    #[test]
    fn thresholds_default_event_rate_values() {
        let t = MonitorThresholds::default();
        assert!((t.max_fallback_rate - 200.0).abs() < f64::EPSILON);
        assert!((t.max_keep_last_rate - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn thresholds_merge_event_rate_overrides() {
        let base = MonitorThresholds::DEFAULT;
        let overrides = ThresholdOverrides {
            max_fallback_rate: Some(50.0),
            max_keep_last_rate: Some(25.0),
            ..Default::default()
        };
        let merged = base.merge(&overrides);
        assert!((merged.max_fallback_rate - 50.0).abs() < f64::EPSILON);
        assert!((merged.max_keep_last_rate - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn thresholds_merge_event_rate_none_keeps_default() {
        let base = MonitorThresholds::DEFAULT;
        let overrides = ThresholdOverrides::NONE;
        let merged = base.merge(&overrides);
        assert!((merged.max_fallback_rate - 200.0).abs() < f64::EPSILON);
        assert!((merged.max_keep_last_rate - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn summary_keep_last_rate_computed() {
        // 2 CPUs, each with keep_last incrementing by 5 per sample.
        // 3 samples over 200ms -> total delta = 2*10 = 20, rate = 20/0.2 = 100.
        let samples = vec![
            sample_with_events(0, 1000, 0, 0),
            sample_with_events(100, 1500, 0, 5),
            sample_with_events(200, 2000, 0, 10),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let deltas = summary.event_deltas.unwrap();
        assert!((deltas.keep_last_rate - 100.0).abs() < f64::EPSILON);
    }

    // -- compute_event_deltas edge cases --

    #[test]
    fn event_deltas_none_without_counters() {
        let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(summary.event_deltas.is_none());
    }

    #[test]
    fn event_deltas_single_sample() {
        // Only one sample with events -> first == last, duration=0, rates=0.
        let samples = vec![sample_with_events(100, 1000, 50, 25)];
        let summary = MonitorSummary::from_samples(&samples);
        let deltas = summary.event_deltas.unwrap();
        assert_eq!(deltas.fallback_rate, 0.0);
        assert_eq!(deltas.keep_last_rate, 0.0);
    }

    #[test]
    fn event_deltas_max_fallback_burst() {
        // 3 samples: burst between samples 1 and 2.
        let samples = vec![
            sample_with_events(0, 1000, 0, 0),
            sample_with_events(100, 1500, 5, 0),
            sample_with_events(200, 2000, 100, 0),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let deltas = summary.event_deltas.unwrap();
        // Per-CPU: burst is (100-5)*2 = 190 across 2 CPUs.
        assert!(deltas.max_fallback_burst > 0);
    }

    #[test]
    fn event_deltas_all_counters_computed() {
        let make = |elapsed_ms, fb, kl, dsq_off, exit, migdis| MonitorSample {
            elapsed_ms,
            cpus: vec![CpuSnapshot {
                nr_running: 1,
                rq_clock: elapsed_ms * 10,
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: fb,
                    dispatch_local_dsq_offline: dsq_off,
                    dispatch_keep_last: kl,
                    enq_skip_exiting: exit,
                    enq_skip_migration_disabled: migdis,
                }),
                ..Default::default()
            }],
        };
        let samples = vec![
            make(100, 10, 20, 30, 40, 50),
            make(200, 110, 120, 130, 140, 150),
        ];
        let summary = MonitorSummary::from_samples(&samples);
        let d = summary.event_deltas.unwrap();
        assert_eq!(d.total_fallback, 100);
        assert_eq!(d.total_dispatch_keep_last, 100);
        assert_eq!(d.total_dispatch_offline, 100);
        assert_eq!(d.total_enq_skip_exiting, 100);
        assert_eq!(d.total_enq_skip_migration_disabled, 100);
    }

    // -- ThresholdOverrides merge --

    #[test]
    fn threshold_overrides_partial_merge() {
        let base = MonitorThresholds::DEFAULT;
        let overrides = ThresholdOverrides {
            max_imbalance_ratio: Some(10.0),
            fail_on_stall: Some(false),
            ..Default::default()
        };
        let merged = base.merge(&overrides);
        assert!((merged.max_imbalance_ratio - 10.0).abs() < f64::EPSILON);
        assert!(!merged.fail_on_stall);
        // Unoverridden fields keep defaults.
        assert_eq!(merged.max_local_dsq_depth, 50);
        assert_eq!(merged.sustained_samples, 5);
    }

    // -- data_looks_valid tests --

    #[test]
    fn data_looks_valid_empty() {
        assert!(MonitorThresholds::data_looks_valid(&[]));
    }

    #[test]
    fn data_looks_valid_normal() {
        let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 2000)];
        assert!(MonitorThresholds::data_looks_valid(&samples));
    }

    #[test]
    fn data_looks_valid_all_same_clocks() {
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        rq_clock: 5000,
                        ..Default::default()
                    },
                ],
            },
        ];
        assert!(!MonitorThresholds::data_looks_valid(&samples));
    }

    #[test]
    fn data_looks_valid_dsq_over_ceiling() {
        let samples = vec![MonitorSample {
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: 50000,
                rq_clock: 1000,
                ..Default::default()
            }],
        }];
        assert!(!MonitorThresholds::data_looks_valid(&samples));
    }

    // -- MonitorSample::imbalance_ratio tests --

    #[test]
    fn imbalance_ratio_empty_cpus() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![],
        };
        assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_single_cpu() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![CpuSnapshot {
                nr_running: 5,
                ..Default::default()
            }],
        };
        assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_balanced() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 3,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 3,
                    ..Default::default()
                },
            ],
        };
        assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_imbalanced() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 8,
                    ..Default::default()
                },
            ],
        };
        assert!((s.imbalance_ratio() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn imbalance_ratio_zero_min() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 5,
                    ..Default::default()
                },
            ],
        };
        // min=0, max(0,1)=1, ratio=5/1=5.0
        assert!((s.imbalance_ratio() - 5.0).abs() < f64::EPSILON);
    }

    // -- MonitorSample::sum_event_field tests --

    #[test]
    fn sum_event_field_none_when_no_counters() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
        };
        assert!(s.sum_event_field(|e| e.select_cpu_fallback).is_none());
    }

    #[test]
    fn sum_event_field_sums_across_cpus() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: 10,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                CpuSnapshot {
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: 20,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
        };
        assert_eq!(s.sum_event_field(|e| e.select_cpu_fallback), Some(30));
    }

    #[test]
    fn sum_event_field_mixed_some_none() {
        let s = MonitorSample {
            elapsed_ms: 0,
            cpus: vec![
                CpuSnapshot {
                    event_counters: Some(ScxEventCounters {
                        dispatch_keep_last: 7,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                CpuSnapshot::default(),
            ],
        };
        assert_eq!(s.sum_event_field(|e| e.dispatch_keep_last), Some(7));
    }

    // -- sample_looks_valid tests --

    #[test]
    fn sample_looks_valid_normal() {
        let s = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: 5,
                ..Default::default()
            }],
        };
        assert!(sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_at_ceiling() {
        let s = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: DSQ_PLAUSIBILITY_CEILING,
                ..Default::default()
            }],
        };
        assert!(sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_over_ceiling() {
        let s = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![CpuSnapshot {
                local_dsq_depth: DSQ_PLAUSIBILITY_CEILING + 1,
                ..Default::default()
            }],
        };
        assert!(!sample_looks_valid(&s));
    }

    #[test]
    fn sample_looks_valid_empty_cpus() {
        let s = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![],
        };
        assert!(sample_looks_valid(&s));
    }

    // -- MonitorSummary field value assertions --

    #[test]
    fn from_samples_fields_sane_values() {
        let samples: Vec<_> = (0..5u64)
            .map(|i| MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: (i as u32 + 1),
                        scx_nr_running: i as u32,
                        local_dsq_depth: (i as u32) % 3,
                        rq_clock: 1000 + i * 500,
                        scx_flags: 0,
                        event_counters: Some(ScxEventCounters {
                            select_cpu_fallback: i as i64 * 2,
                            dispatch_keep_last: i as i64,
                            ..Default::default()
                        }),
                    },
                    CpuSnapshot {
                        nr_running: (i as u32 + 2),
                        scx_nr_running: i as u32 + 1,
                        local_dsq_depth: 0,
                        rq_clock: 1100 + i * 600,
                        scx_flags: 0,
                        event_counters: Some(ScxEventCounters {
                            select_cpu_fallback: i as i64 * 3,
                            dispatch_keep_last: i as i64 * 2,
                            ..Default::default()
                        }),
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        // total_samples matches input count
        assert_eq!(summary.total_samples, 5);
        // max_imbalance_ratio: all samples have nr_running differing by 1,
        // worst case is sample 0: nr_running=[1,2] -> ratio=2.0
        assert!(
            summary.max_imbalance_ratio >= 1.0,
            "ratio must be >= 1.0: {}",
            summary.max_imbalance_ratio
        );
        assert!(
            summary.max_imbalance_ratio <= 10.0,
            "ratio must be reasonable: {}",
            summary.max_imbalance_ratio
        );
        // max_local_dsq_depth: worst is (4 % 3) = 1 on cpu0 at i=4, or (3 % 3)=0 at i=3, (2%3)=2 at i=2
        assert!(
            summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
            "dsq depth must be below plausibility ceiling: {}",
            summary.max_local_dsq_depth
        );
        assert!(
            summary.max_local_dsq_depth <= 10,
            "dsq depth must be small in this controlled test: {}",
            summary.max_local_dsq_depth
        );
        // stall_detected: rq_clock advances each sample, so no stall
        assert!(
            !summary.stall_detected,
            "no stall expected with advancing rq_clock"
        );
        // event_deltas: should be computed
        let deltas = summary
            .event_deltas
            .as_ref()
            .expect("event deltas must be present");
        assert!(
            deltas.total_fallback >= 0,
            "fallback count must be non-negative"
        );
        assert!(
            deltas.total_dispatch_keep_last >= 0,
            "keep_last count must be non-negative"
        );
        assert!(
            deltas.fallback_rate >= 0.0,
            "fallback rate must be non-negative"
        );
        assert!(
            deltas.keep_last_rate >= 0.0,
            "keep_last rate must be non-negative"
        );
    }

    #[test]
    fn from_samples_empty_all_defaults() {
        // Verify every field of MonitorSummary defaults correctly for empty input,
        // including event_deltas which empty_samples_default_summary does not check.
        let summary = MonitorSummary::from_samples(&[]);
        assert_eq!(summary.total_samples, 0);
        assert_eq!(summary.max_imbalance_ratio, 0.0);
        assert_eq!(summary.max_local_dsq_depth, 0);
        assert!(!summary.stall_detected);
        assert!(
            summary.event_deltas.is_none(),
            "empty input must not produce event deltas"
        );
    }

    // ---------------------------------------------------------------
    // Negative tests: verify monitor diagnostics catch controlled failures
    // ---------------------------------------------------------------

    #[test]
    fn neg_tight_imbalance_threshold_catches_mild_imbalance() {
        let t = MonitorThresholds {
            max_imbalance_ratio: 1.0,
            sustained_samples: 2,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 2,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 3,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.max_imbalance_ratio >= 1.5,
            "summary must capture ratio"
        );
        assert!(!summary.stall_detected, "no stall in this scenario");
        assert_eq!(summary.total_samples, 3);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed, "imbalance=1.5 must fail threshold=1.0");
        // Format: "imbalance ratio 1.5 exceeded threshold 1.0 for 2 consecutive samples (ending at sample 2)"
        let detail = v.details.iter().find(|d| d.contains("imbalance")).unwrap();
        assert!(detail.contains("ratio"), "must include 'ratio': {detail}");
        assert!(
            detail.contains("exceeded threshold"),
            "must include threshold: {detail}"
        );
        assert!(
            detail.contains("1.0"),
            "must show threshold value: {detail}"
        );
        assert!(
            detail.contains("consecutive samples"),
            "must show sustained count: {detail}"
        );
        assert!(
            detail.contains("ending at sample"),
            "must show sample index: {detail}"
        );
        assert!(
            v.summary.contains("FAILED"),
            "summary must say FAILED: {}",
            v.summary
        );
    }

    #[test]
    fn neg_tight_dsq_threshold_catches_small_depth() {
        let t = MonitorThresholds {
            max_local_dsq_depth: 1,
            sustained_samples: 2,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| MonitorSample {
                elapsed_ms: i * 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        local_dsq_depth: 3,
                        rq_clock: 1000 + i * 500,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        local_dsq_depth: 0,
                        rq_clock: 1100 + i * 500,
                        ..Default::default()
                    },
                ],
            })
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert_eq!(
            summary.max_local_dsq_depth, 3,
            "summary must capture max depth"
        );
        assert!(
            summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
            "depth must be plausible"
        );
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed, "dsq_depth=3 must fail threshold=1");
        // Format: "local DSQ depth 3 on cpu0 exceeded threshold 1 for 2 consecutive samples (ending at sample 2)"
        let detail = v.details.iter().find(|d| d.contains("DSQ depth")).unwrap();
        assert!(detail.contains("3"), "must show depth value: {detail}");
        assert!(detail.contains("cpu0"), "must show CPU number: {detail}");
        assert!(
            detail.contains("threshold 1"),
            "must show threshold: {detail}"
        );
        assert!(
            detail.contains("consecutive samples"),
            "must show count: {detail}"
        );
    }

    #[test]
    fn neg_stall_detection_catches_frozen_rq_clock() {
        let t = MonitorThresholds {
            fail_on_stall: true,
            sustained_samples: 100,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 6000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 5000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 7000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.stall_detected,
            "summary.stall_detected must be true"
        );
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed, "frozen rq_clock must be detected");
        // Format: "rq_clock stall on cpu0 between samples 0 and 1 (clock=5000)"
        let detail = v
            .details
            .iter()
            .find(|d| d.contains("rq_clock stall"))
            .unwrap();
        assert!(detail.contains("cpu0"), "must name frozen CPU: {detail}");
        assert!(
            detail.contains("between samples 0 and 1"),
            "must name sample indices: {detail}"
        );
        assert!(
            detail.contains("clock=5000"),
            "must include frozen clock value: {detail}"
        );
    }

    #[test]
    fn neg_combined_imbalance_and_stall_both_reported() {
        let t = MonitorThresholds {
            max_imbalance_ratio: 2.0,
            sustained_samples: 1,
            fail_on_stall: true,
            ..Default::default()
        };
        let samples = vec![
            MonitorSample {
                elapsed_ms: 100,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 2000,
                        ..Default::default()
                    },
                ],
            },
            MonitorSample {
                elapsed_ms: 200,
                cpus: vec![
                    CpuSnapshot {
                        nr_running: 1,
                        rq_clock: 1000,
                        ..Default::default()
                    },
                    CpuSnapshot {
                        nr_running: 10,
                        rq_clock: 3000,
                        ..Default::default()
                    },
                ],
            },
        ];
        let summary = MonitorSummary::from_samples(&samples);
        assert!(summary.stall_detected);
        assert!(summary.max_imbalance_ratio >= 10.0);
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed);
        let imb = v.details.iter().find(|d| d.contains("imbalance")).unwrap();
        assert!(
            imb.contains("exceeded threshold 2.0"),
            "imbalance format: {imb}"
        );
        let stall = v
            .details
            .iter()
            .find(|d| d.contains("rq_clock stall"))
            .unwrap();
        assert!(stall.contains("cpu0"), "stall format: {stall}");
        assert!(
            v.details.len() >= 2,
            "both violations must be reported, got {}",
            v.details.len()
        );
        assert!(v.summary.contains("FAILED"), "summary: {}", v.summary);
    }

    #[test]
    fn neg_fallback_rate_threshold_fires() {
        let t = MonitorThresholds {
            sustained_samples: 2,
            max_fallback_rate: 5.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert!(
            summary.event_deltas.is_some(),
            "event deltas must be computed"
        );
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed, "fallback rate must be caught");
        // Format: "fallback rate 200.0/s exceeded threshold 5.0/s for 2 consecutive intervals (ending at sample 2)"
        let detail = v
            .details
            .iter()
            .find(|d| d.contains("fallback rate"))
            .unwrap();
        assert!(detail.contains("/s"), "must include rate unit: {detail}");
        assert!(
            detail.contains("exceeded threshold"),
            "must state threshold: {detail}"
        );
        assert!(
            detail.contains("5.0/s"),
            "must show threshold value: {detail}"
        );
        assert!(
            detail.contains("consecutive intervals"),
            "must show sustained count: {detail}"
        );
    }

    #[test]
    fn neg_keep_last_rate_threshold_fires() {
        let t = MonitorThresholds {
            sustained_samples: 2,
            max_keep_last_rate: 5.0,
            fail_on_stall: false,
            ..Default::default()
        };
        let samples: Vec<_> = (0..3u64)
            .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
            .collect();
        let summary = MonitorSummary::from_samples(&samples);
        assert!(summary.event_deltas.is_some());
        let report = MonitorReport { samples, summary };
        let v = t.evaluate(&report);
        assert!(!v.passed, "keep_last rate must be caught");
        // Format: "keep_last rate .../s exceeded threshold 5.0/s for 2 consecutive intervals ..."
        let detail = v
            .details
            .iter()
            .find(|d| d.contains("keep_last rate"))
            .unwrap();
        assert!(detail.contains("/s"), "must include rate unit: {detail}");
        assert!(
            detail.contains("exceeded threshold"),
            "must state threshold: {detail}"
        );
        assert!(
            detail.contains("5.0/s"),
            "must show threshold value: {detail}"
        );
    }
}
