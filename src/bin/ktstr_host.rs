use std::time::Duration;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use ktstr::cgroup::CgroupManager;
use ktstr::runner::{RunConfig, Runner};
use ktstr::scenario::{self, flags};
use ktstr::topology::TestTopology;

#[derive(Parser)]
#[command(
    name = "ktstr-host",
    about = "Run ktstr scheduler test scenarios on the host"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run test scenarios against a scheduler.
    Run {
        /// Scheduler binary path.
        #[arg(long)]
        scheduler: Option<String>,

        /// Extra arguments passed to the scheduler.
        #[arg(long = "sched-arg", num_args = 1)]
        sched_args: Vec<String>,

        /// Scenario duration in seconds.
        #[arg(long, default_value = "20")]
        duration: u64,

        /// Workers per cgroup.
        #[arg(long, default_value = "4")]
        workers: usize,

        /// Active flags (comma-separated). Omit for all profiles.
        /// Valid: llc, borrow, steal, rebal, reject-pin, no-ctrl.
        #[arg(long, value_delimiter = ',')]
        flags: Option<Vec<String>>,

        /// Filter scenarios by name substring.
        #[arg(long)]
        filter: Option<String>,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Verbose output.
        #[arg(long, short)]
        verbose: bool,

        /// Enable repro mode (attach BPF probes).
        #[arg(long)]
        repro: bool,

        /// Crash stack for auto-probe (file path or comma-separated function names).
        #[arg(long)]
        probe_stack: Option<String>,

        /// Enable auto-repro on crash.
        #[arg(long)]
        auto_repro: bool,

        /// Kernel build directory (for DWARF source locations).
        #[arg(long)]
        kernel_dir: Option<String>,

        /// Override work type for all cgroups.
        /// Valid: CpuSpin, YieldHeavy, Mixed, IoSync, Bursty, PipeIo,
        /// FutexPingPong, CachePressure, CacheYield, CachePipe, FutexFanOut.
        #[arg(long)]
        work_type: Option<String>,

        /// Parent cgroup path.
        #[arg(long, default_value = "/sys/fs/cgroup/ktstr")]
        parent_cgroup: String,
    },
    /// List available scenarios.
    List {
        /// Filter scenarios by name substring.
        #[arg(long)]
        filter: Option<String>,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show host CPU topology.
    Topo,
    /// Clean up leftover cgroups.
    Cleanup {
        /// Parent cgroup path.
        #[arg(long, default_value = "/sys/fs/cgroup/ktstr")]
        parent_cgroup: String,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            scheduler,
            sched_args,
            duration,
            workers,
            flags: flag_arg,
            filter,
            json,
            verbose,
            repro,
            probe_stack,
            auto_repro,
            kernel_dir,
            work_type,
            parent_cgroup,
        } => {
            let active_flags = match flag_arg {
                Some(fs) => {
                    let mut resolved = Vec::new();
                    for f in &fs {
                        match flags::from_short_name(f) {
                            Some(name) => resolved.push(name),
                            None => bail!(
                                "unknown flag: '{f}'. valid flags: {}",
                                flags::ALL.join(", "),
                            ),
                        }
                    }
                    Some(resolved)
                }
                None => None,
            };

            let work_type_override = match work_type {
                Some(ref name) => {
                    let wt = ktstr::workload::WorkType::from_name(name);
                    if wt.is_none() {
                        bail!(
                            "unknown work type: '{name}'. valid types: {}",
                            ktstr::workload::WorkType::ALL_NAMES.join(", "),
                        );
                    }
                    wt
                }
                None => None,
            };

            let config = RunConfig {
                scheduler_bin: scheduler,
                scheduler_args: sched_args,
                parent_cgroup,
                duration: Duration::from_secs(duration),
                workers_per_cgroup: workers,
                json,
                verbose,
                active_flags,
                repro,
                probe_stack,
                auto_repro,
                kernel_dir,
                work_type_override,
                ..Default::default()
            };

            let topo = TestTopology::from_system()?;
            let runner = Runner::new(config, topo)?;

            let scenarios = scenario::all_scenarios();
            let refs: Vec<&scenario::Scenario> = scenarios
                .iter()
                .filter(|s| {
                    filter
                        .as_ref()
                        .map_or(true, |f| s.name.contains(f.as_str()))
                })
                .collect();

            if refs.is_empty() {
                bail!(
                    "no scenarios matched filter. run 'ktstr-host list' to see available scenarios"
                );
            }

            let results = runner.run_scenarios(&refs)?;

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                for r in &results {
                    let status = if r.passed { "PASS" } else { "FAIL" };
                    println!("[{status}] {} ({:.1}s)", r.scenario_name, r.duration_s);
                    for d in &r.details {
                        println!("  {d}");
                    }
                }
                let passed = results.iter().filter(|r| r.passed).count();
                let total = results.len();
                println!("\n{passed}/{total} passed");
            }
        }

        Command::List { filter, json } => {
            let scenarios = scenario::all_scenarios();
            let filtered: Vec<&scenario::Scenario> = scenarios
                .iter()
                .filter(|s| {
                    filter
                        .as_ref()
                        .map_or(true, |f| s.name.contains(f.as_str()))
                })
                .collect();

            if json {
                let entries: Vec<serde_json::Value> = filtered
                    .iter()
                    .map(|s| {
                        let profiles: Vec<String> = s.profiles().iter().map(|p| p.name()).collect();
                        serde_json::json!({
                            "name": s.name,
                            "category": s.category,
                            "description": s.description,
                            "profiles": profiles,
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else {
                for s in &filtered {
                    let profiles: Vec<String> = s.profiles().iter().map(|p| p.name()).collect();
                    println!(
                        "{:<30} [{:<12}] {} (profiles: {})",
                        s.name,
                        s.category,
                        s.description,
                        profiles.join(", "),
                    );
                }
                println!("\n{} scenarios", filtered.len());
            }
        }

        Command::Topo => {
            let topo = TestTopology::from_system()?;
            println!("CPUs:       {}", topo.total_cpus());
            println!("LLCs:       {}", topo.num_llcs());
            println!("NUMA nodes: {}", topo.num_numa_nodes());
            for (i, llc) in topo.llcs().iter().enumerate() {
                println!("  LLC {} (node {}): {:?}", i, llc.numa_node(), llc.cpus(),);
            }
        }

        Command::Cleanup { parent_cgroup } => {
            let cgroups = CgroupManager::new(&parent_cgroup);
            cgroups.cleanup_all()?;
            println!("cleaned up {parent_cgroup}");
        }
    }

    Ok(())
}
