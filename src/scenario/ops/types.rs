//! `Op`, `CgroupDef`, `Step`, and supporting limit types — pure data
//! model extracted from the parent [`super`] module. Re-exported by
//! the parent so external paths remain `crate::scenario::ops::Op` etc.
//! See the parent module for the full module-level documentation
//! (cgroup tooling overview, worked examples, implementation entry
//! points).

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::time::Duration;

use crate::scenario::Ctx;
use crate::workload::{AffinityIntent, WorkSpec, WorkType};

// ---------------------------------------------------------------------------
// Op / CpusetSpec
// ---------------------------------------------------------------------------

/// Atomic operation on the cgroup topology.
///
/// Names use `Cow<'static, str>` so ops can reference compile-time
/// literals (zero-cost) or runtime-generated strings (owned).
///
/// # `#[non_exhaustive]`
///
/// `Op` is `#[non_exhaustive]` — see [`crate::non_exhaustive`] for
/// the cross-crate pattern-match rule. `Op`-specific construction
/// convention: prefer the per-op constructors (e.g. `Op::add_cgroup`,
/// `Op::run_payload`) over naming variants directly; new
/// constructors are added alongside new variants and are the stable
/// surface.
#[derive(Clone, Debug, strum::EnumDiscriminants)]
#[strum_discriminants(name(OpKind))]
#[strum_discriminants(derive(strum::EnumIter))]
#[strum_discriminants(vis(pub))]
#[non_exhaustive]
pub enum Op {
    /// Create a new cgroup under the managed cgroup parent.
    AddCgroup { name: Cow<'static, str> },
    /// Remove a cgroup (stops its workers first).
    RemoveCgroup { cgroup: Cow<'static, str> },
    /// Set a cgroup's cpuset to the resolved CPU set.
    SetCpuset {
        cgroup: Cow<'static, str>,
        cpus: CpusetSpec,
    },
    /// Clear a cgroup's cpuset (allow all CPUs).
    ClearCpuset { cgroup: Cow<'static, str> },
    /// Read both cgroups' cpusets and swap them.
    SwapCpusets {
        a: Cow<'static, str>,
        b: Cow<'static, str>,
    },
    /// Spawn workers and move them into the target cgroup.
    ///
    /// The work type is used as-is; gauntlet `work_type_override` does
    /// not apply. Use [`CgroupDef`] with `swappable(true)` when the
    /// work type should be overridable.
    Spawn {
        cgroup: Cow<'static, str>,
        work: WorkSpec,
    },
    /// Stop all workers in a cgroup (does not remove the cgroup).
    StopCgroup { cgroup: Cow<'static, str> },
    /// Set worker affinity in a cgroup. Resolved at apply time via
    /// [`resolve_affinity_for_cgroup()`](super::resolve_affinity_for_cgroup).
    SetAffinity {
        cgroup: Cow<'static, str>,
        affinity: AffinityIntent,
    },
    /// Spawn workers in the parent cgroup (not in a managed cgroup).
    ///
    /// `WorkSpec` is resolved to a `WorkloadConfig` at apply time, matching
    /// the resolution pattern used by `Op::Spawn`.
    SpawnHost { work: WorkSpec },
    /// Move all tasks from one cgroup to another.
    ///
    /// Each task is moved via `cgroup.procs`. If any move fails, the
    /// error propagates and handle name keys are left unchanged (workers
    /// remain addressed under `from`). On success, handle name keys are
    /// updated to `to` so subsequent ops address the moved workers.
    ///
    /// # Lifetime / ownership-direction asymmetry
    ///
    /// `MoveAllTasks` is asymmetric with respect to cgroup ownership:
    /// the legality of a move depends on the relative lifetimes of
    /// the `from` and `to` cgroups, not just on which one is the
    /// source.
    ///
    /// | `from` ownership      | `to` ownership        | Outcome |
    /// |-----------------------|-----------------------|---------|
    /// | step-local            | step-local            | Allowed; both die at step teardown together. |
    /// | step-local            | Backdrop (persistent) | Allowed; handle ownership transfers from step-local set to Backdrop set so the worker survives step teardown. |
    /// | Backdrop              | Backdrop              | Allowed; both persist for the scenario. |
    /// | Backdrop              | step-local            | **Rejected at apply time.** A persistent worker would be stranded inside a cgroup that gets `rmdir`'d at step boundary; the kernel migrates the orphaned task to the cgroup root with a frozen-task warning in dmesg. The `bail!` diagnostic names the offending pair and tells the operator to either declare the destination in the Backdrop too, or move the worker back into a Backdrop-owned cgroup. |
    ///
    /// The Backdrop→Backdrop and step→step cases are unconditionally
    /// allowed because both endpoints share a lifetime; the
    /// step→Backdrop case is allowed because the kernel moves
    /// reference-count once and the framework's
    /// [`ScenarioState::rename_handles`](super::ScenarioState::rename_handles)
    /// transfers the handle into the persistent slot in the same
    /// step. The Backdrop→step case is the only one that produces
    /// a guaranteed orphan, hence the asymmetric reject.
    ///
    /// # Backdrop-setup exemption
    ///
    /// `MoveAllTasks` ops running INSIDE a Backdrop's `setup_ops`
    /// pass (`state.target_backdrop=true`) are exempt from the
    /// Backdrop→step-local check: at that point, "step-local"
    /// cgroups don't exist yet (the Backdrop is the only cgroup
    /// scope), and the rule reduces to a pure source-ownership
    /// check that the apply path handles already.
    MoveAllTasks {
        from: Cow<'static, str>,
        to: Cow<'static, str>,
    },
    /// Spawn a userspace [`Payload`](crate::test_support::Payload)
    /// binary in the background and track its
    /// [`PayloadHandle`](crate::scenario::payload_run::PayloadHandle)
    /// under the step's payload-handle set.
    ///
    /// Subsequent [`Op::WaitPayload`] / [`Op::KillPayload`] address
    /// the running child by the composite
    /// (`Payload::name`, `cgroup`) key — the same payload can run
    /// concurrently in two different cgroups without a dedup
    /// collision, but the lookup from the waiting op must match
    /// the pair the run op recorded. See [`Op::WaitPayload`] /
    /// [`Op::KillPayload`] for the ambiguity rules when the
    /// waiting op supplies only the name.
    ///
    /// Only [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are spawnable; scheduler-kind payloads are rejected at
    /// apply time with an actionable error.
    ///
    /// `args` is appended to `payload.default_args`. `cgroup`, when
    /// set, places the child in the named cgroup (resolved relative
    /// to the scenario's parent cgroup) via
    /// [`PayloadRun::in_cgroup`](crate::scenario::payload_run::PayloadRun::in_cgroup);
    /// unset inherits the spawning process's cgroup.
    ///
    /// Handles not explicitly consumed by `WaitPayload` / `KillPayload`
    /// are drained at step-teardown by `collect_step` (step-local) or
    /// at scenario end by `collect_backdrop` (when the handle lives on
    /// the Backdrop), matching the [`CgroupDef::workload`] semantics.
    ///
    /// # Scheduler-kind rejection across surfaces
    ///
    /// Three surfaces accept a `&Payload` and each rejects a
    /// scheduler-kind Payload differently — deliberately, to match
    /// the lifecycle of the caller:
    ///
    /// | Surface                                                                                   | Rejection             | When          |
    /// |-------------------------------------------------------------------------------------------|-----------------------|---------------|
    /// | [`PayloadRun::run`](crate::scenario::payload_run::PayloadRun::run) (`ctx.payload(&X)...`) | `Err(anyhow::Error)`  | scenario-time |
    /// | [`CgroupDef::workload`]                                                                   | `panic!`              | declaration-time |
    /// | `Op::RunPayload` (this variant)                                                           | `Err(anyhow::Error)`  | apply-ops-time |
    ///
    /// Rationale: `CgroupDef::workload` is a builder invoked during
    /// test construction (nextest `--list` phase) — a panic there
    /// surfaces the misuse before any VM boot, with a full
    /// backtrace pointing at the offending call. `ctx.payload()`
    /// and `Op::RunPayload` both run inside an executing scenario
    /// where one bad misuse should not crash the whole test run;
    /// they `bail!` with an actionable message and let the
    /// surrounding step-sequence skip to teardown. The three
    /// paths are symmetric in *what* they reject (scheduler-kind
    /// Payloads in non-scheduler slots); they differ only in
    /// *how* the misuse is surfaced, matched to caller context.
    RunPayload {
        payload: &'static crate::test_support::Payload,
        args: Vec<String>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// Block until the payload named `name` exits naturally, then
    /// evaluate its checks and record metrics to the per-test sidecar.
    ///
    /// The target is looked up by composite key (`name`, `cgroup`).
    /// `cgroup: None` matches the unique live copy (whatever its
    /// placement); if two or more copies of the same payload are
    /// live in different cgroups, the lookup bails with an
    /// "ambiguous — specify cgroup" error so the test doesn't
    /// silently wait on the wrong one. Use
    /// [`Op::wait_payload_in_cgroup`] to disambiguate.
    ///
    /// A consumed or unknown `(name, cgroup)` pair returns `Err`
    /// with an actionable message — test authors must not silently
    /// wait for payloads that were never started or have already
    /// been consumed by a prior `WaitPayload`/`KillPayload`.
    ///
    /// **No timeout.** `WaitPayload` waits indefinitely for the
    /// child to exit. A binary that never terminates (e.g. a
    /// benchmark configured without `--runtime=N`, or a stress-ng
    /// run without `--timeout`) will hang the step until the
    /// outer test watchdog fires. For time-boxed long-running
    /// payloads, prefer [`KillPayload`](Self::KillPayload) paired
    /// with a [`HoldSpec::Fixed`] / [`HoldSpec::Frac`] step
    /// boundary that guarantees forward progress; the payload's
    /// own CLI (`--runtime`, `--timeout`) is the reliable way to
    /// cap a single invocation's runtime.
    ///
    /// Check failures from the payload are recorded to the sidecar
    /// for regression analysis but do NOT fail the step or the test
    /// in-process. Use
    /// [`ctx.payload(&X).run()`](crate::scenario::payload_run::PayloadRun::run)
    /// directly if the test body needs to gate on check results.
    WaitPayload {
        name: Cow<'static, str>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// SIGKILL the payload named `name`, reap the child, evaluate
    /// checks, and record metrics. Mirrors the behavior of
    /// step-teardown drain for an explicitly-targeted payload.
    ///
    /// The target is looked up by composite key (`name`, `cgroup`)
    /// — see [`Op::WaitPayload`] for the ambiguity rules.
    ///
    /// A consumed or unknown `(name, cgroup)` pair returns `Err`
    /// with an actionable message, identical to [`Op::WaitPayload`]'s
    /// lookup semantics.
    ///
    /// Check failures from the payload are recorded to the sidecar
    /// for regression analysis but do NOT fail the step or the test
    /// in-process. Use
    /// [`ctx.payload(&X).run()`](crate::scenario::payload_run::PayloadRun::run)
    /// directly if the test body needs to gate on check results.
    KillPayload {
        name: Cow<'static, str>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// Freeze every task in the named cgroup via `cgroup.freeze`.
    ///
    /// Writes `"1"` to the cgroup's `cgroup.freeze` file. The kernel's
    /// `cgroup_freeze_write` dispatches the asynchronous freeze path;
    /// tasks transition to the frozen state without external SIGSTOP,
    /// and `cgroup.events` reaches `frozen 1` once every task has
    /// parked. Idempotent — freezing an already-frozen cgroup is a
    /// no-op.
    ///
    /// # Auto-unfreeze at teardown
    ///
    /// `Op::FreezeCgroup` is paired with [`Op::UnfreezeCgroup`] to
    /// release. A test that omits the unfreeze still tears down
    /// cleanly: [`crate::cgroup::CgroupManager::remove_cgroup`]
    /// auto-unfreezes the cgroup before draining tasks (see the
    /// kernel's `cgroup_freezer_migrate_task`, which clears the
    /// task's freeze state when it migrates to an unfrozen
    /// destination), so step teardown is robust to a stuck-frozen
    /// cgroup. Pair the ops explicitly when the scenario needs
    /// observable unfreeze timing inside the step body.
    ///
    /// # Worked example
    ///
    /// Three-Step suspend/resume sequence: a `Backdrop`-resident
    /// long-running workload is paused mid-scenario and resumed
    /// later, exercising how the scheduler responds to a sudden
    /// idle window.
    ///
    /// ```text
    /// Step 1 (run): apply cgroup; workload spins for 2s.
    /// Step 2 (suspend): Op::freeze_cgroup("workers"); hold 1s.
    ///                   The cgroup's tasks park via cgroup.freeze,
    ///                   schedstat gauges drop to zero, and the
    ///                   scheduler observes a sudden idle subtree.
    /// Step 3 (resume): Op::unfreeze_cgroup("workers"); hold 2s.
    ///                  Tasks return to runnable state, the
    ///                  scheduler must re-pick them onto the
    ///                  cgroup's CPUs without spuriously preempting
    ///                  unrelated workloads.
    /// ```
    ///
    /// # Observer-cgroup deadlock warning
    ///
    /// Do NOT freeze a cgroup that hosts the test's own observation
    /// machinery. The freeze path stops every task in the cgroup —
    /// including any thread that:
    /// - opens `/proc/<pid>/sched` or other procfs entries owned by
    ///   tasks inside the frozen cgroup, then waits on the read,
    /// - holds a futex shared with frozen tasks (the unfreeze must
    ///   land before the wait can complete),
    /// - synchronously waits on a stalled-task pipe whose
    ///   producer is in the frozen cgroup.
    ///
    /// The framework's stimulus-event SHM ring and the `BlkWorker`
    /// epoll loop both run outside the test cgroup tree, so they
    /// are unaffected — but a test author who explicitly places an
    /// observer thread inside the same cgroup as its observation
    /// targets will deadlock the scenario when the freeze fires.
    /// Place observers in a sibling cgroup (or in the parent) so
    /// `cgroup.freeze` is scoped to the workload subtree alone.
    ///
    /// Pair with [`Op::UnfreezeCgroup`] to release. Useful for
    /// scheduler suspend/resume tests where the test body wants to
    /// observe how the scheduler handles a suddenly-frozen workload
    /// and the resumption sequence afterwards.
    ///
    /// Treats a missing cgroup as a step failure: the
    /// `cgroup.freeze` write fails with `ENOENT` and the error
    /// propagates via the `apply_ops` `with_context` chain.
    /// Freezing a non-existent cgroup is NOT a no-op; only
    /// freezing an already-frozen cgroup is.
    FreezeCgroup { cgroup: Cow<'static, str> },
    /// Unfreeze every task in the named cgroup via `cgroup.freeze`.
    ///
    /// Writes `"0"` to the cgroup's `cgroup.freeze` file. Inverse of
    /// [`Op::FreezeCgroup`]. Idempotent.
    UnfreezeCgroup { cgroup: Cow<'static, str> },
    /// Capture a host-side diagnostic snapshot under `name`. The
    /// freeze coordinator pauses every vCPU long enough to read
    /// the BPF map state, vCPU registers, and per-CPU
    /// counters into a
    /// [`FailureDumpReport`](crate::monitor::dump::FailureDumpReport),
    /// then resumes the guest. The report is keyed by `name` on
    /// the active
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge);
    /// downstream test code reads it via
    /// [`Snapshot`](crate::scenario::snapshot::Snapshot).
    ///
    /// On-demand snapshots are orthogonal to the error-class
    /// freeze trigger — the request flows through a separate
    /// channel, does not transition the coordinator's
    /// `freeze_state`, and is serviced even after `Done`. The only
    /// scheduling rule: at most one capture in flight at a time
    /// (each request waits for the previous freeze's vCPUs to
    /// fully resume before issuing).
    ///
    /// **Guest → host wire.** Locked at an in-kernel ioeventfd
    /// doorbell at a dedicated MMIO GPA inside the MMIO gap
    /// (e.g. `MMIO_GAP_START + 0x3000`). The guest writes the tag
    /// into a small SHM-resident slot and then writes the doorbell
    /// GPA via the existing `/dev/mem` mmap pattern that the SHM
    /// ring already uses. KVM dispatches in-kernel
    /// (`KVM_IOEVENTFD`) without a vCPU userspace exit, the
    /// freeze coordinator wakes on `eventfd_signal`, and the
    /// installed `CaptureCallback` returns the resulting report
    /// through a paired reply completion. See
    /// [`CaptureCallback`](crate::scenario::snapshot::CaptureCallback)
    /// for the full protocol.
    ///
    /// **No active bridge ⇒ no-op.** When the executor runs in a
    /// context with no installed
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge)
    /// (e.g. unit tests that exercise the executor without
    /// spinning up a VM), this op emits a `tracing::warn!` and
    /// continues. Existing scenarios that never declare snapshot
    /// ops keep their behavior unchanged.
    ///
    /// # Example
    ///
    /// Declare a snapshot mid-step, fetch the captured report
    /// after the scenario completes, and assert against a
    /// BTF-rendered field:
    ///
    /// ```ignore
    /// use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
    /// use ktstr::scenario::snapshot::{Snapshot, SnapshotBridge};
    ///
    /// // Wire up the bridge before execute_steps runs (host-side
    /// // VM setup typically performs this step automatically).
    /// let bridge = SnapshotBridge::new(/* capture callback */);
    /// let _guard = bridge.clone().set_thread_local();
    ///
    /// let steps = vec![Step {
    ///     setup: vec![CgroupDef::named("workers").workers(2)].into(),
    ///     ops: vec![Op::snapshot("after_spawn")],
    ///     hold: HoldSpec::FULL,
    /// }];
    /// execute_steps(ctx, steps)?;
    ///
    /// // Inspection.
    /// let captured = bridge.drain();
    /// let report = captured.get("after_spawn").expect("snapshot recorded");
    /// let snap = Snapshot::new(report);
    /// let nr_cpus = snap.var("nr_cpus_onln").as_u64()?;
    /// assert!(nr_cpus > 0, "snapshot captured live nr_cpus_onln");
    /// ```
    Snapshot { name: Cow<'static, str> },
    /// Capture a snapshot whenever the guest writes to the named
    /// kernel symbol. The snapshot is tagged with the symbol
    /// itself; one fire = one capture.
    ///
    /// Symbol resolution at op execution time is a verbatim match
    /// against the vmlinux ELF symbol table: the freeze coordinator
    /// walks `Elf::syms` and accepts the symbol whose strtab entry
    /// equals the requested string byte-for-byte. There is no
    /// prefix stripping, BTF lookup, kallsyms walk, or per-CPU
    /// offset arithmetic — the string must match an entry that
    /// `nm vmlinux` would print (e.g. `"jiffies_64"`,
    /// `"scx_watchdog_timestamp"`).
    ///
    /// The `register_watch` callback on a host-side
    /// [`SnapshotBridge`](crate::scenario::snapshot::SnapshotBridge)
    /// is for **host-side unit testing only** — it lets in-process
    /// executor tests record the symbol and return without arming
    /// any hardware. Production in-VM scenarios run via the
    /// virtio-console port 1 `MSG_TYPE_SNAPSHOT_REQUEST` TLV frame
    /// and the host coordinator's `arm_user_watchpoint` path
    /// (`src/vmm/freeze_coord.rs`); the thread-local bridge is
    /// never installed inside the guest.
    ///
    /// # Guard rails
    ///
    /// - **Maximum of 3 watch ops per scenario.** The KVM
    ///   hardware-watchpoint plumbing reserves slot 0 for the
    ///   existing `*scx_root->exit_kind` trigger (used by the
    ///   error-trigger path); only the remaining three user
    ///   watchpoint slots are available for on-demand watches. The
    ///   bridge's `register_watch` rejects a 4th
    ///   `Op::WatchSnapshot` and fails the step when the cap is
    ///   exceeded.
    /// - **Symbol resolution failures bail immediately.** A
    ///   missing symbol or unaligned address surfaces as an `Err`
    ///   from `execute_steps` so the test author notices the
    ///   watch did not attach. Silent degradation would leave the
    ///   scenario running with no captures and look identical to
    ///   a healthy passing run.
    /// - **4-byte alignment.** The resolved KVA must be 4-byte
    ///   aligned: the framework arms 4-byte data-write watches,
    ///   which require `addr & 0x3 == 0` on every supported
    ///   architecture. Mis-aligned addresses bail at setup with
    ///   the resolved KVA in the error.
    ///
    /// **Guest → host wire.** The registration request rides the
    /// same ioeventfd doorbell as [`Op::Snapshot`] (separate tag
    /// namespace), so symbol resolution + user watchpoint slot
    /// allocation + `KVM_SET_GUEST_DEBUG` arming happen on the host
    /// without a vCPU userspace exit. Once armed, the
    /// `KVM_EXIT_DEBUG` dispatch path drives the resulting
    /// captures directly into the freeze coordinator (no
    /// per-fire doorbell write needed). See
    /// [`WatchRegisterCallback`](crate::scenario::snapshot::WatchRegisterCallback)
    /// for the full protocol.
    ///
    /// Note: high-frequency variables (rq counters, jiffies)
    /// will fire watches every few microseconds and fire
    /// thousands of times (each overwriting the prior capture
    /// under the same tag); the framework does not rate-limit
    /// captures, so the test author owns the frequency choice.
    /// Use [`Op::Snapshot`] for time-driven captures when
    /// frequency is the concern.
    WatchSnapshot { symbol: Cow<'static, str> },
}

/// How to compute a cpuset from topology.
///
/// # `#[non_exhaustive]`
///
/// `CpusetSpec` is `#[non_exhaustive]` — see
/// [`crate::non_exhaustive`] for the cross-crate pattern-match and
/// construction rules shared by every such type.
///
/// Variant-specific guidance for `CpusetSpec`: prefer the
/// associated constructor functions — [`Self::llc`], [`Self::numa`],
/// [`Self::range`], [`Self::disjoint`], [`Self::overlap`], and
/// [`Self::exact`] — over naming variant literals like
/// `CpusetSpec::Llc(0)` or `CpusetSpec::Range { start_frac,
/// end_frac }`. Two reasons:
///
/// 1. **Stability across variant reshaping.** A future commit that
///    adds a field to `Range` (e.g. a stride parameter) breaks every
///    caller that spelled out `CpusetSpec::Range { start_frac,
///    end_frac }`; the `Self::range(..)` constructor absorbs the
///    new field behind a defaulted parameter. The `#[non_exhaustive]`
///    attribute is what reserves that freedom for the enum; the
///    constructor convention is how callers opt into benefiting from
///    it.
/// 2. **Semantic consistency with [`Self::exact`].** The `exact`
///    constructor accepts any `IntoIterator<Item = usize>` (arrays,
///    ranges, `Vec`, `BTreeSet`) and converts to `BTreeSet<usize>`
///    internally; callers that bypass it and write
///    `CpusetSpec::Exact(set)` directly must hand-build the
///    `BTreeSet` — duplicate bookkeeping a future-proofed constructor
///    erases.
///
/// Test code that needs to *inspect* a variant via pattern match
/// necessarily references the variant literal (the name is load-
/// bearing for the match), so the construction-side rule is a
/// convention for *production* call sites, not a hard constraint.
/// Inside this crate, matchers obey the pattern-side rule above;
/// constructors obey this rule.
///
/// `Clone + Debug + PartialEq`. `Eq` / `Hash` are impossible
/// because [`Range`](Self::Range) and [`Overlap`](Self::Overlap)
/// carry `f64` fractions; `Default` has no honest value (`Llc(0)`
/// vs. `Range(0..1)` vs. `Exact(empty)` are all different
/// "no-op" semantics).
///
/// Note: `f64::NAN != f64::NAN` per IEEE 754, so a `CpusetSpec`
/// containing NaN fractions will not equal a clone of itself;
/// `validate()` rejects NaN inputs.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CpusetSpec {
    /// All CPUs in a given LLC index.
    Llc(usize),
    /// All CPUs in a given NUMA node index.
    Numa(usize),
    /// Fractional range of usable CPUs [start_frac..end_frac).
    Range { start_frac: f64, end_frac: f64 },
    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    Disjoint { index: usize, of: usize },
    /// Like Disjoint but each set overlaps neighbors by `frac` of its size.
    Overlap { index: usize, of: usize, frac: f64 },
    /// Exact CPU set (no topology resolution).
    Exact(BTreeSet<usize>),
}

impl CpusetSpec {
    /// Construct an `Exact` cpuset from any iterator of CPU indices.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn exact(cpus: impl IntoIterator<Item = usize>) -> Self {
        CpusetSpec::Exact(cpus.into_iter().collect())
    }

    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    pub fn disjoint(index: usize, of: usize) -> Self {
        CpusetSpec::Disjoint { index, of }
    }

    /// Like [`disjoint`](Self::disjoint) but each set overlaps neighbors by `frac` of its size.
    pub fn overlap(index: usize, of: usize, frac: f64) -> Self {
        CpusetSpec::Overlap { index, of, frac }
    }

    /// Fractional range of usable CPUs `[start_frac..end_frac)`.
    pub fn range(start_frac: f64, end_frac: f64) -> Self {
        CpusetSpec::Range {
            start_frac,
            end_frac,
        }
    }

    /// All CPUs in a given LLC index.
    pub fn llc(index: usize) -> Self {
        CpusetSpec::Llc(index)
    }

    /// All CPUs in a given NUMA node index.
    pub fn numa(index: usize) -> Self {
        CpusetSpec::Numa(index)
    }
}

// ---------------------------------------------------------------------------
// Cgroup v2 resource limits
// ---------------------------------------------------------------------------

/// CPU controller limits (`cpu.max` + `cpu.weight`) for a cgroup. All
/// fields default to "inherit from parent" — the framework only writes
/// each knob when its corresponding field is `Some`.
///
/// Set via [`CgroupDef::cpu`]. The kernel allows `quota` and `weight`
/// to coexist (per `Documentation/admin-guide/cgroup-v2.rst`,
/// "CPU Interface Files"): `weight` biases relative CPU share inside
/// `period`, `quota` enforces an absolute ceiling. Surfacing both as
/// independent options lets a test author express "this cgroup gets
/// at most 50% of one CPU AND should lose to a heavier sibling under
/// contention" in a single declaration.
///
/// Validation runs at `apply_setup` time — any violation surfaces as
/// `anyhow::bail!` so a misconfigured CgroupDef fails before any
/// worker spawns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CpuLimits {
    /// `cpu.max` quota and period in microseconds. `quota = None`
    /// means "max" (no upper bound). `quota = Some(q)` allows the
    /// cgroup `q` µs of CPU time per `period`. `q > period` is
    /// legal: it lets the cgroup use multiple CPUs concurrently
    /// (e.g. quota 200_000 / period 100_000 = up to 2 CPUs of
    /// throughput).
    ///
    /// `period` defaults to 100_000 µs (100 ms) when omitted via
    /// the [`CgroupDef::cpu_quota_pct`] convenience builder. Set
    /// via [`CgroupDef::cpu_quota`] when a non-default period is
    /// needed (e.g. tighter control loops with 10 ms periods for
    /// latency-sensitive scheduler tests).
    pub max_quota_us: Option<u64>,
    /// `cpu.max` period component. Required whenever `max_quota_us`
    /// is `Some`; ignored when `max_quota_us` is `None` (the
    /// framework writes `"max <period>"` so the period stays
    /// recorded for diagnostics).
    pub max_period_us: u64,
    /// `cpu.weight` relative-share weight (range 1..=10000, default
    /// 100). `None` leaves the kernel default in place. Larger
    /// values get a larger share when the parent cgroup's CPU is
    /// contended.
    pub weight: Option<u32>,
}

/// Memory controller limits (`memory.max` / `memory.high` /
/// `memory.low` / `memory.swap.max`). Each field is `None` by
/// default (inherit from parent / no limit).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct MemoryLimits {
    /// `memory.max` hard ceiling in bytes. Crossing this triggers
    /// the cgroup OOM killer per `Documentation/admin-guide/
    /// cgroup-v2.rst`'s "Memory Interface Files". `None` writes
    /// `"max"` (no hard limit).
    pub max: Option<u64>,
    /// `memory.high` soft throttle threshold in bytes. Crossing
    /// this triggers reclaim throttling but NOT OOM-kill. `None`
    /// writes `"max"`.
    pub high: Option<u64>,
    /// `memory.low` soft protection threshold in bytes. The kernel
    /// preferentially reclaims FROM other cgroups before reclaiming
    /// this cgroup's memory below `low`. `None` writes `"0"` (no
    /// protection).
    pub low: Option<u64>,
    /// `memory.swap.max` ceiling on the cgroup's swap usage in bytes.
    /// `None` writes `"max"` (no swap cap, the kernel default). The
    /// kernel parses the wire value via `page_counter_memparse` —
    /// either the literal `"max"` or a decimal byte count
    /// (`swap_max_write` in `mm/memcontrol.c`).
    ///
    /// # `CONFIG_SWAP=n` kernel detection
    ///
    /// `memory.swap.max` only exists when the kernel was built with
    /// `CONFIG_SWAP=y`; on swap-disabled builds the file is absent
    /// and the wire-time write returns ENOENT. The framework only
    /// emits the write when `swap_max.is_some()` — the explicit
    /// opt-in matches the per-knob semantics of the pids block, so
    /// tests that never call [`CgroupDef::memory_swap_max`] /
    /// [`CgroupDef::memory_swap_unlimited`] succeed verbatim on a
    /// swap-disabled kernel.
    ///
    /// **`swap_max = Some(N)` on a `CONFIG_SWAP=n` kernel surfaces
    /// as a hard scenario failure**: `apply_setup` propagates the
    /// ENOENT from `set_memory_swap_max`'s `write_with_timeout` up
    /// the error chain with the `memory.swap.max` filename in the
    /// context. Test authors who target the swap controller must
    /// either (a) gate the swap_max call on a host probe, or (b)
    /// require the test kernel be built with `CONFIG_SWAP=y` and
    /// document the requirement on the test.
    ///
    /// # ktstr's kernel config and swap
    ///
    /// `ktstr.kconfig` (the project-level kernel-config fragment that
    /// `cargo ktstr` merges into the test kernel's defconfig) does
    /// NOT pin `CONFIG_SWAP=y` — swap is not a test-framework
    /// requirement, and many test scenarios run faster without it.
    /// Tests that call `memory_swap_max` therefore must either
    /// extend the per-test kconfig fragment (passed alongside
    /// `ktstr.kconfig` at kernel-build time) or detect at
    /// scenario-setup time by reading `/proc/swaps` (a missing
    /// file or empty body indicates no swap subsystem) or
    /// `/proc/config.gz` (search for `CONFIG_SWAP=y`). The framework
    /// does NOT auto-detect because host probing is policy that
    /// belongs to the test author, not the workload runner.
    pub swap_max: Option<u64>,
}

/// Pids controller limits (`pids.max`). `None` is the default
/// (inherit from parent — typically `"max"`, no ceiling).
///
/// Per the kernel's `pids_max_write`, existing tasks are NOT killed
/// when the limit lands below the current task count; only future
/// `fork()` / `clone()` calls are blocked once the cgroup's task
/// count meets the limit. Useful for fork-bomb / task-count-ceiling
/// tests.
///
/// # Per-WorkType thread-budget guidance
///
/// `pids.max` counts every task (process AND thread) inside the
/// cgroup. Sizing the limit below the workload's natural task
/// budget produces silent fork failures that surface as
/// `WorkloadConfig`-level workers refusing to start.
///
/// **The framework spawns exactly one task per worker** — no
/// per-worker helper threads in any variant's
/// [`worker_main`](crate::workload) dispatch arm. Per-worker
/// budget therefore depends only on
/// [`CloneMode`](crate::workload::CloneMode) (whether each worker
/// is a process or a thread sharing the parent's tgid) and
/// whether the variant transiently forks short-lived children
/// inside its own loop. The two columns below capture both:
///
/// | Variant | Steady-state tasks | Transient peak |
/// |---------|--------------------|----------------|
/// | `SpinWait`, `YieldHeavy`, `Mixed` | 1/worker | — |
/// | `Bursty`, `IdleChurn` | 1/worker | — |
/// | `IoSyncWrite`, `IoRandRead`, `IoConvoy` | 1/worker | — |
/// | `CachePressure`, `CacheYield`, `CachePipe` | 1/worker | — |
/// | `PageFaultChurn` | 1/worker | — |
/// | `AffinityChurn`, `PolicyChurn`, `NiceSweep` | 1/worker | — |
/// | `NumaWorkingSetSweep`, `NumaMigrationChurn`, `CgroupChurn` | 1/worker | — |
/// | `Sequence` | 1/worker | — |
/// | `AluHot`, `SmtSiblingSpin`, `IpcVariance` | 1/worker | — |
/// | `PipeIo`, `FutexPingPong`, `AsymmetricWaker`, `SignalStorm` | 1/worker | — |
/// | `FutexFanOut`, `FanOutCompute` | 1/worker | — |
/// | `ThunderingHerd`, `MutexContention`, `WakeChain` | 1/worker | — |
/// | `PriorityInversion`, `ProducerConsumerImbalance` | 1/worker | — |
/// | `RtStarvation`, `PreemptStorm`, `EpollStorm` | 1/worker | — |
/// | `ForkExit` | 1/worker | +1/worker (waitpid'd before next iter) |
/// | `Custom` | 1/worker | depends on user closure (see below) |
///
/// **`CloneMode::Fork`** (the default): each worker is a separate
/// process placed in the cgroup. The cgroup's task count for one
/// `WorkSpec` is exactly `num_workers`; for `ForkExit` the
/// instantaneous peak is `2 × num_workers` (each parent forks one
/// child, waitpid's, repeats).
///
/// **`CloneMode::Thread`**: every worker is a thread sharing the
/// test runner's tgid. The pids controller counts each thread as
/// a task, so the cgroup's task count for one `WorkSpec` is
/// `num_workers + 1` (workers + the parent task). `ForkExit` is
/// rejected at spawn time under Thread mode (see
/// [`WorkType::ForkExit`](crate::workload::WorkType::ForkExit)).
///
/// **`Custom`**: the framework runs the user closure in a single
/// task per worker (1/worker, identical to every other variant).
/// Any fork/clone the closure issues inside its loop adds to the
/// cgroup's task count for as long as the resulting child lives;
/// `pids.max` must reserve headroom equal to the closure's peak
/// child count per worker. Under `CloneMode::Fork` the framework
/// reaps closure-spawned descendants at teardown via
/// `killpg(worker_pid, SIGKILL)` against the worker's per-process
/// group, so transient children are bounded by the closure
/// itself. Under `CloneMode::Thread` the worker shares the test
/// runner's pgid and `killpg`-based cleanup is unavailable, so
/// the closure owns whatever helpers it spawns and must reap
/// them explicitly before returning the
/// [`WorkerReport`](crate::workload::WorkerReport).
///
/// **Sizing rule**: `pids.max ≥ Σ(steady-state + transient)` for
/// every [`WorkSpec`](crate::workload::WorkSpec) in the cgroup,
/// plus headroom for `cgroup.procs` migration scratch tasks and
/// any payload-binary helper processes the test attaches via
/// [`CgroupDef::workload`] (e.g. `stress-ng` spawns one task per
/// `--cpu N`). Tests with composed `WorkSpec` groups must sum
/// across every group — the framework does NOT auto-derive a
/// budget from the work spec.
///
/// # Parent-cgroup hierarchical charging
///
/// `pids.max` is a per-cgroup ceiling, but every fork/clone
/// charges every ancestor up to (but not including) the
/// unified-hierarchy root. The kernel's `pids_can_fork` calls
/// `pids_try_charge`, which loops
/// `for (p = pids; parent_pids(p); p = parent_pids(p))` and
/// charges each level (kernel/cgroup/pids.c) — root is NOT
/// charged per the loop's `parent_pids(p)` termination
/// condition. EAGAIN propagates from the FIRST level
/// (leaf-to-root traversal order) whose post-charge counter
/// exceeds its limit, so a child cgroup with `pids.max = 1024`
/// still hits EAGAIN when a parent two levels up sits at its
/// own ceiling.
///
/// Sizing rule for nested test trees: the *effective* limit is
/// `min(pids.max)` along the path from the test cgroup up to the
/// pids-controlled root, NOT just the value set on the test
/// cgroup itself. When ktstr runs under a delegated parent slice
/// (systemd `user.slice`, container runtime cgroup, ktstr's own
/// build sandbox), inspect the parent's `pids.max` before sizing
/// the test cgroup — a generous test-cgroup setting is silently
/// shadowed by a tighter ancestor.
///
/// # `pids.max(0)` is rejected at apply_setup, not type-level
///
/// `Some(0)` would silently halt every fork/clone inside the
/// cgroup, including the worker spawn itself for `CloneMode::Fork`
/// and the `ForkExit` per-iteration child fork. The kernel accepts
/// the value (it's a legitimate `pids_max_write` input), so
/// `apply_setup` adds the bail at scenario-setup time; promoting
/// it to a type-level invariant (e.g. `NonZeroU64`) would force
/// every numeric literal through a non-`const` constructor and
/// ripple into every test fixture. The runtime bail keeps the
/// surface ergonomic while still surfacing the foot-cannon at
/// construction time (before any worker spawns).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PidsLimits {
    /// `pids.max` task-count ceiling. `None` writes the literal
    /// string `"max"` (the kernel's `PIDS_MAX_STR` sentinel for
    /// unlimited). `Some(n)` writes the decimal `n`. The kernel
    /// rejects negative or `>= PIDS_MAX (PID_MAX_LIMIT + 1, typically ~4M on 64-bit)` values with
    /// EINVAL; the framework's `apply_setup` rejects `Some(0)`
    /// before the syscall (a 0 limit silently halts every fork
    /// or clone inside the cgroup, blocking both worker spawn
    /// under `CloneMode::Fork` and `ForkExit`'s per-iteration
    /// child fork).
    pub max: Option<u64>,
}

/// IO controller limits (`io.weight`). Per-device throughput caps
/// (`io.max`) are intentionally not surfaced here — the per-device
/// interface needs major:minor device-id lookup which has no
/// in-tree consumer; surface it as a follow-up task when a
/// concrete use case lands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct IoLimits {
    /// `io.weight` relative-share weight (range 1..=10000, default
    /// 100). `None` leaves the kernel default in place.
    pub weight: Option<u16>,
}

// ---------------------------------------------------------------------------
// CgroupDef
// ---------------------------------------------------------------------------

/// Declarative cgroup definition: name + cpuset + synthetic
/// [`WorkSpec`] groups + optional userspace [`Payload`](crate::test_support::Payload).
///
/// Bundles the ops that always go together (AddCgroup + SetCpuset +
/// Spawn) into a single value. The executor creates the cgroup, optionally
/// sets its cpuset, spawns workers for each [`WorkSpec`] entry, and moves
/// them into the cgroup.
///
/// Multiple [`WorkSpec`] entries run in parallel within the cgroup. Each
/// entry spawns its own set of worker processes. The optional
/// [`Self::payload`] slot is a *single* userspace binary that runs
/// alongside those synthetic [`WorkSpec`] groups (hence "plural works,
/// singular payload" — the pluralization in the legacy "workload(s)"
/// prose elided this distinction).
///
/// Use `CgroupDef` in `Step::with_defs` for scenarios where cgroups are
/// created once and run for the step duration. Use `Op::AddCgroup` +
/// `Op::Spawn` directly when you need mid-step cgroup creation, removal,
/// or other dynamic operations between spawn and collect.
///
/// # Resource controllers overview
///
/// `CgroupDef` exposes one builder method per cgroup v2 controller
/// knob, each writing the corresponding `cgroup.*` / `*.max` /
/// `*.weight` file at `apply_setup` time. The full surface:
///
/// | Controller | One-line description | Builder methods | Underlying file(s) |
/// |------------|----------------------|-----------------|--------------------|
/// | cpuset | Bind to a CPU subset and NUMA-node memory affinity. | [`Self::with_cpuset`], [`Self::with_cpuset_mems`] | `cpuset.cpus`, `cpuset.mems` |
/// | cpu    | Bandwidth ceiling (`cpu.max` quota/period) plus relative-share weight. | [`Self::cpu_quota_pct`], [`Self::cpu_quota`], [`Self::cpu_unlimited`], [`Self::cpu_weight`] | `cpu.max`, `cpu.weight` |
/// | memory | Hard ceiling, soft throttle threshold, soft protection floor, swap cap. | [`Self::memory_max`], [`Self::memory_high`], [`Self::memory_low`], [`Self::memory_swap_max`], [`Self::memory_swap_unlimited`], [`Self::memory_unlimited`] | `memory.max`, `memory.high`, `memory.low`, `memory.swap.max` |
/// | io     | Relative IO share (BFQ / io.cost) when the io controller is enabled. | [`Self::io_weight`] | `io.weight` |
/// | pids   | Task-count ceiling — fork(2)/clone(2) returns EAGAIN once the cap is hit. | [`Self::pids_max`], [`Self::pids_unlimited`] | `pids.max` |
/// | freeze | Pause/resume every task in the cgroup mid-run via the JOBCTL freeze path. | (Op-level) [`Op::freeze_cgroup`], [`Op::unfreeze_cgroup`] | `cgroup.freeze` |
///
/// `CgroupDef` covers steady-state resource limits — knobs that
/// hold for the cgroup's whole lifetime. The freeze knob is
/// intentionally exposed at the [`Op`] layer instead, because
/// freeze/unfreeze describe transitions over time (suspend
/// mid-step, resume later) rather than the cgroup's identity; see
/// the "See also" section below for the full Op-variants list.
///
/// All builders are additive — a `CgroupDef` accumulates an
/// optional [`CpuLimits`] / [`MemoryLimits`] / [`IoLimits`] /
/// [`PidsLimits`] block. When a block is set (e.g. `def.memory`
/// is `Some`), **all** knobs in that block are written —
/// `None`-valued fields emit their kernel-default sentinel
/// (`"max"` for `memory.max`/`memory.high`, `"0"` for
/// `memory.low`). Only `memory.swap.max` is gated: `None` means
/// no write (for `CONFIG_SWAP=n` compatibility). The "*_unlimited"
/// builders explicitly rewind a knob to its sentinel value
/// (`"max"` / `"0"`) so a base `CgroupDef` factory can cap a
/// resource and a per-test extension can clear that cap without
/// rewriting the whole `CgroupDef`.
///
/// Validation runs at `apply_setup` time (before any worker
/// spawn): out-of-range weights, `cpu.max period == 0`, and
/// `pids.max == Some(0)` all produce actionable bails before the
/// syscall fires. The kernel is the final authority on
/// per-controller numeric ranges; framework-level checks catch
/// only the foot-cannons documented per-builder.
///
/// # See also
///
/// `CgroupDef` only expresses the steady-state shape of a cgroup
/// (name, cpuset, work groups, payload). State changes that need
/// to happen DURING a step — without tearing the cgroup down and
/// recreating it — go through dedicated [`Op`] variants instead:
///
/// * [`Op::FreezeCgroup`] / [`Op::UnfreezeCgroup`] — pause and
///   resume every task in the cgroup via `cgroup.freeze` (the
///   kernel-side asynchronous freeze path; not a SIGSTOP).
///   Useful for scheduler suspend/resume tests that observe
///   how the scheduler handles a workload that goes idle
///   mid-step. **Do not freeze a cgroup hosting the test's own
///   observers** — see the deadlock warning on
///   [`Op::FreezeCgroup`].
/// * [`Op::SetCpuset`] — re-pin an existing cgroup's cpuset to
///   exercise the scheduler's response to a moving CPU mask
///   without disrupting the worker tasks themselves.
/// * [`Op::AddCgroup`] / [`Op::RemoveCgroup`] — add or destroy
///   cgroups mid-step when a `CgroupDef`'s lifecycle is
///   tied to step duration but the test wants a different
///   (e.g. nested) cgroup to appear or disappear partway
///   through.
///
/// These describe transitions over time rather than the cgroup's
/// identity, which is why they live as `Op` variants alongside
/// the rest of the operation vocabulary rather than as
/// `CgroupDef` builders.
///
/// ```
/// # use ktstr::scenario::ops::{CgroupDef, CpusetSpec};
/// # use ktstr::workload::{WorkSpec, WorkType};
/// // Single work group via convenience methods.
/// let def = CgroupDef::named("workers")
///     .with_cpuset(CpusetSpec::disjoint(0, 2))
///     .workers(4)
///     .work_type(WorkType::SpinWait);
///
/// assert_eq!(def.name, "workers");
/// assert_eq!(def.works[0].num_workers, Some(4));
///
/// // Multiple concurrent work groups via .work().
/// let def = CgroupDef::named("mixed")
///     .work(WorkSpec::default().workers(4).work_type(WorkType::SpinWait))
///     .work(WorkSpec::default().workers(2).work_type(WorkType::YieldHeavy));
///
/// assert_eq!(def.works.len(), 2);
///
/// // Synthetic work + userspace binary side-by-side via .workload(&X).
/// // The binary runs inside the same cgroup as the WorkSpec handles;
/// // both spawn in apply_setup, the WorkSpec groups first, then the
/// // Payload after the cpuset settles.
/// # use ktstr::test_support::{OutputFormat, Payload, PayloadKind};
/// # const BENCH: Payload = Payload {
/// #     name: "bench",
/// #     kind: PayloadKind::Binary("bench"),
/// #     output: OutputFormat::ExitCode,
/// #     default_args: &[],
/// #     default_checks: &[],
/// #     metrics: &[],
/// #     include_files: &[],
/// #     uses_parent_pgrp: false,
/// #     known_flags: None,
/// # };
/// let def = CgroupDef::named("io_and_spin")
///     .with_cpuset(CpusetSpec::disjoint(0, 2))
///     .workers(2)
///     .work_type(WorkType::SpinWait)
///     .workload(&BENCH);
///
/// assert!(def.payload.is_some());
/// assert_eq!(def.works[0].num_workers, Some(2));
/// ```
#[derive(Clone, Debug)]
pub struct CgroupDef {
    /// Cgroup name relative to the scenario's parent cgroup. Must be a
    /// valid cgroupfs filename.
    pub name: Cow<'static, str>,
    /// Optional cpuset assignment. `None` inherits the parent cgroup's
    /// cpuset (typically the scenario's usable CPU set).
    pub cpuset: Option<CpusetSpec>,
    /// WorkSpec groups to spawn. Empty means use a single default WorkSpec
    /// (SpinWait, Normal, ctx.workers_per_cgroup workers).
    pub works: Vec<WorkSpec>,
    /// When true, the gauntlet work_type override replaces each WorkSpec's
    /// work_type (applied per-WorkSpec via resolve_work_type).
    pub swappable: bool,
    /// Optional userspace [`Payload`](crate::test_support::Payload) to
    /// launch inside this cgroup.
    ///
    /// **Spawn order within `apply_setup`**: the cgroup is created
    /// (`add_cgroup_no_cpuset`), its cpuset is resolved + set, then
    /// each `WorkSpec` entry is spawned and moved into the cgroup in
    /// declaration order, and finally — after every synthetic
    /// `WorkSpec` handle has started — the `Payload` is spawned via
    /// `PayloadRun::new(ctx, p).in_cgroup(name).spawn()`. This
    /// fixed order lets the cgroup cpuset and mempolicy settle on
    /// the `WorkSpec` handles before the binary inherits placement, so
    /// the binary sees a stable topology. Once spawned, all three
    /// (cgroup, works, payload) run concurrently until teardown.
    ///
    /// Only
    /// [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are accepted — scheduler-kind payloads are rejected
    /// at construction time via [`Self::workload`]. The payload is
    /// killed at step-teardown (before cgroup removal) so the cgroup
    /// removal does not fail with EBUSY.
    pub payload: Option<&'static crate::test_support::Payload>,
    /// Optional cpuset.mems NUMA node binding. `None` inherits the
    /// parent cgroup's `cpuset.mems`. Set via
    /// [`Self::with_cpuset_mems`].
    pub cpuset_mems: Option<BTreeSet<usize>>,
    /// Optional cpu controller limits (`cpu.max`, `cpu.weight`).
    /// `None` leaves both kernel defaults in place. Set via
    /// [`Self::cpu_quota_pct`] / [`Self::cpu_quota`] /
    /// [`Self::cpu_weight`].
    pub cpu: Option<CpuLimits>,
    /// Optional memory controller limits (`memory.max`,
    /// `memory.high`, `memory.low`, `memory.swap.max`). `None`
    /// leaves all four at the kernel defaults. Set via
    /// [`Self::memory_max`] / [`Self::memory_high`] /
    /// [`Self::memory_low`] / [`Self::memory_swap_max`].
    pub memory: Option<MemoryLimits>,
    /// Optional io controller limits (`io.weight`). `None` leaves
    /// the kernel default in place. Set via [`Self::io_weight`].
    pub io: Option<IoLimits>,
    /// Optional pids controller limits (`pids.max`). `None` leaves
    /// the kernel default in place (no ceiling). Set via
    /// [`Self::pids_max`].
    pub pids: Option<PidsLimits>,
}

impl CgroupDef {
    /// Create a CgroupDef with defaults (empty works, no cpuset).
    ///
    /// **Worker-spawning default:** `CgroupDef::named("cg_0")` alone
    /// still spawns workers at execution time — `apply_setup` fills
    /// an empty `works` slice with one default [`WorkSpec`] (SpinWait,
    /// SCHED_NORMAL, `ctx.workers_per_cgroup` workers). To express
    /// an empty move-target cgroup with NO workers, declare it via
    /// [`Op::AddCgroup`] at step or Backdrop level instead of using
    /// a `CgroupDef`.
    #[must_use = "dropping a CgroupDef discards the cgroup specification"]
    pub fn named(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the cpuset for this cgroup. Use when defining cgroups in step
    /// setup (initial topology). For mid-run cpuset changes, use [`Op::SetCpuset`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn with_cpuset(mut self, cpus: CpusetSpec) -> Self {
        self.cpuset = Some(cpus);
        self
    }

    /// Add a work group. Can be called multiple times for concurrent
    /// work groups within this cgroup.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work(mut self, w: WorkSpec) -> Self {
        self.works.push(w);
        self
    }

    /// Ensure works[0] exists for single-WorkSpec builder methods.
    fn ensure_default_work(&mut self) {
        if self.works.is_empty() {
            self.works.push(WorkSpec::default());
        }
    }

    /// Set the number of workers (convenience for single WorkSpec).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workers(mut self, n: usize) -> Self {
        self.ensure_default_work();
        self.works[0].num_workers = Some(n);
        self
    }

    /// Set the work type (convenience for single WorkSpec).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.ensure_default_work();
        self.works[0].work_type = wt;
        self
    }

    /// Set the scheduling policy (convenience for single WorkSpec).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn sched_policy(mut self, p: crate::workload::SchedPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].sched_policy = p;
        self
    }

    /// Set the per-worker affinity (convenience for single WorkSpec).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn affinity(mut self, a: crate::workload::AffinityIntent) -> Self {
        self.ensure_default_work();
        self.works[0].affinity = a;
        self
    }

    /// Set the NUMA memory placement policy (convenience for single WorkSpec).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mem_policy(mut self, p: crate::workload::MemPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags (convenience for single WorkSpec).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn mpol_flags(mut self, f: crate::workload::MpolFlags) -> Self {
        self.ensure_default_work();
        self.works[0].mpol_flags = f;
        self
    }

    /// Set the per-worker nice value for all WorkSpec groups in this
    /// cgroup.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn nice(mut self, n: i32) -> Self {
        self.ensure_default_work();
        for w in &mut self.works {
            w.nice = n;
        }
        self
    }

    /// Set the worker process name for all WorkSpec groups in this
    /// cgroup. Every worker forked from any group calls
    /// `prctl(PR_SET_NAME)` with this name.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn comm(mut self, name: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.ensure_default_work();
        let name = name.into();
        for w in &mut self.works {
            w.comm = Some(name.clone());
        }
        self
    }

    /// Set the worker effective UID for all WorkSpec groups.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn uid(mut self, uid: u32) -> Self {
        self.ensure_default_work();
        for w in &mut self.works {
            w.uid = Some(uid);
        }
        self
    }

    /// Set the worker effective GID for all WorkSpec groups.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn gid(mut self, gid: u32) -> Self {
        self.ensure_default_work();
        for w in &mut self.works {
            w.gid = Some(gid);
        }
        self
    }

    /// Restrict all workers to a NUMA node's CPU set.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn numa_node(mut self, node: u32) -> Self {
        self.ensure_default_work();
        for w in &mut self.works {
            w.numa_node = Some(node);
        }
        self
    }

    /// When true, the gauntlet work_type override replaces each WorkSpec's work type.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn swappable(mut self, swappable: bool) -> Self {
        self.swappable = swappable;
        self
    }

    /// Attach a userspace payload binary that runs inside this cgroup
    /// alongside any synthetic [`WorkSpec`] groups. The payload spawns
    /// when the step enters `apply_setup` and is killed during
    /// step-teardown so the cgroup can be removed cleanly.
    ///
    /// Only
    /// [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are accepted; passing a scheduler-kind
    /// [`Payload`](crate::test_support::Payload) panics with an
    /// actionable message.
    ///
    /// **Why panic at declaration time, not at spawn time?** Three
    /// reasons, all of which favor failing fast:
    /// 1. **Discovery-time surfacing.** `CgroupDef` builders run
    ///    during test construction, which nextest's `--list`
    ///    invocation reaches BEFORE any VM boot. A panic here
    ///    emits a full backtrace inside the test binary and
    ///    surfaces the offending call site immediately; a deferred
    ///    runtime error would require a KVM-capable host + a
    ///    kernel image + an initramfs build to observe — a 30+
    ///    second feedback loop for what is purely a
    ///    typed-API misuse.
    /// 2. **No side effects.** The panic happens before
    ///    `CgroupDef.payload = Some(p)` assignment runs, so the
    ///    in-progress builder is left in its prior (no-payload)
    ///    state. A caller that catches the panic via
    ///    `catch_unwind` sees a valid CgroupDef either way.
    /// 3. **Scheduler-kind is always a programming error here.**
    ///    `Payload::KERNEL_DEFAULT` in `CgroupDef::workload` is never a
    ///    legitimate use case — it means the author confused the
    ///    `scheduler` slot (test-level) with the `workload` slot
    ///    (cgroup-level). There is no recovery path; the only
    ///    resolution is editing the source.
    ///
    /// Scheduler-kind payloads in the step-level `Op::RunPayload`
    /// path bail with an `anyhow::Error` instead of panicking —
    /// that path runs during scenario execution where one bad op
    /// should not crash a whole test run.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn workload(mut self, p: &'static crate::test_support::Payload) -> Self {
        assert!(
            !p.is_scheduler(),
            "CgroupDef::workload called with a scheduler-kind Payload ({}); \
             CgroupDef.workload is for userspace binary payloads only. \
             Use #[ktstr_test(scheduler = ...)] for scheduler placement.",
            p.name,
        );
        self.payload = Some(p);
        self
    }

    /// Bind `cpuset.mems` for this cgroup. Mirrors
    /// [`Self::with_cpuset`] for NUMA memory placement: the cgroup's
    /// tasks may only allocate memory on the listed NUMA nodes.
    /// `None` (default) inherits the parent's `cpuset.mems`.
    ///
    /// Required when the cgroup spans CPUs on a NUMA node whose
    /// memory is NOT in the parent's `cpuset.mems` — without it,
    /// allocations from the cgroup's tasks fail with `ENOMEM` or
    /// migrate per the kernel's `cpuset_update_task_spread` path.
    /// The framework writes `cpuset.mems` immediately after
    /// `cpuset.cpus` so the binding is in effect before any worker
    /// is moved into the cgroup.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn with_cpuset_mems(mut self, nodes: BTreeSet<usize>) -> Self {
        self.cpuset_mems = Some(nodes);
        self
    }

    /// Set `cpu.max` quota as a percentage of one CPU's
    /// throughput, with a default 100 ms `period`. `100` means
    /// "one full CPU" (quota=100_000, period=100_000); `200` means
    /// "two CPUs". Use [`Self::cpu_quota`] for non-default periods.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_quota_pct(mut self, pct: u32) -> Self {
        let cpu = self.cpu.get_or_insert_with(default_cpu_limits);
        cpu.max_period_us = 100_000;
        cpu.max_quota_us = Some((pct as u64) * 1_000);
        self
    }

    /// Set `cpu.max` quota and period directly. `quota` may exceed
    /// `period` (multi-CPU concurrency, see [`CpuLimits::max_quota_us`]).
    /// Both arguments are converted to microseconds; sub-microsecond
    /// fractions in the supplied [`Duration`]s are truncated.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_quota(mut self, quota: Duration, period: Duration) -> Self {
        let cpu = self.cpu.get_or_insert_with(default_cpu_limits);
        cpu.max_quota_us = Some(quota.as_micros() as u64);
        cpu.max_period_us = period.as_micros() as u64;
        self
    }

    /// Clear any previously-set `cpu.max` quota (writes `"max"`),
    /// leaving `cpu.weight` (if set) intact. Useful when a base
    /// CgroupDef builder applied a default cap and the test wants
    /// only weight-based bias.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_unlimited(mut self) -> Self {
        let cpu = self.cpu.get_or_insert_with(default_cpu_limits);
        cpu.max_quota_us = None;
        self
    }

    /// Set `cpu.weight` (range 1..=10000, default 100 in the
    /// kernel). Larger values get a larger CPU share when the
    /// parent cgroup is contended. Independent of `cpu.max`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cpu_weight(mut self, weight: u32) -> Self {
        let cpu = self.cpu.get_or_insert_with(default_cpu_limits);
        cpu.weight = Some(weight);
        self
    }

    /// Set `memory.max` hard ceiling in bytes. Crossing this
    /// triggers the cgroup OOM killer.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_max(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.max = Some(bytes);
        self
    }

    /// Set `memory.high` soft throttle threshold in bytes.
    /// Crossing this triggers reclaim throttling but NOT OOM-kill.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_high(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.high = Some(bytes);
        self
    }

    /// Set `memory.low` soft protection threshold in bytes.
    /// Reclaim prefers other cgroups before this one's memory
    /// drops below `low`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_low(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.low = Some(bytes);
        self
    }

    /// Clear all three memory limits (writes `"max"` for max/high
    /// and `"0"` for low). Equivalent to leaving `memory` unset
    /// at construction; provided for symmetry with
    /// [`Self::cpu_unlimited`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_unlimited(mut self) -> Self {
        self.memory = Some(MemoryLimits::default());
        self
    }

    /// Set `io.weight` (range 1..=10000, default 100 in the
    /// kernel). Biases relative IO share across sibling cgroups
    /// when the io controller is enabled. `io.max` per-device caps
    /// are not surfaced here — see [`IoLimits`] for the rationale.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn io_weight(mut self, weight: u16) -> Self {
        let io = self.io.get_or_insert_with(IoLimits::default);
        io.weight = Some(weight);
        self
    }

    /// Set `memory.swap.max` ceiling in bytes. The kernel parses the
    /// wire value via `page_counter_memparse` and accepts a decimal
    /// byte count (`swap_max_write` in `mm/memcontrol.c`). Distinct
    /// from `memory.max`: this caps how much of the cgroup's memory
    /// can spill to swap, separate from total memory consumption.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_swap_max(mut self, bytes: u64) -> Self {
        let m = self.memory.get_or_insert_with(MemoryLimits::default);
        m.swap_max = Some(bytes);
        self
    }

    /// Clear any previously-set `memory.swap.max` (writes `"max"`).
    /// Mirrors [`Self::cpu_unlimited`] / [`Self::memory_unlimited`]
    /// for a single memory-knob unset; useful when a base
    /// `CgroupDef` builder applied a swap cap and the test wants to
    /// remove only that knob while preserving `memory.max`/`high`/
    /// `low`.
    ///
    /// No-ops when `self.memory == None` — the default state already
    /// means "no swap cap" (apply_setup emits no memory writes for an
    /// unset `memory` field), so creating a fresh `MemoryLimits` just
    /// to set `swap_max = None` would (a) be redundant and (b)
    /// trigger 3 unwanted writes for `memory.max` / `memory.high` /
    /// `memory.low` at apply_setup time. The no-op short-circuit
    /// keeps "fresh CgroupDef + memory_swap_unlimited()" semantically
    /// identical to "fresh CgroupDef".
    #[must_use = "builder methods consume self; bind the result"]
    pub fn memory_swap_unlimited(mut self) -> Self {
        if let Some(m) = self.memory.as_mut() {
            m.swap_max = None;
        }
        self
    }

    /// Set `pids.max` task-count ceiling. `n` is the maximum number
    /// of processes the cgroup may host before subsequent
    /// `fork()` / `clone()` calls return EAGAIN. Existing tasks are
    /// NOT killed when the limit lands below the current count
    /// (per the `pids_max_write` kernel comment: "Limit updates
    /// don't need to be mutex'd, since it isn't critical that any
    /// racing fork()s follow the new limit").
    ///
    /// `n = 0` is rejected at `apply_setup` time: a 0-limit cgroup
    /// halts every fork/clone inside, including the worker spawn
    /// under `CloneMode::Fork` and the `ForkExit` per-iteration
    /// child fork. There is no kernel sentinel for "no fork ever";
    /// `pids_max=0` silently fails every `fork()` inside with
    /// `EAGAIN`, which is almost certainly a configuration bug.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn pids_max(mut self, n: u64) -> Self {
        let pids = self.pids.get_or_insert_with(PidsLimits::default);
        pids.max = Some(n);
        self
    }

    /// Clear any previously-set `pids.max` (writes `"max"`).
    /// Mirrors [`Self::cpu_unlimited`] / [`Self::memory_unlimited`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn pids_unlimited(mut self) -> Self {
        let pids = self.pids.get_or_insert_with(PidsLimits::default);
        pids.max = None;
        self
    }
}

/// Constructor for the default [`CpuLimits`] used by the cgroup
/// builders — `cpu.max` quota off, period 100 ms (the kernel
/// default for `cpu.max`'s second column), `cpu.weight` unset.
/// Extracted so the four builders that ensure_cpu_limits share
/// one initial state and a future change to the default period
/// (e.g. shorter for latency-sensitive tests) only edits here.
fn default_cpu_limits() -> CpuLimits {
    CpuLimits {
        max_quota_us: None,
        max_period_us: 100_000,
        weight: None,
    }
}

impl Default for CgroupDef {
    fn default() -> Self {
        Self {
            name: Cow::Borrowed("cg_0"),
            cpuset: None,
            works: vec![],
            swappable: false,
            payload: None,
            cpuset_mems: None,
            cpu: None,
            memory: None,
            io: None,
            pids: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Step / HoldSpec
// ---------------------------------------------------------------------------

/// How to produce the CgroupDefs for a step's setup phase.
pub enum Setup {
    /// Static list of cgroup definitions.
    Defs(Vec<CgroupDef>),
    /// Factory that generates definitions from the runtime context.
    Factory(fn(&Ctx) -> Vec<CgroupDef>),
}

impl Clone for Setup {
    fn clone(&self) -> Self {
        match self {
            Setup::Defs(defs) => Setup::Defs(defs.clone()),
            Setup::Factory(f) => Setup::Factory(*f),
        }
    }
}

impl std::fmt::Debug for Setup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Setup::Defs(defs) => f.debug_tuple("Defs").field(defs).finish(),
            Setup::Factory(_) => f
                .debug_tuple("Factory")
                .field(&"fn(&Ctx) -> Vec<CgroupDef>")
                .finish(),
        }
    }
}

impl Setup {
    pub(super) fn resolve(&self, ctx: &Ctx) -> Vec<CgroupDef> {
        match self {
            Setup::Defs(defs) => defs.clone(),
            Setup::Factory(f) => f(ctx),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        match self {
            Setup::Defs(defs) => defs.is_empty(),
            Setup::Factory(_) => false,
        }
    }
}

impl From<Vec<CgroupDef>> for Setup {
    fn from(defs: Vec<CgroupDef>) -> Self {
        Setup::Defs(defs)
    }
}

/// A sequence of ops followed by a hold period.
///
/// For non-`Loop` steps, `ops` are applied first, then `setup` cgroups
/// are created, configured, and populated. For `Loop` steps, `setup`
/// runs once before the ops loop. Use `Step::new` to create a step
/// with only ops (no setup).
#[derive(Clone, Debug)]
pub struct Step {
    /// Cgroup setup applied before (non-`Loop`) or once above (`Loop`)
    /// the ops list. Runtime cgroups are spawned from this spec.
    pub setup: Setup,
    /// Ordered operations applied each time the step body runs:
    /// cpuset edits, task moves, spawn/despawn, etc.
    pub ops: Vec<Op>,
    /// How long, and whether to loop, after the ops finish one pass.
    pub hold: HoldSpec,
}

impl Step {
    /// Create a step with ops only (no CgroupDef setup).
    #[must_use = "dropping a Step discards its ops and hold for that scenario phase"]
    pub fn new(ops: Vec<Op>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(vec![]),
            ops,
            hold,
        }
    }

    /// Create a step with CgroupDef setup and a hold period.
    ///
    /// Most steps only need cgroup definitions and a hold duration.
    /// Use [`set_ops`](Step::set_ops) to chain ops onto the step.
    #[must_use = "dropping a Step discards its CgroupDef setup and hold for that scenario phase"]
    pub fn with_defs(defs: Vec<CgroupDef>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(defs),
            ops: vec![],
            hold,
        }
    }

    /// Replace the ops for a step, consuming and returning it.
    ///
    /// Named `set_ops` rather than `with_ops` because the semantics
    /// are REPLACE, not EXTEND — contrast
    /// [`Backdrop::with_ops`](crate::scenario::backdrop::Backdrop::with_ops),
    /// which appends. A chained `Step::new(ops).set_ops(more)`
    /// drops `ops` and keeps only `more`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn set_ops(mut self, ops: Vec<Op>) -> Self {
        self.ops = ops;
        self
    }

    /// Create a step that spawns a single userspace
    /// [`Payload`](crate::test_support::Payload) binary in the
    /// background and holds for the given duration before teardown.
    ///
    /// Shorthand for `Step::new(vec![Op::run_payload(payload,
    /// vec![])], hold)`. The returned step is chainable — add
    /// `.set_ops(...)` to replace the ops vec (note the
    /// REPLACE-not-EXTEND semantics), or use
    /// `Op::wait_payload(name)` / `Op::kill_payload(name)` on later
    /// steps to control the spawned child.
    ///
    /// Test authors who want the payload placed in a named cgroup
    /// should use `Op::run_payload_in_cgroup` directly; this
    /// convenience targets the common "one payload, whole step"
    /// shape.
    #[must_use = "dropping a Step discards its payload and hold for that scenario phase"]
    pub fn with_payload(payload: &'static crate::test_support::Payload, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(vec![]),
            ops: vec![Op::run_payload(payload, vec![])],
            hold,
        }
    }
}

/// How a step advances after its ops are applied. `Frac` and `Fixed`
/// hold for a duration; `Loop` repeatedly re-applies `Step::ops` at a
/// fixed interval instead of holding.
#[derive(Clone, Debug)]
pub enum HoldSpec {
    /// Fraction of the total scenario duration.
    Frac(f64),
    /// Fixed duration.
    Fixed(Duration),
    /// Repeat the step's ops in a loop at the given interval until the
    /// remaining scenario time is exhausted.
    Loop { interval: Duration },
}

impl HoldSpec {
    /// Hold for the full scenario duration (`Frac(1.0)`).
    pub const FULL: HoldSpec = HoldSpec::Frac(1.0);

    /// Reject hold values that are vacuous (no-op step) or would
    /// panic downstream.
    ///
    /// Rules:
    /// - `Fixed(Duration::ZERO)` — the step applies ops and then
    ///   immediately advances; workers get no run time before the
    ///   next step. Almost always a typo; reject.
    /// - `Frac(f)` with `!f.is_finite()` (NaN/Inf) — propagates into
    ///   `Duration::from_secs_f64(f)` which panics.
    /// - `Frac(f)` with `f <= 0.0` — zero is vacuous, negative
    ///   panics in `Duration::from_secs_f64`.
    /// - `Loop { interval: Duration::ZERO }` — busy-polls the
    ///   deadline loop without yielding; almost always a typo.
    pub fn validate(&self) -> std::result::Result<(), String> {
        match self {
            HoldSpec::Fixed(d) if d.is_zero() => {
                Err("HoldSpec::Fixed(Duration::ZERO) is vacuous — workers \
                     get no run time before the next step; use at least a \
                     few ms or drop the step entirely"
                    .into())
            }
            HoldSpec::Frac(f) if !f.is_finite() => Err(format!(
                "HoldSpec::Frac({f}) is not finite (NaN/Inf) — would \
                     panic in Duration::from_secs_f64"
            )),
            HoldSpec::Frac(f) if *f <= 0.0 => Err(format!(
                "HoldSpec::Frac({f}) must be > 0.0; negative values \
                     panic in Duration::from_secs_f64 and zero is vacuous"
            )),
            HoldSpec::Loop { interval } if interval.is_zero() => {
                Err("HoldSpec::Loop { interval: Duration::ZERO } would \
                     busy-spin the deadline check without yielding; use a \
                     non-zero interval"
                    .into())
            }
            _ => Ok(()),
        }
    }
}

impl Op {
    /// Return a unique bit index for each Op variant (for op_kinds bitmask).
    ///
    /// Dispatched via [`OpKind`] — the auto-generated fieldless shadow
    /// enum from `#[derive(strum::EnumDiscriminants)]` on [`Op`]. The
    /// indirection is load-bearing: `OpKind` also derives `EnumIter`,
    /// so `op_kind_bit_indices_are_unique_and_contiguous` can
    /// exhaustively verify every `OpKind` maps to a distinct,
    /// contiguous bit index — guarding against a new variant slipping
    /// in with a duplicated or gap-leaving index.
    pub(super) fn discriminant(&self) -> u32 {
        OpKind::from(self).bit_index()
    }
}

impl OpKind {
    /// Unique bit index per variant, used by [`Op::discriminant`] for
    /// the `op_kinds` bitmask. Contiguous from 0 — the
    /// `op_kind_bit_indices_are_unique_and_contiguous` test iterates
    /// every variant via `EnumIter` and pins this.
    pub(super) fn bit_index(self) -> u32 {
        match self {
            OpKind::AddCgroup => 0,
            OpKind::RemoveCgroup => 1,
            OpKind::SetCpuset => 2,
            OpKind::ClearCpuset => 3,
            OpKind::SwapCpusets => 4,
            OpKind::Spawn => 5,
            OpKind::StopCgroup => 6,
            OpKind::SetAffinity => 7,
            OpKind::SpawnHost => 8,
            OpKind::MoveAllTasks => 9,
            OpKind::RunPayload => 10,
            OpKind::WaitPayload => 11,
            OpKind::KillPayload => 12,
            OpKind::FreezeCgroup => 13,
            OpKind::UnfreezeCgroup => 14,
            OpKind::Snapshot => 15,
            OpKind::WatchSnapshot => 16,
        }
    }
}

impl Op {
    /// Create a new cgroup.
    pub fn add_cgroup(name: impl Into<Cow<'static, str>>) -> Self {
        Op::AddCgroup { name: name.into() }
    }

    /// Remove a cgroup (stops its workers first).
    pub fn remove_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::RemoveCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set a cgroup's cpuset.
    pub fn set_cpuset(cgroup: impl Into<Cow<'static, str>>, cpus: CpusetSpec) -> Self {
        Op::SetCpuset {
            cgroup: cgroup.into(),
            cpus,
        }
    }

    /// Clear a cgroup's cpuset (allow all CPUs).
    pub fn clear_cpuset(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::ClearCpuset {
            cgroup: cgroup.into(),
        }
    }

    /// Swap cpusets between two cgroups.
    pub fn swap_cpusets(a: impl Into<Cow<'static, str>>, b: impl Into<Cow<'static, str>>) -> Self {
        Op::SwapCpusets {
            a: a.into(),
            b: b.into(),
        }
    }

    /// Spawn workers in a cgroup.
    pub fn spawn(cgroup: impl Into<Cow<'static, str>>, work: WorkSpec) -> Self {
        Op::Spawn {
            cgroup: cgroup.into(),
            work,
        }
    }

    /// Stop all workers in a cgroup.
    pub fn stop_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::StopCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set worker affinity in a cgroup.
    pub fn set_affinity(cgroup: impl Into<Cow<'static, str>>, affinity: AffinityIntent) -> Self {
        Op::SetAffinity {
            cgroup: cgroup.into(),
            affinity,
        }
    }

    /// Spawn workers in the parent cgroup.
    pub fn spawn_host(work: WorkSpec) -> Self {
        Op::SpawnHost { work }
    }

    /// Move all tasks from one cgroup to another.
    pub fn move_all_tasks(
        from: impl Into<Cow<'static, str>>,
        to: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::MoveAllTasks {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Spawn a [`Payload`](crate::test_support::Payload) binary in the
    /// background. `args` is appended to `payload.default_args`.
    /// Placement is inherited from the caller; use
    /// [`run_payload_in_cgroup`](Self::run_payload_in_cgroup) to put
    /// the child into a named cgroup.
    pub fn run_payload(payload: &'static crate::test_support::Payload, args: Vec<String>) -> Self {
        Op::RunPayload {
            payload,
            args,
            cgroup: None,
        }
    }

    /// Spawn a [`Payload`](crate::test_support::Payload) in the
    /// background and place the child in a cgroup (relative to the
    /// scenario's parent cgroup).
    pub fn run_payload_in_cgroup(
        payload: &'static crate::test_support::Payload,
        args: Vec<String>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::RunPayload {
            payload,
            args,
            cgroup: Some(cgroup.into()),
        }
    }

    /// Block until the payload named `name` exits, evaluate checks,
    /// and record metrics. Matches whichever cgroup the payload is
    /// in when exactly one copy of the name is live; bails when two
    /// or more copies are live (use
    /// [`wait_payload_in_cgroup`](Self::wait_payload_in_cgroup) to
    /// disambiguate).
    pub fn wait_payload(name: impl Into<Cow<'static, str>>) -> Self {
        Op::WaitPayload {
            name: name.into(),
            cgroup: None,
        }
    }

    /// Block until the payload named `name` that's running inside
    /// the given `cgroup` exits. Use this form when two or more
    /// copies of the same payload are live in different cgroups
    /// and a cgroup-less `wait_payload` would be ambiguous. An
    /// empty-string `cgroup` matches payloads that inherited their
    /// parent's placement (spawned via `Op::run_payload(..., cgroup:
    /// None)`); explicit names match payloads placed via
    /// [`Op::run_payload_in_cgroup`] or
    /// [`CgroupDef::workload`](crate::scenario::ops::CgroupDef::workload).
    pub fn wait_payload_in_cgroup(
        name: impl Into<Cow<'static, str>>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::WaitPayload {
            name: name.into(),
            cgroup: Some(cgroup.into()),
        }
    }

    /// SIGKILL the payload named `name`, evaluate checks, and record
    /// metrics. Matches the unique live copy by name; bails on
    /// ambiguity. See [`wait_payload`](Self::wait_payload) for the
    /// full ambiguity rules and
    /// [`kill_payload_in_cgroup`](Self::kill_payload_in_cgroup)
    /// for the disambiguating form.
    pub fn kill_payload(name: impl Into<Cow<'static, str>>) -> Self {
        Op::KillPayload {
            name: name.into(),
            cgroup: None,
        }
    }

    /// SIGKILL the payload named `name` that's running inside the
    /// given `cgroup`. See
    /// [`wait_payload_in_cgroup`](Self::wait_payload_in_cgroup) for
    /// the placement-matching contract.
    pub fn kill_payload_in_cgroup(
        name: impl Into<Cow<'static, str>>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::KillPayload {
            name: name.into(),
            cgroup: Some(cgroup.into()),
        }
    }

    /// Freeze every task in a cgroup via `cgroup.freeze`.
    pub fn freeze_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::FreezeCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Unfreeze every task in a cgroup via `cgroup.freeze`.
    pub fn unfreeze_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::UnfreezeCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Capture a host-side diagnostic snapshot under `name`. See
    /// [`Op::Snapshot`] for the full request/reply protocol and
    /// no-bridge fallback semantics.
    pub fn snapshot(name: impl Into<Cow<'static, str>>) -> Self {
        Op::Snapshot { name: name.into() }
    }

    /// Register a write-driven snapshot watch on `symbol`. See
    /// [`Op::WatchSnapshot`] for the symbol-resolution rules and
    /// guard rails (max 3 watches per scenario, verbatim vmlinux
    /// ELF symtab match, 4-byte alignment requirement).
    pub fn watch_snapshot(symbol: impl Into<Cow<'static, str>>) -> Self {
        Op::WatchSnapshot {
            symbol: symbol.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// CpusetSpec resolution
// ---------------------------------------------------------------------------

impl CpusetSpec {
    /// Check whether this spec can produce a non-empty cpuset for the
    /// given topology. Returns `Err` with a human-readable reason on
    /// failure.
    pub fn validate(&self, ctx: &Ctx) -> std::result::Result<(), String> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) if *idx >= ctx.topo.num_llcs() => Err(format!(
                "Llc({idx}) out of range: topology has {} LLCs",
                ctx.topo.num_llcs()
            )),
            CpusetSpec::Numa(node) if *node >= ctx.topo.num_numa_nodes() => Err(format!(
                "Numa({node}) out of range: topology has {} NUMA nodes",
                ctx.topo.num_numa_nodes()
            )),
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. } if *of == 0 => {
                Err("partition count (of) must be > 0".into())
            }
            CpusetSpec::Disjoint { index, of, .. } | CpusetSpec::Overlap { index, of, .. }
                if *index >= *of =>
            {
                Err(format!("index {index} >= partition count {of}"))
            }
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if !start_frac.is_finite() || !end_frac.is_finite() => Err(format!(
                "Range start_frac ({start_frac}) or end_frac ({end_frac}) is not finite"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if *start_frac < 0.0 || *end_frac > 1.0 => Err(format!(
                "Range fracs must lie in [0.0, 1.0]: start_frac={start_frac}, end_frac={end_frac}"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if start_frac >= end_frac => Err(format!(
                "Range start_frac ({start_frac}) >= end_frac ({end_frac})"
            )),
            CpusetSpec::Overlap { frac, .. } if !frac.is_finite() => {
                Err(format!("Overlap frac ({frac}) is not finite"))
            }
            CpusetSpec::Overlap { frac, .. } if *frac < 0.0 || *frac > 1.0 => {
                Err(format!("Overlap frac ({frac}) must lie in [0.0, 1.0]"))
            }
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. }
                if usable.len() < *of =>
            {
                Err(format!(
                    "not enough usable CPUs ({}) for {} partitions",
                    usable.len(),
                    of
                ))
            }
            CpusetSpec::Exact(cpus) if cpus.is_empty() => {
                Err("CpusetSpec::Exact(empty) would assign no CPUs to the \
                 cgroup; cpuset.cpus rejects an empty mask and the \
                 cgroup would become unschedulable"
                    .into())
            }
            CpusetSpec::Exact(cpus) => {
                // Reject only CPUs the topology doesn't physically have
                // (`all_cpuset`), not the ones outside `usable_cpuset`.
                // A scheduler author may intentionally pin to an
                // isolated CPU (e.g. the root-reserved one) for
                // testing; writing it to cpuset.cpus is a legitimate
                // operation and the kernel is the final authority on
                // whether the write succeeds. Only truly-nonexistent
                // CPU indices are guaranteed to produce EINVAL.
                let all = ctx.topo.all_cpuset();
                let missing: Vec<usize> =
                    cpus.iter().copied().filter(|c| !all.contains(c)).collect();
                if !missing.is_empty() {
                    return Err(format!(
                        "CpusetSpec::Exact contains CPU(s) {missing:?} \
                         outside the topology's physical CPU set (max \
                         CPU index: {}); writing them to cpuset.cpus \
                         would fail with EINVAL",
                        all.iter().next_back().copied().unwrap_or(0),
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Resolve to a concrete CPU set given the topology.
    ///
    /// **Callers MUST run [`Self::validate`] first and propagate its error.**
    /// `apply_setup` and `apply_ops::SetCpuset` do so via `anyhow::bail!`.
    ///
    /// Defense-in-depth: every malformed input that `validate`
    /// rejects (out-of-range `Llc`/`Numa`, partition `of == 0`,
    /// `index >= of`, inverted or non-finite `Range.start_frac` /
    /// `end_frac`, out-of-bounds `Overlap.frac`) also has a
    /// panic-free fallback here — out-of-range indices clamp to the
    /// last valid index with a `tracing::warn!`, `of == 0` returns
    /// an empty set with a warn, and inverted/non-finite fracs clamp
    /// to `[0, len]` so the resulting slice never inverts. Skipping
    /// `validate` therefore degrades into a usable (possibly empty)
    /// cpuset rather than crashing the caller.
    pub fn resolve(&self, ctx: &Ctx) -> BTreeSet<usize> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) => {
                if *idx >= ctx.topo.num_llcs() {
                    // Graceful fallback: clamp to last LLC instead of panicking.
                    let clamped = ctx.topo.num_llcs().saturating_sub(1);
                    tracing::warn!(
                        llc_idx = idx,
                        num_llcs = ctx.topo.num_llcs(),
                        clamped,
                        "CpusetSpec::Llc index out of range, clamping",
                    );
                    ctx.topo.llc_aligned_cpuset(clamped)
                } else {
                    ctx.topo.llc_aligned_cpuset(*idx)
                }
            }
            CpusetSpec::Numa(idx) => {
                if *idx >= ctx.topo.num_numa_nodes() {
                    let clamped = ctx.topo.num_numa_nodes().saturating_sub(1);
                    tracing::warn!(
                        numa_node = idx,
                        num_numa_nodes = ctx.topo.num_numa_nodes(),
                        clamped,
                        "CpusetSpec::Numa index out of range, clamping",
                    );
                    ctx.topo.numa_aligned_cpuset(clamped)
                } else {
                    ctx.topo.numa_aligned_cpuset(*idx)
                }
            }
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } => {
                let len = usable.len();
                // Defense-in-depth: clamp non-finite fracs to 0 (NaN
                // would saturate to 0 via `as usize` anyway; explicit
                // check matches validate's rejection reason).
                let sf = if start_frac.is_finite() {
                    *start_frac
                } else {
                    0.0
                };
                let ef = if end_frac.is_finite() { *end_frac } else { 0.0 };
                let start = (len as f64 * sf) as usize;
                let end = (len as f64 * ef) as usize;
                // Guard against inverted Range (start_frac > end_frac)
                // — `&usable[start..end]` panics when start > end even
                // if both are clamped to `len`. `start.min(end)` keeps
                // the slice empty in that case instead of panicking.
                let s = start.min(len);
                let e = end.min(len).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Disjoint { index, of } => {
                if *of == 0 {
                    // Defense-in-depth: `validate` rejects of==0 with a
                    // clear error. If a caller reaches `resolve` with
                    // of==0 anyway (skipped validate, or used a
                    // malformed programmatic spec), returning an empty
                    // set is safer than the div-by-zero panic.
                    tracing::warn!("CpusetSpec::Disjoint with of=0 — returning empty cpuset");
                    return BTreeSet::new();
                }
                let chunk = usable.len() / of;
                let start = index * chunk;
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    (index + 1) * chunk
                };
                let s = start.min(usable.len());
                let e = end.min(usable.len()).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Overlap { index, of, frac } => {
                if *of == 0 {
                    tracing::warn!("CpusetSpec::Overlap with of=0 — returning empty cpuset");
                    return BTreeSet::new();
                }
                let chunk = usable.len() / of;
                // Clamp non-finite / out-of-range frac to 0 so the
                // overlap computation stays bounded.
                let frac = if frac.is_finite() {
                    frac.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let overlap = (chunk as f64 * frac) as usize;
                let start = if *index == 0 {
                    0
                } else {
                    (index * chunk).saturating_sub(overlap)
                };
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    ((index + 1) * chunk + overlap).min(usable.len())
                };
                let s = start.min(usable.len());
                let e = end.min(usable.len()).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Exact(cpus) => cpus.clone(),
        }
    }
}
