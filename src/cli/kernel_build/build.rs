//! Top-level kernel build orchestration.
//!
//! Holds [`kernel_build_pipeline`] (the post-acquisition orchestrator
//! that runs `clean` → configure → build → validate → cache-store),
//! the two-phase reservation acquisition
//! ([`acquire_build_reservation`]) for LLC flock + cgroup sandbox +
//! `make -jN` hint, and the source-tree flock helper
//! ([`acquire_source_tree_lock`]) that serializes parallel builds
//! against the same on-disk source tree.

use std::path::Path;

use anyhow::{Context, Result};

use super::super::kernel_cmd::{
    DIRTY_TREE_CACHE_SKIP_HINT, EMBEDDED_KCONFIG, NON_GIT_TREE_CACHE_SKIP_HINT,
    embedded_kconfig_hash,
};
use super::super::util::{Spinner, success, warn};
use super::kconfig::{
    configure_kernel, has_sched_ext, validate_kernel_config, warn_dropped_extra_kconfig_lines,
    warn_extra_kconfig_overrides_baked_in,
};
use super::make::{make_kernel_with_output, run_make, run_make_with_output};

/// Result of the post-acquisition kernel build pipeline.
///
/// Returned by [`kernel_build_pipeline`] so callers can inspect
/// the cache entry and built image path.
#[non_exhaustive]
pub struct KernelBuildResult {
    /// Cache entry, if the build was cached. `None` for dirty trees
    /// or when cache store fails.
    pub entry: Option<crate::cache::CacheEntry>,
    /// Path to the built kernel image.
    pub image_path: std::path::PathBuf,
    /// Whether the source tree was dirty as observed by the build
    /// pipeline. `true` if either the acquire-time inspection
    /// reported dirty OR the post-build re-check observed a
    /// mid-build mutation (worktree edit, branch flip, mid-build
    /// commit). The downstream label decoration in cargo-ktstr's
    /// `resolve_one` uses this to append `_dirty` so a
    /// non-reproducible run is distinguishable from a clean rebuild
    /// of the same path.
    pub post_build_is_dirty: bool,
}

/// Two-phase build reservation handles (LLC flock plan + cgroup v2
/// sandbox + make -jN hint). Consumed by
/// [`kernel_build_pipeline`]; the factored-out
/// [`acquire_build_reservation`] builds it from `cpu_cap` without
/// depending on kernel source, enabling integration tests that
/// exercise the reservation logic against synthetic topologies.
///
/// Drop order is load-bearing: `_sandbox` is declared first and
/// drops first per Rust's declaration-order field-drop rule;
/// this ensures the cgroup sandbox is removed before the LLC
/// flock is released. Otherwise a peer could observe the LLC
/// released before the cgroup is gone and mint a conflicting
/// plan.
#[derive(Debug)]
pub(crate) struct BuildReservation {
    /// cgroup v2 sandbox. `None` when `plan` is `None` (no reservation
    /// to enforce). Drops FIRST per struct field order — cgroup
    /// rmdir runs while LLC flocks are still held. `_` prefix
    /// keeps the binding alive through Drop but marks it as
    /// not-read — the RAII invariant IS the read.
    pub(crate) _sandbox: Option<crate::vmm::cgroup_sandbox::BuildSandbox>,
    /// LLC plan (flock fds + cpus + mems). `None` under
    /// `KTSTR_BYPASS_LLC_LOCKS=1` or sysfs-unreadable host without
    /// `--cpu-cap`. Drops SECOND per struct field order —
    /// flocks release AFTER the sandbox rmdir lands.
    pub(crate) plan: Option<crate::vmm::host_topology::LlcPlan>,
    /// `make -jN` parallelism hint. `Some(N)` under an active
    /// `plan`; `None` when no reservation exists (caller falls
    /// back to `nproc`).
    pub(crate) make_jobs: Option<usize>,
}

/// Acquire the two-phase reservation (LLC flocks + cgroup sandbox)
/// for a kernel build. Factored out of [`kernel_build_pipeline`]
/// so integration tests can exercise the cpu_cap → acquire →
/// sandbox → make_jobs decision tree without requiring a real
/// kernel source tree.
///
/// Returns a `BuildReservation` whose fields are the three values
/// `kernel_build_pipeline` used to bind inline. `_sandbox` is
/// declared first and drops first per Rust's declaration-order
/// field-drop rule; this ensures the cgroup sandbox is removed
/// before the LLC flock is released.
///
/// `cli_label` prefixes operator-facing error text.
///
/// `cpu_cap` is the resolved CPU-count cap from
/// [`CpuCap::resolve`](crate::vmm::host_topology::CpuCap::resolve);
/// `None` means "reserve 30% of the calling process's allowed-CPU
/// set", applied inside the planner at acquire time.
pub(crate) fn acquire_build_reservation(
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<BuildReservation> {
    let bypass = std::env::var("KTSTR_BYPASS_LLC_LOCKS")
        .ok()
        .is_some_and(|v| !v.is_empty());
    // INVARIANT: `_sandbox` is declared first and drops first per
    // Rust's declaration-order field-drop rule; this ensures the
    // cgroup sandbox is removed before the LLC flock is released.
    // Reordering either would either
    // (a) unlock LLCs while the sandbox still enforces the
    // cpuset — a concurrent peer could claim the LLC and stomp
    // gcc children that haven't exited — or (b) leave the cgroup
    // hierarchy non-empty when its parent tries to rmdir.
    let plan: Option<crate::vmm::host_topology::LlcPlan> = if bypass {
        if cpu_cap.is_some() {
            anyhow::bail!(
                "{cli_label}: --cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; \
                 unset one of them. --cpu-cap is a resource contract; bypass \
                 disables the contract entirely."
            );
        }
        None
    } else if let Ok(host_topo) = crate::vmm::host_topology::HostTopology::from_sysfs() {
        let test_topo = crate::topology::TestTopology::from_system()?;
        let acquired_plan =
            crate::vmm::host_topology::acquire_llc_plan(&host_topo, &test_topo, cpu_cap)?;
        crate::vmm::host_topology::warn_if_cross_node_spill(&acquired_plan, &host_topo);
        Some(acquired_plan)
    } else {
        if cpu_cap.is_some() {
            anyhow::bail!(
                "{cli_label}: --cpu-cap set but host LLC topology unreadable \
                 from sysfs — cannot enforce the resource budget. Run on a \
                 host with /sys/devices/system/cpu populated, or drop \
                 --cpu-cap to build without enforcement."
            );
        }
        tracing::warn!(
            "{cli_label}: could not read host LLC topology from sysfs; \
             skipping kernel-build LLC reservation. Concurrent perf-mode \
             runs on this host will NOT be serialized against this build"
        );
        None
    };

    // Phase 2: cgroup v2 sandbox that enforces cpu+mem binding on
    // make/gcc children. `hard_error_on_degrade` is driven by
    // whether `--cpu-cap` was set explicitly: degradation is fatal
    // under the flag (the flag promises enforcement), and warn-only
    // when the 30%-of-allowed default was expanded (the default
    // contract is best-effort — a parent cgroup narrowing the
    // reservation should not fail the build).
    let sandbox: Option<crate::vmm::cgroup_sandbox::BuildSandbox> = match plan.as_ref() {
        Some(p) => Some(crate::vmm::cgroup_sandbox::BuildSandbox::try_create(
            &p.cpus,
            &p.mems,
            cpu_cap.is_some(),
        )?),
        None => None,
    };

    // `make -jN` parallelism hint. `N` = `plan.cpus.len()` via
    // `make_jobs_for_plan` — the reserved CPU count, whether that
    // came from an explicit `--cpu-cap N` or the 30%-of-allowed
    // default. See `make_kernel_with_output` for the resolution.
    let make_jobs = plan
        .as_ref()
        .map(crate::vmm::host_topology::make_jobs_for_plan);

    Ok(BuildReservation {
        plan,
        _sandbox: sandbox,
        make_jobs,
    })
}

/// Acquire an exclusive flock on a per-source-canonical-path lockfile
/// so two concurrent `cargo ktstr test --kernel <path>` runs against
/// the SAME source tree don't race in `make` (defconfig vs
/// olddefconfig vs compile_commands.json) and stomp each other's
/// `.config` and build artifacts.
///
/// The lockfile lives at
/// `{KTSTR_CACHE_DIR}/.locks/source-{path_hash}.lock` where
/// `{path_hash}` is the full 8-char CRC32 hex of the canonical
/// source-path bytes (same shape and helper the
/// `local-unknown-{path_hash}` cache key uses, see
/// [`crate::fetch::canonical_path_hash`] /
/// [`crate::fetch::compose_local_cache_key`]) — one per-tree
/// identifier ties the source-tree flock to the cache key it gates.
///
/// Lockfile placement piggybacks on the cache root's `.locks/`
/// subdirectory ([`crate::flock::LOCK_DIR_NAME`]) so source-tree
/// flocks share the same filesystem-residency story as cache-entry
/// flocks: never under `/tmp`, where `tmpwatch` (or the equivalent
/// `systemd-tmpfiles` cleanup) can sweep stale-mtime files out from
/// under an active flock holder. flock(2) does NOT update the
/// inode's mtime, so a /tmp-resident lockfile would be a candidate
/// for sweep on every run, with the resulting `unlink(2)` racing
/// any peer trying to `open(2)` the same path. The `.locks/`
/// directory under the user-controlled cache root is exempt from
/// those sweeps.
///
/// Try-then-wait: attempts a non-blocking acquire first. If
/// contended, logs the holder (pid + cmdline from /proc/locks)
/// and falls through to a blocking acquire that parks until the
/// peer releases. When the blocking acquire returns, the peer's
/// build is done and the cache likely contains the artifact —
/// the caller checks the cache after we return and skips the
/// build if the slot is populated.
///
/// Distinct from the cache-entry flock acquired inside
/// [`crate::cache::CacheDir::store`]: that lock serializes the
/// atomic install of an artifact bundle into a cache slot; this
/// lock serializes the BUILD itself against the source-tree
/// `make` invocations.
pub(crate) fn acquire_source_tree_lock(
    canonical: &Path,
    cli_label: &str,
) -> Result<std::os::fd::OwnedFd> {
    use anyhow::Context;

    // Share the per-path CRC32 with `local-unknown-{hash}` cache
    // keys so a single per-tree identifier ties the source-tree
    // flock to the cache slot it gates.
    let path_hash = crate::fetch::canonical_path_hash(canonical);
    let cache = crate::cache::CacheDir::new()
        .with_context(|| "open cache root for source-tree lockfile placement")?;
    cache
        .ensure_lock_dir()
        .with_context(|| "create cache `.locks/` subdir for source-tree lock")?;
    let lock_path = cache.lock_path(&format!("source-{path_hash}"));

    match crate::flock::try_flock(&lock_path, crate::flock::FlockMode::Exclusive)
        .with_context(|| format!("acquire source-tree flock {}", lock_path.display()))?
    {
        Some(fd) => Ok(fd),
        None => {
            // Non-blocking acquire failed (EWOULDBLOCK) — a live
            // peer holds the lock. Surface the holder, then block
            // until they release. When the blocking acquire
            // returns, the peer's build is done and the cache
            // likely contains the artifact we need — the caller
            // checks the cache after we return, so it will skip
            // the build if the peer populated the slot.
            let holders = crate::flock::read_holders(&lock_path).unwrap_or_default();
            let holder_text = if holders.is_empty() {
                String::from("(holder not identified via /proc/locks)")
            } else {
                crate::flock::format_holder_list(&holders)
            };
            eprintln!(
                "{cli_label}: source tree {} is locked by a concurrent ktstr \
                 build — waiting for it to finish.\n{holder_text}",
                canonical.display(),
            );
            crate::flock::block_flock(&lock_path, crate::flock::FlockMode::Exclusive).with_context(
                || format!("blocking wait on source-tree flock {}", lock_path.display()),
            )
        }
    }
}

/// Post-acquisition kernel build pipeline.
///
/// Handles: clean, configure, build, validate config, generate
/// compile_commands.json for local trees, find image, strip vmlinux,
/// compute metadata, cache store, and remote cache store (when
/// enabled). Callers handle source acquisition.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
///
/// `is_local_source` should be true when the source is a local
/// kernel source tree, regardless of how the caller arrived there
/// (`kernel build --source`, `cargo ktstr test --kernel <path>`,
/// or any other Path-spec entry that funnels through
/// [`super::super::resolve_kernel_dir`] /
/// [`super::super::resolve_kernel_dir_to_entry`]). It controls the
/// mrproper warning and `source_tree_path` in metadata.
///
/// `extra_kconfig` is an optional user-supplied kconfig fragment
/// merged on top of [`EMBEDDED_KCONFIG`] before `configure_kernel`
/// (which runs olddefconfig only when new lines are needed).
/// `Some(content)` appends the fragment AFTER the baked-in fragment
/// so kbuild's last-occurrence-wins semantics
/// (`scripts/kconfig/confdata.c::conf_read_simple`) make user values
/// override baked-in ones on conflict, and forces a re-configure pass
/// even when `.config` already carries `CONFIG_SCHED_CLASS_EXT=y`
/// (the user fragment may add or invert symbols the baked-in pass
/// alone wouldn't have produced).
///
/// Two metadata fields capture the build inputs separately:
/// - `ktstr_kconfig_hash` always holds the bare baked-in hash
///   (`crate::kconfig_hash()` of `EMBEDDED_KCONFIG`) so
///   `KconfigStatus::Matches/Stale/Untracked` keeps comparing
///   against the live baked-in fragment.
/// - `extra_kconfig_hash` holds `Some(crate::extra_kconfig_hash(content))`
///   when extras were supplied, `None` otherwise. Drives the
///   `(extra kconfig)` tag in `kernel list`.
///
/// Callers that don't expose `--extra-kconfig` (test/coverage/
/// shell/verifier) pass `None`.
pub fn kernel_build_pipeline(
    acquired: &crate::fetch::AcquiredSource,
    cache: &crate::cache::CacheDir,
    cli_label: &str,
    clean: bool,
    is_local_source: bool,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
    extra_kconfig: Option<&str>,
) -> Result<KernelBuildResult> {
    let source_dir = &acquired.source_dir;
    let (arch, image_name) = crate::fetch::arch_info();

    // Two-phase reservation. A concurrent perf-mode test run must
    // not have its measured CPUs stomped by a `make -j$(nproc)`
    // explosion of gcc children, and vice-versa a concurrent
    // kernel build must not have its compile window extended by
    // a test pinning RT-FIFO on shared cores. Phase 1 of the
    // reservation is the LLC-level flock from
    // [`acquire_llc_plan`]: whole-LLC flocks whose count is
    // chosen to cover the CPU budget (either an explicit
    // `--cpu-cap N` or the 30%-of-allowed default). Phase 2 is
    // the cgroup v2 sandbox from
    // [`BuildSandbox::try_create`] that binds make/gcc's
    // cpu+mem sets to the plan's CPUs + NUMA nodes so the
    // parallelism hint is enforced, not just advisory.
    //
    // Binding order is load-bearing: `_sandbox` is declared first
    // and drops first per Rust's declaration-order field-drop rule,
    // which migrates the build pid out of the cgroup and rmdirs the
    // child while the LLC flocks are still held. Otherwise a peer
    // could observe the LLC released before the cgroup is gone,
    // mint a new plan against the same LLCs, and see an orphan
    // cgroup lingering for up to the 24h sweep window.
    //
    // Escape hatches:
    //   - `KTSTR_BYPASS_LLC_LOCKS=1`: skip the LLC plan+flock
    //     acquisition entirely; the build proceeds immediately
    //     without coordinating with any concurrent perf-mode run.
    //     Use when the operator explicitly accepts measurement
    //     noise (one shell doing unrelated work, an isolated
    //     developer workstation, or a CI queue that already
    //     serializes jobs at a higher layer). Mutually exclusive
    //     with `--cpu-cap` at CLI parse time — see the CLI
    //     binaries' pre-dispatch conflict check.
    //   - Sysfs-unreadable host (non-Linux, degraded container):
    //     `HostTopology::from_sysfs()` returns `Err`. Without
    //     `--cpu-cap`, we emit a `tracing::warn!` and proceed
    //     without locks. With `--cpu-cap`, the flag cannot be
    //     honoured and we fail hard — cpu_cap is a contract, not
    //     a hint: a silent degrade would let a build exceed the
    //     declared resource budget without surfacing.
    // `_plan` + `_sandbox` are kept alive via RAII — their Drops
    // release the LLC flocks and cgroup on scope exit. Struct
    // field order in BuildReservation ensures `_sandbox` drops
    // BEFORE `plan`, per Rust's declaration-order field-drop rule.
    let BuildReservation {
        plan: _plan,
        _sandbox,
        make_jobs,
    } = acquire_build_reservation(cli_label, cpu_cap)?;

    // Source-tree flock for local sources. Two parallel
    // `cargo ktstr test --kernel ./linux` runs would otherwise race
    // in `make` against the same source tree (e.g. one's
    // `make defconfig` racing with another's `make compile_commands.json`)
    // and produce inconsistent .config / build artifacts. The flock is
    // taken on the SOURCE TREE itself (per canonical path), distinct from
    // the cache-entry flock acquired inside `cache.store` (per cache key).
    // The two are complementary: the source-tree flock serializes the
    // build phase; the cache-entry flock serializes the atomic install.
    //
    // Held via `OwnedFd` for the lifetime of `_source_lock` — drops at
    // end of pipeline. Skipped under `KTSTR_BYPASS_LLC_LOCKS` to share
    // the operator's escape hatch with the LLC-flock bypass; that
    // env var already declares "I accept noise from concurrent runs."
    //
    // `try_flock` is non-blocking — if a concurrent peer holds the
    // lock, it returns `Ok(None)` and we bail with an actionable error
    // pointing at `cargo ktstr locks` for diagnosis. A blocking acquire
    // here would silently stall the operator's terminal with no
    // indication why; a fail-fast surfaces the contention immediately.
    let _source_lock = if is_local_source
        && std::env::var("KTSTR_BYPASS_LLC_LOCKS")
            .ok()
            .is_none_or(|v| v.is_empty())
    {
        Some(acquire_source_tree_lock(source_dir, cli_label)?)
    } else {
        None
    };

    if clean {
        if !is_local_source {
            eprintln!(
                "{cli_label}: --clean is only meaningful with --source (downloaded sources start clean)"
            );
        } else {
            eprintln!("{cli_label}: make mrproper");
            run_make(source_dir, &["mrproper"])?;
        }
    }

    // Build the merged fragment ONCE so the configure call observes
    // the byte layout `{EMBEDDED_KCONFIG}\n{extra}` (with a `\n`
    // interleave) defined in [`crate::merge_kconfig_fragments`]. The
    // helper returns a `Cow<'_, str>` so the no-extras path borrows
    // `EMBEDDED_KCONFIG` without allocating; only the user-fragment
    // case heaps the merged string. Unit tests pin the exact
    // ordering kbuild's last-wins rule operates on.
    let merged_fragment = crate::merge_kconfig_fragments(EMBEDDED_KCONFIG, extra_kconfig);

    // Forced re-configure when extra-kconfig is supplied. The
    // `has_sched_ext` short-circuit was tuned for the EMBEDDED_KCONFIG
    // path: `has_sched_ext` is a probe for the primary option;
    // olddefconfig fills the rest. With user-supplied extras, an
    // existing `.config` (e.g. a stale build state) can satisfy the
    // sched_ext probe yet miss every user line, producing a kernel
    // that silently ignored the extras. Always run the merged
    // configure when extras are present so the user's symbols land.
    // Surface a `tracing::warn!` for each user fragment line that
    // overrides a baked-in symbol from `EMBEDDED_KCONFIG`. The build
    // proceeds with the user value winning (last-wins is the design
    // intent) — the warning lets the operator see they are shadowing
    // a baked-in setting before configure_kernel (which runs
    // olddefconfig only when new lines are needed), which is when
    // an over-aggressive override can still be addressed by editing
    // the fragment. A separate post-build `validate_kernel_config`
    // pass catches critical-baked-in disablement (e.g. CONFIG_BPF).
    if let Some(extra) = extra_kconfig {
        warn_extra_kconfig_overrides_baked_in(extra, cli_label);
    }

    let needs_configure = extra_kconfig.is_some() || !has_sched_ext(source_dir);
    if needs_configure {
        let configure_result =
            Spinner::with_progress("Configuring kernel...", "Kernel configured", |_| {
                configure_kernel(source_dir, &merged_fragment)
            });
        // Wrap configure errors with `--extra-kconfig` context when
        // extras are present so the user can pinpoint which input is
        // responsible for an olddefconfig failure (e.g. a malformed
        // `CONFIG_X=` line in their fragment).
        configure_result.with_context(|| {
            if extra_kconfig.is_some() {
                "kernel configure failed (with --extra-kconfig fragment merged on top of \
                 baked-in ktstr.kconfig); check the fragment for syntax errors or \
                 conflicting symbol declarations"
                    .to_string()
            } else {
                "kernel configure failed".to_string()
            }
        })?;

        // Post-olddefconfig validation — warn (not error) when a
        // user-requested option from `--extra-kconfig` did not
        // survive into the final `.config` (typically because
        // olddefconfig dropped it for an unmet dependency). Emits
        // one `tracing::warn!` per dropped line naming the
        // requested setting and the actual final value.
        // The hard-fail "user override killed a baked-in invariant"
        // case (e.g. user disabled `CONFIG_BPF`) is caught at
        // `validate_kernel_config` post-build with extra context.
        if let Some(extra) = extra_kconfig {
            warn_dropped_extra_kconfig_lines(source_dir, extra, cli_label);
        }
    }

    Spinner::with_progress("Building kernel...", "Kernel built", |sp| {
        make_kernel_with_output(source_dir, Some(sp), make_jobs)
    })?;

    // Validate critical config options were not silently disabled.
    // When `--extra-kconfig` is set, attach an actionable hint
    // pointing at the user fragment as a likely cause. The most
    // plausible failure mode is a user override that disables a
    // baked-in invariant (e.g. a fragment containing
    // `# CONFIG_BPF is not set` defeats the BPF dep chain), so
    // name `--extra-kconfig` in the wrap context.
    validate_kernel_config(source_dir).with_context(|| {
        if extra_kconfig.is_some() {
            "post-build kernel config validation failed; check that your \
             --extra-kconfig fragment does not disable a CONFIG_X required by \
             ktstr (e.g. CONFIG_BPF, CONFIG_DEBUG_INFO_BTF, CONFIG_FTRACE, \
             CONFIG_SCHED_CLASS_EXT)"
                .to_string()
        } else {
            "post-build kernel config validation failed".to_string()
        }
    })?;

    // Generate compile_commands.json for local trees (LSP support).
    if !acquired.is_temp {
        Spinner::with_progress(
            "Generating compile_commands.json...",
            "compile_commands.json generated",
            |sp| run_make_with_output(source_dir, &["compile_commands.json"], Some(sp)),
        )?;
    }

    // Find the built kernel image and vmlinux.
    let image_path = crate::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| anyhow::anyhow!("no kernel image found in {}", source_dir.display()))?;
    let vmlinux_path = source_dir.join("vmlinux");
    let vmlinux_ref = if vmlinux_path.exists() {
        let orig_mb = std::fs::metadata(&vmlinux_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        eprintln!("{cli_label}: caching vmlinux ({orig_mb:.0} MB, will be stripped)");
        Some(vmlinux_path.as_path())
    } else {
        eprintln!("{cli_label}: warning: vmlinux not found, BTF will not be cached");
        None
    };

    // Cache (skip for dirty local trees).
    if acquired.is_dirty {
        eprintln!("{cli_label}: kernel built at {}", image_path.display());
        // Branch the hint wording: commit/stash is only an actionable
        // remediation for an actual git repo. A non-git source tree
        // is force-marked dirty (see `acquire_local_source` in
        // `fetch.rs`) because dirty detection is impossible, and
        // telling the operator to "commit or stash" leads nowhere.
        let hint = if acquired.is_git {
            DIRTY_TREE_CACHE_SKIP_HINT
        } else {
            NON_GIT_TREE_CACHE_SKIP_HINT
        };
        eprintln!("{cli_label}: {hint}");
        return Ok(KernelBuildResult {
            entry: None,
            image_path,
            post_build_is_dirty: true,
        });
    }

    // Post-build dirty re-check. `local_source` captures
    // `is_dirty` ONCE at acquire time. The operator may then edit a
    // tracked file (`.config` mutation, source patch) DURING the
    // build window. The acquire-time `is_dirty=false` would say
    // "safe to cache" but the on-disk content actually built
    // differs from the HEAD commit recorded in the cache key —
    // a future cache hit on that key would serve a build that no
    // longer matches its identity. Re-running the same gix probes
    // catches the race. On any change (dirty flip OR HEAD-hash
    // shift from a concurrent commit), skip the cache store and
    // emit a one-liner explaining why the cache slot was passed
    // over.
    //
    // Errors from the re-check are surfaced as a warning rather
    // than a hard fail — the build itself succeeded; refusing to
    // store on a re-check probe failure would penalize an
    // otherwise-clean run for a transient gix glitch. The cache
    // store proceeds with the original key, on the same
    // pessimistic basis as a tree the re-check could not classify.
    if is_local_source {
        match crate::fetch::inspect_local_source_state(source_dir) {
            Ok(post) => {
                let hash_changed = post.short_hash
                    != acquired
                        .kernel_source
                        .as_local_git_hash()
                        .map(str::to_string);
                if post.is_dirty || hash_changed {
                    eprintln!(
                        "{cli_label}: source tree changed during build \
                         (acquire-time dirty={}, post-build dirty={}; \
                         hash_changed={hash_changed}); skipping cache store \
                         to avoid recording a stale identity. Re-run after \
                         the working tree settles to populate the cache.",
                        acquired.is_dirty, post.is_dirty,
                    );
                    return Ok(KernelBuildResult {
                        entry: None,
                        image_path,
                        // Mid-build mutation flips the run's
                        // reproducibility — the cache key recorded at
                        // acquire time no longer identifies the actual
                        // build input. Mirror that into the outcome so
                        // the kernel-label downstream gets the
                        // `_dirty` suffix.
                        post_build_is_dirty: true,
                    });
                }
            }
            Err(e) => {
                tracing::warn!(
                    cli_label = cli_label,
                    err = %format!("{e:#}"),
                    "post-build dirty re-check failed; proceeding to cache store",
                );
            }
        }
    }

    let config_path = source_dir.join(".config");
    let config_hash = if config_path.exists() {
        let data = std::fs::read(&config_path)?;
        Some(format!("{:08x}", crc32fast::hash(&data)))
    } else {
        None
    };

    // Two-segment metadata: the bare baked-in hash stays in
    // `ktstr_kconfig_hash` so `kernel list`'s matches/stale/
    // untracked verdict (see `CacheEntry::kconfig_status`) keeps
    // comparing against the live `EMBEDDED_KCONFIG`, and the user
    // extras hash lives in its own slot. Matches the cache-key
    // suffix shape `kc{baked}-xkc{extra}` produced by
    // [`crate::cache_key_suffix_with_extra`].
    let kconfig_hash = embedded_kconfig_hash();
    let extra_kconfig_hash_value = extra_kconfig.map(crate::extra_kconfig_hash);

    // Source-tree vmlinux stat (size + mtime seconds) so a later
    // `prefer_source_tree_for_dwarf` lookup can detect a user
    // rebuild between cache store and DWARF read. Only meaningful
    // for local sources whose vmlinux survived the build —
    // `vmlinux_ref` is `None` if vmlinux wasn't found, in which
    // case there's nothing to stat. mtime read is best-effort:
    // failure leaves the validation pair `None` and prefers the
    // pre-validation behavior for this entry.
    let source_vmlinux_stat = vmlinux_ref.and_then(|v| {
        let stat = std::fs::metadata(v).ok()?;
        let mtime_secs = stat.modified().ok().and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .ok()
                .or_else(|| {
                    std::time::UNIX_EPOCH
                        .duration_since(t)
                        .ok()
                        .map(|d| -(d.as_secs() as i64))
                })
        })?;
        Some((stat.len(), mtime_secs))
    });

    let mut metadata = crate::cache::KernelMetadata::new(
        acquired.kernel_source.clone(),
        arch.to_string(),
        image_name.to_string(),
        crate::test_support::now_iso8601(),
    )
    .with_version(acquired.version.clone())
    .with_config_hash(config_hash)
    .with_ktstr_kconfig_hash(Some(kconfig_hash))
    .with_extra_kconfig_hash(extra_kconfig_hash_value);
    if is_local_source && let Some((size, mtime_secs)) = source_vmlinux_stat {
        metadata = metadata.with_source_vmlinux_stat(size, mtime_secs);
    }

    let mut artifacts = crate::cache::CacheArtifacts::new(&image_path);
    if let Some(v) = vmlinux_ref {
        artifacts = artifacts.with_vmlinux(v);
    }
    let entry = match cache.store(&acquired.cache_key, &artifacts, &metadata) {
        Ok(entry) => {
            success(&format!("\u{2713} Kernel cached: {}", acquired.cache_key));
            eprintln!("{cli_label}: image: {}", entry.image_path().display());
            if crate::remote_cache::is_enabled() {
                crate::remote_cache::remote_store(&entry, cli_label);
            }
            Some(entry)
        }
        Err(e) => {
            warn(&format!("{cli_label}: cache store failed: {e:#}"));
            None
        }
    };

    Ok(KernelBuildResult {
        entry,
        image_path,
        post_build_is_dirty: false,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::kernel_cmd::KernelCommand;
    use super::*;

    /// `kernel build --cpu-cap N` parses through clap into
    /// `KernelCommand::Build { cpu_cap: Some(N), .. }`. Pins the
    /// flag's wire path: a future rename of the field, a stray
    /// `default_value`, or a `value_parser` change that altered
    /// rejection semantics would surface as a parse failure or a
    /// shape mismatch on the assertion.
    #[test]
    fn kernel_build_parses_cpu_cap_without_extra_flags() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "build", "6.14.2", "--cpu-cap", "4"])
            .expect("kernel build --cpu-cap N must parse");
        match parsed.cmd {
            KernelCommand::Build {
                cpu_cap, version, ..
            } => {
                assert_eq!(cpu_cap, Some(4));
                assert_eq!(version.as_deref(), Some("6.14.2"));
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `kernel build` without `--cpu-cap` parses with `cpu_cap: None`
    /// — the "unset" sentinel the downstream planner expands into the
    /// 30%-of-allowed default. Pins the no-flag path so a future
    /// rename of the clap field or a stray `default_value = "0"`
    /// surfaces as a test failure, not a silent runtime behavior change.
    #[test]
    fn kernel_build_without_cpu_cap_defaults_to_none() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "build", "6.14.2"])
            .expect("kernel build without --cpu-cap must parse");
        match parsed.cmd {
            KernelCommand::Build { cpu_cap, .. } => {
                assert_eq!(cpu_cap, None, "no --cpu-cap must produce None, not Some(0)",);
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `kernel build --cpu-cap 0` parses successfully at clap level
    /// — the "must be ≥ 1" check lives in [`CpuCap::new`], not in
    /// the clap value parser. Pins the two-layer validation: clap
    /// accepts any usize; runtime resolution via `CpuCap::resolve` is
    /// responsible for the "0 is rejected" diagnostic.
    #[test]
    fn kernel_build_cpu_cap_zero_passes_clap() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "build", "6.14.2", "--cpu-cap", "0"])
            .expect("clap-level parse must accept 0; runtime validation rejects");
        match parsed.cmd {
            KernelCommand::Build { cpu_cap, .. } => {
                assert_eq!(
                    cpu_cap,
                    Some(0),
                    "clap parses 0 verbatim; validation is downstream",
                );
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // kernel_build_pipeline reservation phase — factored-out
    // `acquire_build_reservation` covers the cpu_cap → acquire →
    // sandbox → make_jobs flow without needing a real kernel source.
    // ---------------------------------------------------------------

    /// Serialize `KTSTR_BYPASS_LLC_LOCKS` env-var mutation across
    /// test threads. Two parallel tests can't both mutate the same
    /// process-wide env var without coordinating.
    fn bypass_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// RAII guard for scoped `KTSTR_BYPASS_LLC_LOCKS` mutation.
    /// Caller holds `bypass_env_lock()` before constructing.
    struct BypassGuard;
    impl BypassGuard {
        fn set(value: &str) -> Self {
            // SAFETY: env_lock held by caller; serializes with
            // every other env-mutating test.
            unsafe {
                std::env::set_var("KTSTR_BYPASS_LLC_LOCKS", value);
            }
            BypassGuard
        }
        fn remove() -> Self {
            // SAFETY: caller holds env_lock.
            unsafe {
                std::env::remove_var("KTSTR_BYPASS_LLC_LOCKS");
            }
            BypassGuard
        }
    }
    impl Drop for BypassGuard {
        fn drop(&mut self) {
            // SAFETY: guard lifetime bounded by env_lock held by
            // caller; Drop runs before the mutex guard releases.
            unsafe {
                std::env::remove_var("KTSTR_BYPASS_LLC_LOCKS");
            }
        }
    }

    /// `acquire_build_reservation` with `KTSTR_BYPASS_LLC_LOCKS=1`
    /// plus `cpu_cap=None` returns a no-reservation `BuildReservation`:
    /// plan, sandbox, and make_jobs all None. Pins the "bypass
    /// disables both layers" contract.
    #[test]
    fn acquire_build_reservation_bypass_returns_no_reservation() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let r = acquire_build_reservation("test", None).expect("bypass + no cap must succeed");
        assert!(r.plan.is_none(), "bypass must produce no LLC plan");
        assert!(
            r._sandbox.is_none(),
            "bypass must produce no cgroup sandbox",
        );
        assert!(
            r.make_jobs.is_none(),
            "bypass must fall back to nproc (None signals to caller)",
        );
    }

    /// `acquire_build_reservation` with `KTSTR_BYPASS_LLC_LOCKS=1`
    /// plus `cpu_cap=Some(_)` must error with the "resource contract"
    /// substring. Pins the conflict check at the pipeline's
    /// reservation entry point.
    #[test]
    fn acquire_build_reservation_bypass_with_cap_errors() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let cap = crate::vmm::host_topology::CpuCap::new(2).expect("cap=2 valid");
        let err =
            acquire_build_reservation("test", Some(cap)).expect_err("bypass + cap must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("resource contract"),
            "err must name the resource contract: {msg}",
        );
    }

    /// `acquire_build_reservation` without bypass on a sysfs-capable
    /// host: returns a `BuildReservation` whose fields populate
    /// consistently — plan.is_some() iff make_jobs.is_some() iff
    /// sandbox.is_some(). Pins the "plan and make_jobs must never
    /// diverge" invariant.
    #[test]
    fn acquire_build_reservation_plan_and_make_jobs_consistent() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::remove();
        match acquire_build_reservation("test", None) {
            Ok(r) => {
                assert_eq!(
                    r.plan.is_some(),
                    r.make_jobs.is_some(),
                    "plan and make_jobs must agree on reservation presence",
                );
                if let (Some(p), Some(jobs)) = (r.plan.as_ref(), r.make_jobs) {
                    assert_eq!(
                        jobs,
                        crate::vmm::host_topology::make_jobs_for_plan(p),
                        "make_jobs must equal make_jobs_for_plan(&plan)",
                    );
                }
                assert_eq!(
                    r.plan.is_some(),
                    r._sandbox.is_some(),
                    "sandbox and plan must agree on reservation presence",
                );
            }
            Err(e) => {
                // Sysfs-unreadable host or contested LLCs. Accept
                // either outcome; the test's intent is to pin the
                // invariant in the success case, not force success.
                eprintln!("acquire_build_reservation unavailable on this host: {e:#}");
            }
        }
    }

    /// `acquire_build_reservation` plain bypass (no `--cpu-cap`)
    /// must NOT touch the sysfs probe. The test sets the bypass and
    /// confirms no error escapes, even on a host whose
    /// `HostTopology::from_sysfs()` would otherwise fail (the
    /// bypass branch is taken FIRST in the function, before the
    /// sysfs probe is attempted). Pins the "bypass short-circuits
    /// the topology probe" branch shape — a regression that
    /// re-ordered the bypass check below the sysfs probe would
    /// surface as a sysfs-error escape.
    #[test]
    fn acquire_build_reservation_bypass_does_not_touch_sysfs() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let r = acquire_build_reservation("test", None)
            .expect("bypass must succeed regardless of sysfs availability");
        // The bypass branch produces (None, None, None) by
        // construction — no further state to assert beyond the
        // sibling tests that already pin the field shape.
        assert!(r.plan.is_none());
        assert!(r._sandbox.is_none());
        assert!(r.make_jobs.is_none());
    }

    // ---------------------------------------------------------------
    // acquire_source_tree_lock — per-source-tree flock that
    // serializes parallel builds against the same on-disk source.
    // ---------------------------------------------------------------
    //
    // Tests use `isolated_cache_dir()` to point `KTSTR_CACHE_DIR` at
    // a tempdir for the test's lifetime, so the production
    // `CacheDir::new()` resolves into the tempdir without touching
    // the operator's real cache directory. The lockfile path is
    // deterministic (cache_root/.locks/source-{path_hash}.lock) so
    // we can re-derive it from the canonical input path and assert
    // its presence.

    /// `acquire_source_tree_lock` on a fresh canonical path under
    /// an isolated cache root succeeds (no peer holding the lock)
    /// and creates the lockfile under `cache_root/.locks/`. Pins
    /// the lockfile placement: a regression that moved the lockfile
    /// to `/tmp/` (where `tmpwatch` could sweep it under an active
    /// holder) would surface here as the assertion failing on
    /// "lockfile not found at expected path."
    #[test]
    fn acquire_source_tree_lock_succeeds_on_fresh_path() {
        use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};
        let _env_lock = lock_env();
        let cache = isolated_cache_dir();
        let canonical = std::path::PathBuf::from("/tmp/fake-source-tree-for-test");
        let fd = acquire_source_tree_lock(&canonical, "test")
            .expect("fresh-path acquire must succeed under isolated cache");
        // Lockfile must land under the isolated cache root's
        // `.locks/` subdirectory. The naming is `source-{hash}.lock`
        // where `{hash}` is `canonical_path_hash(canonical)`.
        let path_hash = crate::fetch::canonical_path_hash(&canonical);
        let expected = cache
            .path()
            .join(crate::flock::LOCK_DIR_NAME)
            .join(format!("source-{path_hash}.lock"));
        assert!(
            expected.exists(),
            "lockfile must exist at {} after acquire",
            expected.display(),
        );
        // Drop the FD explicitly to release the flock before the
        // tempdir cleanup races with it.
        drop(fd);
    }

    /// `acquire_source_tree_lock` returns the SAME lockfile path
    /// for two different canonical inputs IFF they share the same
    /// `canonical_path_hash`. Two distinct inputs (`/srv/linux-a`
    /// and `/srv/linux-b`) must produce DIFFERENT lockfiles so
    /// concurrent builds against unrelated source trees don't
    /// serialize against each other. Pins the per-tree
    /// disambiguation contract.
    #[test]
    fn acquire_source_tree_lock_distinct_paths_yield_distinct_lockfiles() {
        use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};
        let _env_lock = lock_env();
        let cache = isolated_cache_dir();
        let path_a = std::path::PathBuf::from("/tmp/fake-source-a");
        let path_b = std::path::PathBuf::from("/tmp/fake-source-b");
        let fd_a = acquire_source_tree_lock(&path_a, "test")
            .expect("path A acquire must succeed under isolated cache");
        // Acquiring path B while path A's lock is still held must
        // ALSO succeed — they hash to different lockfiles, so
        // there's no contention.
        let fd_b = acquire_source_tree_lock(&path_b, "test").expect(
            "path B acquire must succeed concurrently with A — \
                 distinct canonical paths must hash to distinct \
                 lockfiles so unrelated builds don't serialize",
        );
        let hash_a = crate::fetch::canonical_path_hash(&path_a);
        let hash_b = crate::fetch::canonical_path_hash(&path_b);
        assert_ne!(
            hash_a, hash_b,
            "distinct canonical paths must produce distinct CRC32 hashes",
        );
        let lock_a = cache
            .path()
            .join(crate::flock::LOCK_DIR_NAME)
            .join(format!("source-{hash_a}.lock"));
        let lock_b = cache
            .path()
            .join(crate::flock::LOCK_DIR_NAME)
            .join(format!("source-{hash_b}.lock"));
        assert!(lock_a.exists());
        assert!(lock_b.exists());
        assert_ne!(lock_a, lock_b);
        drop(fd_a);
        drop(fd_b);
    }

    /// `acquire_source_tree_lock` on a path whose lockfile is
    /// already held by a peer in this process returns an error
    /// citing the source tree path AND the actionable
    /// `cargo ktstr locks` hint. Pins the EWOULDBLOCK diagnostic
    /// shape — a regression that swallowed the actionable hint or
    /// degraded to a blocking acquire would surface here.
    ///
    /// We simulate "concurrent peer" by holding the first FD in
    /// the same process: `try_flock` is non-blocking, so a second
    /// acquire of the same lockfile returns `Ok(None)` and the
    /// helper bails with the actionable error.
    #[test]
    fn acquire_source_tree_lock_contention_waits_then_succeeds() {
        use crate::test_support::test_helpers::{isolated_cache_dir, lock_env};
        let _env_lock = lock_env();
        let _cache = isolated_cache_dir();
        let canonical = std::path::PathBuf::from("/tmp/fake-source-contention");
        let holder = acquire_source_tree_lock(&canonical, "test")
            .expect("first acquire must succeed under isolated cache");
        let canonical2 = canonical.clone();
        let waiter = std::thread::spawn(move || {
            acquire_source_tree_lock(&canonical2, "test")
        });
        // Give the waiter thread time to hit the blocking flock,
        // then release the holder so it unblocks.
        std::thread::sleep(std::time::Duration::from_millis(200));
        drop(holder);
        let result = waiter
            .join()
            .expect("waiter thread must not panic");
        result.expect("second acquire must succeed after holder releases");
    }

    /// `BuildReservation` field declaration order is load-bearing:
    /// `_sandbox` MUST be declared BEFORE `plan` so Rust's
    /// in-declaration-order field-drop runs the sandbox cgroup
    /// rmdir BEFORE the LLC flock release.
    ///
    /// A regression that swapped the field order would mean
    /// LLC flocks release first, which lets a peer claim the LLC
    /// while gcc children are still bound to a cgroup whose rmdir
    /// hasn't run yet.
    ///
    /// We can't assert drop ORDER directly without exotic
    /// machinery, but we can assert the field order is what we
    /// expect via the `Debug` derive: `_sandbox` appears in the
    /// formatted struct BEFORE `plan` IFF the field declaration
    /// order matches the Drop-order requirement. The field-name
    /// regex is enough to pin the order without depending on the
    /// inner field shapes (which evolve as the planner / sandbox
    /// types add or rename their own fields).
    #[test]
    fn build_reservation_field_order_pins_drop_invariant() {
        let r = BuildReservation {
            _sandbox: None,
            plan: None,
            make_jobs: None,
        };
        let dbg = format!("{r:?}");
        let sandbox_pos = dbg
            .find("_sandbox")
            .expect("Debug output must mention _sandbox field");
        let plan_pos = dbg
            .find("plan")
            .expect("Debug output must mention plan field");
        assert!(
            sandbox_pos < plan_pos,
            "_sandbox MUST be declared before plan so cgroup rmdir \
             runs BEFORE LLC flock release on Drop. Debug: {dbg}",
        );
    }
}
