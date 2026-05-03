//! `cargo ktstr stats` subcommand dispatch helpers.
//!
//! Houses the dispatcher [`run_stats`] for the read-only stats
//! surface (list, list-metrics, list-values, show-host,
//! explain-sidecar, compare), the `--{a,b}-{project,kernel}-commit`
//! revspec resolver [`resolve_commit_specs`] (HEAD~N / branch /
//! tag / `A..B` ranges → 7-char short hashes the sidecar pool
//! stores), and the [`BuildCompareFilters`] sugar resolver that
//! folds shared `--X` and per-side `--{a,b}-X` filter flags into
//! the two `RowFilter` instances `cli::compare_partitions`
//! consumes.

use crate::cli::StatsCommand;
use ktstr::cli;

pub(crate) fn run_stats(command: &Option<StatsCommand>) -> Result<(), String> {
    match command {
        None => {
            if let Some(output) = cli::print_stats_report() {
                print!("{output}");
            }
            Ok(())
        }
        Some(StatsCommand::List) => cli::list_runs().map_err(|e| format!("{e:#}")),
        Some(StatsCommand::ListMetrics { json }) => match cli::list_metrics(*json) {
            Ok(s) => {
                print!("{s}");
                Ok(())
            }
            Err(e) => Err(format!("{e:#}")),
        },
        Some(StatsCommand::ListValues { json, dir }) => {
            match cli::list_values(*json, dir.as_deref()) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::ShowHost { run, dir }) => {
            match cli::show_run_host(run, dir.as_deref()) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::ExplainSidecar { run, dir, json }) => {
            match cli::explain_sidecar(run, dir.as_deref(), *json) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::Compare {
            filter,
            threshold,
            policy,
            dir,
            kernel,
            project_commit,
            kernel_commit,
            run_source,
            scheduler,
            topology,
            work_type,
            flags,
            a_kernel,
            a_project_commit,
            a_kernel_commit,
            a_run_source,
            a_scheduler,
            a_topology,
            a_work_type,
            a_flags,
            b_kernel,
            b_project_commit,
            b_kernel_commit,
            b_run_source,
            b_scheduler,
            b_topology,
            b_work_type,
            b_flags,
            no_average,
        }) => {
            // Resolve `--threshold N` / `--policy PATH` / neither
            // into a single `ComparisonPolicy`. Clap's
            // `conflicts_with` guarantees at most one of
            // (threshold, policy) is set, so the three branches
            // are exhaustive on user-visible input.
            let resolved_policy = match (threshold, policy.as_ref()) {
                (Some(t), None) => {
                    let p = ktstr::cli::ComparisonPolicy::uniform(*t);
                    // `uniform` is infallible, but the user-supplied
                    // percent still needs a sign check. `validate`
                    // rejects negatives before they reach
                    // `compare_rows`' dual-gate math.
                    p.validate().map_err(|e| format!("{e:#}"))?;
                    p
                }
                (None, Some(path)) => {
                    ktstr::cli::ComparisonPolicy::load_json(path).map_err(|e| format!("{e:#}"))?
                }
                (None, None) => ktstr::cli::ComparisonPolicy::default(),
                (Some(_), Some(_)) => {
                    // Defence-in-depth: clap's `conflicts_with` is
                    // load-bearing here, but a regression that
                    // dropped either attribute would silently pick
                    // one path and ignore the other. Panic loudly.
                    unreachable!(
                        "clap `conflicts_with` on --threshold / --policy \
                         must enforce mutual exclusion at parse time",
                    );
                }
            };
            // Resolve git revspecs in `--project-commit` /
            // `--kernel-commit` flags (HEAD~1, tags, branch names,
            // `A..B` ranges) into 7-char short hashes BEFORE
            // constructing BuildCompareFilters. The shared and
            // per-side vecs each go through `resolve_commit_specs`
            // independently so a per-side override that uses a
            // revspec ("--a-project-commit HEAD~1") expands the
            // same way a shared one would.
            //
            // Project-side repo: `gix::discover` from cwd, mirroring
            // `detect_project_commit`'s open mode at sidecar-write
            // time. `gix::discover` walks parents until it finds a
            // `.git` marker so the operator can run `cargo ktstr
            // stats compare` from anywhere inside the project tree;
            // failure (cwd outside any repo) collapses to `None`
            // and every input passes through as a literal — matching
            // the documented "literal fallback" contract.
            //
            // Kernel-side repo: `gix::open` against the path in
            // `KTSTR_KERNEL`, mirroring `detect_kernel_commit`'s
            // open mode. `KTSTR_KERNEL` unset / empty / non-git
            // collapses to `None` — every input passes through as
            // a literal. `KTSTR_KERNEL` may carry a Version /
            // CacheKey (not a path) on the test/coverage paths;
            // here on the stats path the caller is doing post-hoc
            // analysis, so a path is the relevant interpretation.
            // A non-path value falls through to literal too.
            let project_repo = std::env::current_dir()
                .ok()
                .and_then(|cwd| gix::discover(cwd).ok());
            // Probe `KTSTR_KERNEL` for the kernel-side git repo.
            // Path-spec resolution in cargo-ktstr now exports the
            // CACHE ENTRY directory for clean source trees (see
            // [`ktstr::cli::resolve_kernel_dir_to_entry`]); a
            // direct `gix::open` against that dir would fail
            // because the cache entry is not a git repo. The
            // shared `recover_local_source_tree` helper reads
            // `metadata.json` and returns the recorded
            // `source_tree_path` when present; when absent (the
            // env value is itself a source tree — the dirty
            // build-every-time path), fall back to opening the
            // env value verbatim.
            let kernel_repo = ktstr::ktstr_kernel_env()
                .map(std::path::PathBuf::from)
                .and_then(|p| {
                    let target = ktstr::cache::recover_local_source_tree(&p).unwrap_or(p);
                    gix::open(target).ok()
                });
            let project_commit =
                resolve_commit_specs(project_repo.as_ref(), project_commit, "project-commit");
            let kernel_commit =
                resolve_commit_specs(kernel_repo.as_ref(), kernel_commit, "kernel-commit");
            let a_project_commit =
                resolve_commit_specs(project_repo.as_ref(), a_project_commit, "a-project-commit");
            let a_kernel_commit =
                resolve_commit_specs(kernel_repo.as_ref(), a_kernel_commit, "a-kernel-commit");
            let b_project_commit =
                resolve_commit_specs(project_repo.as_ref(), b_project_commit, "b-project-commit");
            let b_kernel_commit =
                resolve_commit_specs(kernel_repo.as_ref(), b_kernel_commit, "b-kernel-commit");
            // Construct the BuildCompareFilters from the raw CLI
            // inputs. Sugar logic (shared `--X` pins both sides;
            // per-side `--a-X` / `--b-X` REPLACES the shared value
            // for that side) lives inside `build()` so it's
            // unit-testable in isolation. The dispatch site stays
            // a dumb data carrier.
            let build = BuildCompareFilters {
                shared_kernel: kernel.clone(),
                shared_project_commit: project_commit,
                shared_kernel_commit: kernel_commit,
                shared_run_source: run_source.clone(),
                shared_scheduler: scheduler.clone(),
                shared_topology: topology.clone(),
                shared_work_type: work_type.clone(),
                shared_flags: flags.clone(),
                a_kernel: a_kernel.clone(),
                a_project_commit,
                a_kernel_commit,
                a_run_source: a_run_source.clone(),
                a_scheduler: a_scheduler.clone(),
                a_topology: a_topology.clone(),
                a_work_type: a_work_type.clone(),
                a_flags: a_flags.clone(),
                b_kernel: b_kernel.clone(),
                b_project_commit,
                b_kernel_commit,
                b_run_source: b_run_source.clone(),
                b_scheduler: b_scheduler.clone(),
                b_topology: b_topology.clone(),
                b_work_type: b_work_type.clone(),
                b_flags: b_flags.clone(),
            };
            let (filter_a, filter_b) = build.build();
            let exit = cli::compare_partitions(
                &filter_a,
                &filter_b,
                filter.as_deref(),
                &resolved_policy,
                dir.as_deref(),
                *no_average,
            )
            .map_err(|e| format!("{e:#}"))?;
            if exit != 0 {
                std::process::exit(exit);
            }
            Ok(())
        }
    }
}

/// Match the on-disk `project_commit` / `kernel_commit` shape:
/// `^[0-9a-f]{7,40}(-dirty)?$`. Used to gate the rev_parse-Err
/// warning in [`resolve_commit_specs`] so legitimate literal
/// hashes (the common case for `--project-commit abc1234`) do
/// not produce noisy "did not resolve as a revspec" lines.
///
/// SHAPE rationale: `detect_project_commit` /
/// `detect_kernel_commit` write `to_hex_with_len(7)` (7-char
/// short hash) optionally followed by `-dirty`. The lower bound
/// 7 matches that emitter exactly — anything shorter is an
/// abbreviated revspec (`HE`, `v6`) or a typo, not a stored
/// commit identifier. The upper bound 40 matches the SHA-1
/// hex width gix's repo objects use today.
///
/// SHA-256 gap: when gix migrates a repository to SHA-256 object
/// IDs (gix's experimental `sha256` feature, not yet enabled in
/// the workspace), full hashes become 64 hex chars and 41–64
/// hashes will be misclassified by this predicate as revspecs.
/// At that point the upper bound must lift to 64 and a sibling
/// test must pin both the SHA-1 and SHA-256 widths. Until then
/// the 40-char ceiling is correct: every commit hash we read
/// from a sidecar fits in 40 chars.
///
/// Anything outside this shape — `HEAD`, `HEAD~1`, branch/tag
/// names, `..`-ranges, mixed-case, longer than 40 chars — is
/// considered an attempted revspec and warrants the warning.
///
/// Case-insensitive hex: `detect_*_commit`'s `to_hex_with_len`
/// produces lowercase and the sidecar pool stores lowercase, so
/// an uppercase input like `ABC1234` will not match a stored row
/// — but it is still recognizably a hash, not a revspec, and
/// silently emitting "did not resolve as a revspec" on it is
/// noise. Pasted-from-elsewhere uppercase / mixed-case hex
/// strings pass the predicate; the suppressed warning means the
/// downstream filter sees the literal verbatim and the
/// (lowercase) pool simply produces an empty match — the same
/// outcome as a legitimate-but-unknown short hash.
pub(crate) fn looks_like_literal_hash(input: &str) -> bool {
    let core = input.strip_suffix("-dirty").unwrap_or(input);
    let len = core.len();
    if !(7..=40).contains(&len) {
        return false;
    }
    core.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Resolve git revspecs (HEAD~1, tags, branch names, ranges
/// `A..B`) in `--project-commit` / `--kernel-commit` filter
/// inputs into the 7-char short hashes the sidecar's
/// `project_commit` / `kernel_commit` fields are written with.
///
/// Each entry in `raw` is treated independently. `repo` is the
/// repository against which to resolve (cwd-discovered for
/// project commits, opened from `KTSTR_KERNEL` for kernel
/// commits); when `None` (no repo available — cwd outside any
/// git tree, or `KTSTR_KERNEL` unset) every input passes
/// through verbatim as a literal. `flag_name` is the
/// user-visible CLI flag (e.g. `"project-commit"`,
/// `"a-kernel-commit"`) included in stderr warnings for
/// fall-through arms so the operator can identify which input
/// did not resolve as expected.
///
/// Per-input resolution:
/// - `repo.rev_parse(input)` succeeds with `Spec::Include(id)`
///   or `Spec::ExcludeParents(id)` (single-commit revspec):
///   push the commit's 7-char short hex. When the resolved OID
///   equals current HEAD AND the worktree is dirty, append
///   `-dirty` to the short hex so the filter matches the
///   suffixed entries `detect_project_commit` /
///   `detect_kernel_commit` write at sidecar-write time.
///   Historical commits (Include / ExcludeParents resolving to
///   non-HEAD OIDs) never get the suffix — the operator named
///   that specific commit and current worktree state has no
///   bearing on its identity in the recorded pool.
/// - `repo.rev_parse(input)` succeeds with `Spec::Range { from,
///   to }` (an `A..B` revspec): walk via `repo.rev_walk([to])
///   .with_hidden([from]).all()` and push each yielded
///   commit's 7-char short hex. Range commits never get the
///   `-dirty` suffix even if the walk includes HEAD — `A..B`
///   is a content-set query, not a current-state query. A
///   walk-init failure prints a stderr warning naming the
///   input and the gix error, then falls through to literal.
/// - `repo.rev_parse(input)` succeeds with another spec kind
///   (`Exclude`, `Merge`, `IncludeOnlyParents`): push the input
///   unchanged. Those forms have no useful single-side
///   expansion for an OR-combined exact-match filter, and a
///   stderr warning is emitted so the operator notices that the
///   input was not expanded.
/// - `repo.rev_parse(input)` returns `Err`: push the input
///   unchanged (literal fallback). This preserves the existing
///   exact-match contract for hand-typed short hashes — a
///   `--project-commit abc1234-dirty` argument never resolves
///   as a revspec (revspec parsing rejects the `-dirty`
///   suffix) and lands as a literal, matching the `-dirty`-
///   suffixed entries in the sidecar pool. The warning fires
///   only when [`looks_like_literal_hash`] returns false —
///   legitimate `<hex>` and `<hex>-dirty` shapes pass through
///   silently, while revspec-shaped inputs that failed
///   resolution surface a diagnostic.
///
/// Resolution happens BEFORE [`BuildCompareFilters`]
/// construction so the sugar logic in `build()` operates on
/// already-expanded vecs and stays free of repo I/O. The
/// expansion is order-preserving on the input, with each
/// range entry yielding its commits in walk order; downstream
/// `RowFilter::matches` is OR-combined so order does not affect
/// the matched set.
pub(crate) fn resolve_commit_specs(
    repo: Option<&gix::Repository>,
    raw: &[String],
    flag_name: &str,
) -> Vec<String> {
    let Some(repo) = repo else {
        return raw.to_vec();
    };
    // Resolve once: the `-dirty` suffix only applies when the
    // resolved OID equals current HEAD AND the worktree is
    // dirty. Both are repo-global properties, so compute them
    // upfront and reuse across every input. Either probe
    // failing (no HEAD, no readable tree) collapses
    // `head_dirty` to false — the documented "treat as clean
    // on probe failure" policy that the canonical
    // [`ktstr::test_support::repo_is_dirty`] already encodes,
    // matching what the sidecar writer applies at sidecar-write
    // time so the filter shape lines up with pool entries.
    let head_oid: Option<gix::ObjectId> = repo.head_id().ok().map(|id| id.detach());
    let head_dirty: bool = head_oid
        .as_ref()
        .and_then(|_| ktstr::test_support::repo_is_dirty(repo))
        .unwrap_or(false);
    let format_short = |id: gix::ObjectId| -> String {
        let short = id.to_hex_with_len(7).to_string();
        // Dirty-suffix only when (a) the worktree is dirty and
        // (b) the resolved OID is current HEAD. A historical
        // commit that happens to be referenced via Include /
        // ExcludeParents (e.g. an explicit short-hash that
        // resolves to a non-HEAD object) does NOT get -dirty,
        // because the operator named that specific commit and
        // the current worktree state has no bearing on its
        // identity in the recorded sidecar pool.
        if head_dirty && head_oid == Some(id) {
            format!("{short}-dirty")
        } else {
            short
        }
    };
    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    for input in raw {
        match repo.rev_parse(input.as_str()) {
            Ok(spec) => match spec.detach() {
                gix::revision::plumbing::Spec::Include(id)
                | gix::revision::plumbing::Spec::ExcludeParents(id) => {
                    out.push(format_short(id));
                }
                gix::revision::plumbing::Spec::Range { from, to } => {
                    match repo.rev_walk([to]).with_hidden([from]).all() {
                        Ok(walk) => {
                            // `walk.flatten()` silently drops any
                            // per-element traversal `Err` (a single
                            // unreadable commit object inside the
                            // range walk). The cost is one filter
                            // entry per failure; the alternative —
                            // bailing on the whole range — would
                            // discard every successfully-yielded
                            // sibling commit. Since the filter is
                            // OR-combined, a missing entry only
                            // narrows the match set rather than
                            // producing wrong rows, so silent drop
                            // is the safe degradation.
                            //
                            // Range commits are historical — they
                            // never get the -dirty suffix even when
                            // the walk happens to include HEAD,
                            // because the operator's intent in
                            // `A..B` is "match every commit in this
                            // range" which is a content-set, not a
                            // current-state query. Use raw
                            // `to_hex_with_len` here (bypass the
                            // `format_short` closure) to skip the
                            // dirty-suffix decision.
                            for info in walk.flatten() {
                                out.push(info.id.to_hex_with_len(7).to_string());
                            }
                        }
                        Err(err) => {
                            eprintln!(
                                "cargo ktstr: --{flag_name} range '{input}' could \
                                 not be expanded: {err}; using as literal filter",
                            );
                            out.push(input.clone());
                        }
                    }
                }
                gix::revision::plumbing::Spec::Exclude(_)
                | gix::revision::plumbing::Spec::Merge { .. }
                | gix::revision::plumbing::Spec::IncludeOnlyParents(_) => {
                    eprintln!(
                        "cargo ktstr: --{flag_name} '{input}' uses an unsupported \
                         revspec form (Exclude/Merge/IncludeOnlyParents); using \
                         as literal filter",
                    );
                    out.push(input.clone());
                }
            },
            Err(_) => {
                // Suppress the warning when `input` already looks
                // like a literal hash (`^[0-9a-fA-F]{7,40}(-dirty)?$`).
                // Lowercase is the SHAPE the sidecar writer produces;
                // uppercase / mixed-case is what an operator pastes
                // from `git log` output or another tool's UI. Both
                // are recognizably hashes and the operator typing
                // them at the CLI is the common case — emitting
                // "did not resolve as a revspec" on every legitimate
                // `--project-commit abc1234` invocation is pure
                // noise. Only warn for inputs that look like an
                // attempted revspec (alpha beyond hex, ~, .., ^,
                // longer than 40 chars, or other non-hash shapes)
                // so a typo'd revspec ("HEAD~XYZ", "main") still
                // surfaces a diagnostic.
                if !looks_like_literal_hash(input) {
                    eprintln!(
                        "cargo ktstr: --{flag_name} '{input}' did not resolve as \
                         a revspec; using as literal filter",
                    );
                }
                out.push(input.clone());
            }
        }
    }
    out
}

/// Symmetric-sugar resolver for `cargo ktstr stats compare`'s
/// shared `--X` and per-side `--a-X` / `--b-X` filter flags.
///
/// CLI flag semantics:
/// - Shared `--X` pins BOTH sides to the same value(s). E.g.
///   `--kernel 6.14` is equivalent to
///   `--a-kernel 6.14 --b-kernel 6.14`.
/// - Per-side `--a-X` REPLACES the shared `--X` value for the A
///   side only (and `--b-X` replaces for B only). "More-specific
///   replaces" — the per-side flag takes precedence over the
///   shared default for that side, but does not affect the
///   other side.
///
/// Constructed from the raw clap-parsed values; `build()` does
/// the sugar resolution and returns `(filter_a, filter_b)`. The
/// struct is unit-testable in isolation so the sugar logic does
/// not require booting a real comparison.
///
/// Project- and kernel-commit fields hold ALREADY-EXPANDED
/// commit lists: any git revspecs (HEAD~1, tags, branch names,
/// `A..B` ranges) the user typed are resolved to 7-char short
/// hashes by [`resolve_commit_specs`] BEFORE construction so
/// `build()` stays a pure data-shuffler with no repo I/O.
#[derive(Debug, Clone, Default)]
pub(crate) struct BuildCompareFilters {
    pub(crate) shared_kernel: Vec<String>,
    pub(crate) shared_project_commit: Vec<String>,
    pub(crate) shared_kernel_commit: Vec<String>,
    pub(crate) shared_run_source: Vec<String>,
    pub(crate) shared_scheduler: Vec<String>,
    pub(crate) shared_topology: Vec<String>,
    pub(crate) shared_work_type: Vec<String>,
    pub(crate) shared_flags: Vec<String>,
    pub(crate) a_kernel: Vec<String>,
    pub(crate) a_project_commit: Vec<String>,
    pub(crate) a_kernel_commit: Vec<String>,
    pub(crate) a_run_source: Vec<String>,
    pub(crate) a_scheduler: Vec<String>,
    pub(crate) a_topology: Vec<String>,
    pub(crate) a_work_type: Vec<String>,
    pub(crate) a_flags: Vec<String>,
    pub(crate) b_kernel: Vec<String>,
    pub(crate) b_project_commit: Vec<String>,
    pub(crate) b_kernel_commit: Vec<String>,
    pub(crate) b_run_source: Vec<String>,
    pub(crate) b_scheduler: Vec<String>,
    pub(crate) b_topology: Vec<String>,
    pub(crate) b_work_type: Vec<String>,
    pub(crate) b_flags: Vec<String>,
}

impl BuildCompareFilters {
    /// Resolve sugar into per-side `RowFilter` instances.
    /// "More-specific replaces": a per-side Vec is applied
    /// verbatim when non-empty, otherwise the shared Vec is
    /// used. Every dimension on `RowFilter` is now a `Vec<String>`
    /// (after the conversion from `Option<String>` to repeatable
    /// Vec for scheduler/topology/work_type), so a single `pick_vec`
    /// helper handles every dim uniformly — the prior `pick_opt`
    /// branch is no longer reachable.
    pub(crate) fn build(&self) -> (ktstr::cli::RowFilter, ktstr::cli::RowFilter) {
        let pick_vec = |a: &[String], shared: &[String]| -> Vec<String> {
            if a.is_empty() {
                shared.to_vec()
            } else {
                a.to_vec()
            }
        };
        let filter_a = ktstr::cli::RowFilter {
            kernels: pick_vec(&self.a_kernel, &self.shared_kernel),
            project_commits: pick_vec(&self.a_project_commit, &self.shared_project_commit),
            kernel_commits: pick_vec(&self.a_kernel_commit, &self.shared_kernel_commit),
            run_sources: pick_vec(&self.a_run_source, &self.shared_run_source),
            schedulers: pick_vec(&self.a_scheduler, &self.shared_scheduler),
            topologies: pick_vec(&self.a_topology, &self.shared_topology),
            work_types: pick_vec(&self.a_work_type, &self.shared_work_type),
            flags: pick_vec(&self.a_flags, &self.shared_flags),
        };
        let filter_b = ktstr::cli::RowFilter {
            kernels: pick_vec(&self.b_kernel, &self.shared_kernel),
            project_commits: pick_vec(&self.b_project_commit, &self.shared_project_commit),
            kernel_commits: pick_vec(&self.b_kernel_commit, &self.shared_kernel_commit),
            run_sources: pick_vec(&self.b_run_source, &self.shared_run_source),
            schedulers: pick_vec(&self.b_scheduler, &self.shared_scheduler),
            topologies: pick_vec(&self.b_topology, &self.shared_topology),
            work_types: pick_vec(&self.b_work_type, &self.shared_work_type),
            flags: pick_vec(&self.b_flags, &self.shared_flags),
        };
        (filter_a, filter_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- BuildCompareFilters: symmetric sugar resolution --

    /// Empty input → both sides default. No filters populated
    /// anywhere; the dispatch site rejects this with the
    /// "specify at least one --a-X" error, but the builder
    /// itself just returns two empty filters.
    #[test]
    fn build_compare_filters_empty_yields_default_default() {
        let b = BuildCompareFilters::default();
        let (fa, fb) = b.build();
        assert!(fa.kernels.is_empty());
        assert!(fa.project_commits.is_empty());
        assert!(fa.kernel_commits.is_empty());
        assert!(fa.run_sources.is_empty());
        assert!(fa.schedulers.is_empty());
        assert!(fa.topologies.is_empty());
        assert!(fa.work_types.is_empty());
        assert!(fa.flags.is_empty());
        assert_eq!(fa.kernels, fb.kernels);
        assert_eq!(fa.project_commits, fb.project_commits);
        assert_eq!(fa.kernel_commits, fb.kernel_commits);
        assert_eq!(fa.run_sources, fb.run_sources);
        assert_eq!(fa.schedulers, fb.schedulers);
        assert_eq!(fa.topologies, fb.topologies);
        assert_eq!(fa.work_types, fb.work_types);
    }

    /// Per-side `--a-kernel-commit` overrides shared
    /// `--kernel-commit` for A only; B retains the shared value.
    /// Same "more-specific replaces" semantics as `--a-kernel`.
    /// The per-side override path is what populates the slicing
    /// dim on `KernelCommit` — without it, two sides with
    /// different live kernel HEADs cannot be contrasted in one
    /// `compare` invocation.
    #[test]
    fn build_compare_filters_per_side_kernel_commit_overrides_shared() {
        let b = BuildCompareFilters {
            shared_kernel_commit: vec!["abcdef1".to_string(), "fedcba2".to_string()],
            a_kernel_commit: vec!["111aaaa".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(
            fa.kernel_commits,
            vec!["111aaaa"],
            "A overrides shared kernel-commit",
        );
        assert_eq!(
            fb.kernel_commits,
            vec!["abcdef1", "fedcba2"],
            "B retains shared kernel-commit default",
        );
    }

    /// `--a-kernel-commit X --b-kernel-commit Y` slices on the
    /// `KernelCommit` dimension. Pins the slicing-dim derivation
    /// for the kernel-commit axis so a regression that dropped
    /// the dim from `derive_slicing_dims` lands here.
    #[test]
    fn build_compare_filters_disjoint_per_side_kernel_commit_slices() {
        let b = BuildCompareFilters {
            a_kernel_commit: vec!["abcdef1".to_string()],
            b_kernel_commit: vec!["fedcba2".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernel_commits, vec!["abcdef1"]);
        assert_eq!(fb.kernel_commits, vec!["fedcba2"]);
        let slicing = ktstr::cli::derive_slicing_dims(&fa, &fb);
        assert_eq!(
            slicing,
            vec![ktstr::cli::Dimension::KernelCommit],
            "differing per-side kernel-commit must derive as a single \
             KernelCommit slicing dim",
        );
    }

    /// Shared `--kernel V` pins BOTH sides to the same vec.
    /// Sugar for `--a-kernel V --b-kernel V`.
    #[test]
    fn build_compare_filters_shared_kernel_pins_both_sides() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.14"]);
    }

    /// Per-side `--a-kernel` overrides shared `--kernel` for A
    /// only; B retains the shared value. "More-specific
    /// replaces" semantics.
    #[test]
    fn build_compare_filters_per_side_overrides_shared_for_that_side_only() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string(), "6.15".to_string()],
            a_kernel: vec!["6.13".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.13"], "A overrides shared");
        assert_eq!(fb.kernels, vec!["6.14", "6.15"], "B retains shared default",);
    }

    /// Per-side overrides on the SAME dimension on BOTH sides
    /// produce the disjoint per-side filters the dispatch
    /// expects. This is the typical "slice on kernel" call shape:
    /// `--a-kernel A --b-kernel B`.
    #[test]
    fn build_compare_filters_disjoint_per_side_kernel_yields_two_filters() {
        let b = BuildCompareFilters {
            a_kernel: vec!["6.14".to_string()],
            b_kernel: vec!["6.15".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.15"]);
    }

    /// Per-side `--a-scheduler` overrides shared `--scheduler` for
    /// A only. Sibling test for the scheduler dimension after the
    /// conversion from `Option<String>` to repeatable
    /// `Vec<String>` — the override semantics now mirror every
    /// other Vec dim ("non-empty per-side replaces shared
    /// verbatim"), so this test pins the same shape every other
    /// override-test pins for kernel / commit / source / etc.
    #[test]
    fn build_compare_filters_per_side_scheduler_overrides_shared() {
        let b = BuildCompareFilters {
            shared_scheduler: vec!["scx_default".to_string()],
            a_scheduler: vec!["scx_alpha".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(
            fa.schedulers,
            vec!["scx_alpha".to_string()],
            "A overrides shared scheduler",
        );
        assert_eq!(
            fb.schedulers,
            vec!["scx_default".to_string()],
            "B retains shared scheduler when only --a-scheduler overrides",
        );
    }

    /// Multi-dim sugar: shared `--kernel` pins both sides AND
    /// per-side `--a-scheduler` / `--b-scheduler` slice on
    /// scheduler. The resulting filters share kernel but slice
    /// on scheduler — exactly what the
    /// "narrow scope, slice on one axis" workflow needs.
    #[test]
    fn build_compare_filters_shared_pin_plus_per_side_slice() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string()],
            a_scheduler: vec!["scx_alpha".to_string()],
            b_scheduler: vec!["scx_beta".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.14"]);
        assert_eq!(fa.schedulers, vec!["scx_alpha".to_string()]);
        assert_eq!(fb.schedulers, vec!["scx_beta".to_string()]);
        // The slicing-dim derivation for these two filters
        // returns just [Scheduler] — kernel pins both sides
        // so the comparison joins on kernel and contrasts on
        // scheduler.
        let slicing = ktstr::cli::derive_slicing_dims(&fa, &fb);
        assert_eq!(slicing, vec![ktstr::cli::Dimension::Scheduler]);
    }

    /// `--a-flag` / `--b-flag` (AND-combined Vec) compose the
    /// same way as `--a-kernel` / `--b-kernel` (OR-combined
    /// Vec) — per-side empty defers to shared, per-side non-
    /// empty replaces. Pin the shape for the AND-combined dim
    /// to ensure no accidental special-case for OR-vs-AND.
    #[test]
    fn build_compare_filters_per_side_flag_overrides_shared() {
        let b = BuildCompareFilters {
            shared_flags: vec!["llc".to_string()],
            a_flags: vec!["steal".to_string(), "borrow".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.flags, vec!["steal", "borrow"]);
        assert_eq!(fb.flags, vec!["llc"]);
    }

    /// Sibling of `build_compare_filters_empty_yields_default_default`
    /// for the `run_sources` field. The existing empty-default test
    /// asserts on `run_sources` already — this companion adds the
    /// cross-side equality check that ensures
    /// `fa.run_sources == fb.run_sources` under the empty default,
    /// matching the same pattern other dimensions have. A regression
    /// that diverged the per-side `run_sources` defaults (e.g. by
    /// forgetting to thread `shared_run_source` into BOTH
    /// constructors in `BuildCompareFilters::build`) would surface
    /// here.
    #[test]
    fn build_compare_filters_empty_run_sources_field_equal_on_both_sides() {
        let b = BuildCompareFilters::default();
        let (fa, fb) = b.build();
        assert!(
            fa.run_sources.is_empty(),
            "empty BuildCompareFilters must produce A-side filter with empty run_sources",
        );
        assert!(
            fb.run_sources.is_empty(),
            "empty BuildCompareFilters must produce B-side filter with empty run_sources",
        );
        assert_eq!(
            fa.run_sources, fb.run_sources,
            "both sides must agree on empty run_sources",
        );
    }

    /// Per-side `--a-run-source` / `--b-run-source` produce
    /// disjoint per-side filters with the shared `run_sources`
    /// left empty. Mirrors
    /// `build_compare_filters_disjoint_per_side_kernel_yields_two_filters`
    /// for the run-source dimension. Pins the wiring of the
    /// `run_sources` field through `build()` so a regression that
    /// dropped it from the per-side branch — silently leaving
    /// `fa.run_sources` / `fb.run_sources` empty under per-side
    /// input — surfaces here.
    #[test]
    fn build_compare_filters_disjoint_per_side_source_yields_two_filters() {
        let b = BuildCompareFilters {
            a_run_source: vec!["ci".to_string()],
            b_run_source: vec!["local".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.run_sources, vec!["ci".to_string()]);
        assert_eq!(fb.run_sources, vec!["local".to_string()]);
    }

    /// Shared `--run-source` pins BOTH sides to the same vec.
    /// Sugar for `--a-run-source V --b-run-source V`. Mirrors
    /// `build_compare_filters_shared_kernel_pins_both_sides` for
    /// the run-source dimension.
    #[test]
    fn build_compare_filters_shared_source_pins_both_sides() {
        let b = BuildCompareFilters {
            shared_run_source: vec!["ci".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.run_sources, vec!["ci".to_string()]);
        assert_eq!(fb.run_sources, vec!["ci".to_string()]);
    }

    /// Per-side `--a-run-source` overrides shared `--run-source`
    /// for A only; B retains the shared value. "More-specific
    /// replaces" semantics — same shape as the existing
    /// `per_side_overrides_shared_for_that_side_only` for kernels.
    /// Pins the override resolution path for the run-source
    /// dimension.
    #[test]
    fn build_compare_filters_per_side_source_overrides_shared_for_that_side_only() {
        let b = BuildCompareFilters {
            shared_run_source: vec!["local".to_string(), "archive".to_string()],
            a_run_source: vec!["ci".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.run_sources, vec!["ci".to_string()], "A overrides shared");
        assert_eq!(
            fb.run_sources,
            vec!["local".to_string(), "archive".to_string()],
            "B retains shared default",
        );
    }

    // -- resolve_commit_specs: revspec resolution --

    /// Build a chain of `n` commits in `dir` and return their
    /// `ObjectId`s in order (oldest first). Each commit gets a
    /// fresh tree containing one blob whose contents differ per
    /// commit so the trees never collide. Mirrors the structure
    /// of `init_clean_repo_with_file` in `test_support::sidecar`'s
    /// test mod but extends it to a multi-commit chain.
    ///
    /// `gix::Repository::commit` requires both an author and a
    /// committer signature. `committer_or_set_generic_fallback`
    /// only writes the committer fallback; without `user.name`/
    /// `user.email` in the runner's git config, the author cascade
    /// (author -> user) yields `None` and `commit` bails with
    /// `AuthorMissing`. CI runners that do not pre-seed `user.name`
    /// hit this. Plant `gitoxide.author.nameFallback` /
    /// `emailFallback` directly so the author cascade has a value
    /// regardless of ambient git config — same shape gix uses for
    /// the committer fallback in `committer_or_set_generic_fallback`
    /// (see `gix::config::tree::gitoxide::Author` for the keys).
    fn init_repo_with_chain(dir: &std::path::Path, n: usize) -> Vec<gix::ObjectId> {
        let mut repo = gix::init(dir).expect("gix::init");
        let _ = repo
            .committer_or_set_generic_fallback()
            .expect("committer fallback");
        // Author fallback: mirror the committer-fallback pattern from
        // gix-0.81 `committer_or_set_generic_fallback` against the
        // Author keys.
        {
            use gix::config::tree::gitoxide;
            let mut cfg = gix::config::File::new(gix::config::file::Metadata::api());
            cfg.set_raw_value(&gitoxide::Author::NAME_FALLBACK, "ktstr-test")
                .expect("set author name fallback");
            cfg.set_raw_value(
                &gitoxide::Author::EMAIL_FALLBACK,
                "ktstr-test@example.invalid",
            )
            .expect("set author email fallback");
            let mut snap = repo.config_snapshot_mut();
            snap.append(cfg);
        }
        let mut chain: Vec<gix::ObjectId> = Vec::with_capacity(n);
        for i in 0..n {
            let blob_id: gix::ObjectId = repo
                .write_blob(format!("v{i}\n").as_bytes())
                .expect("write blob")
                .detach();
            let tree = gix::objs::Tree {
                entries: vec![gix::objs::tree::Entry {
                    mode: gix::objs::tree::EntryKind::Blob.into(),
                    filename: "file.txt".into(),
                    oid: blob_id,
                }],
            };
            let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
            let parents: Vec<gix::ObjectId> = chain.last().copied().into_iter().collect();
            let commit_id: gix::ObjectId = repo
                .commit("HEAD", format!("c{i}"), tree_id, parents)
                .expect("commit")
                .detach();
            chain.push(commit_id);
        }
        chain
    }

    /// `repo: None` (no repo available — cwd outside any git tree
    /// or `KTSTR_KERNEL` unset / non-git) is a documented contract:
    /// every input passes through verbatim as a literal. Pins that
    /// `resolve_commit_specs` does not require a repo to function —
    /// the bare-string fallback is the load-bearing default for
    /// stats analysis on a host that's not the original build
    /// machine.
    #[test]
    fn resolve_commit_specs_no_repo_passes_through_literal() {
        let raw = vec![
            "abc1234".to_string(),
            "main".to_string(),
            "HEAD".to_string(),
        ];
        let out = resolve_commit_specs(None, &raw, "test");
        assert_eq!(out, raw, "no repo → every input lands as-is");
    }

    /// `HEAD` resolves to the 7-char short hex of the tip commit.
    /// Pins the `Spec::Include(id)` arm against a real repo: a
    /// regression that returned the input string verbatim instead
    /// of the resolved hash would silently break revspec resolution
    /// even when the repo is available.
    #[test]
    fn resolve_commit_specs_head_resolves_to_short_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 3);
        let head = *chain.last().unwrap();
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec!["HEAD".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec![head.to_hex_with_len(7).to_string()],
            "HEAD must resolve to the tip commit's 7-char short hex",
        );
    }

    /// `HEAD~1` resolves to the parent of HEAD. Pins the navigate-
    /// up form which is the most common revspec users write when
    /// pointing at "the prior commit". Same `Spec::Include(id)`
    /// arm, just driven through a non-trivial revspec.
    #[test]
    fn resolve_commit_specs_head_tilde_resolves_to_parent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 3);
        // chain[0] is oldest; chain[2] is HEAD; chain[1] is HEAD~1.
        let head_tilde_1 = chain[1];
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec!["HEAD~1".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec![head_tilde_1.to_hex_with_len(7).to_string()],
            "HEAD~1 must resolve to the parent commit's 7-char short hex",
        );
    }

    /// `A..B` (range revspec) expands to every commit reachable
    /// from B but not from A. With three commits c0→c1→c2 and the
    /// range `c0..HEAD`, the result must include c1 and c2 (c0 is
    /// excluded; HEAD is included). Pins the `Spec::Range` arm and
    /// the `rev_walk([to]).with_hidden([from]).all()` walk.
    #[test]
    fn resolve_commit_specs_range_expands_inclusive_of_to() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 3);
        let c0 = chain[0];
        let c1 = chain[1];
        let c2 = chain[2];
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec![format!("{}..HEAD", c0.to_hex_with_len(40))];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        // The walk yields c2 (HEAD) and c1 in some order; c0 is
        // hidden. Verify membership (order is walk-implementation
        // dependent and not pinned).
        assert!(
            out.contains(&c1.to_hex_with_len(7).to_string()),
            "range result must include c1 (the parent of HEAD); got {out:?}",
        );
        assert!(
            out.contains(&c2.to_hex_with_len(7).to_string()),
            "range result must include c2 (HEAD); got {out:?}",
        );
        assert!(
            !out.contains(&c0.to_hex_with_len(7).to_string()),
            "range result must NOT include c0 (the hidden side); got {out:?}",
        );
        assert_eq!(
            out.len(),
            2,
            "range c0..HEAD over 3-commit chain must yield exactly 2 commits; got {out:?}",
        );
    }

    /// A non-revspec input round-trips unchanged. Pins the
    /// literal-fallback arm: the existing exact-match contract
    /// for hand-typed strings must keep working when the input
    /// can't be parsed as a revspec at all.
    /// `gix::Repository::rev_parse` returns an error for any
    /// input that is neither a known ref nor a valid hex hash
    /// prefix; that error falls through to `out.push(input
    /// .clone())`. `"zzzzzzz"` is non-hex (z is outside
    /// `[0-9a-fA-F]`), so revspec parsing definitionally rejects
    /// it without needing to consult the object database.
    #[test]
    fn resolve_commit_specs_unknown_hash_falls_through_to_literal() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_chain(tmp.path(), 1);
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec!["zzzzzzz".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec!["zzzzzzz".to_string()],
            "non-hex input must pass through as literal",
        );
    }

    /// A literal `<hash>-dirty` form falls through to the literal
    /// arm. `gix::Repository::rev_parse` rejects the `-dirty`
    /// suffix as an invalid revspec; the function lands the input
    /// unchanged so it matches the `-dirty`-suffixed entries the
    /// sidecar writer produces. Pins the design contract that
    /// hand-typed dirty filters keep working alongside revspec
    /// expansion.
    #[test]
    fn resolve_commit_specs_dirty_suffix_falls_through_to_literal() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_chain(tmp.path(), 1);
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec!["abc1234-dirty".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec!["abc1234-dirty".to_string()],
            "-dirty-suffixed input must pass through as literal",
        );
    }

    /// Empty input → empty output, no work done. Pins the no-op
    /// pass for the common case where a side has no commit filter
    /// — the function returns an empty Vec without exercising
    /// any rev_parse machinery.
    #[test]
    fn resolve_commit_specs_empty_input_yields_empty_output() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_chain(tmp.path(), 1);
        let repo = gix::open(tmp.path()).expect("gix::open");
        let out = resolve_commit_specs(Some(&repo), &[], "test");
        assert!(out.is_empty(), "empty input must yield empty output");
    }

    /// Mixed input: some revspecs resolve, some fall through to
    /// literals, all in the same call. Pins the per-input
    /// independence — one literal does not poison sibling
    /// resolutions, and one resolved revspec does not consume the
    /// literal fallback for its siblings. Mirrors the realistic
    /// case where a user types `--project-commit HEAD --project-commit
    /// abc1234-dirty` to combine "current HEAD" with a known dirty
    /// run from history.
    #[test]
    fn resolve_commit_specs_mixed_inputs_resolve_per_entry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 1);
        let head = chain[0];
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec!["HEAD".to_string(), "abc1234-dirty".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec![
                head.to_hex_with_len(7).to_string(),
                "abc1234-dirty".to_string(),
            ],
            "HEAD resolves; -dirty input lands literal; order preserved",
        );
    }

    /// `HEAD` against a dirty worktree resolves to
    /// `<short>-dirty`, mirroring what `detect_project_commit`
    /// writes at sidecar-write time. Pins the dirty-suffix
    /// behavior on the Include/ExcludeParents arm: a regression
    /// that emitted just `<short>` would silently miss every
    /// sidecar row written from this same dirty worktree, since
    /// `RowFilter::matches` is strict equality on the
    /// commit-string field.
    #[test]
    fn resolve_commit_specs_head_in_dirty_repo_appends_dirty_suffix() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 1);
        let head = chain[0];
        // Mutate the tracked file so index-vs-worktree diverges.
        // `init_repo_with_chain` writes the blob to ODB but does
        // not populate a worktree file or index — so create the
        // index from the tree, write the file, then mutate it.
        let repo = gix::open(tmp.path()).expect("gix::open");
        let head_tree = repo.head_tree().expect("head_tree").id;
        let mut idx = repo.index_from_tree(&head_tree).expect("index_from_tree");
        idx.write(gix::index::write::Options::default())
            .expect("write index");
        std::fs::write(tmp.path().join("file.txt"), b"original\n").unwrap();
        // Now mutate the tracked file to dirty the worktree.
        std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
        let raw = vec!["HEAD".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        let expected_short = head.to_hex_with_len(7).to_string();
        assert_eq!(
            out,
            vec![format!("{expected_short}-dirty")],
            "HEAD in a dirty repo must resolve to <short>-dirty",
        );
    }

    /// A historical commit (HEAD~1 in a 2-commit chain) does NOT
    /// get the `-dirty` suffix even when the worktree is dirty.
    /// Pins the design constraint: only the resolved OID equal to
    /// current HEAD reflects worktree state; named historical
    /// commits keep their plain identity. A regression that
    /// propagated dirt to non-HEAD resolutions would fingerprint
    /// historical filters with the current local state,
    /// mis-matching every sidecar in the pool.
    #[test]
    fn resolve_commit_specs_non_head_does_not_get_dirty_suffix_in_dirty_repo() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 2);
        let parent = chain[0];
        // Same dirty-worktree setup as the prior test.
        let repo = gix::open(tmp.path()).expect("gix::open");
        let head_tree = repo.head_tree().expect("head_tree").id;
        let mut idx = repo.index_from_tree(&head_tree).expect("index_from_tree");
        idx.write(gix::index::write::Options::default())
            .expect("write index");
        std::fs::write(tmp.path().join("file.txt"), b"v1\n").unwrap();
        std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
        let raw = vec!["HEAD~1".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec![parent.to_hex_with_len(7).to_string()],
            "HEAD~1 (historical commit) must NOT get -dirty suffix \
             even when worktree is dirty",
        );
    }

    /// `<oid>^!` (ExcludeParents revspec, "this commit
    /// specifically") resolves to the commit's own 7-char short
    /// hex. gix maps `^!` to `Spec::ExcludeParents(id)`; the
    /// resolver groups it with `Spec::Include(id)` because both
    /// arms describe a single object the operator named directly.
    /// Pins the second arm of the Include/ExcludeParents match
    /// branch; without this test the branch could collapse to
    /// "Include only" without anyone noticing.
    #[test]
    fn resolve_commit_specs_exclude_parents_resolves_like_include() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 2);
        let head = chain[1];
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec![format!("{}^!", head.to_hex_with_len(40))];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec![head.to_hex_with_len(7).to_string()],
            "<oid>^! must resolve to the same 7-char short hex as <oid>",
        );
    }

    /// A branch name resolves to the commit it points at. Pins
    /// that the resolver accepts ref-name inputs (not just OIDs)
    /// — without this test the branch-creation API change in a
    /// future gix version could silently drop branch resolution.
    #[test]
    fn resolve_commit_specs_branch_name_resolves_to_tip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 2);
        let parent = chain[0];
        let repo = gix::open(tmp.path()).expect("gix::open");
        // Create `refs/heads/feature` pointing at the parent
        // commit (chain[0]) so the test can distinguish it from
        // HEAD (chain[1]).
        repo.reference(
            "refs/heads/feature",
            parent,
            gix::refs::transaction::PreviousValue::MustNotExist,
            "create feature branch for test",
        )
        .expect("create branch");
        let raw = vec!["feature".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec![parent.to_hex_with_len(7).to_string()],
            "branch name must resolve to its tip commit",
        );
    }

    /// A tag name resolves to the commit it points at. Same
    /// shape as the branch test, against `refs/tags/<name>` —
    /// gix `rev_parse` looks up tags through the same ref-name
    /// resolver, so this exercises the ref-resolution code path
    /// for the tag namespace specifically.
    #[test]
    fn resolve_commit_specs_tag_name_resolves_to_target() {
        let tmp = tempfile::TempDir::new().unwrap();
        let chain = init_repo_with_chain(tmp.path(), 2);
        let parent = chain[0];
        let repo = gix::open(tmp.path()).expect("gix::open");
        // Lightweight tag (a ref under `refs/tags/`) pointing at
        // the parent commit. Distinct from an annotated tag
        // (which is its own object kind); rev_parse handles both
        // by peeling, but lightweight tags are simpler to set up
        // in a fixture without writing a Tag object.
        repo.reference(
            "refs/tags/v0",
            parent,
            gix::refs::transaction::PreviousValue::MustNotExist,
            "create v0 tag for test",
        )
        .expect("create tag");
        let raw = vec!["v0".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec![parent.to_hex_with_len(7).to_string()],
            "tag name must resolve to its target commit",
        );
    }

    /// `HEAD..HEAD` (empty range — `from == to`) yields zero
    /// commits because the walk hides every reachable commit
    /// from `from` and there are no remaining commits in the
    /// `to`-side. Pins that an empty range lands as an empty
    /// expansion (not a literal fallback), so a downstream
    /// filter built from this Vec excludes every row — which
    /// matches the operator's intent ("commits in `HEAD..HEAD`"
    /// is the empty set).
    #[test]
    fn resolve_commit_specs_empty_range_yields_no_entries() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_chain(tmp.path(), 2);
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec!["HEAD..HEAD".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert!(
            out.is_empty(),
            "HEAD..HEAD must expand to zero commits; got {out:?}",
        );
    }

    /// A valid-hex 7-char prefix that does not match any commit
    /// in the repo (`deadbee`) falls through to literal. Pins
    /// the rev_parse-Err arm against a hex-shaped input —
    /// distinct from the prior `zzzzzzz` non-hex case, this one
    /// reaches the object-database lookup before failing, so it
    /// exercises a deeper rev_parse code path. The literal
    /// `deadbee` lands in the output because the warning is
    /// suppressed by [`looks_like_literal_hash`]; either way
    /// the value passes through.
    #[test]
    fn resolve_commit_specs_valid_hex_nonexistent_prefix_falls_through_to_literal() {
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_chain(tmp.path(), 1);
        let repo = gix::open(tmp.path()).expect("gix::open");
        let raw = vec!["deadbee".to_string()];
        let out = resolve_commit_specs(Some(&repo), &raw, "test");
        assert_eq!(
            out,
            vec!["deadbee".to_string()],
            "valid-hex non-existent prefix must pass through as literal",
        );
    }

    /// `looks_like_literal_hash` accepts the on-disk shape that
    /// `detect_*_commit` writes: 7..=40 lowercase hex chars,
    /// optionally followed by `-dirty`. Pins the gating predicate
    /// against the canonical writer's output so a regression that
    /// tightened the predicate (e.g. 7-only, no -dirty) would
    /// surface noisy warnings on legitimate inputs.
    #[test]
    fn looks_like_literal_hash_accepts_canonical_shapes() {
        // Bare 7-char hash (the most common operator-typed shape).
        assert!(looks_like_literal_hash("abc1234"));
        // 40-char full hash (upper bound).
        assert!(looks_like_literal_hash(
            "abcdef0123456789abcdef0123456789abcdef01"
        ));
        // 7-char + -dirty (the dirty-suffixed sidecar entry).
        assert!(looks_like_literal_hash("abc1234-dirty"));
        // 40-char + -dirty.
        assert!(looks_like_literal_hash(
            "abcdef0123456789abcdef0123456789abcdef01-dirty"
        ));
    }

    /// `looks_like_literal_hash` rejects revspec-shaped inputs
    /// so the gated rev_parse-Err warning still fires for them.
    /// Pins the negative side of the predicate against every
    /// shape the gating spec calls out: alpha beyond hex, ~, ..,
    /// ^, and out-of-bound lengths. Mixed-case hex is now ACCEPTED
    /// — see
    /// [`looks_like_literal_hash_accepts_uppercase_and_mixed_case`].
    #[test]
    fn looks_like_literal_hash_rejects_revspec_shapes() {
        // Alpha beyond hex.
        assert!(!looks_like_literal_hash("HEAD"));
        assert!(!looks_like_literal_hash("main"));
        // Tilde-form revspec.
        assert!(!looks_like_literal_hash("HEAD~1"));
        // Range form.
        assert!(!looks_like_literal_hash("HEAD~3..HEAD"));
        // Caret form.
        assert!(!looks_like_literal_hash("HEAD^"));
        // Below 7-char minimum.
        assert!(!looks_like_literal_hash("abc123"));
        // Above 40-char maximum (e.g. SHA-256 prefix or paste of
        // a longer string).
        assert!(!looks_like_literal_hash(
            "abcdef0123456789abcdef0123456789abcdef0123"
        ));
        // Empty input.
        assert!(!looks_like_literal_hash(""));
        // -dirty without enough hex prefix.
        assert!(!looks_like_literal_hash("abc-dirty"));
    }

    /// `looks_like_literal_hash` accepts uppercase and mixed-case
    /// hex. The sidecar writer produces lowercase, but operators
    /// commonly paste hashes from `git log` UIs or other tools that
    /// uppercase. Suppressing the rev_parse-Err warning on these
    /// keeps CLI noise down; the literal still passes through, and
    /// the (lowercase) sidecar pool simply produces no match — the
    /// same outcome as a legitimate-but-unknown short hash.
    #[test]
    fn looks_like_literal_hash_accepts_uppercase_and_mixed_case() {
        // Pure uppercase hex.
        assert!(looks_like_literal_hash("ABC1234"));
        // Mixed-case hex.
        assert!(looks_like_literal_hash("AbC1234"));
        // Uppercase + -dirty.
        assert!(looks_like_literal_hash("ABC1234-dirty"));
        // 40-char uppercase full hash.
        assert!(looks_like_literal_hash(
            "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        ));
    }
}
