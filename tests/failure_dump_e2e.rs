//! End-to-end test for the failure-dump pipeline.
//!
//! Boots scx-ktstr with `--stall-after=1`, lets the BPF probe latch
//! the resulting `SCX_EXIT_ERROR_STALL` exit, and asserts that the
//! freeze coordinator's host-side dump captures BTF-rendered fields
//! from the scheduler's `.bss` section AND from the `BPF_MAP_TYPE_ARENA`
//! map that scx-ktstr's `sdt_alloc`-backed per-task contexts populate.
//!
//! The freeze coordinator writes the JSON-pretty `FailureDumpReport`
//! to a per-test path inside the run's sidecar directory
//! (`{sidecar_dir()}/{test_name}.failure-dump.json`). The test
//! framework's primary dispatch
//! (`test_support::eval::run_ktstr_test_inner`) attaches that
//! path on every VM builder it constructs — no env var required,
//! no per-scenario setup beyond reading the path back here after
//! the run.
//!
//! User-facing test bar (per project memory): "I see variable names
//! and values in the logs when a scheduler stalls." This test
//! enforces the host-side half of that bar on three axes:
//!   1. the file at the sidecar-dir path contains `stall`, `crash`
//!      and other BTF-resolved field names from `scx-ktstr`'s global
//!      section, not hex offsets;
//!   2. a `BPF_MAP_TYPE_ARENA` map is present in the dump and at least
//!      one captured page contains the `KTSTR_ARENA_MAGIC` sentinel
//!      (proves live-data capture, not zero pages);
//!   3. `ktstr_alloc_count` in `.bss` is non-zero (cross-validates
//!      that the alloc path executed before the stall).

mod common;

use anyhow::Result;
use common::dump_paths::failure_dump_path;
use ktstr::assert::AssertResult;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

fn scenario_failure_dump_renders_bss_fields(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    // The freeze coordinator's file sink is wired by the test
    // framework (see `test_support::eval::run_ktstr_test_inner`,
    // which attaches the primary dump path on every VM builder
    // it constructs) — no env-var dance, no `set_var` race
    // against parallel tests. This scenario just reads the file
    // back from the same sidecar dir keyed by `test_name`.
    let dump_path = failure_dump_path("failure_dump_renders_bss_fields");

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    // Read the dump file written by the freeze coordinator. The
    // file must exist when the scenario reaches here because:
    //   1. `--stall-after=1` triggers an SCX_EXIT_ERROR_STALL inside
    //      the guest within ~1 second.
    //   2. The probe BPF tracepoint fires on the error-class exit.
    //   3. The freeze coordinator's .bss-poll observes the latch,
    //      runs `dump_state`, and writes the JSON to the
    //      builder-configured path before clearing `freeze`.
    //   4. The watchdog tears the VM down, `execute_steps` returns,
    //      and we land here on the host side of the same process.
    //
    // A theoretical race exists between the guest-side sched_ext
    // teardown and the host freeze rendezvous — if the guest
    // userspace dropped the BPF skeleton before the host paused
    // vCPUs, the arena map would be absent from the IDR walk. In
    // practice, the vCPU freeze preempts guest userspace reaction:
    // the error-exit latch flips inside the probe BPF program
    // (sched_ext_exit tracepoint), the host's .bss poll observes it,
    // and the freeze coordinator pauses every vCPU before the
    // guest's userspace `Drop` path can deliberately unload the
    // skeleton.
    //
    // If the file is missing, surface the failure in the
    // AssertResult details rather than panicking — `expect_err: true`
    // would otherwise mask a missing-dump regression as a pass.
    let json = match std::fs::read_to_string(&dump_path) {
        Ok(s) => s,
        Err(e) => {
            result.passed = false;
            result.details.push(ktstr::assert::AssertDetail::new(
                ktstr::assert::DetailKind::Other,
                format!(
                    "failure dump file missing at {}: {e} (freeze coordinator did \
                     not write — either the SCX_EXIT_ERROR_STALL latch did not \
                     fire, owned_accessor / dump_btf was None, or the file \
                     write failed silently)",
                    dump_path.display()
                ),
            ));
            anyhow::bail!(
                "failure dump file missing at {} — freeze coordinator did not \
                 write the JSON dump",
                dump_path.display()
            );
        }
    };

    // Parse as a generic JSON value. The dump schema is now an
    // explicit discriminant — `FailureDumpReportAny::from_json`
    // rejects any blob without a `schema` field — so the test
    // short-circuits on the schema before reaching the variant-
    // specific shape. The full-dump happy path expects
    // [`SCHEMA_SINGLE`]; SCHEMA_DEGRADED on this happy-path test
    // indicates a regression in the freeze coordinator's
    // capture-vs-degraded dispatch.
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}"))?;
    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing top-level `schema` field — the dispatcher \
                 at `FailureDumpReportAny::from_json` requires an explicit \
                 discriminant"
            )
        })?;
    anyhow::ensure!(
        schema == "single",
        "happy-path dump must carry schema=\"single\"; got schema={schema} \
         (a `degraded` schema here means the freeze coordinator's gate \
         cross-reference or rendezvous-timeout path fired when it should not have)"
    );

    // Top-level shape: {"maps": [...]}. `non_exhaustive` does not
    // affect serde output, so the field name is stable.
    let maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `maps` array"))?;

    // Find the scx-ktstr `.bss` map. libbpf composes
    // `<obj_name>.bss` for the global-section map, and scx-ktstr's
    // BPF object is `bpf` (per `scx-ktstr/build.rs`'s
    // `enable_skel("src/bpf/main.bpf.c", "bpf")` call), so the dump
    // should carry an entry whose name ends with `.bss` and is NOT
    // one of the framework probes filtered by `KTSTR_INTERNAL_MAPS`.
    let bss_map = maps
        .iter()
        .find(|m| {
            m.get("name")
                .and_then(|n| n.as_str())
                .map(|n| {
                    n.ends_with(".bss")
                        && !n.starts_with("probe_bp.")
                        && !n.starts_with("fentry_p.")
                })
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump has no scheduler `.bss` map (got {} maps): {json}",
                maps.len()
            )
        })?;

    // The rendered value is a Struct whose members enumerate the
    // BTF-resolved global names. Serde tags the variant via
    // `kind = "struct"`; members are at `.value.members[]`. Each
    // member is `{ "name": "<field>", "value": {...} }`.
    let value_field = bss_map
        .get("value")
        .ok_or_else(|| anyhow::anyhow!(".bss map has no `value` field"))?;
    let kind = value_field
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("");
    if kind != "struct" {
        anyhow::bail!(
            "expected .bss value to render as a Struct (kind=\"struct\"), got kind={kind:?}: \
             {value_field}"
        );
    }
    let members = value_field
        .get("members")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!(".bss Struct has no `members` array"))?;

    // The user-facing test bar: BTF field names must appear in the
    // rendered output, NOT hex offsets. scx-ktstr's main.bpf.c
    // declares `stall`, `crash`, `degrade_rt`, `degrade_cnt`,
    // `slow_cnt` (and others) at file scope. At minimum the trigger
    // field `stall` and the headline error fields must be visible —
    // the others may shift across scx-ktstr versions, so don't
    // pin the full set.
    let names: std::collections::HashSet<&str> = members
        .iter()
        .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
        .collect();
    for required in ["stall", "crash"] {
        if !names.contains(required) {
            anyhow::bail!(
                "BTF-rendered .bss missing required field `{required}` — \
                 either the field was renamed in scx-ktstr's main.bpf.c \
                 or the renderer fell through to an Unsupported branch \
                 instead of recursing into the Struct. members: {names:?}"
            );
        }
    }

    // Stall flag must be a non-zero unsigned integer — proves the
    // dump captured the LIVE state at error-exit time, not a
    // pre-init zero. scx-ktstr writes `stall = 1` from
    // `--stall-after=1` before the watchdog fires.
    let stall_value = members
        .iter()
        .find(|m| m.get("name").and_then(|n| n.as_str()) == Some("stall"))
        .and_then(|m| m.get("value"))
        .ok_or_else(|| anyhow::anyhow!("`stall` member found but has no `value`"))?;
    let stall_kind = stall_value
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("");
    let stall_int = stall_value
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`stall` value is not a numeric u64 (kind={stall_kind:?}): {stall_value}"
            )
        })?;
    if stall_int == 0 {
        anyhow::bail!(
            "`stall` rendered as 0 — the scheduler-side stall flag never flipped, \
             or the freeze coordinator captured pre-stall state. Full value: {stall_value}"
        );
    }

    // Per-vCPU register snapshots: the freeze coordinator
    // attaches a `vcpu_regs` array (BSP at index 0, APs at
    // 1..N). Each entry is `null` when capture failed for that
    // vCPU OR a `{instruction_pointer, stack_pointer,
    // page_table_root}` object otherwise. The test asserts the
    // array exists, has at least one populated entry, and that
    // entry's `instruction_pointer` is non-zero (a zero RIP/PC
    // would mean the snapshot was captured but holds garbage —
    // possibly indicating the vCPU thread crashed before the
    // capture or the kernel/userspace VA was uninitialized).
    //
    // `vcpu_regs` is opt-out via serde's `skip_serializing_if =
    // "Vec::is_empty"`, so its absence here would mean the
    // freeze coordinator's regs-attach path didn't fire — a
    // genuine regression on the host-side capture wiring.
    let vcpu_regs = value
        .get("vcpu_regs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing top-level `vcpu_regs` array — \
                 freeze coordinator did not attach per-vCPU register \
                 snapshots after rendezvous"
            )
        })?;
    if vcpu_regs.is_empty() {
        anyhow::bail!(
            "dump JSON `vcpu_regs` array is empty — expected at least \
             one entry (BSP idx 0 plus N APs)"
        );
    }
    let populated_with_ip: Vec<&serde_json::Value> = vcpu_regs
        .iter()
        .filter(|slot| {
            slot.is_object()
                && slot
                    .get("instruction_pointer")
                    .and_then(|ip| ip.as_u64())
                    .is_some_and(|ip| ip != 0)
        })
        .collect();
    if populated_with_ip.is_empty() {
        anyhow::bail!(
            "dump JSON `vcpu_regs` has no entry with non-zero \
             instruction_pointer — every slot is null or has zero IP. \
             Capture-on-vCPU-thread path may be broken or rendezvous \
             timed out before any vCPU completed handle_freeze. \
             Full vcpu_regs: {vcpu_regs:?}"
        );
    }

    // user_page_table_root is arch-conditional:
    //   x86_64: always None → JSON key absent
    //     (skip_serializing_if = "Option::is_none").
    //   aarch64: best-effort Some(ttbr0_el1) when the TTBR0_EL1
    //     KVM_GET_ONE_REG read succeeds; otherwise still absent.
    //
    // Pin per-arch behaviour so a future field rename or a regression
    // (e.g. accidentally always populating on x86_64) is caught.
    #[cfg(target_arch = "x86_64")]
    {
        for slot in &populated_with_ip {
            assert!(
                slot.get("user_page_table_root").is_none(),
                "x86_64 vcpu_regs entry must NOT carry user_page_table_root \
                 (CR3 alone identifies the active mm); got: {slot}"
            );
        }
    }
    // (aarch64 doesn't get a hard requirement here because the
    // sysreg read can be gated by the host kernel — best-effort
    // capture per the design. The serde test inside exit_dispatch
    // already pins the JSON-key contract for both states.)

    // Arena map presence + live-data assertions.
    //
    // A BPF arena (BPF_MAP_TYPE_ARENA) is a sparse, page-granular
    // memory region the host walker translates page-by-page from the
    // guest's kernel page tables. scx-ktstr uses sdt_alloc-backed
    // arena memory: every task gets a `struct ktstr_arena_ctx`
    // allocated via `scx_task_alloc` in `ktstr_init_task` (see
    // `scx-ktstr/src/bpf/main.bpf.c`). Each allocation stamps:
    //   - magic   = KTSTR_ARENA_MAGIC (0xDEADBEEFCAFEBABE)
    //   - counter = KTSTR_TASK_COUNTER (42; not separately asserted —
    //               magic alone proves liveness)
    // and increments `ktstr_alloc_count` (u64 in .bss). After
    // `--stall-after=1` triggers the watchdog, tasks have already
    // run through `init_task`, so the dump must capture:
    //   1. a non-zero `ktstr_alloc_count` member in `.bss` — proves
    //      the alloc path executed and the counter was captured live.
    //   2. an arena map (BPF_MAP_TYPE_ARENA = 33) by map_type — the
    //      arena is declared bare-named ("arena") via the __weak
    //      SEC(".maps") declaration in lib/arena_map.h, with no
    //      libbpf <obj>.<section> prefix.
    //   3. at least one captured page in arena.pages — proves the
    //      walker translated user_addr → kern_vm and read live
    //      memory, not an empty snapshot.
    //   4. the magic constant inside at least one page's bytes — the
    //      bar test ("LIVE data, not zeros") requires content
    //      verification, not just non-empty pages.
    //
    // BPF_MAP_TYPE_ARENA is hardcoded as 33 here to match the test's
    // existing pattern of not importing crate internals (the test
    // operates on JSON shape, not Rust types).

    // Read `ktstr_alloc_count` first — it cross-validates the alloc
    // path independently of the arena walker, and the magic-scan bail
    // below uses its value to narrow down the failure mode (alloc ran
    // but capture broken, vs alloc never ran).
    let alloc_count_value = members
        .iter()
        .find(|m| m.get("name").and_then(|n| n.as_str()) == Some("ktstr_alloc_count"))
        .and_then(|m| m.get("value"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "BTF-rendered .bss missing `ktstr_alloc_count` — \
                 either the field was renamed in scx-ktstr's main.bpf.c, \
                 or the BTF Datasec walker did not surface it. members: \
                 {names:?}"
            )
        })?;
    let alloc_count_kind = alloc_count_value
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("");
    let alloc_count_int = alloc_count_value
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`ktstr_alloc_count` value is not a numeric u64 \
                 (kind={alloc_count_kind:?}): {alloc_count_value}"
            )
        })?;
    if alloc_count_int == 0 {
        anyhow::bail!(
            "`ktstr_alloc_count` rendered as 0 — the alloc path never \
             ran (no `__sync_fetch_and_add` in ktstr_init_task), or \
             the dump captured pre-init state. Full value: \
             {alloc_count_value}"
        );
    }

    const BPF_MAP_TYPE_ARENA: u64 = 33;
    let arena_map = maps
        .iter()
        .find(|m| {
            m.get("map_type")
                .and_then(|t| t.as_u64())
                .is_some_and(|t| t == BPF_MAP_TYPE_ARENA)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump has no BPF_MAP_TYPE_ARENA (33) map — scx-ktstr \
                 declares one via lib/arena_map.h, so either the dump \
                 path filtered it out, the map enumeration missed it, \
                 or the scheduler failed to load the arena. Got {} \
                 maps total: {json}",
                maps.len()
            )
        })?;
    // Arena map JSON shape:
    //   {"map_type": 33, "arena": {"pages": [{"user_addr": N,
    //    "bytes": [u8, u8, ...]}, ...], "declared_pages": N,
    //    "truncated"?: bool, "span_capped"?: bool}, ...}.
    // `pages` is `skip_serializing_if = "Vec::is_empty"`, so an empty
    // page set means the key is absent from JSON entirely. Both flags
    // and the inner `arena` object can also be absent depending on
    // the snapshot path.
    let arena_field = arena_map.get("arena").ok_or_else(|| {
        anyhow::anyhow!(
            "arena map present but `arena` field absent — render_map's \
             BPF_MAP_TYPE_ARENA arm did not populate ArenaSnapshot \
             (likely arena_offsets was None: kernel BTF lacks \
             struct bpf_arena, or BpfArenaOffsets::from_btf failed). \
             arena map JSON: {arena_map}"
        )
    })?;

    // ArenaSnapshot.pages uses `skip_serializing_if = "Vec::is_empty"`,
    // so an empty pages vector is absent from JSON entirely, not present
    // as `[]`. Treat both shapes (absent + present-but-empty) as the
    // same "no pages captured" failure mode — the bar is non-empty.
    let arena_pages: &[serde_json::Value] = match arena_field.get("pages") {
        Some(p) => p.as_array().map(|a| a.as_slice()).ok_or_else(|| {
            anyhow::anyhow!(
                "arena.pages is present but not an array — \
                 ArenaSnapshot serde shape changed. arena field: {arena_field}"
            )
        })?,
        None => &[],
    };
    if arena_pages.is_empty() {
        anyhow::bail!(
            "arena.pages is empty (absent or zero-length) — snapshot_arena \
             returned no pages. Either the PTE walker found no mapped \
             pgoffs (kern_vm translation failed for every page), \
             max_entries is 0, or scx_task_alloc never ran on any task \
             (alloc_count={alloc_count_int}). arena field: {arena_field}"
        );
    }

    // declared_pages sanity: must be > 0 (proves max_entries was
    // readable from `struct bpf_map` at dump time) and must be at
    // least as large as the captured page set (a captured page count
    // exceeding the declared capacity would mean the walker over-
    // walked or the snapshot accidentally accumulated stale entries).
    // Absent key falls back to 0 — the default for ArenaSnapshot.
    let declared_pages = arena_field
        .get("declared_pages")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if declared_pages == 0 {
        anyhow::bail!(
            "arena.declared_pages is 0 (or absent) — \
             ArenaWalkPlan computed a zero-page span, meaning \
             `info.max_entries` was unreadable or zero at dump time. \
             arena field: {arena_field}"
        );
    }
    if (arena_pages.len() as u64) > declared_pages {
        anyhow::bail!(
            "arena.pages.len() ({}) exceeds declared_pages ({}) — \
             walker invariant violated; ArenaWalkPlan should never \
             emit more pages than the declared capacity. arena field: \
             {arena_field}",
            arena_pages.len(),
            declared_pages
        );
    }

    // Magic-byte scan: the LE bytes of KTSTR_ARENA_MAGIC must appear
    // in at least one captured page. Each ArenaPage.bytes serializes
    // as a JSON array of u8 (serde's default Vec<u8> serialization);
    // collect into Vec<u8> per page and use windows() to find the
    // 8-byte LE pattern. The constant is derived from the u64 source
    // value so a future change to KTSTR_ARENA_MAGIC in main.bpf.c
    // only needs the matching update here, no manual byte-reversal.
    //
    // Per-page scan (vs. cross-page concatenation): scx-ktstr's
    // sdt_alloc slot layout is 24 bytes (8-byte sdt_data header +
    // 16-byte ktstr_arena_ctx) and slots align within pages — no slot
    // crosses a page boundary, so the magic u64 is always contiguous
    // within a single captured page.
    const KTSTR_ARENA_MAGIC: u64 = 0xDEADBEEFCAFEBABE;
    const KTSTR_ARENA_MAGIC_LE: [u8; 8] = KTSTR_ARENA_MAGIC.to_le_bytes();
    let mut magic_hits = 0usize;
    let mut total_bytes = 0usize;
    for page in arena_pages {
        let bytes = page
            .get("bytes")
            .and_then(|b| b.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "arena page missing `bytes` array — \
                     ArenaPage serde shape changed. page: {page}"
                )
            })?;
        // Each element must be a u8; collect into a flat Vec<u8>.
        let raw: Vec<u8> = bytes
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or_else(|| anyhow::anyhow!("arena page byte is not a u8 (0..=255): {v}"))
            })
            .collect::<Result<Vec<u8>>>()?;
        total_bytes += raw.len();
        if raw
            .windows(KTSTR_ARENA_MAGIC_LE.len())
            .any(|w| w == KTSTR_ARENA_MAGIC_LE)
        {
            magic_hits += 1;
        }
    }
    if magic_hits == 0 {
        anyhow::bail!(
            "no arena page contained KTSTR_ARENA_MAGIC \
             (0x{KTSTR_ARENA_MAGIC:016x}) — pages were captured but \
             contain no live-stamped data. Most diagnostic case: \
             alloc_count={alloc_count_int} (>0 means the alloc path \
             ran, so the magic stamp was lost OR the walker captured \
             the wrong pages); alloc_count=0 would mean no tasks \
             were initialized in the first place. {} pages totalling \
             {} bytes scanned.",
            arena_pages.len(),
            total_bytes
        );
    }

    // Confirming detail so the test log shows the captured value.
    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "failure-dump file at {} contains scheduler .bss render with \
             stall={stall_int}, ktstr_alloc_count={alloc_count_int}, \
             member count={}, vcpu_regs entries={} ({} populated with \
             non-zero IP), arena pages={} ({total_bytes} bytes, \
             {magic_hits} pages with KTSTR_ARENA_MAGIC sentinel, \
             declared_pages={declared_pages})",
            dump_path.display(),
            members.len(),
            vcpu_regs.len(),
            populated_with_ip.len(),
            arena_pages.len(),
        ),
    ));

    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_FAILURE_DUMP_BSS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "failure_dump_renders_bss_fields",
        func: scenario_failure_dump_renders_bss_fields,
        scheduler: &KTSTR_SCHED,
        // --stall-after=1 makes the scheduler return early from
        // dispatch after 1 second of operation, triggering
        // SCX_EXIT_ERROR_STALL via the kernel watchdog.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays under the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // The scenario itself returns Err to surface a missing-dump
        // regression as a real failure. Successful rendering returns
        // a failed AssertResult (the stall is the expected behaviour);
        // expect_err inverts that to PASS.
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Asserts that the freeze coordinator's host-side capture modules
/// (`crate::vmm::capture_scx`, `crate::vmm::capture_tasks`,
/// `crate::vmm::capture_numa`) populate
/// [`crate::monitor::dump::FailureDumpReport`] with non-default
/// data when the `--stall-after=1` SCX_EXIT_ERROR_STALL path
/// triggers a freeze.
///
/// User-facing test bar (per project memory): "captures must
/// always produce data" — when scx-ktstr is loaded and tasks are
/// runnable, the dump should carry per-CPU rq->scx state, at
/// least the global DSQ, the scx_sched scalar state, and at
/// least one task enrichment record. NUMA stats either populate
/// (CONFIG_NUMA=y kernel) or carry the diagnostic reason that
/// explains why they didn't.
///
/// Distinct from `scenario_failure_dump_renders_bss_fields`:
/// that test exercises the BTF / arena render path; this one
/// exercises the live-walker captures wired into freeze_coord
/// at #68/#69/#70.
fn scenario_failure_dump_renders_capture_modules(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let dump_path = failure_dump_path("failure_dump_renders_capture_modules");
    let num_cpus = ctx.topo.total_cpus();

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = match std::fs::read_to_string(&dump_path) {
        Ok(s) => s,
        Err(e) => {
            result.passed = false;
            result.details.push(ktstr::assert::AssertDetail::new(
                ktstr::assert::DetailKind::Other,
                format!(
                    "failure dump file missing at {}: {e} (freeze coordinator did \
                     not write — either the SCX_EXIT_ERROR_STALL latch did not \
                     fire or the file write failed silently)",
                    dump_path.display()
                ),
            ));
            anyhow::bail!(
                "failure dump file missing at {} — freeze coordinator did not \
                 write the JSON dump",
                dump_path.display()
            );
        }
    };

    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}"))?;

    // -- scx_walker capture (rq_scx_states / dsq_states / scx_sched_state) --
    //
    // The walker pushes one entry per CPU whose rq + scx_rq + task
    // sub-group offsets resolved. With CONFIG_SCHED_CLASS_EXT=y and
    // a debug-info kernel (per ktstr.kconfig) every CPU resolves, so
    // the vec length must equal num_cpus. Surface the absent /
    // partial state diagnostic when the walker fails so the failure
    // mode is identifiable from the dump alone.
    if let Some(reason) = value.get("scx_walker_unavailable").and_then(|r| r.as_str()) {
        anyhow::bail!(
            "scx_walker_unavailable={reason:?} — capture_scx::build returned \
             None or the walker reached no state. Captures must always \
             produce data when scx-ktstr is loaded. Full JSON: {json}"
        );
    }
    let rq_scx_states = value
        .get("rq_scx_states")
        .and_then(|s| s.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing `rq_scx_states` array — capture_scx \
                 wiring did not populate the field. Full JSON: {json}"
            )
        })?;
    if rq_scx_states.len() != num_cpus {
        anyhow::bail!(
            "rq_scx_states.len()={} but expected num_cpus={num_cpus} — \
             walk_rq_scx skipped some CPUs (sub-group offset resolution \
             failed or per-CPU rq translate failed). Full rq_scx_states: \
             {rq_scx_states:?}",
            rq_scx_states.len(),
        );
    }
    // At least one CPU must show evidence of scheduler activity:
    // either a non-zero `nr_running` (tasks queued on rq->scx) or a
    // non-zero `flags` (any scx_rq.flags bit set). Both being zero
    // across every CPU would mean the walker ran but read pre-init
    // state — an empty walker is no better than no walker.
    let any_active = rq_scx_states.iter().any(|s| {
        let nr = s.get("nr_running").and_then(|v| v.as_u64()).unwrap_or(0);
        let flags = s.get("flags").and_then(|v| v.as_u64()).unwrap_or(0);
        nr > 0 || flags != 0
    });
    if !any_active {
        anyhow::bail!(
            "no rq_scx_states entry has nr_running>0 OR flags!=0 — every \
             CPU's rq->scx scalar read came back zero, meaning the walker \
             ran but every per-CPU scx_rq is empty. Either no scx tasks \
             were ever runnable or the rq_pa translate produced wrong \
             addresses. Full rq_scx_states: {rq_scx_states:?}"
        );
    }

    let dsq_states = value
        .get("dsq_states")
        .and_then(|s| s.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing `dsq_states` array — capture_scx \
                 wiring did not populate the field. Full JSON: {json}"
            )
        })?;
    if dsq_states.is_empty() {
        anyhow::bail!(
            "dsq_states is empty — walk_dsqs reached no DSQs. The \
             global DSQ (SCX_DSQ_GLOBAL per-node) must always be \
             reachable when *scx_root is non-null. Full JSON: {json}"
        );
    }

    if value.get("scx_sched_state").is_none()
        || value.get("scx_sched_state").is_some_and(|v| v.is_null())
    {
        anyhow::bail!(
            "scx_sched_state is absent or null — read_scx_sched_state \
             returned None. *scx_root was unreadable or the BTF offsets \
             didn't resolve. Full JSON: {json}"
        );
    }

    // -- task_enrichments capture --
    //
    // The runnable_list walker pushes one entry per task on each
    // CPU's rq->scx.runnable_list. With workers_per_cgroup=2 driving
    // active workloads, at least one task should be runnable at the
    // freeze instant. An empty enrichment vec when scx-ktstr is
    // loaded means the walker missed every task — a real defect.
    if let Some(reason) = value
        .get("task_enrichments_unavailable")
        .and_then(|r| r.as_str())
    {
        anyhow::bail!(
            "task_enrichments_unavailable={reason:?} — capture_tasks::build \
             returned None or the walker yielded zero tasks. Captures \
             must always produce data when scx tasks are runnable. Full \
             JSON: {json}"
        );
    }
    let task_enrichments = value
        .get("task_enrichments")
        .and_then(|s| s.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing `task_enrichments` array — \
                 capture_tasks wiring did not populate the field. \
                 Full JSON: {json}"
            )
        })?;
    if task_enrichments.is_empty() {
        anyhow::bail!(
            "task_enrichments is empty — runnable_list walker found no \
             tasks. With workers_per_cgroup>0 driving load, at least \
             one task must be runnable at freeze time. Full JSON: {json}"
        );
    }
    // At least one enrichment must carry an identity that proves the
    // task_struct read produced live data: non-empty comm AND pid > 0.
    // pid==0 is the swapper / idle task — possible but not proof of
    // liveness; insist on a real userspace task slip through the
    // walker. comm is null-terminated and skipped when zero-length
    // wouldn't be skip-serialized but a zero-byte read would surface
    // as an empty string, not absent.
    let has_real_task = task_enrichments.iter().any(|t| {
        let pid = t.get("pid").and_then(|v| v.as_i64()).unwrap_or(0);
        let comm = t.get("comm").and_then(|v| v.as_str()).unwrap_or("");
        pid > 0 && !comm.is_empty()
    });
    if !has_real_task {
        anyhow::bail!(
            "no task_enrichment entry has pid>0 AND non-empty comm — \
             every task_struct read produced pid<=0 or empty comm, \
             meaning the slab translate fell back to garbage memory. \
             Full task_enrichments: {task_enrichments:?}"
        );
    }

    // -- per_node_numa capture --
    //
    // ktstr.kconfig sets CONFIG_NUMA=y, so capture_numa::build runs.
    // With nr_nodes=1 (default topology) it walks node 0 and emits
    // one PerNodeNumaStats row. If for any reason the walker bails
    // (symbol absent, BTF offsets unresolved, pgdat translate failed),
    // per_node_numa stays empty and per_node_numa_unavailable carries
    // the diagnostic. Both shapes are acceptable; what's NOT
    // acceptable is the empty vec without a diagnostic.
    let per_node_numa = value
        .get("per_node_numa")
        .and_then(|s| s.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);
    let per_node_numa_unavailable = value
        .get("per_node_numa_unavailable")
        .and_then(|r| r.as_str());
    if per_node_numa.is_empty() && per_node_numa_unavailable.is_none() {
        anyhow::bail!(
            "per_node_numa is empty AND per_node_numa_unavailable is \
             absent — the dump pipeline broke its own contract that \
             one of the two must be populated. Full JSON: {json}"
        );
    }

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "failure-dump file at {} contains capture-module data: \
             rq_scx_states.len()={} (num_cpus={num_cpus}), \
             dsq_states.len()={}, scx_sched_state present, \
             task_enrichments.len()={}, per_node_numa.len()={} \
             (unavailable={:?})",
            dump_path.display(),
            rq_scx_states.len(),
            dsq_states.len(),
            task_enrichments.len(),
            per_node_numa.len(),
            per_node_numa_unavailable,
        ),
    ));

    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_FAILURE_DUMP_CAPTURES: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "failure_dump_renders_capture_modules",
        func: scenario_failure_dump_renders_capture_modules,
        scheduler: &KTSTR_SCHED,
        // --stall-after=1 makes the scheduler return early from
        // dispatch after 1 second of operation, triggering
        // SCX_EXIT_ERROR_STALL via the kernel watchdog.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays under the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // The scenario returns a non-failed AssertResult on success
        // (the stall is the expected trigger that produces the dump);
        // any capture defect is reported via anyhow::bail! and bubbles
        // up as an Err. expect_err inverts the AssertResult fail-on-stall
        // to a pass.
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Asserts that the failure dump's `probe_counters` field captures
/// non-zero `trigger_count` after an SCX_EXIT_ERROR_STALL fires.
///
/// User-facing test bar (per project memory): the BPF probe's
/// per-CPU diagnostic counters must surface in the failure dump
/// with values that prove each tracepoint actually fired during
/// the run. After the per-CPU conversion landed (replacing N
/// shared-global counters with a `[MAX_CPUS][KTSTR_PCPU_NR]`
/// 2D array in `.bss`), this test pins:
///   1. `probe_counters` is present and structured (not absent /
///      null in the JSON);
///   2. `probe_counters.trigger_count > 0` — the
///      `tp_btf/sched_ext_exit` handler fired at least once during
///      the stall, which proves the per-CPU sum reaches the host;
///   3. `probe_counters.probe_count > 0` — kprobes attached and
///      fired (confirms the host-side sum walks the array, since
///      a stub-empty array would produce 0 even on a working run).
///
/// Distinct from `scenario_failure_dump_renders_bss_fields` (which
/// asserts the scheduler's own `.bss` BTF render) and
/// `scenario_failure_dump_renders_capture_modules` (which asserts
/// the live walker captures): this test exercises the host-side
/// host-side `decode_probe_counters_snapshot` reader specifically.
fn scenario_failure_dump_renders_probe_counters(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let dump_path = failure_dump_path("failure_dump_renders_probe_counters");

    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    let json = match std::fs::read_to_string(&dump_path) {
        Ok(s) => s,
        Err(e) => {
            result.passed = false;
            result.details.push(ktstr::assert::AssertDetail::new(
                ktstr::assert::DetailKind::Other,
                format!(
                    "failure dump file missing at {}: {e} (freeze coordinator did \
                     not write — either the SCX_EXIT_ERROR_STALL latch did not \
                     fire or the file write failed silently)",
                    dump_path.display()
                ),
            ));
            anyhow::bail!(
                "failure dump file missing at {} — freeze coordinator did not \
                 write the JSON dump",
                dump_path.display()
            );
        }
    };

    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}"))?;

    // `probe_counters` is `skip_serializing_if = "Option::is_none"`,
    // so its absence in the JSON means the host-side decoder
    // returned None. That's a regression — when the probe has
    // attached and fired (which the stall scenario guarantees),
    // the decoder must produce a populated struct.
    let probe_counters = value.get("probe_counters").ok_or_else(|| {
        anyhow::anyhow!(
            "dump JSON missing `probe_counters` field — \
             decode_probe_counters_snapshot returned None. \
             Probe `.bss` map absent, BTF lookup failed, or the \
             `ktstr_pcpu_counters` array offset didn't resolve. \
             Full JSON: {json}"
        )
    })?;
    if probe_counters.is_null() {
        anyhow::bail!(
            "`probe_counters` is null — decoder ran but produced None; \
             same prerequisite-missing failure modes as above. \
             Full JSON: {json}"
        );
    }

    // `trigger_count` is the structural assertion — a stall
    // scenario is guaranteed to fire `tp_btf/sched_ext_exit`
    // (the SCX kernel emits SCX_EXIT_ERROR_STALL through the
    // tracepoint), so a zero value here means either (a) the
    // probe didn't attach the trigger handler, (b) the handler
    // fired but the per-CPU slot bump didn't land, or (c) the
    // host-side cross-CPU sum walked the wrong slot index.
    let trigger_count = probe_counters
        .get("trigger_count")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`probe_counters.trigger_count` missing or non-numeric — \
                 ProbeBssCounters serde shape changed. \
                 probe_counters: {probe_counters}"
            )
        })?;
    if trigger_count == 0 {
        anyhow::bail!(
            "`probe_counters.trigger_count == 0` — `tp_btf/sched_ext_exit` \
             never fired (or the per-CPU slot didn't increment). The stall \
             scenario must produce at least one tracepoint fire. \
             probe_counters: {probe_counters}"
        );
    }

    // `probe_count` cross-validates the array walk: the kprobe
    // handler is attached to multiple kernel functions (sched
    // entry / dispatch path) and fires throughout the run, so a
    // healthy stall scenario produces hundreds-to-millions of
    // fires. A non-zero value here proves the host-side reader
    // walked the per-CPU slots (rather than reading a stub-zero
    // value from index 0 of an empty array).
    let probe_count = probe_counters
        .get("probe_count")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`probe_counters.probe_count` missing or non-numeric — \
                 ProbeBssCounters serde shape changed. \
                 probe_counters: {probe_counters}"
            )
        })?;
    if probe_count == 0 {
        anyhow::bail!(
            "`probe_counters.probe_count == 0` — kprobe path never fired \
             across the run. Either probe attach failed, ktstr_enabled \
             never flipped to true, or the host-side sum walked the wrong \
             slot index. probe_counters: {probe_counters}"
        );
    }

    result.details.push(ktstr::assert::AssertDetail::new(
        ktstr::assert::DetailKind::Other,
        format!(
            "failure-dump file at {} contains probe_counters with \
             trigger_count={trigger_count}, probe_count={probe_count} \
             (per-CPU sum walked across CPUs in `.bss` \
             `ktstr_pcpu_counters` array)",
            dump_path.display(),
        ),
    ));

    Ok(result)
}

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_FAILURE_DUMP_PROBE_COUNTERS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "failure_dump_renders_probe_counters",
        func: scenario_failure_dump_renders_probe_counters,
        scheduler: &KTSTR_SCHED,
        // --stall-after=1 fires SCX_EXIT_ERROR_STALL on watchdog
        // timeout. The probe's tp_btf/sched_ext_exit handler
        // bumps `KTSTR_PCPU_TRIGGER_COUNT` on every fire, so a
        // single stall produces a non-zero cross-CPU sum.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays under the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // Stall scenarios surface as a failed AssertResult — the
        // test framework's `expect_err: true` flips that into a
        // pass, so the scenario itself only returns Err when the
        // dump renders incorrectly (missing/zero counter).
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
