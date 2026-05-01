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

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec, sidecar_dir};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// Compute the per-test failure-dump path. Mirrors the path
/// `test_support::eval::run_ktstr_test_inner` configures on the
/// VM builder for every test (the primary dispatch attaches this
/// path before booting):
/// `{sidecar_dir()}/{test_name}.failure-dump.json`. Both sites
/// must agree — if `run_ktstr_test_inner` changes the naming
/// convention, this helper must follow.
fn failure_dump_path(test_name: &str) -> std::path::PathBuf {
    sidecar_dir().join(format!("{test_name}.failure-dump.json"))
}

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

    // Parse as a generic JSON value to avoid an unbounded
    // dependency on the (pub(crate)) `FailureDumpReport` type from
    // outside the crate.
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}"))?;

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
        scheduler: &KTSTR_SCHED_PAYLOAD,
        // --stall-after=1 makes the scheduler return early from
        // dispatch after 1 second of operation, triggering
        // SCX_EXIT_ERROR_STALL via the kernel watchdog.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays under the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        workers_per_cgroup: 2,
        // The scenario itself returns Err to surface a missing-dump
        // regression as a real failure. Successful rendering returns
        // a failed AssertResult (the stall is the expected behaviour);
        // expect_err inverts that to PASS.
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
