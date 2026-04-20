//! Ready-made [`Payload`](ktstr::Payload) fixtures for the two
//! benchmark binaries that dominate scheduler-regression testing:
//! `fio` (disk IO throughput, emits JSON) and `stress-ng`
//! (synthetic CPU/memory stressors, exit-code only).
//!
//! Each fixture is declared via
//! [`#[derive(Payload)]`](ktstr::Payload), the same path downstream
//! test authors use — so this module doubles as an end-to-end
//! exercise of the derive macro. The emitted `const` follows the
//! derive's naming convention: `struct FooPayload` produces
//! `const FOO: Payload`.
//!
//! These fixtures live under `tests/common/` rather than inside the
//! library's `src/` tree because they are TEST SCAFFOLDING, not
//! shipped API. A downstream scheduler author who wants the same
//! `fio` / `stress-ng` shapes should either copy the declarations
//! below into their own crate or write their own via
//! `#[derive(Payload)]`. The library does not ship fio or stress-ng
//! binaries — the `kind = PayloadKind::Binary(name)` just declares
//! the name; host-side include-files resolution picks the path up
//! at test time.
//!
//! The two fixtures illustrate the two ends of the
//! [`OutputFormat`](ktstr::test_support::OutputFormat) spectrum:
//!
//! - [`FIO`] and [`FIO_JSON`] declare `OutputFormat::Json` with a
//!   set of [`MetricHint`](ktstr::test_support::MetricHint)s
//!   describing the canonical read/write throughput + latency paths.
//!   Extracted metrics land with correct polarity/unit automatically.
//! - [`STRESS_NG`] uses `OutputFormat::ExitCode` with a single
//!   `exit_code_eq(0)` default — stress-ng reports via exit code
//!   (bogo_ops land in stderr and are not machine-extractable
//!   without `--metrics-brief --yaml`).
//!
//! Both fixtures use short, stable `name` fields (`"fio"`,
//! `"stress-ng"`) matching the binary names that ktstr's
//! include-files infrastructure resolves inside the guest.

use ktstr::Payload;

/// `fio` — flexible IO tester. Canonical workload for disk/IO
/// scheduler regressions.
///
/// Output format: JSON. Supply `--output-format=json` at the call
/// site (via `.arg(...)` on the runtime builder, or via a scheduler
/// default_args entry) or use [`FIO_JSON`] which bakes it into
/// `default_args` for the common "just give me metrics" path.
///
/// **Caveat:** `FIO` leaves `default_args` empty, so invoking it
/// without `--output-format=json` causes `fio` to emit its
/// human-readable output, `extract_metrics` finds no JSON region,
/// and the check pass records every referenced metric as missing
/// without otherwise failing. Prefer [`FIO_JSON`] unless the test
/// author intentionally overrides the output mode.
///
/// Metric hints cover the first-job read/write leaf names. Fio's
/// JSON output is deeply nested (`jobs[N].read.iops`,
/// `.write.iops`, `.read.lat_ns.mean`, etc.); the hints pin the
/// four most-commonly-asserted paths. Unhinted paths land as
/// [`Polarity::Unknown`](ktstr::test_support::Polarity::Unknown)
/// and are still extracted for sidecar regression tracking.
#[derive(Payload)]
#[payload(binary = "fio", output = Json)]
#[default_check(exit_code_eq(0))]
#[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.write.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.read.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[metric(name = "jobs.0.write.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[allow(dead_code)]
pub struct FioPayload;

/// `fio` with `--output-format=json` pre-baked into `default_args`.
///
/// Compared to [`FIO`], this fixture differs in exactly two
/// fields:
///
/// 1. **`name`** — `"fio_json"` instead of `"fio"`. The distinct
///    name lets both fixtures register as workloads in the same
///    `#[ktstr_test(workloads = [FIO, FIO_JSON])]` attribute
///    without hitting the pairwise-dedup rejection. The `binary`
///    field (the name resolved by the include-files
///    infrastructure) is still `"fio"` in both.
/// 2. **`default_args`** — `&["--output-format=json"]` instead of
///    `&[]`. Everything else — `kind`, `output`, `default_checks`,
///    `metrics` — is character-for-character identical to [`FIO`].
///
/// **Caveat: simultaneous FIO + FIO_JSON.** Both fixtures have
/// `kind = PayloadKind::Binary("fio")`, so a scenario that lists
/// `#[ktstr_test(workloads = [FIO, FIO_JSON])]` spawns the `fio`
/// binary TWICE — each with its own argv set, inside whatever
/// cgroup the framework places each fixture in. The pairwise-dedup
/// on the `workloads` attribute only rejects identical Payload
/// paths; two distinct Payload constants that happen to share a
/// binary are NOT deduped. Test authors who want the same fio
/// binary once should pick ONE of the two fixtures, and extend it
/// via `ctx.payload(&FIO).arg("--output-format=json")` if the
/// `FIO_JSON` preset's args don't match their scenario.
#[derive(Payload)]
#[payload(binary = "fio", name = "fio_json", output = Json)]
#[default_args("--output-format=json")]
#[default_check(exit_code_eq(0))]
#[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.write.iops", polarity = HigherBetter, unit = "iops")]
#[metric(name = "jobs.0.read.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[metric(name = "jobs.0.write.lat_ns.mean", polarity = LowerBetter, unit = "ns")]
#[allow(dead_code)]
pub struct FioJsonPayload;

/// `stress-ng` — synthetic load generator (CPU, memory, IO, VM,
/// etc.). Canonical workload for exercising scheduler decisions
/// under configurable contention.
///
/// Output format: `ExitCode`. stress-ng emits human-readable
/// progress lines to stderr; metrics require `--metrics-brief
/// --yaml`, which produces YAML on stderr (not stdout). Since the
/// extraction pipeline only consumes stdout, even `--yaml` does
/// not currently feed `extract_metrics`; the fixture stays in
/// exit-code mode and the happy path is a zero exit.
///
/// **Caveat:** `default_args` is empty, so invoking `STRESS_NG`
/// without at least one stressor flag (e.g. `--cpu 1`, `--vm 1`)
/// causes stress-ng to print usage and exit nonzero on some
/// versions. Always append a stressor via `.arg(...)` on the
/// runtime builder.
///
/// Tests that want bogo_ops/sec metrics should declare their own
/// custom `Payload` via [`#[derive(Payload)]`](ktstr::Payload) and
/// pair it with a post-hoc stderr-to-stdout bridge, or wait for
/// the LlmExtract backend to land and declare
/// `output = LlmExtract("bogo ops")`.
#[derive(Payload)]
#[payload(binary = "stress-ng")]
#[default_check(exit_code_eq(0))]
#[allow(dead_code)]
pub struct StressNgPayload;
