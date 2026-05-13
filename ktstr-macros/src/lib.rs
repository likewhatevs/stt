use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Data, DeriveInput, Fields, ItemFn, Meta, MetaNameValue, parse::Parser, parse_macro_input,
};

// `kernel_path` is mirrored from the parent ktstr crate via
// `build.rs` (see that file for the rationale). It exposes
// `KernelId::parse` + `KernelId::validate` — the same parser the
// verifier uses at runtime — so `declare_scheduler!` can reject
// obviously-malformed `kernels = [..]` entries at macro expand time
// instead of letting them surface as "cache key not found" errors
// inside the verifier.
#[allow(dead_code)]
mod kernel_path {
    include!(concat!(env!("OUT_DIR"), "/kernel_path.rs"));
}

/// Emit `Some(value)` or `None` as token streams.
fn option_tokens<T: ToTokens>(opt: &Option<T>) -> proc_macro2::TokenStream {
    match opt {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    }
}

/// Default topology and memory for ktstr_test-annotated functions.
const DEFAULT_LLCS: u32 = 1;
const DEFAULT_CORES: u32 = 2;
const DEFAULT_THREADS: u32 = 1;
const DEFAULT_MEMORY_MB: u32 = 2048;

/// Attribute macro that registers a function as a ktstr integration test.
///
/// The annotated function must have signature `fn(&ktstr::scenario::Ctx) ->
/// anyhow::Result<ktstr::assert::AssertResult>`. The macro:
///
/// 1. Renames the original function to `__ktstr_inner_{name}`.
/// 2. Registers it in the `KTSTR_TESTS` distributed slice via linkme.
/// 3. Emits a `#[test]` wrapper that boots a VM and runs the function
///    inside it.
///
/// Every key=value attribute is optional. The accepted attributes and
/// their defaults are the fields of
/// [`ktstr::test_support::KtstrTestEntry`] (runtime metadata) and
/// [`ktstr::assert::Assert`] (checking thresholds). A few are
/// worth calling out because their names differ from the underlying
/// field or because they have nontrivial defaults:
///
///   - `llcs = N` — number of LLCs (default: inherited from
///     scheduler, or 1).
///   - `cores = N` (default: inherited from scheduler, or 2)
///   - `threads = N` (default: inherited from scheduler, or 1)
///   - `numa_nodes = N` (default: inherited from scheduler, or 1)
///   - `memory_mb = N` (default: 2048)
///   - `duration_s = N` — scenario run duration in seconds; maps
///     onto `KtstrTestEntry::duration`
///   - `watchdog_timeout_s = N` — watchdog fire threshold in
///     seconds; maps onto `KtstrTestEntry::watchdog_timeout`
///   - `cleanup_budget_ms = N` — sub-watchdog cap on host-side VM
///     teardown wall time; maps onto `KtstrTestEntry::cleanup_budget`
///     as `Duration::from_millis(N)`. Default: `None` (unenforced).
///   - `num_snapshots = N` — fire `N` periodic
///     `freeze_and_capture(false)` boundaries inside the workload's
///     10 %–90 % window, stored on the host
///     `SnapshotBridge` under `periodic_NNN`. `0` (default)
///     disables periodic capture entirely. Maps onto
///     `KtstrTestEntry::num_snapshots`; runtime
///     `KtstrTestEntry::validate` rejects values past the bridge
///     cap (`MAX_STORED_SNAPSHOTS`), `host_only = true`, and
///     duration / `N` settings that would land boundaries closer
///     than 100 ms apart.
///   - `scheduler = PATH` — path to a `const Scheduler` (typically
///     produced by `declare_scheduler!(...)`). Maps onto
///     `KtstrTestEntry::scheduler`, which is typed
///     `&'static Scheduler`. Default: `&Scheduler::EEVDF`, the
///     no-scx placeholder that runs under the kernel's default
///     scheduler.
///   - `payload = PATH` — path to a `const Payload` used as the
///     primary binary workload (must be `PayloadKind::Binary`;
///     runtime-enforced). Default: `None` (scheduler-only test).
///     Coexists with `scheduler = PATH` — the payload runs *under*
///     the selected scheduler.
///   - `workloads = [PATH, PATH, ...]` — additional `const Payload`
///     references composed with the primary via `Ctx::payload` in
///     the test body. Default: `&[]`. Must not contain the same
///     path as `payload` — reject at expansion time to catch the
///     common "fio as primary AND workload" slip.
///   - `auto_repro = bool` (default: `true`)
///   - `host_only = bool` (default: `false`) — run the test function
///     on the host instead of inside a VM
///   - `no_perf_mode = bool` (default: `false`) — decouple the
///     virtual topology from host hardware. The VM is built with
///     the declared `numa_nodes` / `llcs` / `cores` / `threads`
///     even on smaller hosts; vCPU pinning, hugepages, NUMA mbind,
///     RT scheduling, and KVM exit suppression are skipped, and
///     gauntlet preset filtering relaxes host-topology checks
///     to the single "host has enough total CPUs" inequality.
///     Mutually exclusive with `performance_mode = true`. Maps onto
///     `KtstrTestEntry::no_perf_mode`.
///   - `post_vm = PATH` — host-side callback invoked after
///     `vm.run()` returns, with access to the full `VmResult`.
///     Use for assertions that need host-side state — e.g.
///     draining `VmResult.snapshot_bridge` after a snapshot
///     capture pipeline fires inside the guest. The function
///     must have signature
///     `fn(&ktstr::vmm::VmResult) -> anyhow::Result<()>`.
///     Default: `None` (no callback).
///   - `config = EXPR` — inline scheduler config content, written
///     into the guest at the path declared by the scheduler's
///     `config_file_def`. `EXPR` is either a string literal or a
///     path to a `const &'static str` (e.g. `LAYERED_CONFIG`).
///     Maps onto `KtstrTestEntry::config_content`. Required when
///     the scheduler declares `config_file_def`; rejected when the
///     scheduler does not. The pairing is enforced at compile time
///     via a `const` assertion against `Payload::config_file_def`,
///     and again at runtime by `KtstrTestEntry::validate` so direct
///     programmatic-entry construction sees the same gate.
#[proc_macro_attribute]
pub fn ktstr_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let orig_name = &input.sig.ident;
    let inner_name = format_ident!("__ktstr_inner_{}", orig_name);
    let entry_name = format_ident!("__KTSTR_ENTRY_{}", orig_name.to_string().to_uppercase());
    let name_str = orig_name.to_string();

    // Parse attributes
    let mut llcs = DEFAULT_LLCS;
    let mut cores = DEFAULT_CORES;
    let mut threads = DEFAULT_THREADS;
    let mut numa_nodes: u32 = 1;
    let mut llcs_set = false;
    let mut cores_set = false;
    let mut threads_set = false;
    let mut numa_nodes_set = false;
    let mut memory_mb = DEFAULT_MEMORY_MB;
    let mut memory_mb_set = false;
    let mut scheduler: Option<syn::Path> = None;
    let mut payload: Option<syn::Path> = None;
    let mut payload_set = false;
    let mut workloads: Vec<syn::Path> = Vec::new();
    let mut workloads_set = false;
    let mut auto_repro = true;
    let mut auto_repro_set = false;
    let mut not_starved: Option<bool> = None;
    let mut isolation: Option<bool> = None;
    let mut max_gap_ms: Option<u64> = None;
    let mut max_spread_pct: Option<f64> = None;
    let mut max_imbalance_ratio: Option<f64> = None;
    let mut max_local_dsq_depth: Option<u32> = None;
    let mut fail_on_stall: Option<bool> = None;
    let mut sustained_samples: Option<usize> = None;
    let mut max_throughput_cv: Option<f64> = None;
    let mut min_work_rate: Option<f64> = None;
    let mut max_fallback_rate: Option<f64> = None;
    let mut max_keep_last_rate: Option<f64> = None;
    let mut max_p99_wake_latency_ns: Option<u64> = None;
    let mut max_wake_latency_cv: Option<f64> = None;
    let mut min_iteration_rate: Option<f64> = None;
    let mut max_migration_ratio: Option<f64> = None;
    let mut min_page_locality: Option<f64> = None;
    let mut max_cross_node_migration_ratio: Option<f64> = None;
    let mut max_slow_tier_ratio: Option<f64> = None;
    let mut extra_sched_args: Vec<String> = Vec::new();
    let mut extra_include_files: Vec<String> = Vec::new();
    let mut min_numa_nodes: u32 = 1;
    let mut min_numa_nodes_set = false;
    let mut min_llcs: u32 = 1;
    let mut min_llcs_set = false;
    let mut requires_smt: bool = false;
    let mut requires_smt_set = false;
    let mut min_cpus: u32 = 1;
    let mut min_cpus_set = false;
    let mut max_llcs: Option<u32> = Some(12);
    let mut max_llcs_set = false;
    let mut max_numa_nodes: Option<u32> = Some(1);
    let mut max_numa_nodes_set = false;
    let mut max_cpus: Option<u32> = Some(192);
    let mut max_cpus_set = false;
    let mut watchdog_timeout_s: u64 = 4;
    let mut watchdog_timeout_s_set = false;
    let mut performance_mode: bool = false;
    let mut performance_mode_set = false;
    let mut no_perf_mode: bool = false;
    let mut no_perf_mode_set = false;
    let mut duration_s: u64 = 2;
    let mut duration_s_set = false;
    let mut workers_per_cgroup: u32 = 2;
    let mut workers_per_cgroup_set = false;
    let mut num_snapshots: u32 = 0;
    let mut num_snapshots_set = false;
    let mut bpf_map_write: Option<syn::Path> = None;
    let mut expect_err: bool = false;
    let mut expect_err_set = false;
    let mut host_only: bool = false;
    let mut host_only_set = false;
    let mut cleanup_budget_ms: Option<u64> = None;
    let mut post_vm: Option<syn::Path> = None;
    let mut config_expr: Option<proc_macro2::TokenStream> = None;
    let mut config_set = false;

    let attr_parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
    let parsed_attrs = match attr_parser.parse(attr) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    for meta in &parsed_attrs {
        match meta {
            Meta::NameValue(MetaNameValue { path, value, .. }) => {
                let ident = match path.get_ident() {
                    Some(id) => id.to_string(),
                    None => {
                        return syn::Error::new_spanned(path, "expected identifier")
                            .to_compile_error()
                            .into();
                    }
                };
                match ident.as_str() {
                    "scheduler" => {
                        let p = match value {
                            syn::Expr::Path(ep) => ep.path.clone(),
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected path for scheduler (e.g. MITOSIS or crate::MITOSIS)",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        scheduler = Some(p);
                    }
                    "payload" => {
                        if payload_set {
                            return syn::Error::new_spanned(
                                path,
                                "duplicate `payload = ...` — each test declares at \
                                 most one primary payload; extras belong in \
                                 `workloads = [..]`",
                            )
                            .to_compile_error()
                            .into();
                        }
                        let p = match value {
                            syn::Expr::Path(ep) => ep.path.clone(),
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected path for payload (e.g. FIO or crate::FIO)",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        payload = Some(p);
                        payload_set = true;
                    }
                    "workloads" => {
                        if workloads_set {
                            return syn::Error::new_spanned(
                                path,
                                "duplicate `workloads = [...]` — combine all \
                                 entries into a single array",
                            )
                            .to_compile_error()
                            .into();
                        }
                        let arr = match value {
                            syn::Expr::Array(ea) => ea,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected array of Payload paths for workloads \
                                     (e.g. [FIO, STRESS_NG])",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        for elem in &arr.elems {
                            match elem {
                                syn::Expr::Path(ep) => workloads.push(ep.path.clone()),
                                _ => {
                                    return syn::Error::new_spanned(
                                        elem,
                                        "expected Payload path in workloads array",
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            }
                        }
                        workloads_set = true;
                    }
                    "bpf_map_write" => {
                        let p = match value {
                            syn::Expr::Path(ep) => ep.path.clone(),
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected path for bpf_map_write (e.g. BPF_CRASH)",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        bpf_map_write = Some(p);
                    }
                    "post_vm" => {
                        let p = match value {
                            syn::Expr::Path(ep) => ep.path.clone(),
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected path for post_vm (e.g. my_post_vm_check)",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        post_vm = Some(p);
                    }
                    "config" => {
                        if config_set {
                            return syn::Error::new_spanned(
                                path,
                                "duplicate `config = ...` — each test declares at \
                                 most one inline scheduler config",
                            )
                            .to_compile_error()
                            .into();
                        }
                        // Accept either a string literal (`config = "..."`) or a
                        // path to a `const &'static str` (`config = MY_CONFIG`).
                        // The field is `Option<&'static str>`, so any other
                        // expression shape would either not borrow as `'static`
                        // or fail to coerce — reject early with a targeted error
                        // instead of letting rustc surface a confusing borrow /
                        // type-mismatch diagnostic at the spread site.
                        let tokens = match value {
                            syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(_),
                                ..
                            }) => quote! { #value },
                            syn::Expr::Path(_) => quote! { #value },
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected string literal or path to a \
                                     `const &'static str` for `config` (e.g. \
                                     `config = \"{...}\"` or `config = MY_CONFIG`)",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        config_expr = Some(tokens);
                        config_set = true;
                    }
                    "auto_repro" | "not_starved" | "isolation" | "performance_mode"
                    | "no_perf_mode" | "requires_smt" | "expect_err" | "fail_on_stall"
                    | "host_only" => {
                        let lit_bool = match value {
                            syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Bool(lb),
                                ..
                            }) => lb,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    format!("expected bool literal for {ident}"),
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        match ident.as_str() {
                            "auto_repro" => {
                                auto_repro = lit_bool.value();
                                auto_repro_set = true;
                            }
                            "not_starved" => not_starved = Some(lit_bool.value()),
                            "isolation" => isolation = Some(lit_bool.value()),
                            "performance_mode" => {
                                performance_mode = lit_bool.value();
                                performance_mode_set = true;
                            }
                            "no_perf_mode" => {
                                no_perf_mode = lit_bool.value();
                                no_perf_mode_set = true;
                            }
                            "requires_smt" => {
                                requires_smt = lit_bool.value();
                                requires_smt_set = true;
                            }
                            "expect_err" => {
                                expect_err = lit_bool.value();
                                expect_err_set = true;
                            }
                            "fail_on_stall" => fail_on_stall = Some(lit_bool.value()),
                            "host_only" => {
                                host_only = lit_bool.value();
                                host_only_set = true;
                            }
                            _ => unreachable!(),
                        }
                    }
                    "llcs"
                    | "cores"
                    | "threads"
                    | "numa_nodes"
                    | "memory_mb"
                    | "sustained_samples"
                    | "max_gap_ms"
                    | "watchdog_timeout_s"
                    | "duration_s"
                    | "workers_per_cgroup"
                    | "max_local_dsq_depth"
                    | "min_numa_nodes"
                    | "min_llcs"
                    | "min_cpus"
                    | "max_llcs"
                    | "max_numa_nodes"
                    | "max_cpus"
                    | "max_p99_wake_latency_ns"
                    | "cleanup_budget_ms"
                    | "num_snapshots" => {
                        let lit_int = match value {
                            syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Int(li),
                                ..
                            }) => li,
                            _ => {
                                return syn::Error::new_spanned(value, "expected integer literal")
                                    .to_compile_error()
                                    .into();
                            }
                        };
                        match ident.as_str() {
                            "llcs" => {
                                llcs = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                llcs_set = true;
                            }
                            "numa_nodes" => {
                                numa_nodes = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                numa_nodes_set = true;
                            }
                            "cores" => {
                                cores = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                cores_set = true;
                            }
                            "threads" => {
                                threads = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                threads_set = true;
                            }
                            "memory_mb" => {
                                memory_mb = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                memory_mb_set = true;
                            }
                            "sustained_samples" => {
                                sustained_samples = Some(
                                    lit_int
                                        .base10_parse::<usize>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                )
                            }
                            "max_gap_ms" => {
                                max_gap_ms = Some(
                                    lit_int
                                        .base10_parse::<u64>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                )
                            }
                            "cleanup_budget_ms" => {
                                cleanup_budget_ms = Some(
                                    lit_int
                                        .base10_parse::<u64>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                )
                            }
                            "watchdog_timeout_s" => {
                                watchdog_timeout_s = lit_int
                                    .base10_parse::<u64>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                watchdog_timeout_s_set = true;
                            }
                            "duration_s" => {
                                duration_s = lit_int
                                    .base10_parse::<u64>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                duration_s_set = true;
                            }
                            "workers_per_cgroup" => {
                                workers_per_cgroup = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                workers_per_cgroup_set = true;
                            }
                            "num_snapshots" => {
                                num_snapshots = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                num_snapshots_set = true;
                            }
                            "max_local_dsq_depth" => {
                                max_local_dsq_depth = Some(
                                    lit_int
                                        .base10_parse::<u32>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                )
                            }
                            "min_numa_nodes" => {
                                min_numa_nodes = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                min_numa_nodes_set = true;
                            }
                            "min_llcs" => {
                                min_llcs = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                min_llcs_set = true;
                            }
                            "min_cpus" => {
                                min_cpus = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                min_cpus_set = true;
                            }
                            "max_llcs" => {
                                max_llcs = Some(
                                    lit_int
                                        .base10_parse::<u32>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                );
                                max_llcs_set = true;
                            }
                            "max_numa_nodes" => {
                                max_numa_nodes = Some(
                                    lit_int
                                        .base10_parse::<u32>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                );
                                max_numa_nodes_set = true;
                            }
                            "max_cpus" => {
                                max_cpus = Some(
                                    lit_int
                                        .base10_parse::<u32>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                );
                                max_cpus_set = true;
                            }
                            "max_p99_wake_latency_ns" => {
                                max_p99_wake_latency_ns = Some(
                                    lit_int
                                        .base10_parse::<u64>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                )
                            }
                            _ => unreachable!(),
                        }
                    }
                    "max_imbalance_ratio"
                    | "max_fallback_rate"
                    | "max_keep_last_rate"
                    | "max_spread_pct"
                    | "max_throughput_cv"
                    | "min_work_rate"
                    | "max_wake_latency_cv"
                    | "min_iteration_rate"
                    | "max_migration_ratio"
                    | "min_page_locality"
                    | "max_cross_node_migration_ratio"
                    | "max_slow_tier_ratio" => {
                        let lit_float = match value {
                            syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Float(lf),
                                ..
                            }) => lf,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    format!("expected float literal for {ident}"),
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        let v = lit_float
                            .base10_parse::<f64>()
                            .unwrap_or_else(|e| panic!("{e}"));
                        match ident.as_str() {
                            "max_imbalance_ratio" => max_imbalance_ratio = Some(v),
                            "max_fallback_rate" => max_fallback_rate = Some(v),
                            "max_keep_last_rate" => max_keep_last_rate = Some(v),
                            "max_spread_pct" => max_spread_pct = Some(v),
                            "max_throughput_cv" => max_throughput_cv = Some(v),
                            "min_work_rate" => min_work_rate = Some(v),
                            "max_wake_latency_cv" => max_wake_latency_cv = Some(v),
                            "min_iteration_rate" => min_iteration_rate = Some(v),
                            "max_migration_ratio" => max_migration_ratio = Some(v),
                            "min_page_locality" => min_page_locality = Some(v),
                            "max_cross_node_migration_ratio" => {
                                max_cross_node_migration_ratio = Some(v)
                            }
                            "max_slow_tier_ratio" => max_slow_tier_ratio = Some(v),
                            _ => unreachable!(),
                        }
                    }
                    "extra_sched_args" => {
                        let arr = match value {
                            syn::Expr::Array(ea) => ea,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected array of string literals for extra_sched_args",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        for elem in &arr.elems {
                            match elem {
                                syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(ls),
                                    ..
                                }) => extra_sched_args.push(ls.value()),
                                _ => {
                                    return syn::Error::new_spanned(
                                        elem,
                                        "expected string literal in extra_sched_args",
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            }
                        }
                    }
                    "extra_include_files" => {
                        let arr = match value {
                            syn::Expr::Array(ea) => ea,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected array of string literals for extra_include_files",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        for elem in &arr.elems {
                            match elem {
                                syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(ls),
                                    ..
                                }) => extra_include_files.push(ls.value()),
                                _ => {
                                    return syn::Error::new_spanned(
                                        elem,
                                        "expected string literal in extra_include_files",
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            }
                        }
                    }
                    _ => {
                        return syn::Error::new_spanned(
                            path,
                            format!("unknown attribute `{ident}`, expected: llcs, cores, threads, numa_nodes, memory_mb, scheduler, payload, workloads, auto_repro, not_starved, isolation, max_gap_ms, max_spread_pct, max_throughput_cv, min_work_rate, max_p99_wake_latency_ns, max_wake_latency_cv, min_iteration_rate, max_migration_ratio, max_imbalance_ratio, max_local_dsq_depth, fail_on_stall, sustained_samples, max_fallback_rate, max_keep_last_rate, min_page_locality, max_cross_node_migration_ratio, max_slow_tier_ratio, extra_sched_args, min_numa_nodes, min_llcs, requires_smt, min_cpus, max_llcs, max_numa_nodes, max_cpus, watchdog_timeout_s, performance_mode, no_perf_mode, duration_s, workers_per_cgroup, bpf_map_write, expect_err, host_only, cleanup_budget_ms, post_vm, config, num_snapshots"),
                        )
                        .to_compile_error()
                        .into();
                    }
                }
            }
            other => {
                return syn::Error::new_spanned(other, "expected `key = value`")
                    .to_compile_error()
                    .into();
            }
        }
    }

    // Mutual exclusion: the primary payload must not also appear
    // in the workloads array. A Rust `syn::Path` is compared via
    // `ToTokens` so `FIO` and `crate::FIO` remain distinct strings —
    // this catches the common in-file alias case (same ident both
    // places) but not resolved-path-identity, which is impossible
    // to verify at macro expansion time. Runtime validation can
    // add path-deduplication after `payload`/`workloads` are read
    // back from the registered KtstrTestEntry.
    if let Some(primary) = payload.as_ref() {
        let primary_repr = primary.to_token_stream().to_string();
        for w in &workloads {
            if w.to_token_stream().to_string() == primary_repr {
                return syn::Error::new_spanned(
                    w,
                    format!(
                        "`{primary_repr}` appears in both `payload = ...` and \
                         `workloads = [..]` — pick one. The primary payload \
                         runs as the test's main workload; entries in \
                         `workloads` are composed alongside it."
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // Pairwise dedup inside the workloads array itself. `[FIO, FIO]`
    // is almost always a typo — a test author meant to list two
    // distinct payloads and accidentally repeated one. Same caveat
    // as the payload/workloads cross-check: token-string equality
    // catches in-file aliases (`FIO` == `FIO`) but not resolved-path
    // identity (`FIO` vs `crate::FIO`).
    for (i, wi_path) in workloads.iter().enumerate() {
        let wi = wi_path.to_token_stream().to_string();
        for wj_path in workloads.iter().skip(i + 1) {
            let wj = wj_path.to_token_stream().to_string();
            if wi == wj {
                return syn::Error::new_spanned(
                    wj_path,
                    format!(
                        "`{wi}` appears twice in `workloads = [..]` — each \
                         workload entry must be distinct. Remove the \
                         duplicate or compose the payload once and rely \
                         on runtime scheduling to spread it across cgroups."
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // Reject zero values at compile time (only for explicitly set values).
    if llcs_set && llcs == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "llcs must be > 0 (a topology with zero LLCs has zero CPUs — \
             `total_cpus = llcs * cores * threads` — so the VM would boot \
             with no addressable processors)",
        )
        .to_compile_error()
        .into();
    }
    if cores_set && cores == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "cores must be > 0 (a topology with zero cores per LLC has \
             zero CPUs — `total_cpus = llcs * cores * threads` — so the \
             VM would boot with no addressable processors)",
        )
        .to_compile_error()
        .into();
    }
    if threads_set && threads == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "threads must be > 0 (a topology with zero threads per core \
             has zero CPUs — `total_cpus = llcs * cores * threads` — so \
             the VM would boot with no addressable processors)",
        )
        .to_compile_error()
        .into();
    }
    if numa_nodes_set && numa_nodes == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "numa_nodes must be > 0 (a topology with zero NUMA nodes has \
             nothing to attach LLCs or memory to; every downstream \
             accessor would observe an empty node set)",
        )
        .to_compile_error()
        .into();
    }
    if llcs_set && numa_nodes_set && !llcs.is_multiple_of(numa_nodes) {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("llcs ({llcs}) must be divisible by numa_nodes ({numa_nodes})"),
        )
        .to_compile_error()
        .into();
    }
    if memory_mb == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "memory_mb must be > 0 (a VM with zero memory cannot boot)",
        )
        .to_compile_error()
        .into();
    }
    if duration_s == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "duration_s must be > 0 (a zero-duration run never exercises the \
             scheduler and produces no data for assertions)",
        )
        .to_compile_error()
        .into();
    }
    if workers_per_cgroup == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "workers_per_cgroup must be > 0 (a zero-worker cgroup emits no \
             WorkerReports and assertions vacuously pass)",
        )
        .to_compile_error()
        .into();
    }
    if cleanup_budget_ms == Some(0) {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "cleanup_budget_ms must be > 0 — a zero budget would \
             reject every successful run (any measurable cleanup \
             duration overshoots zero). Omit the attribute to \
             disable the check.",
        )
        .to_compile_error()
        .into();
    }
    // Validate explicitly set constraint values. When a field is
    // inherited from the scheduler, the proc macro doesn't know the
    // value so cross-field validation is deferred to runtime.
    if max_llcs_set && max_llcs == Some(0) {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "max_llcs must be > 0 (a zero cap excludes every host from \
             the gauntlet — use a non-zero cap, or omit the field to \
             use the default)",
        )
        .to_compile_error()
        .into();
    }
    if max_numa_nodes_set && max_numa_nodes == Some(0) {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "max_numa_nodes must be > 0 (a zero cap excludes every host \
             from the gauntlet — use a non-zero cap, or omit the field \
             to inherit the scheduler-level default)",
        )
        .to_compile_error()
        .into();
    }
    if max_cpus_set && max_cpus == Some(0) {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "max_cpus must be > 0 (a zero cap excludes every host from \
             the gauntlet — use a non-zero cap, or omit the field to \
             use the default)",
        )
        .to_compile_error()
        .into();
    }
    if min_llcs_set && max_llcs_set && matches!(max_llcs, Some(m) if m < min_llcs) {
        let m = max_llcs.unwrap();
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("min_llcs ({min_llcs}) exceeds max_llcs ({m}). Set max_llcs explicitly."),
        )
        .to_compile_error()
        .into();
    }
    if min_numa_nodes_set
        && max_numa_nodes_set
        && matches!(max_numa_nodes, Some(m) if m < min_numa_nodes)
    {
        let m = max_numa_nodes.unwrap();
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "min_numa_nodes ({min_numa_nodes}) exceeds max_numa_nodes ({m}). Set max_numa_nodes explicitly."
            ),
        )
        .to_compile_error()
        .into();
    }
    if min_cpus_set && max_cpus_set && matches!(max_cpus, Some(m) if m < min_cpus) {
        let m = max_cpus.unwrap();
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("min_cpus ({min_cpus}) exceeds max_cpus ({m}). Set max_cpus explicitly."),
        )
        .to_compile_error()
        .into();
    }

    // Build the scheduler reference token. The `scheduler` slot on
    // `KtstrTestEntry` is `&'static Scheduler`; callers pass either a
    // `NAME` const emitted by `declare_scheduler!` or `Scheduler::EEVDF`
    // directly. The default is the kernel-default EEVDF placeholder.
    let scheduler_tokens = match &scheduler {
        Some(p) => {
            quote! { &#p }
        }
        None => {
            quote! { &::ktstr::test_support::Scheduler::EEVDF }
        }
    };

    // Build topology tokens. Each dimension independently inherits from
    // the scheduler's topology when not explicitly set. `Scheduler.topology`
    // is a direct field (a `Topology` struct), so the field-of-field
    // access below remains valid inside a `const` initializer.
    let llcs_tokens = if llcs_set {
        let l = llcs;
        quote! { #l }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology.llcs }
    } else {
        let l = llcs;
        quote! { #l }
    };
    let cores_tokens = if cores_set {
        let c = cores;
        quote! { #c }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology.cores_per_llc }
    } else {
        let c = cores;
        quote! { #c }
    };
    let threads_tokens = if threads_set {
        let t = threads;
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology.threads_per_core }
    } else {
        let t = threads;
        quote! { #t }
    };
    let numa_nodes_tokens = if numa_nodes_set {
        let n = numa_nodes;
        quote! { #n }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology.numa_nodes }
    } else {
        let n = numa_nodes;
        quote! { #n }
    };
    let topology_tokens = quote! {
        ::ktstr::test_support::Topology {
            llcs: #llcs_tokens,
            cores_per_llc: #cores_tokens,
            threads_per_core: #threads_tokens,
            numa_nodes: #numa_nodes_tokens,
            nodes: None,
            distances: None,
        }
    };

    // Build the renamed inner function
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;
    let attrs = &input.attrs;
    let inner_sig = syn::Signature {
        ident: inner_name.clone(),
        ..sig.clone()
    };

    // Build Assert field tokens.
    let not_starved_tokens = option_tokens(&not_starved);
    let isolation_tokens = option_tokens(&isolation);
    let gap_tokens = option_tokens(&max_gap_ms);
    let spread_tokens = option_tokens(&max_spread_pct);
    let imbalance_tokens = option_tokens(&max_imbalance_ratio);
    let dsq_tokens = option_tokens(&max_local_dsq_depth);
    let stall_tokens = option_tokens(&fail_on_stall);
    let sustained_tokens = option_tokens(&sustained_samples);
    let throughput_cv_tokens = option_tokens(&max_throughput_cv);
    let work_rate_tokens = option_tokens(&min_work_rate);
    let fallback_rate_tokens = option_tokens(&max_fallback_rate);
    let keep_last_rate_tokens = option_tokens(&max_keep_last_rate);
    let p99_wake_tokens = option_tokens(&max_p99_wake_latency_ns);
    let wake_cv_tokens = option_tokens(&max_wake_latency_cv);
    let iter_rate_tokens = option_tokens(&min_iteration_rate);
    let mig_ratio_tokens = option_tokens(&max_migration_ratio);
    let page_locality_tokens = option_tokens(&min_page_locality);
    let cross_node_mig_tokens = option_tokens(&max_cross_node_migration_ratio);
    let slow_tier_tokens = option_tokens(&max_slow_tier_ratio);

    // `cleanup_budget_ms` lives on the macro side as `Option<u64>` of
    // milliseconds; the entry field is `Option<Duration>`, so wrap
    // the literal in `Duration::from_millis(...)` at emission time.
    let cleanup_budget_tokens = match cleanup_budget_ms {
        Some(ms) => {
            quote! { ::core::option::Option::Some(::std::time::Duration::from_millis(#ms)) }
        }
        None => quote! { ::core::option::Option::None },
    };

    let bpf_map_write_tokens = match &bpf_map_write {
        Some(p) => quote! { &[&#p] },
        None => quote! { &[] },
    };

    // Emit `Option<&'static Payload>` for the primary payload. The
    // user supplies a path (`&FIO` equivalent in source), so we
    // wrap it in `Some(&#p)` at emission time to preserve the
    // `entry.payload: Option<&'static Payload>` field type.
    let payload_tokens = match &payload {
        Some(p) => quote! { Some(&#p) },
        None => quote! { None },
    };
    // Emit `&'static [&'static Payload]` for workloads. Each path
    // the user supplied is a `const Payload`; we take `&` on each
    // to match the stored type.
    let workload_refs: Vec<proc_macro2::TokenStream> =
        workloads.iter().map(|p| quote! { &#p }).collect();
    let workloads_tokens = quote! { &[#(#workload_refs),*] };

    // Conditionally-emitted KtstrTestEntry fields. Each block is
    // either an empty TokenStream (so the field is left to
    // `..KtstrTestEntry::DEFAULT` in the spread) or a `field: VAL,`
    // pair when the macro must override the default. This pattern
    // means new struct fields with sane defaults need no macro
    // change — adding to KtstrTestEntry::DEFAULT alone is enough.
    let memory_mb_field = if memory_mb_set {
        quote! { memory_mb: #memory_mb, }
    } else {
        quote! {}
    };
    let payload_field = if payload_set {
        quote! { payload: #payload_tokens, }
    } else {
        quote! {}
    };
    let workloads_field = if workloads_set {
        quote! { workloads: #workloads_tokens, }
    } else {
        quote! {}
    };
    let auto_repro_field = if auto_repro_set {
        quote! { auto_repro: #auto_repro, }
    } else {
        quote! {}
    };
    // Any of the per-check assert fields supplied by the attribute
    // forces emission of the full `assert: Assert { .. }` block. When
    // none are set the spread inherits `Assert::NO_OVERRIDES` from
    // `KtstrTestEntry::DEFAULT`, which is bit-for-bit identical to
    // the all-`None` Assert the prior unconditional emission produced.
    let any_assert_set = not_starved.is_some()
        || isolation.is_some()
        || max_gap_ms.is_some()
        || max_spread_pct.is_some()
        || max_throughput_cv.is_some()
        || min_work_rate.is_some()
        || max_p99_wake_latency_ns.is_some()
        || max_wake_latency_cv.is_some()
        || min_iteration_rate.is_some()
        || max_migration_ratio.is_some()
        || max_imbalance_ratio.is_some()
        || max_local_dsq_depth.is_some()
        || fail_on_stall.is_some()
        || sustained_samples.is_some()
        || max_fallback_rate.is_some()
        || max_keep_last_rate.is_some()
        || min_page_locality.is_some()
        || max_cross_node_migration_ratio.is_some()
        || max_slow_tier_ratio.is_some();
    let assert_field = if any_assert_set {
        quote! {
            assert: ::ktstr::assert::Assert {
                not_starved: #not_starved_tokens,
                isolation: #isolation_tokens,
                max_gap_ms: #gap_tokens,
                max_spread_pct: #spread_tokens,
                max_throughput_cv: #throughput_cv_tokens,
                min_work_rate: #work_rate_tokens,
                max_p99_wake_latency_ns: #p99_wake_tokens,
                max_wake_latency_cv: #wake_cv_tokens,
                min_iteration_rate: #iter_rate_tokens,
                max_migration_ratio: #mig_ratio_tokens,
                max_imbalance_ratio: #imbalance_tokens,
                max_local_dsq_depth: #dsq_tokens,
                fail_on_stall: #stall_tokens,
                sustained_samples: #sustained_tokens,
                max_fallback_rate: #fallback_rate_tokens,
                max_keep_last_rate: #keep_last_rate_tokens,
                min_page_locality: #page_locality_tokens,
                max_cross_node_migration_ratio: #cross_node_mig_tokens,
                max_slow_tier_ratio: #slow_tier_tokens,
            },
        }
    } else {
        quote! {}
    };
    let extra_sched_args_field = if extra_sched_args.is_empty() {
        quote! {}
    } else {
        quote! { extra_sched_args: &[#(#extra_sched_args),*], }
    };
    let watchdog_timeout_field = if watchdog_timeout_s_set {
        quote! { watchdog_timeout: ::std::time::Duration::from_secs(#watchdog_timeout_s), }
    } else {
        quote! {}
    };
    let bpf_map_write_field = if bpf_map_write.is_some() {
        quote! { bpf_map_write: #bpf_map_write_tokens, }
    } else {
        quote! {}
    };
    let performance_mode_field = if performance_mode_set {
        quote! { performance_mode: #performance_mode, }
    } else {
        quote! {}
    };
    let no_perf_mode_field = if no_perf_mode_set {
        quote! { no_perf_mode: #no_perf_mode, }
    } else {
        quote! {}
    };
    let duration_field = if duration_s_set {
        quote! { duration: ::std::time::Duration::from_secs(#duration_s), }
    } else {
        quote! {}
    };
    let workers_per_cgroup_field = if workers_per_cgroup_set {
        quote! { workers_per_cgroup: #workers_per_cgroup, }
    } else {
        quote! {}
    };
    let num_snapshots_field = if num_snapshots_set {
        quote! { num_snapshots: #num_snapshots, }
    } else {
        quote! {}
    };
    let expect_err_field = if expect_err_set {
        quote! { expect_err: #expect_err, }
    } else {
        quote! {}
    };
    let host_only_field = if host_only_set {
        quote! { host_only: #host_only, }
    } else {
        quote! {}
    };
    let extra_include_files_field = if extra_include_files.is_empty() {
        quote! {}
    } else {
        quote! { extra_include_files: &[#(#extra_include_files),*], }
    };
    let cleanup_budget_field = if cleanup_budget_ms.is_some() {
        quote! { cleanup_budget: #cleanup_budget_tokens, }
    } else {
        quote! {}
    };
    // The user-supplied path resolves to a `fn(&VmResult) ->
    // Result<()>`. Wrap in `Some(...)` so the entry's
    // `Option<fn(&VmResult) -> Result<()>>` field accepts it.
    let post_vm_field = if let Some(ref p) = post_vm {
        quote! { post_vm: Some(#p), }
    } else {
        quote! {}
    };
    // `config = EXPR` lands in `KtstrTestEntry::config_content`, which
    // is `Option<&'static str>`. Wrap the user-supplied expression in
    // `Some(...)` at emission so the spread site sees a typed Option.
    let config_content_field = if let Some(ref tokens) = config_expr {
        quote! { config_content: ::core::option::Option::Some(#tokens), }
    } else {
        quote! {}
    };

    // Compile-time assert: `config = ...` must be paired with a
    // scheduler that declares `config_file_def`, and vice versa. The
    // macro can't read the scheduler const's value (it sees only a
    // path), but both `Payload::config_file_def` and `Option::is_some`
    // are `const fn`, so a `const _: () = assert!(...)` block can
    // verify the pairing at compile time. The `KtstrTestEntry::validate`
    // method enforces the same gate at runtime so direct programmatic
    // construction doesn't bypass the macro path.
    let config_set_lit = config_set;
    let pairing_assert_const_name = format_ident!(
        "__KTSTR_CONFIG_PAIRING_{}",
        orig_name.to_string().to_uppercase()
    );
    let pairing_assert = quote! {
        const #pairing_assert_const_name: () = {
            let has_def = (#scheduler_tokens).config_file_def.is_some();
            let has_content: bool = #config_set_lit;
            if has_def && !has_content {
                panic!(
                    "scheduler declares `config_file_def` but the test \
                     does not supply `config = ...`; provide an inline \
                     scheduler config or remove `config_file_def` from \
                     the scheduler definition"
                );
            }
            if !has_def && has_content {
                panic!(
                    "test supplies `config = ...` but the scheduler does \
                     not declare `config_file_def`; remove `config = ...` \
                     or add `config_file_def(arg_template, guest_path)` \
                     to the scheduler definition"
                );
            }
        };
    };

    let test_body = if expect_err {
        quote! {
            match ::ktstr::test_support::run_ktstr_test(&#entry_name) {
                Ok(_) => panic!("expected test to fail but it passed"),
                Err(e) if ::ktstr::test_support::is_kernel_unavailable(&e) => {
                    // Harness not configured (no kernel resolved):
                    // running outside `cargo ktstr test` produces no
                    // expected failure either, because the test
                    // never ran. Skip cleanly so a developer running
                    // `cargo nextest run` directly sees a SKIP
                    // banner rather than a confusing "no kernel
                    // found" panic.
                    eprintln!("ktstr: SKIP: harness not configured: {e:#}");
                    return;
                }
                Err(e) if ::ktstr::test_support::is_resource_contention(&e) => {
                    // Resource contention is host-infra, not a test
                    // outcome: emit the canonical SKIP banner and
                    // early-return so libtest sees pass. Otherwise an
                    // expect_err test would mask host contention as a
                    // satisfied "expected failure", hiding the
                    // contention from stats tooling and producing a
                    // false-positive pass for the wrong reason.
                    //
                    // KTSTR_NO_SKIP_MODE inverts the policy: CI runs
                    // that demand every test execute against the
                    // available hardware promote contention to a
                    // hard failure so a misconfigured host surfaces
                    // instead of silently passing.
                    if ::std::env::var_os("KTSTR_NO_SKIP_MODE").is_some() {
                        panic!(
                            "ktstr: FAIL: resource contention under --no-skip-mode: {e:#}. \
                             Either provision hardware that satisfies the test's topology \
                             requirement, or drop --no-skip-mode / KTSTR_NO_SKIP_MODE to \
                             accept the skip."
                        );
                    }
                    eprintln!("ktstr: SKIP: resource contention: {e:#}");
                    return;
                }
                Err(_) => {}
            }
        }
    } else {
        quote! {
            match ::ktstr::test_support::run_ktstr_test(&#entry_name) {
                Ok(_) => {}
                Err(e) if ::ktstr::test_support::is_kernel_unavailable(&e) => {
                    // Harness not configured (no kernel resolved):
                    // the binary was likely invoked outside
                    // `cargo ktstr test`, which builds and injects a
                    // kernel automatically. Skip cleanly so a
                    // developer running `cargo nextest run` directly
                    // sees a SKIP banner rather than a confusing
                    // "no kernel found" panic.
                    eprintln!("ktstr: SKIP: harness not configured: {e:#}");
                    return;
                }
                Err(e) if ::ktstr::test_support::is_resource_contention(&e) => {
                    // Resource contention is host-infra, not a test
                    // failure: emit the canonical SKIP banner and
                    // early-return so libtest sees pass. The skip
                    // sidecar is recorded inside `run_ktstr_test_inner`
                    // at every contention site, so stats tooling still
                    // sees the skip without a panic-driven retry.
                    //
                    // KTSTR_NO_SKIP_MODE inverts the policy: CI runs
                    // that demand every test execute against the
                    // available hardware promote contention to a
                    // hard failure so a misconfigured host surfaces
                    // instead of silently passing.
                    if ::std::env::var_os("KTSTR_NO_SKIP_MODE").is_some() {
                        panic!(
                            "ktstr: FAIL: resource contention under --no-skip-mode: {e:#}. \
                             Either provision hardware that satisfies the test's topology \
                             requirement, or drop --no-skip-mode / KTSTR_NO_SKIP_MODE to \
                             accept the skip."
                        );
                    }
                    eprintln!("ktstr: SKIP: resource contention: {e:#}");
                    return;
                }
                Err(e) => panic!("{e:#}"),
            }
        }
    };

    // Build constraint tokens. Each field independently inherits from
    // the scheduler's constraints when not explicitly set, following
    // the same pattern as topology inheritance.
    let min_numa_nodes_tokens = if min_numa_nodes_set {
        let v = min_numa_nodes;
        quote! { #v }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints.min_numa_nodes }
    } else {
        let v = min_numa_nodes;
        quote! { #v }
    };
    let max_numa_nodes_tokens = if max_numa_nodes_set {
        let t = option_tokens(&max_numa_nodes);
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints.max_numa_nodes }
    } else {
        let t = option_tokens(&max_numa_nodes);
        quote! { #t }
    };
    let min_llcs_tokens = if min_llcs_set {
        let v = min_llcs;
        quote! { #v }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints.min_llcs }
    } else {
        let v = min_llcs;
        quote! { #v }
    };
    let max_llcs_tokens = if max_llcs_set {
        let t = option_tokens(&max_llcs);
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints.max_llcs }
    } else {
        let t = option_tokens(&max_llcs);
        quote! { #t }
    };
    let requires_smt_tokens = if requires_smt_set {
        quote! { #requires_smt }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints.requires_smt }
    } else {
        quote! { #requires_smt }
    };
    let min_cpus_tokens = if min_cpus_set {
        let v = min_cpus;
        quote! { #v }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints.min_cpus }
    } else {
        let v = min_cpus;
        quote! { #v }
    };
    let max_cpus_tokens = if max_cpus_set {
        let t = option_tokens(&max_cpus);
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints.max_cpus }
    } else {
        let t = option_tokens(&max_cpus);
        quote! { #t }
    };

    let expanded = quote! {
        #(#attrs)*
        #vis #inner_sig #block

        #[::ktstr::__private::linkme::distributed_slice(::ktstr::test_support::KTSTR_TESTS)]
        #[linkme(crate = ::ktstr::__private::linkme)]
        static #entry_name: ::ktstr::test_support::KtstrTestEntry = ::ktstr::test_support::KtstrTestEntry {
            // Always-emit fields. `name`/`func` are macro-generated;
            // `topology`/`constraints` inherit from the scheduler
            // via field access that the spread cannot
            // recover; `scheduler` substitutes
            // `Scheduler::EEVDF` when no `scheduler = ...`
            // attribute was supplied. Every remaining field below
            // (memory_mb, payload, workloads, auto_repro, assert,
            // extra_sched_args, ..., disk, and any future addition)
            // falls through to `..KtstrTestEntry::DEFAULT` when the
            // attribute did not specify a value, so future fields
            // with sane defaults need no macro change at all.
            name: #name_str,
            func: #inner_name,
            topology: #topology_tokens,
            constraints: ::ktstr::test_support::TopologyConstraints {
                min_numa_nodes: #min_numa_nodes_tokens,
                max_numa_nodes: #max_numa_nodes_tokens,
                min_llcs: #min_llcs_tokens,
                max_llcs: #max_llcs_tokens,
                requires_smt: #requires_smt_tokens,
                min_cpus: #min_cpus_tokens,
                max_cpus: #max_cpus_tokens,
            },
            scheduler: #scheduler_tokens,
            #memory_mb_field
            #payload_field
            #workloads_field
            #auto_repro_field
            #assert_field
            #extra_sched_args_field
            #watchdog_timeout_field
            #bpf_map_write_field
            #performance_mode_field
            #no_perf_mode_field
            #duration_field
            #workers_per_cgroup_field
            #num_snapshots_field
            #expect_err_field
            #host_only_field
            #extra_include_files_field
            #cleanup_budget_field
            #post_vm_field
            #config_content_field
            ..::ktstr::test_support::KtstrTestEntry::DEFAULT
        };

        #pairing_assert

        #[test]
        fn #orig_name() {
            #test_body
        }
    };

    expanded.into()
}

/// Convert a CamelCase identifier to SCREAMING_SNAKE_CASE.
///
/// Handles acronyms (consecutive uppercase): a separator is inserted
/// before the last letter of a run when followed by lowercase.
///
/// `Llc` -> `"LLC"`, `RejectPin` -> `"REJECT_PIN"`, `NoCtrl` -> `"NO_CTRL"`,
/// `LLC` -> `"LLC"`, `HTTPServer` -> `"HTTP_SERVER"`.
fn camel_to_screaming_snake(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, &ch) in chars.iter().enumerate() {
        if ch.is_uppercase() && i > 0 {
            let prev_upper = chars[i - 1].is_uppercase();
            let next_lower = chars.get(i + 1).is_some_and(|c| c.is_lowercase());
            if !prev_upper || next_lower {
                out.push('_');
            }
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

/// Function-style macro that registers a `Scheduler` const.
///
/// # Syntax
///
/// ```rust,ignore
/// use ktstr::prelude::*;
///
/// declare_scheduler!(MITOSIS, {
///     name = "mitosis",
///     binary = "scx_mitosis",
///     cgroup_parent = "/ktstr",
///     sched_args = ["--exit-dump-len", "1048576"],
///     kernels = ["6.14", "7.0..=7.2"],
///     constraints = TopologyConstraints {
///         min_llcs: 1, max_llcs: Some(8), max_cpus: Some(64),
///         ..TopologyConstraints::DEFAULT
///     },
/// });
/// ```
///
/// # Generated items
///
/// Given `declare_scheduler!(MITOSIS, { ... })`:
///
/// - `pub static MITOSIS: ::ktstr::test_support::Scheduler` — the declared
///   scheduler value. No `_PAYLOAD` suffix; the const IS the
///   `Scheduler`.
/// - A hidden `static __KTSTR_SCHED_REG_MITOSIS: &'static Scheduler`
///   registered in [`KTSTR_SCHEDULERS`](ktstr::test_support::KTSTR_SCHEDULERS)
///   via linkme so the verifier can discover the declaration by
///   spawning the test binary with `--ktstr-list-schedulers`.
///
/// # Visibility prefix
///
/// An optional Rust visibility prefix may precede the const name:
///
/// ```rust,ignore
/// declare_scheduler!(MY_SCHED, { ... });             // defaults to `pub`
/// declare_scheduler!(pub MY_SCHED, { ... });          // explicit `pub`
/// declare_scheduler!(pub(crate) MY_SCHED, { ... });   // crate-local
/// declare_scheduler!(pub(super) MY_SCHED, { ... });   // parent-module
/// declare_scheduler!(pub(in crate::test_support) MY_SCHED, { ... });
/// ```
///
/// Omitting the prefix defaults to `pub` — schedulers are normally
/// public so the verifier and other crates can reference them; an
/// explicit prefix is needed only when the declaration sits inside
/// a module that wants to narrow the exposed name. (Field content
/// shown above as `{ ... }` is elided; consult the Syntax example
/// for the required fields.) The hidden registry static (see
/// Generated items above) is always `static` (private) regardless
/// of the user-facing const's visibility — `linkme` gathers it via
/// link-section walking, not Rust name resolution, so the slice
/// mechanism works at every visibility level.
///
/// # Accepted fields
///
/// Exactly one scheduler-source must be declared: `binary`,
/// `binary_path`, or the `kernel_builtin_enable` + `kernel_builtin_disable`
/// pair. The three options select between the matching
/// [`SchedulerSpec`] variants. To run under the kernel default
/// instead, reference [`ktstr::test_support::Scheduler::EEVDF`]
/// directly rather than declaring a new scheduler.
///
/// | Field | Required | Description |
/// |---|---|---|
/// | `name = "..."` | yes | Scheduler name (sidecar / logs). |
/// | `binary = "..."` | one source | Binary name → `SchedulerSpec::Discover(...)`. Matched against `[[bin]]` names in `target/{debug,release}/`, the test binary's directory, or `KTSTR_SCHEDULER` env var. Often equal to the cargo package name but not required to be. |
/// | `binary_path = "/abs/path"` | one source | Absolute filesystem path → `SchedulerSpec::Path(...)`. The runtime does not auto-build this variant: the file must already exist at the path when the test runs. Use for prebuilt binaries that live outside the cargo discovery cascade. Macro-time validation rejects empty strings, relative paths, and `~`-prefixed paths (no compile-time tilde expansion); existence is the runtime's job. |
/// | `kernel_builtin_enable = [..]` + `kernel_builtin_disable = [..]` | one source | Two string-array literals that together select `SchedulerSpec::KernelBuiltin { enable: &[..], disable: &[..] }`. The framework writes the enable commands to the guest's `/sched_enable` and the disable commands to `/sched_disable` (see `src/vmm/initramfs.rs`), and the guest interpreter runs each entry once at scenario start / teardown. Both fields must be set together — setting only one is rejected. The interpreter (`src/vmm/rust_init.rs`) accepts EXACTLY ONE shell-line shape: `echo VALUE > /path` (plus blank lines and `#` comments). Pipes, `>>`, `;`, variable expansion, and any other syntax silently no-ops at runtime, so the macro rejects entries that don't match `echo … > /…` at expand time. At least one of the two arrays must be non-empty: a pair that supplies neither enable nor disable commands is equivalent to the EEVDF baseline — reference [`Scheduler::EEVDF`] for that. Note: `cargo ktstr export` currently bails on KernelBuiltin schedulers (`src/export.rs`); declarations using this variant cannot be reproduced via the export-to-shar workflow until that limitation is lifted. |
/// | `topology = (numa, llcs, cores, threads)` | no | Default VM topology. Default: `(1, 1, 2, 1)` (from `Scheduler::new`). Validated at compile time: each value must be non-zero, and `llcs` must be a multiple of `numa`. |
/// | `cgroup_parent = "..."` | no | Cgroup parent path (must begin with `/`). |
/// | `sched_args = [..]` | no | Scheduler CLI args prepended before per-test `extra_sched_args`. |
/// | `sysctls = [Sysctl::new("k", "v"), ..]` | no | Guest sysctls. |
/// | `kargs = [..]` | no | Extra guest kernel cmdline args. |
/// | `kernels = ["6.14", "7.0..=7.2", ..]` | no | Kernel specs the verifier sweeps. Same parser as the `--kernel` CLI flag — accepts exact versions, ranges (`..` or `..=`, both inclusive), git refs (`git+URL#REF`), paths, and cache keys. Each entry is validated at macro-expand time via the same `KernelId::parse` + `validate` the verifier uses at runtime; empty entries, inverted ranges, and `..`-containing strings whose endpoints aren't version-shaped (e.g. `"abc..def"`) are rejected. |
/// | `constraints = TopologyConstraints { .. }` | no | Gauntlet preset constraints — maps directly onto [`Scheduler::constraints`]. Filters which gauntlet topology presets exercise this scheduler. When given as a struct literal, the macro additionally cross-checks each literal field against the effective topology (explicit `topology` field if present, otherwise the `(1, 1, 2, 1)` default from `Scheduler::new`) and rejects infeasible pairings; non-struct-literal forms (e.g. `OTHER::CONST_CONSTRAINTS`) skip that check. |
/// | `assert = Assert::NO_OVERRIDES.method().chain()` | no | Scheduler-wide assertion overrides — maps directly onto [`Scheduler::assert`]. Merged with `Assert::default_checks()` and the per-test `assert` at runtime (`default ← scheduler ← per-test`). Accepts any const-evaluable expression: a const path like `Assert::NO_OVERRIDES`, a const-fn call like `Assert::default_checks()`, or a chain of const-fn setters like `Assert::NO_OVERRIDES.check_not_starved().max_gap_ms(50)`. The macro accepts MethodCall chains and Path-rooted (type/module-prefixed) Calls — only bare single-segment lowercase Calls like `helper()` are rejected as non-const free-fn patterns; non-const methods on a Path receiver slip through and surface as a deep const-eval failure at the spread site. |
/// | `config_file = "..."` | no | Host-side config file path. |
/// | `config_file_def = ("--config {file}", "/include-files/cfg.json")` | no | Inline-config plumbing — maps directly onto [`Scheduler::config_file_def`]. 2-tuple of string literals: arg_template (CLI arg with `{file}` placeholder substituted at run time) and guest_path (absolute path where the framework writes the JSON inside the guest). Distinct from `config_file` (which references a pre-existing host file). The macro validates: tuple-arity = 2, both elements non-empty string literals, `{file}` placeholder present in arg_template, guest_path absolute. |
///
/// # Const naming rules
///
/// The first argument must be a SCREAMING_SNAKE_CASE identifier and
/// must NOT be one of the reserved built-in names (`EEVDF`,
/// `KERNEL_DEFAULT`). The macro emits a `compile_error!` if either rule
/// is violated.
#[proc_macro]
pub fn declare_scheduler(input: TokenStream) -> TokenStream {
    match declare_scheduler_inner(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn declare_scheduler_inner(
    input: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    struct DeclareSchedulerInput {
        visibility: syn::Visibility,
        const_name: syn::Ident,
        fields: Vec<(syn::Ident, syn::Expr)>,
    }

    impl syn::parse::Parse for DeclareSchedulerInput {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            // Optional visibility prefix: `pub`, `pub(crate)`,
            // `pub(super)`, `pub(in path)`, or none. `syn::Visibility`
            // returns `Visibility::Inherited` when no prefix is given;
            // the emit treats this as the default-pub case so call
            // sites without a visibility prefix produce `pub static`.
            let visibility: syn::Visibility = input.parse()?;
            let const_name: syn::Ident = input.parse()?;
            let _: syn::Token![,] = input.parse()?;
            let body;
            syn::braced!(body in input);
            let mut fields = Vec::new();
            while !body.is_empty() {
                let key: syn::Ident = body.parse()?;
                let _: syn::Token![=] = body.parse()?;
                let value: syn::Expr = body.parse()?;
                fields.push((key, value));
                if body.peek(syn::Token![,]) {
                    let _: syn::Token![,] = body.parse()?;
                }
            }
            Ok(DeclareSchedulerInput {
                visibility,
                const_name,
                fields,
            })
        }
    }

    let DeclareSchedulerInput {
        visibility,
        const_name,
        fields,
    } = syn::parse2(input)?;

    // Validate const name: SCREAMING_SNAKE_CASE + not reserved.
    let const_name_str = const_name.to_string();
    if const_name_str != const_name_str.to_uppercase() {
        return Err(syn::Error::new(
            const_name.span(),
            format!(
                "declare_scheduler!: const name `{const_name_str}` must be SCREAMING_SNAKE_CASE"
            ),
        ));
    }
    // Reserve the const names that match the built-in `Scheduler::EEVDF`
    // and `Payload::KERNEL_DEFAULT` baselines so user code cannot shadow
    // either symbol. Match by exact identifier — the spelling is
    // case-sensitive in Rust so the lowercase form (e.g. `eevdf`) is
    // already rejected by the SCREAMING_SNAKE_CASE check above. The
    // companion string-name reservation (handled on the `name = "..."`
    // arm below) is case-insensitive because wire names typically
    // lowercase.
    match const_name_str.as_str() {
        "EEVDF" => {
            return Err(syn::Error::new(
                const_name.span(),
                format!(
                    "declare_scheduler!: const name `{const_name_str}` is reserved \
                     for the built-in Scheduler::EEVDF baseline; pick a different identifier"
                ),
            ));
        }
        "KERNEL_DEFAULT" => {
            return Err(syn::Error::new(
                const_name.span(),
                format!(
                    "declare_scheduler!: const name `{const_name_str}` is reserved \
                     for the built-in Payload::KERNEL_DEFAULT baseline; pick a different identifier"
                ),
            ));
        }
        _ => {}
    }

    // Parse fields.
    let mut sched_name: Option<String> = None;
    let mut sched_binary: Option<String> = None;
    let mut sched_binary_path: Option<String> = None;
    let mut sched_kernel_builtin_enable: Option<Vec<String>> = None;
    let mut sched_kernel_builtin_disable: Option<Vec<String>> = None;
    let mut sched_topology: Option<(u32, u32, u32, u32)> = None;
    let mut sched_cgroup_parent: Option<String> = None;
    let mut sched_args: Vec<String> = Vec::new();
    let mut sched_args_set = false;
    let mut sched_sysctls: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut sched_sysctls_set = false;
    let mut sched_kargs: Vec<String> = Vec::new();
    let mut sched_kargs_set = false;
    let mut sched_kernels: Vec<String> = Vec::new();
    let mut sched_kernels_set = false;
    let mut sched_constraints: Option<syn::Expr> = None;
    let mut sched_config_file: Option<String> = None;
    let mut sched_assert: Option<syn::Expr> = None;
    let mut sched_config_file_def: Option<(String, String)> = None;

    let mut seen_fields = std::collections::HashSet::<String>::new();
    for (key, value) in fields {
        let key_str = key.to_string();
        if !seen_fields.insert(key_str.clone()) {
            return Err(syn::Error::new(
                key.span(),
                format!("declare_scheduler!: duplicate field `{key_str}`"),
            ));
        }
        match key_str.as_str() {
            "name" => {
                let lit = expect_str_lit(&value, &key, "name")?;
                // Mirror the const-name reservation above: the string
                // names of the built-in `Scheduler::EEVDF` (`"eevdf"`)
                // and `Payload::KERNEL_DEFAULT` (`"kernel_default"`)
                // cannot be reused. A `declare_scheduler!` whose
                // `name = "eevdf"` would silently shadow the baseline
                // in `find_scheduler` lookups and sidecar comparisons.
                // Case-insensitive: `"EEVDF"`, `"Eevdf"`, etc. are all
                // the same wire name to a consumer that lowercases.
                let lit_lower = lit.to_lowercase();
                if matches!(lit_lower.as_str(), "eevdf" | "kernel_default") {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "declare_scheduler!: `name = \"{lit}\"` is reserved \
                             for the built-in {} baseline; pick a different name",
                            if lit_lower == "eevdf" {
                                "Scheduler::EEVDF"
                            } else {
                                "Payload::KERNEL_DEFAULT"
                            }
                        ),
                    ));
                }
                sched_name = Some(lit);
            }
            "binary" => {
                let lit = expect_str_lit(&value, &key, "binary")?;
                // An empty binary name flows into
                // `SchedulerSpec::Discover("")` and fails confusingly
                // at runtime inside `build_and_find_binary("")`. Reject
                // at macro-time so the error surfaces at the call site
                // — symmetric to the empty-name check below, except
                // that this one underlines the offending literal via
                // `new_spanned` so the caret lands on the empty `""`.
                if lit.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &value,
                        "declare_scheduler!: `binary` must be a non-empty string",
                    ));
                }
                sched_binary = Some(lit);
            }
            "binary_path" => {
                let lit = expect_str_lit(&value, &key, "binary_path")?;
                // Empty path flows into `SchedulerSpec::Path("")` and
                // fails confusingly at `resolve_scheduler` (anyhow
                // ensure on path.exists()). Reject at macro time.
                if lit.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &value,
                        "declare_scheduler!: `binary_path` must be a \
                         non-empty string",
                    ));
                }
                // Tilde expansion does not happen at compile time and
                // the runtime does not expand it either — `path.exists()`
                // checks the literal `~/foo` against the filesystem, which
                // never matches. Reject up-front with the actionable fix.
                if lit.starts_with('~') {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "declare_scheduler!: `binary_path = \"{lit}\"` \
                             starts with `~` — tilde paths are not expanded \
                             at compile time or by the runtime. Use an \
                             absolute path (e.g. `\"/home/user/bin/scx_foo\"`)."
                        ),
                    ));
                }
                // Relative paths are ambiguous between "sibling file" and
                // "discover-by-name" intent. Force the operator to commit:
                // if they want discovery, use `binary = "name"`; if they
                // want a specific file, write the absolute path.
                if !lit.starts_with('/') {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "declare_scheduler!: `binary_path = \"{lit}\"` \
                             must be absolute (start with `/`). For \
                             discovery-by-name, use `binary = \"...\"` \
                             instead; for a specific file, write the \
                             absolute path."
                        ),
                    ));
                }
                sched_binary_path = Some(lit);
            }
            "kernel_builtin_enable" => {
                let arr = expect_array(&value, &key, "kernel_builtin_enable")?;
                let mut cmds = Vec::with_capacity(arr.elems.len());
                for elem in &arr.elems {
                    let s = expect_str_lit_element(elem, "kernel_builtin_enable")?;
                    validate_kernel_builtin_cmd(elem, &s, "enable")?;
                    cmds.push(s);
                }
                sched_kernel_builtin_enable = Some(cmds);
            }
            "kernel_builtin_disable" => {
                let arr = expect_array(&value, &key, "kernel_builtin_disable")?;
                let mut cmds = Vec::with_capacity(arr.elems.len());
                for elem in &arr.elems {
                    let s = expect_str_lit_element(elem, "kernel_builtin_disable")?;
                    validate_kernel_builtin_cmd(elem, &s, "disable")?;
                    cmds.push(s);
                }
                sched_kernel_builtin_disable = Some(cmds);
            }
            "topology" => {
                if let syn::Expr::Tuple(t) = &value {
                    let mut parts = [0u32; 4];
                    if t.elems.len() != 4 {
                        return Err(syn::Error::new_spanned(
                            t,
                            "topology must be a 4-tuple (numa_nodes, llcs, cores, threads)",
                        ));
                    }
                    for (i, e) in t.elems.iter().enumerate() {
                        parts[i] = expect_u32_lit(e, &key, "topology")?;
                    }
                    sched_topology = Some((parts[0], parts[1], parts[2], parts[3]));
                } else {
                    return Err(syn::Error::new_spanned(
                        &value,
                        "topology must be a tuple expression: topology = (numa_nodes, llcs, cores, threads)",
                    ));
                }
            }
            "cgroup_parent" => {
                let lit = expect_str_lit(&value, &key, "cgroup_parent")?;
                sched_cgroup_parent = Some(lit);
            }
            "sched_args" => {
                sched_args_set = true;
                let arr = expect_array(&value, &key, "sched_args")?;
                for elem in &arr.elems {
                    sched_args.push(expect_str_lit_element(elem, "sched_args")?);
                }
            }
            "sysctls" => {
                sched_sysctls_set = true;
                let arr = expect_array(&value, &key, "sysctls")?;
                for elem in &arr.elems {
                    sched_sysctls.push(elem.to_token_stream());
                }
            }
            "kargs" => {
                sched_kargs_set = true;
                let arr = expect_array(&value, &key, "kargs")?;
                for elem in &arr.elems {
                    sched_kargs.push(expect_str_lit_element(elem, "kargs")?);
                }
            }
            "kernels" => {
                sched_kernels_set = true;
                let arr = expect_array(&value, &key, "kernels")?;
                for elem in &arr.elems {
                    let s = expect_str_lit_element(elem, "kernels")?;
                    // Empty kernel strings parse as `CacheKey("")` and
                    // fail confusingly at verifier runtime with "cache
                    // key not found". Reject up-front so the diagnostic
                    // lands on the literal in the source.
                    if s.is_empty() {
                        return Err(syn::Error::new_spanned(
                            elem,
                            "declare_scheduler!: `kernels` entry must \
                             be a non-empty string. Accepted forms: \
                             exact version (`6.14`), inclusive range \
                             (`6.14..7.0` or `6.14..=7.0`), git source \
                             (`git+URL#REF`), absolute or `~`-prefixed \
                             path, or cache key.",
                        ));
                    }
                    // Run the same `KernelId::parse` + `validate` the
                    // verifier uses so any malformed entry — inverted
                    // range, suspicious `..` substring that fails the
                    // range grammar — is caught at the call site
                    // rather than as a confusing runtime "cache key
                    // not found" error.
                    let parsed = kernel_path::KernelId::parse(&s);
                    if let Err(msg) = parsed.validate() {
                        return Err(syn::Error::new_spanned(
                            elem,
                            format!("declare_scheduler!: invalid kernel \
                                     spec `{s}`: {msg}"),
                        ));
                    }
                    // A literal containing `..` that did not classify
                    // as a Range almost always indicates a typo'd
                    // range spec (e.g. `"abc..def"` where neither
                    // endpoint is version-shaped, or `"6.14..xyz"`).
                    // Per-variant disambiguation: `CacheKey("a..b")`
                    // is wrong because cache keys are content-addressed
                    // identifiers that don't carry `..` separators;
                    // `Path("foo/..bar")` is fine (file paths legally
                    // contain `..`) and already matched the Path arm.
                    if s.contains("..")
                        && matches!(parsed, kernel_path::KernelId::CacheKey(_))
                    {
                        return Err(syn::Error::new_spanned(
                            elem,
                            format!(
                                "declare_scheduler!: `kernels` entry `{s}` \
                                 contains `..` but the endpoints aren't both \
                                 version-shaped (`MAJOR.MINOR[.PATCH][-rcN]`). \
                                 If this was meant as a version range, \
                                 fix the endpoints (e.g. `6.14..7.0`). \
                                 If this is a literal cache key, remove \
                                 the `..` — cache keys do not use \
                                 range syntax.",
                            ),
                        ));
                    }
                    sched_kernels.push(s);
                }
            }
            "constraints" => {
                // `constraints` lands in a `pub static`, so the
                // expression must be const-evaluable. Reject the
                // common typo of passing a non-const helper call
                // (`build_constraints()`, `default_topology().min_llcs(4)`,
                // `Foo::derive(...).constraints`) up-front so the
                // diagnostic explains the constraint instead of
                // letting rustc surface a deep, confusing
                // const-eval-failure chain at the spread site.
                //
                // Accepted shapes:
                //   - `TopologyConstraints { ..TopologyConstraints::DEFAULT }`
                //     (struct literal — the canonical form, used by
                //     every in-tree call site)
                //   - `TopologyConstraints::DEFAULT` (path expression
                //     — bare DEFAULT or any other `const` path)
                //   - `( … )` (parenthesized const-eligible expression
                //     — pass-through to the underlying form so a user
                //     who wraps for clarity is not punished)
                //   - reference / unary on top of any accepted form
                //
                // Calls and method chains are rejected with a hint
                // describing the const-eligible alternatives.
                validate_const_eligible(
                    &value,
                    "constraints",
                    CONSTRAINTS_ACCEPTED_SHAPES,
                    ConstEligibility::StructLiteralOnly,
                )?;
                sched_constraints = Some(value);
            }
            "config_file" => {
                let lit = expect_str_lit(&value, &key, "config_file")?;
                sched_config_file = Some(lit);
            }
            "assert" => {
                // `assert` lands in a `pub static`, so the expression
                // must be const-evaluable. Unlike `constraints`, the
                // canonical Assert pattern is METHOD-CHAINING on const
                // fns (`Assert::NO_OVERRIDES.check_not_starved()...`),
                // so the assert validator accepts MethodCall chains
                // and Path-rooted Calls (`Assert::default_checks()`,
                // `Some(x)`). Only bare single-segment lowercase
                // Calls (`helper()`) are rejected as the free-fn
                // pattern; non-const methods on a Path receiver
                // slip through and surface as a deep const-eval
                // failure at the spread site.
                // See `validate_const_eligible` with
                // `ConstEligibility::AllowConstMethodChains`.
                validate_const_eligible(
                    &value,
                    "assert",
                    ASSERT_ACCEPTED_SHAPES,
                    ConstEligibility::AllowConstMethodChains,
                )?;
                sched_assert = Some(value);
            }
            "config_file_def" => {
                // `config_file_def` is `Option<(arg_template,
                // guest_path)>`. The macro accepts a 2-tuple of string
                // literals and auto-wraps in `Some` via the existing
                // `.config_file_def(arg, path)` builder. Validate at
                // expand time: tuple-arity = 2, each element is a
                // non-empty string literal, arg_template contains the
                // `{file}` placeholder (the runtime substitutes the
                // guest path at that position; a template without it
                // silently fails at dispatch), and guest_path is
                // absolute (the runtime writes the config there, and
                // a relative path breaks the `mkdir -p` invariant).
                let tup = if let syn::Expr::Tuple(t) = &value {
                    t
                } else {
                    return Err(syn::Error::new_spanned(
                        &value,
                        "declare_scheduler!: `config_file_def` must be a \
                         2-tuple of string literals: `(arg_template, guest_path)`. \
                         Example: `(\"--config {file}\", \"/include-files/cfg.json\")`.",
                    ));
                };
                if tup.elems.len() != 2 {
                    return Err(syn::Error::new_spanned(
                        tup,
                        format!(
                            "declare_scheduler!: `config_file_def` must be a \
                             2-tuple of string literals (`(arg_template, guest_path)`), \
                             got {}-tuple.",
                            tup.elems.len()
                        ),
                    ));
                }
                let arg_template = expect_str_lit_element(&tup.elems[0], "config_file_def")?;
                let guest_path = expect_str_lit_element(&tup.elems[1], "config_file_def")?;
                if arg_template.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[0],
                        "declare_scheduler!: `config_file_def` arg_template \
                         (element 0) must be a non-empty string. Example: \
                         `\"--config {file}\"`.",
                    ));
                }
                if guest_path.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[1],
                        "declare_scheduler!: `config_file_def` guest_path \
                         (element 1) must be a non-empty string. Example: \
                         `\"/include-files/cfg.json\"`.",
                    ));
                }
                if !arg_template.contains("{file}") {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[0],
                        format!(
                            "declare_scheduler!: `config_file_def` arg_template \
                             `{arg_template}` is missing the `{{file}}` placeholder \
                             — the framework substitutes the guest path at \
                             that position when invoking the scheduler. \
                             Add `{{file}}` (e.g. `\"--config {{file}}\"`)."
                        ),
                    ));
                }
                if !guest_path.starts_with('/') {
                    return Err(syn::Error::new_spanned(
                        &tup.elems[1],
                        format!(
                            "declare_scheduler!: `config_file_def` guest_path \
                             `{guest_path}` must be absolute (start with `/`). \
                             The framework writes the config file at this path \
                             inside the guest, and a relative path breaks the \
                             `mkdir -p` invariant."
                        ),
                    ));
                }
                sched_config_file_def = Some((arg_template, guest_path));
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("declare_scheduler!: unknown field `{other}`"),
                ));
            }
        }
    }

    let sched_name = sched_name.ok_or_else(|| {
        syn::Error::new(
            const_name.span(),
            "declare_scheduler!: missing required field `name`",
        )
    })?;
    if sched_name.is_empty() {
        return Err(syn::Error::new(
            const_name.span(),
            "declare_scheduler!: `name` must be a non-empty string",
        ));
    }
    // The two `kernel_builtin_*` fields are paired — setting one
    // without the other is always a typo. The KernelBuiltin variant
    // carries both `enable` and `disable` lists in the same struct, so
    // requiring both fields at the macro site mirrors the type-level
    // invariant and prevents a half-specified scheduler from compiling.
    match (
        sched_kernel_builtin_enable.is_some(),
        sched_kernel_builtin_disable.is_some(),
    ) {
        (true, false) => {
            return Err(syn::Error::new(
                const_name.span(),
                "declare_scheduler!: `kernel_builtin_enable` set without \
                 `kernel_builtin_disable`. Both fields must be set \
                 together (or both omitted). Add a `kernel_builtin_disable \
                 = [\"echo ...\"]` line that restores the kernel default \
                 at teardown.",
            ));
        }
        (false, true) => {
            return Err(syn::Error::new(
                const_name.span(),
                "declare_scheduler!: `kernel_builtin_disable` set without \
                 `kernel_builtin_enable`. Both fields must be set \
                 together (or both omitted). Add a `kernel_builtin_enable \
                 = [\"echo ...\"]` line that switches the kernel into the \
                 chosen policy at scenario start.",
            ));
        }
        _ => {}
    }
    let kernel_builtin_set =
        sched_kernel_builtin_enable.is_some() || sched_kernel_builtin_disable.is_some();
    // Both arrays empty would register a KernelBuiltin scheduler that
    // does nothing — functionally identical to the EEVDF baseline.
    if let (Some(en), Some(di)) = (
        sched_kernel_builtin_enable.as_ref(),
        sched_kernel_builtin_disable.as_ref(),
    ) && en.is_empty()
        && di.is_empty()
    {
        return Err(syn::Error::new(
            const_name.span(),
            "declare_scheduler!: `kernel_builtin_enable = []` paired \
             with `kernel_builtin_disable = []` has no commands on \
             either side — that is functionally identical to the \
             kernel-default baseline. Reference \
             `ktstr::test_support::Scheduler::EEVDF` directly instead \
             of declaring a KernelBuiltin scheduler with no commands.",
        ));
    }
    // Exactly one scheduler-source must be set. The three options are
    // `binary` (SchedulerSpec::Discover), `binary_path` (Path), and
    // the paired `kernel_builtin_enable` + `kernel_builtin_disable`
    // (KernelBuiltin). Setting more than one is ambiguous; setting
    // none is rejected so any user wanting the kernel-default baseline
    // references `Scheduler::EEVDF` directly rather than declaring
    // a scheduler with no source.
    let source_count = [
        sched_binary.is_some(),
        sched_binary_path.is_some(),
        kernel_builtin_set,
    ]
    .iter()
    .filter(|b| **b)
    .count();
    if source_count == 0 {
        return Err(syn::Error::new(
            const_name.span(),
            "declare_scheduler!: no scheduler source declared. Pick one of:\n  \
             - `binary = \"scx_my_sched\"` (discover the binary by name)\n  \
             - `binary_path = \"/abs/path/to/scx_custom\"` (absolute filesystem path)\n  \
             - `kernel_builtin_enable = [\"echo 1 > /sys/...\"]` + \
             `kernel_builtin_disable = [\"echo 0 > /sys/...\"]` \
             (in-kernel scheduling policy toggled via shell commands)\n\
             To test under the kernel-default EEVDF baseline, reference \
             `ktstr::test_support::Scheduler::EEVDF` directly instead \
             of declaring a new scheduler.",
        ));
    }
    if source_count > 1 {
        return Err(syn::Error::new(
            const_name.span(),
            "declare_scheduler!: more than one scheduler source declared. \
             Pick exactly one of `binary`, `binary_path`, or the \
             `kernel_builtin_enable` + `kernel_builtin_disable` pair. \
             Each maps to a different `SchedulerSpec` variant \
             (`Discover`, `Path`, `KernelBuiltin`) and they cannot stack.",
        ));
    }
    // String-name reservation extension for KernelBuiltin: the variant's
    // display_name is the literal `"kernel"` (see `SchedulerSpec::display_name`
    // in `src/test_support/entry.rs`), so a user scheduler whose `name`
    // field also resolves to `"kernel"` would collide with the variant
    // label in failure-dump headers and sidecar comparisons. Reserve
    // case-insensitively, matching the existing reservation of
    // `"eevdf"` / `"kernel_default"`.
    if kernel_builtin_set && sched_name.to_lowercase() == "kernel" {
        return Err(syn::Error::new(
            const_name.span(),
            format!(
                "declare_scheduler!: `name = \"{sched_name}\"` collides with \
                 the KernelBuiltin variant's display_name (`\"kernel\"`). \
                 Pick a different name so failure dumps and sidecar \
                 entries can distinguish this scheduler from the \
                 variant label."
            ),
        ));
    }

    // Validate topology.
    if let Some((n, l, c, t)) = sched_topology {
        if n == 0 || l == 0 || c == 0 || t == 0 {
            return Err(syn::Error::new(
                const_name.span(),
                "declare_scheduler!: topology values must all be > 0",
            ));
        }
        if l % n != 0 {
            return Err(syn::Error::new(
                const_name.span(),
                format!(
                    "declare_scheduler!: topology: llcs ({l}) must \
                     be divisible by numa_nodes ({n})"
                ),
            ));
        }
    }

    // Sanity-check the effective topology vs explicit
    // struct-literal constraints. Without this, both
    // `topology = (1, 2, 4, 1)` AND an omitted topology paired
    // with `constraints = TopologyConstraints { min_llcs: 100, .. }`
    // are silently accepted: every gauntlet preset rejects the
    // test at runtime because the effective topology violates
    // the declared minimum (100 LLCs), and the test never runs.
    //
    // When `topology` is omitted the runtime falls back to
    // `Scheduler::new`'s default (numa_nodes=1, llcs=1,
    // cores_per_llc=2, threads_per_core=1, total_cpus=2) — see
    // `Scheduler::new` in `src/test_support/entry.rs`. The macro
    // checks against the same default so infeasible constraints
    // are caught regardless of whether the caller pinned a
    // topology.
    //
    // The macro can only walk the constraint fields when the
    // expression is a struct literal — non-struct-literal forms
    // (`TopologyConstraints::DEFAULT`, a const path) carry values
    // the macro cannot inspect at expand time, so the check no-ops
    // for those shapes.
    if let Some(constraints_expr) = sched_constraints.as_ref()
        && let syn::Expr::Struct(es) = constraints_expr
    {
        let topology_is_default = sched_topology.is_none();
        let (n, l, c, t) = sched_topology.unwrap_or((1, 1, 2, 1));
        let total = (l as u64) * (c as u64) * (t as u64);
        check_constraint_field_against_topology(es, n, l, total, t, topology_is_default)?;
    }

    // Build the Scheduler const expression via the builder chain.
    let sched_name_str = sched_name;
    let mut builder_chain = quote! {
        ::ktstr::test_support::Scheduler::new(#sched_name_str)
    };

    let binary_spec = if let Some(name) = &sched_binary {
        quote! { ::ktstr::test_support::SchedulerSpec::Discover(#name) }
    } else if let Some(path) = &sched_binary_path {
        quote! { ::ktstr::test_support::SchedulerSpec::Path(#path) }
    } else if kernel_builtin_set {
        // Both fields are guaranteed set together by the pair check above.
        let enable = sched_kernel_builtin_enable
            .as_ref()
            .expect("kernel_builtin pair check requires both fields set");
        let disable = sched_kernel_builtin_disable
            .as_ref()
            .expect("kernel_builtin pair check requires both fields set");
        quote! {
            ::ktstr::test_support::SchedulerSpec::KernelBuiltin {
                enable: &[#(#enable),*],
                disable: &[#(#disable),*],
            }
        }
    } else {
        unreachable!("source_count check above proves at least one source set")
    };
    builder_chain = quote! {
        #builder_chain.binary(#binary_spec)
    };
    if let Some((n, l, c, t)) = sched_topology {
        builder_chain = quote! { #builder_chain.topology(#n, #l, #c, #t) };
    }
    if let Some(parent) = &sched_cgroup_parent {
        builder_chain = quote! { #builder_chain.cgroup_parent(#parent) };
    }
    if sched_args_set {
        builder_chain = quote! { #builder_chain.sched_args(&[#(#sched_args),*]) };
    }
    if sched_sysctls_set {
        let entries = &sched_sysctls;
        builder_chain = quote! { #builder_chain.sysctls(&[#(#entries),*]) };
    }
    if sched_kargs_set {
        builder_chain = quote! { #builder_chain.kargs(&[#(#sched_kargs),*]) };
    }
    if sched_kernels_set {
        builder_chain = quote! { #builder_chain.kernels(&[#(#sched_kernels),*]) };
    }
    if let Some(tc) = &sched_constraints {
        builder_chain = quote! { #builder_chain.constraints(#tc) };
    }
    if let Some(cf) = &sched_config_file {
        builder_chain = quote! { #builder_chain.config_file(#cf) };
    }
    if let Some(a) = &sched_assert {
        builder_chain = quote! { #builder_chain.assert(#a) };
    }
    if let Some((arg, path)) = &sched_config_file_def {
        builder_chain = quote! { #builder_chain.config_file_def(#arg, #path) };
    }

    let registry_ident = format_ident!("__KTSTR_SCHED_REG_{}", const_name);

    // Default the emitted const's visibility to `pub` when the user
    // omits a prefix. Explicit prefixes (`pub`, `pub(crate)`,
    // `pub(super)`, `pub(in path)`) flow through verbatim via
    // `quote!`'s `ToTokens` impl for `syn::Visibility`.
    let effective_visibility = match &visibility {
        syn::Visibility::Inherited => quote! { pub },
        v => quote! { #v },
    };

    let expanded = quote! {
        // Suppress `missing_docs` on the emitted static so consumer
        // crates that set `#![deny(missing_docs)]` can still invoke
        // `declare_scheduler!`. The const name is the user-supplied
        // identifier and the macro itself is the documented entry
        // point — requiring a doc comment per declaration would force
        // boilerplate at every call site.
        #[allow(missing_docs)]
        #effective_visibility static #const_name: ::ktstr::test_support::Scheduler = #builder_chain;

        // The registry static stays plain `static` regardless of the
        // user-facing const's visibility — linkme gathers it via
        // link-section walking (not Rust name resolution), so its
        // visibility is irrelevant to the slice mechanism. Keeping it
        // private keeps the registry symbol opaque even when the
        // user-facing const is `pub`.
        #[::ktstr::__private::linkme::distributed_slice(::ktstr::test_support::KTSTR_SCHEDULERS)]
        #[linkme(crate = ::ktstr::__private::linkme)]
        static #registry_ident: &'static ::ktstr::test_support::Scheduler = &#const_name;
    };

    Ok(expanded)
}

fn expect_str_lit(expr: &syn::Expr, key: &syn::Ident, field: &str) -> syn::Result<String> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(ls),
            ..
        }) => Ok(ls.value()),
        _ => Err(syn::Error::new(
            key.span(),
            format!("declare_scheduler!: `{field}` must be a string literal"),
        )),
    }
}

fn expect_str_lit_element(expr: &syn::Expr, field: &str) -> syn::Result<String> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(ls),
            ..
        }) => Ok(ls.value()),
        _ => Err(syn::Error::new_spanned(
            expr,
            format!("declare_scheduler!: element of `{field}` must be a string literal"),
        )),
    }
}

fn expect_u32_lit(expr: &syn::Expr, key: &syn::Ident, field: &str) -> syn::Result<u32> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(li),
            ..
        }) => li.base10_parse(),
        _ => Err(syn::Error::new(
            key.span(),
            format!("declare_scheduler!: `{field}` element must be an integer literal"),
        )),
    }
}

fn expect_array<'a>(
    expr: &'a syn::Expr,
    key: &syn::Ident,
    field: &str,
) -> syn::Result<&'a syn::ExprArray> {
    if let syn::Expr::Array(arr) = expr {
        Ok(arr)
    } else {
        Err(syn::Error::new(
            key.span(),
            format!("declare_scheduler!: `{field}` must be an array literal `[..]`"),
        ))
    }
}

/// Validate a single `kernel_builtin_enable` or `kernel_builtin_disable`
/// command string against the grammar accepted by the guest
/// interpreter at `src/vmm/rust_init.rs`'s `exec_shell_line`. Anything
/// else (`>>`, pipes, `;`, variable expansion, sysctl -w, etc.) silently
/// no-ops at runtime, so the macro rejects up-front. Accepted shapes:
///
/// - `echo VALUE > /path` — writes VALUE+newline to /path
/// - blank line (skipped)
/// - `#`-prefixed comment (skipped)
fn validate_kernel_builtin_cmd(
    elem: &syn::Expr,
    cmd: &str,
    slot: &str,
) -> syn::Result<()> {
    let trimmed = cmd.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(());
    }
    if !trimmed.starts_with("echo ") {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` does not start with `echo ` — the guest \
                 interpreter accepts only `echo VALUE > /path` (plus \
                 blank lines and `#` comments). Other shell syntax \
                 (`>>`, pipes, `;`, variable expansion, sysctl) \
                 silently no-ops at runtime."
            ),
        ));
    }
    // Reject append `>>` explicitly. The guest's split_once(\" > \") would
    // miss the substring on `echo X >> /path` (no space between `>>`)
    // and fall through to the unsupported-command no-op; surface the
    // intent at expand time with the append-specific diagnostic.
    if trimmed.contains(">>") {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` uses `>>` (append) — the guest interpreter \
                 only handles single-`>` truncating writes. Use `>` \
                 instead."
            ),
        ));
    }
    let rest = &trimmed["echo ".len()..];
    let (value, path) = match rest.split_once(" > ") {
        Some((v, p)) => (v.trim(), p.trim()),
        None => {
            return Err(syn::Error::new_spanned(
                elem,
                format!(
                    "declare_scheduler!: `kernel_builtin_{slot}` command \
                     `{cmd}` is missing the ` > ` (space-greater-space) \
                     redirect — the guest interpreter requires \
                     `echo VALUE > /path` with literal spaces around \
                     `>` (`exec_shell_line` in `src/vmm/rust_init.rs` \
                     uses `split_once(\" > \")`)."
                ),
            ));
        }
    };
    if value.is_empty() {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` writes an empty value — `echo > /path` is \
                 valid shell but useless. Provide the value to write \
                 (e.g. `echo 1 > /sys/...`)."
            ),
        ));
    }
    if !path.starts_with('/') {
        return Err(syn::Error::new_spanned(
            elem,
            format!(
                "declare_scheduler!: `kernel_builtin_{slot}` command \
                 `{cmd}` writes to relative path `{path}` — the guest \
                 interpreter writes via `std::fs::write`, which resolves \
                 relative to the guest init's cwd (`/`). Use an absolute \
                 path to be explicit."
            ),
        ));
    }
    Ok(())
}

/// Policy axis for `validate_const_eligible`. Different
/// `declare_scheduler!` fields have different canonical
/// const-construction patterns and need different MethodCall + Call
/// tolerances.
#[derive(Clone, Copy)]
enum ConstEligibility {
    /// Used for `constraints`. Rejects `MethodCall(...)` and rejects
    /// `Call(...)` whose function path tail is not PascalCase. The
    /// canonical pattern is a struct literal (`TopologyConstraints
    /// { .. }`) or a const path (`TopologyConstraints::DEFAULT`) —
    /// method chains in that position are always wrong because
    /// `TopologyConstraints` has no const-fn builder.
    StructLiteralOnly,
    /// Used for `assert`. Accepts `MethodCall(...)` and recurses
    /// into receiver + args; accepts `Call(...)` with multi-segment
    /// or PascalCase function path and recurses into args; rejects
    /// `Call(...)` with single-segment lowercase path (bare local
    /// helper). Required because `Assert`'s canonical const
    /// constructors are snake_case: `Assert::NO_OVERRIDES`,
    /// `Assert::default_checks()`, and the
    /// `Assert::NO_OVERRIDES.check_not_starved()` chain pattern.
    AllowConstMethodChains,
}

/// Reject a `declare_scheduler!` field whose value cannot be
/// const-evaluated. The field lands in a `pub static`, so non-const
/// helper calls yield deep const-eval failures at the spread site
/// that are hard to map back to the original mistake. This validator
/// catches them at expand time with a per-field tailored diagnostic.
///
/// Recurses into struct-literal field values and the `..rest`
/// spread; both PascalCase Call args and MethodCall args (when
/// allowed by `mode`) are recursed too.
fn validate_const_eligible(
    expr: &syn::Expr,
    field_name: &str,
    accepted_shapes: &str,
    mode: ConstEligibility,
) -> syn::Result<()> {
    let recurse = |e: &syn::Expr| validate_const_eligible(e, field_name, accepted_shapes, mode);
    match expr {
        syn::Expr::Struct(es) => {
            for fv in &es.fields {
                recurse(&fv.expr)?;
            }
            if let Some(rest) = &es.rest {
                recurse(rest)?;
            }
            Ok(())
        }
        syn::Expr::Path(_) => Ok(()),
        syn::Expr::Paren(p) => recurse(&p.expr),
        syn::Expr::Reference(r) => recurse(&r.expr),
        syn::Expr::Unary(u) => recurse(&u.expr),
        syn::Expr::Binary(b) => {
            recurse(&b.left)?;
            recurse(&b.right)?;
            Ok(())
        }
        syn::Expr::Lit(_) => Ok(()),
        syn::Expr::MethodCall(mc) => match mode {
            ConstEligibility::StructLiteralOnly => {
                Err(field_not_const_error(field_name, accepted_shapes, expr, true))
            }
            ConstEligibility::AllowConstMethodChains => {
                recurse(&mc.receiver)?;
                for arg in &mc.args {
                    recurse(arg)?;
                }
                Ok(())
            }
        },
        syn::Expr::Call(call) => match mode {
            ConstEligibility::StructLiteralOnly => {
                if call_func_is_pascal_constructor(&call.func) {
                    for arg in &call.args {
                        recurse(arg)?;
                    }
                    Ok(())
                } else {
                    Err(field_not_const_error(field_name, accepted_shapes, expr, true))
                }
            }
            ConstEligibility::AllowConstMethodChains => {
                if call_func_is_single_segment_lowercase(&call.func) {
                    Err(field_not_const_error(field_name, accepted_shapes, expr, true))
                } else {
                    for arg in &call.args {
                        recurse(arg)?;
                    }
                    Ok(())
                }
            }
        },
        syn::Expr::Block(_) => Err(field_block_not_const_error(field_name, expr)),
        _ => Err(field_not_const_error(field_name, accepted_shapes, expr, false)),
    }
}

/// Heuristic: does this call expression look like a tuple-struct
/// or enum-variant constructor (`Some(x)`, `MyVariant(x)`,
/// `Foo::Bar(x)`)? Rust naming convention reserves PascalCase for
/// types and variants; a function-path whose last segment starts
/// with an uppercase ASCII letter is therefore very likely a
/// const-eligible constructor, while a lowercase last segment
/// (`build_value()`) is the snake_case free-fn pattern.
fn call_func_is_pascal_constructor(func: &syn::Expr) -> bool {
    let syn::Expr::Path(ep) = unwrap_parens(func) else {
        return false;
    };
    path_last_segment_ident(&ep.path).is_some_and(|ident| {
        ident
            .to_string()
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_uppercase())
    })
}

/// Return the last segment's identifier of a path, or `None` for an
/// empty path. Used by helpers that match on a path's tail name
/// (`Some`, `BTreeSet`, etc.) without forcing the caller to import
/// the full path.
fn path_last_segment_ident(path: &syn::Path) -> Option<&syn::Ident> {
    path.segments.last().map(|s| &s.ident)
}

/// Heuristic: is this call expression a single-segment lowercase
/// function path (`build_helper()`, `default()`, snake_case-style)?
/// Used by `validate_const_eligible` under
/// `ConstEligibility::AllowConstMethodChains` to reject bare local
/// helpers while accepting type/module-prefixed const-fn calls
/// (`Assert::default_checks()`, `Some(x)`, `path::to::helper()`).
fn call_func_is_single_segment_lowercase(func: &syn::Expr) -> bool {
    let syn::Expr::Path(ep) = unwrap_parens(func) else {
        return false;
    };
    if ep.path.segments.len() != 1 {
        return false;
    }
    path_last_segment_ident(&ep.path).is_some_and(|ident| {
        ident
            .to_string()
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase())
    })
}

/// Strip wrapping `Expr::Paren` layers from an expression so
/// heuristics that match on shape (`Expr::Path` for constructor or
/// snake-case detection) see through `(build_helper)()` style
/// parenthesization. Without this, a deliberately parenthesized
/// bare ident would bypass the lowercase-bare-call rejection.
fn unwrap_parens(expr: &syn::Expr) -> &syn::Expr {
    let mut cur = expr;
    while let syn::Expr::Paren(p) = cur {
        cur = &p.expr;
    }
    cur
}

/// Emit the shared `declare_scheduler!`: `<field>` not-const-eligible
/// diagnostic. `accepted_shapes` is the field-specific sentence
/// listing the accepted shapes (e.g. struct literal vs const path).
/// `append_call_hint` adds the trailing sentence about helper calls
/// and method chains failing at the spread site — used for the
/// `Call`/`MethodCall` arm where that hint is load-bearing.
fn field_not_const_error(
    field_name: &str,
    accepted_shapes: &str,
    expr: &syn::Expr,
    append_call_hint: bool,
) -> syn::Error {
    let header = format!(
        "declare_scheduler!: `{field_name}` must be a const-evaluable \
         expression (emitted into a `pub static`). {accepted_shapes}"
    );
    let msg = if append_call_hint {
        format!(
            "{header} Non-const helper calls and method chains are not \
             const-eligible and would fail with a deep const-eval \
             diagnostic at the spread site."
        )
    } else {
        header
    };
    syn::Error::new_spanned(expr, msg)
}

/// Emit the shared `declare_scheduler!`: `<field>` block-expression
/// rejection diagnostic. Block expressions need tailored guidance
/// (drop the braces / use a const binding) that the generic
/// non-const-eligible message doesn't carry.
fn field_block_not_const_error(field_name: &str, expr: &syn::Expr) -> syn::Error {
    syn::Error::new_spanned(
        expr,
        format!(
            "declare_scheduler!: `{field_name}` must be a const-evaluable \
             expression (emitted into a `pub static`). Block expressions \
             like `{{ ... }}` are not const-eligible here — for a single \
             literal value, drop the braces. For shared values, use a \
             const binding (`const MY_VAL: T = ...;` then reference \
             `MY_VAL`)."
        ),
    )
}

/// Field-specific accepted-shapes sentence for `constraints`.
const CONSTRAINTS_ACCEPTED_SHAPES: &str =
    "Use a struct literal `TopologyConstraints { ..TopologyConstraints::DEFAULT }` \
     or a const path like `TopologyConstraints::DEFAULT`.";

/// Field-specific accepted-shapes sentence for `assert`.
const ASSERT_ACCEPTED_SHAPES: &str =
    "Use a const path like `Assert::NO_OVERRIDES`, a const-fn call like \
     `Assert::default_checks()`, or a chain of const-fn setters like \
     `Assert::NO_OVERRIDES.check_not_starved().max_gap_ms(50)`.";

/// Walk a `TopologyConstraints { .. }` struct literal and reject
/// fields whose literal values make the declared scheduler topology
/// `(numa, llcs, _, threads)` infeasible (total CPUs = `total`,
/// threads_per_core = `threads`).
///
/// Only literal-valued fields are checked — non-literal expressions
/// (paths, calls) carry values the macro cannot evaluate. Fields
/// dropped via `..TopologyConstraints::DEFAULT` are also not
/// validated against the DEFAULT values; doing so would silently
/// reject test authors who pair an explicit non-default topology
/// with the default constraint set on the assumption that those
/// defaults match. Limiting the check to fields the user explicitly
/// wrote keeps the diagnostic targeted: it fires only when an
/// explicit constraint contradicts an explicit topology.
fn check_constraint_field_against_topology(
    es: &syn::ExprStruct,
    numa: u32,
    llcs: u32,
    total_cpus: u64,
    threads_per_core: u32,
    topology_is_default: bool,
) -> syn::Result<()> {
    // When `topology` was omitted, the macro inferred Scheduler::new
    // defaults. Reading "effective topology llcs (1)" without
    // context makes a user wonder where the 1 came from — they
    // didn't write a topology field. Append a tail that names the
    // fallback source + the override syntax.
    let topology_origin_tail = if topology_is_default {
        " (`topology` field omitted; macro fell back to \
         Scheduler::new's default `(numa=1, llcs=1, \
         cores=2, threads=1)`. Add an explicit \
         `topology = (numa, llcs, cores, threads)` to \
         override.)"
    } else {
        ""
    };
    for fv in &es.fields {
        let syn::Member::Named(ident) = &fv.member else {
            continue;
        };
        let name = ident.to_string();
        match name.as_str() {
            "min_llcs" => {
                if let Some(v) = u64_from_lit_expr(&fv.expr)
                    && v > llcs as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.min_llcs \
                             ({v}) exceeds effective topology llcs \
                             ({llcs}); every gauntlet preset would \
                             reject this test at runtime and the test \
                             would never execute. Lower min_llcs to \
                             {llcs} or fewer, or raise topology llcs.\
                             {topology_origin_tail}",
                        ),
                    ));
                }
            }
            "max_llcs" => {
                if let Some(v) = u64_from_option_some_lit(&fv.expr)
                    && v < llcs as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.max_llcs \
                             (Some({v})) is below effective topology \
                             llcs ({llcs}); every gauntlet preset \
                             would reject this test at runtime and \
                             the test would never execute. Raise \
                             max_llcs to {llcs} or higher, or lower \
                             topology llcs.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "min_numa_nodes" => {
                if let Some(v) = u64_from_lit_expr(&fv.expr)
                    && v > numa as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.min_numa_nodes \
                             ({v}) exceeds effective topology numa_nodes \
                             ({numa}); every gauntlet preset would reject \
                             this test at runtime and the test would \
                             never execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "max_numa_nodes" => {
                if let Some(v) = u64_from_option_some_lit(&fv.expr)
                    && v < numa as u64
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.max_numa_nodes \
                             (Some({v})) is below effective topology \
                             numa_nodes ({numa}); every gauntlet preset \
                             would reject this test at runtime and the \
                             test would never execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "min_cpus" => {
                if let Some(v) = u64_from_lit_expr(&fv.expr)
                    && v > total_cpus
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.min_cpus \
                             ({v}) exceeds effective topology total_cpus \
                             ({total_cpus} = llcs * cores * threads); \
                             every gauntlet preset would reject this \
                             test at runtime and the test would never \
                             execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "max_cpus" => {
                if let Some(v) = u64_from_option_some_lit(&fv.expr)
                    && v < total_cpus
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.max_cpus \
                             (Some({v})) is below effective topology \
                             total_cpus ({total_cpus} = llcs * cores * \
                             threads); every gauntlet preset would \
                             reject this test at runtime and the test \
                             would never execute.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            "requires_smt" => {
                if let Some(true) = bool_from_lit_expr(&fv.expr)
                    && threads_per_core < 2
                {
                    return Err(syn::Error::new_spanned(
                        &fv.expr,
                        format!(
                            "declare_scheduler!: constraints.requires_smt \
                             = true but effective topology \
                             threads_per_core = {threads_per_core}; SMT \
                             requires threads_per_core >= 2. Set topology \
                             threads_per_core to 2 (or higher) or drop \
                             the requires_smt constraint.{topology_origin_tail}",
                        ),
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract a `u64` from a literal integer expression. Returns `None`
/// for any other shape so non-literal field values pass through the
/// macro-time check.
fn u64_from_lit_expr(expr: &syn::Expr) -> Option<u64> {
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(li),
        ..
    }) = expr
    else {
        return None;
    };
    li.base10_parse().ok()
}

/// Extract a `u64` from a `Some(<int literal>)` expression. Returns
/// `None` for `None`, paths, or anything else — non-literal forms
/// pass through unchecked.
fn u64_from_option_some_lit(expr: &syn::Expr) -> Option<u64> {
    let syn::Expr::Call(call) = expr else {
        return None;
    };
    let syn::Expr::Path(ep) = &*call.func else {
        return None;
    };
    if path_last_segment_ident(&ep.path)? != "Some" {
        return None;
    }
    if call.args.len() != 1 {
        return None;
    }
    u64_from_lit_expr(&call.args[0])
}

/// Extract a `bool` from a literal boolean expression. Returns `None`
/// for any other shape.
fn bool_from_lit_expr(expr: &syn::Expr) -> Option<bool> {
    let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Bool(lb),
        ..
    }) = expr
    else {
        return None;
    };
    Some(lb.value())
}

/// Derive macro that generates a `Payload` const from an annotated
/// struct for a userspace binary workload (stress-ng, fio, and
/// similar tools test authors compose under a scheduler).
///
/// # Required struct-level attributes (`#[payload(...)]`)
///
/// - `binary = "..."` — the binary name resolved by the guest's
///   include-files infrastructure (required). Becomes
///   [`PayloadKind::Binary(name)`](ktstr::test_support::PayloadKind::Binary),
///   and is also auto-prepended to the emitted `include_files` slice
///   so the binary is packaged into the initramfs without needing a
///   separate `#[include_files("...")]` entry. Extra auxiliary files
///   (helpers, configs, fixtures) still go on `#[include_files(...)]`.
///
/// # Optional struct-level attributes
///
/// - `name = "..."` — short name used in logs and sidecar records.
///   Defaults to the binary name.
/// - `output = Json | ExitCode | LlmExtract("hint")` — how the
///   framework extracts metrics from the payload's stdout. The
///   variant names match the `OutputFormat` enum and the `Polarity`
///   kwarg grammar. Defaults to `ExitCode`. The `LlmExtract` form
///   accepts an optional string literal focus hint appended to the
///   default LLM prompt; bare `LlmExtract` with no parenthesized
///   argument is a shorthand for `LlmExtract()` (no hint).
///
/// # Optional outer attributes
///
/// - `#[default_args("--a", "--b", ...)]` — variadic string
///   literals appended to the binary's argv when the payload runs.
///   May repeat across multiple `#[default_args(...)]` attrs; entries
///   accumulate in source order.
/// - `#[default_check(...)]` — one [`MetricCheck`](ktstr::test_support::MetricCheck)
///   construction expression (e.g. `min("iops", 1000.0)`,
///   `exit_code_eq(0)`). May repeat; entries accumulate in source
///   order. Both `min(...)` and `MetricCheck::min(...)` are accepted: the
///   macro prepends `::ktstr::test_support::MetricCheck::` when the
///   expression doesn't already spell `MetricCheck::` on its callee path,
///   so bare constructors work without an import and qualified
///   constructors read naturally in modules that already have
///   `MetricCheck` in scope.
/// - `#[metric(name = "...", polarity = ..., unit = "...")]` —
///   kwarg form. `polarity` is one of `HigherBetter`, `LowerBetter`,
///   `TargetValue(f64)`, `Unknown`. May repeat; entries accumulate.
/// - `#[include_files("helper", "config.json", ...)]` — variadic
///   string literals appended to the emitted `include_files` slice
///   after the auto-injected binary entry. Each entry passes through
///   the same resolver used by the CLI `-i` flag (bare names search
///   host `PATH`; explicit paths must exist; directories are walked).
///   The primary binary is already packaged automatically, so this
///   attribute is only needed for auxiliary files the payload
///   depends on.
///
/// # Const name derivation
///
/// Strip trailing `"Payload"` suffix (if present), then convert to
/// `SCREAMING_SNAKE_CASE`. `FioPayload` → `FIO`,
/// `StressNgPayload` → `STRESS_NG`, `Fio` (no suffix) → `FIO`.
///
/// # Example
///
/// ```rust,ignore
/// use ktstr::prelude::*;
///
/// #[derive(Payload)]
/// #[payload(binary = "fio", output = Json)]
/// #[default_args("--output-format=json", "--minimal")]
/// #[default_check(exit_code_eq(0))]
/// #[metric(name = "jobs.0.read.iops", polarity = HigherBetter, unit = "iops")]
/// struct FioPayload;
/// ```
#[proc_macro_derive(Payload, attributes(payload, default_args, default_check, metric))]
pub fn derive_payload(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_payload_inner(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn derive_payload_inner(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;
    // Inherit the input struct's visibility so the emitted `const`
    // matches: `pub struct FooPayload` → `pub const FOO: Payload`.
    // Private structs produce private consts, preserving the
    // previous behavior for in-crate tests that rely on it.
    let struct_vis = &input.vis;

    // Reject non-struct inputs; the payload attribute grammar is
    // struct-only, keeping the attribute space unambiguous.
    if !matches!(&input.data, Data::Struct(_)) {
        return Err(syn::Error::new_spanned(
            struct_name,
            "Payload can only be derived for structs",
        ));
    }

    let mut binary: Option<String> = None;
    let mut name_override: Option<String> = None;
    // `None` means "not specified" → default ExitCode at emit time.
    // `Some(tokens)` holds the fully-qualified OutputFormat variant
    // the user selected (possibly with an LlmExtract hint expression).
    let mut output_tokens: Option<proc_macro2::TokenStream> = None;

    for attr in &input.attrs {
        if !attr.path().is_ident("payload") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("binary") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                binary = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("name") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                name_override = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("output") {
                let value = meta.value()?;
                let expr: syn::Expr = value.parse()?;
                output_tokens = Some(output_from_expr(&expr)?);
                Ok(())
            } else {
                Err(meta.error(format!(
                    "unknown payload attribute `{}`",
                    meta.path
                        .get_ident()
                        .map(|i| i.to_string())
                        .unwrap_or_default()
                )))
            }
        })?;
    }

    let binary = binary.ok_or_else(|| {
        syn::Error::new_spanned(struct_name, "missing `binary = \"...\"` in #[payload(...)]")
    })?;

    // Default output = ExitCode. Resolve once here to a canonical
    // TokenStream so the emitter only has one path.
    let output_tokens = output_tokens.unwrap_or_else(|| {
        quote! { ::ktstr::test_support::OutputFormat::ExitCode }
    });

    // `name` falls back to `binary` when omitted.
    let payload_name = name_override.unwrap_or_else(|| binary.clone());

    // Walk outer `#[default_args(...)]` / `#[default_check(...)]` /
    // `#[metric(...)]` / `#[include_files(...)]` attrs in source
    // order so the emitted slices match the declaration.
    let mut default_args: Vec<String> = Vec::new();
    let mut default_checks: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut metrics: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut seen_metric_names: Vec<String> = Vec::new();
    let mut include_files: Vec<String> = Vec::new();

    for attr in &input.attrs {
        if attr.path().is_ident("default_args") {
            // Variadic string literals: `#[default_args("--a", "--b")]`.
            let parser =
                syn::punctuated::Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated;
            let parsed = attr.parse_args_with(parser).map_err(|e| {
                syn::Error::new(
                    e.span(),
                    "default_args must be one or more string literals separated by `,`",
                )
            })?;
            for lit in parsed {
                default_args.push(lit.value());
            }
        } else if attr.path().is_ident("default_check") {
            // Single MetricCheck-constructing expression. Two forms accepted:
            //   - bare: `min("iops", 1000.0)` — the macro prepends
            //     `::ktstr::test_support::MetricCheck::` so users don't have
            //     to import `MetricCheck` in every module that derives.
            //   - qualified: `MetricCheck::min("iops", 1000.0)` — the user
            //     wrote `MetricCheck::` themselves; emit the expression
            //     unchanged so the user's own path resolution wins
            //     (and a double `MetricCheck::MetricCheck::` prefix can't happen).
            let expr: syn::Expr = attr.parse_args().map_err(|e| {
                syn::Error::new(
                    e.span(),
                    "default_check must be a MetricCheck constructor expression (e.g. min(\"iops\", 1000.0))",
                )
            })?;
            if expr_has_check_prefix(&expr) {
                default_checks.push(quote! { #expr });
            } else {
                default_checks.push(quote! { ::ktstr::test_support::MetricCheck::#expr });
            }
        } else if attr.path().is_ident("metric") {
            // Kwarg form: name = "...", polarity = ..., unit = "...".
            let (metric_name, tokens) = parse_metric_attr(attr)?;
            // Reject duplicate metric names — two `#[metric(name = "x", ...)]`
            // lines with the same name are almost certainly a copy-paste
            // typo; the runtime pipeline's `resolve_polarities` uses
            // last-wins semantics, so a duplicate silently shadows the
            // first hint without any signal to the test author.
            if let Some(existing) = seen_metric_names.iter().find(|n| *n == &metric_name) {
                return Err(syn::Error::new_spanned(
                    attr,
                    format!(
                        "duplicate metric name `{existing}` — each \
                         `#[metric(name = \"...\")]` declaration must name a \
                         distinct metric. Remove the duplicate or rename one \
                         of them."
                    ),
                ));
            }
            seen_metric_names.push(metric_name);
            metrics.push(tokens);
        } else if attr.path().is_ident("include_files") {
            // Variadic string literals: `#[include_files("helper",
            // "config.json")]`. Each entry is passed through to
            // `Payload::include_files` verbatim; the runtime
            // resolver (`resolve_include_files`) interprets bare
            // names vs explicit paths vs directories the same way
            // the CLI `-i` flag does. Order is preserved so the
            // user's declaration order is visible in the emitted
            // slice — useful when the resolver's dedup policy
            // reports a conflict, as the first-declared entry
            // wins.
            let parser =
                syn::punctuated::Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated;
            let parsed = attr.parse_args_with(parser).map_err(|e| {
                syn::Error::new(
                    e.span(),
                    "include_files must be one or more string literals separated by `,`",
                )
            })?;
            for lit in parsed {
                include_files.push(lit.value());
            }
        }
    }

    // Derive the const name: strip "Payload" suffix and uppercase.
    let struct_str = struct_name.to_string();
    let base = struct_str.strip_suffix("Payload").unwrap_or(&struct_str);
    if base.is_empty() {
        return Err(syn::Error::new(
            struct_name.span(),
            "struct name cannot be just \"Payload\"",
        ));
    }
    let const_name = format_ident!("{}", camel_to_screaming_snake(base));

    // Auto-inject the `binary` spec as the first entry in the emitted
    // `include_files` slice so `#[payload(binary = "X")]` alone is
    // enough to package `X` into the initramfs — no separate
    // `#[include_files("X")]` required. The runtime's
    // `dedupe_include_files` canonicalizes host paths, so a user who
    // also writes `#[include_files("X")]` (or lists the same binary
    // on `#[ktstr_test(extra_include_files = [..])]`) still works:
    // the duplicate collapses silently. User-declared entries follow
    // in source order — preserving the existing first-declared-wins
    // behavior within the user's own list.
    include_files.insert(0, binary.clone());

    let expanded = quote! {
        #struct_vis const #const_name: ::ktstr::test_support::Payload =
            ::ktstr::test_support::Payload::new(
                #payload_name,
                ::ktstr::test_support::PayloadKind::Binary(#binary),
                #output_tokens,
                &[#(#default_args),*],
                &[#(#default_checks),*],
                &[#(#metrics),*],
                &[#(#include_files),*],
                false,
                None,
                None,
            );
    };

    Ok(expanded)
}

/// Translate the user-facing `output = ...` expression into a
/// fully-qualified `OutputFormat` variant token stream. Accepts
/// the variant names as they appear on the `OutputFormat` enum,
/// so the attribute reads identically to `Polarity` below:
///
/// - `Json` / `ExitCode` — bare idents.
/// - `LlmExtract` — bare ident (no hint).
/// - `LlmExtract("hint")` — call with a single string literal.
/// - `LlmExtract()` — call with no args (no hint).
fn output_from_expr(expr: &syn::Expr) -> syn::Result<proc_macro2::TokenStream> {
    match expr {
        syn::Expr::Path(ep) => {
            let ident = ep.path.get_ident().ok_or_else(|| {
                syn::Error::new_spanned(expr, "expected `Json`, `ExitCode`, or `LlmExtract`")
            })?;
            match ident.to_string().as_str() {
                "Json" => Ok(quote! { ::ktstr::test_support::OutputFormat::Json }),
                "ExitCode" => Ok(quote! { ::ktstr::test_support::OutputFormat::ExitCode }),
                "LlmExtract" => {
                    Ok(quote! { ::ktstr::test_support::OutputFormat::LlmExtract(None) })
                }
                other => Err(syn::Error::new_spanned(
                    expr,
                    format!(
                        "unknown output format `{other}` (expected `Json`, `ExitCode`, or `LlmExtract`)"
                    ),
                )),
            }
        }
        syn::Expr::Call(call) => {
            // Only `LlmExtract(...)` is callable.
            let ident = match &*call.func {
                syn::Expr::Path(ep) => ep.path.get_ident().ok_or_else(|| {
                    syn::Error::new_spanned(expr, "expected `LlmExtract(...)` call form")
                })?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        expr,
                        "expected `LlmExtract(...)` call form",
                    ));
                }
            };
            if ident != "LlmExtract" {
                return Err(syn::Error::new_spanned(
                    expr,
                    format!(
                        "unknown output format `{ident}(...)` (only `LlmExtract(...)` takes arguments)"
                    ),
                ));
            }
            match call.args.len() {
                0 => Ok(quote! { ::ktstr::test_support::OutputFormat::LlmExtract(None) }),
                1 => {
                    let arg = &call.args[0];
                    match arg {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(ls),
                            ..
                        }) => {
                            let hint = ls.value();
                            Ok(quote! {
                                ::ktstr::test_support::OutputFormat::LlmExtract(Some(#hint))
                            })
                        }
                        _ => Err(syn::Error::new_spanned(
                            arg,
                            "LlmExtract argument must be a string literal hint",
                        )),
                    }
                }
                _ => Err(syn::Error::new_spanned(
                    expr,
                    "LlmExtract takes at most one string literal argument",
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "output must be `Json`, `ExitCode`, `LlmExtract`, or `LlmExtract(\"hint\")`",
        )),
    }
}

/// Parse one `#[metric(name = "...", polarity = ..., unit = "...")]`
/// attribute into a `(name, MetricHint { ... } token stream)` pair.
/// The name is returned separately so the caller can check for
/// duplicate `#[metric(name = ...)]` declarations across the struct.
///
/// `polarity` accepts bare idents `HigherBetter`, `LowerBetter`,
/// `Unknown`, and the call form `TargetValue(<float literal>)`. The
/// float literal is stamped into a `Polarity::TargetValue(lit)` so
/// the generated const is const-evaluable.
fn parse_metric_attr(attr: &syn::Attribute) -> syn::Result<(String, proc_macro2::TokenStream)> {
    let mut name: Option<String> = None;
    let mut polarity: Option<proc_macro2::TokenStream> = None;
    let mut unit: String = String::new();
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            name = Some(lit.value());
            Ok(())
        } else if meta.path.is_ident("polarity") {
            let value = meta.value()?;
            let expr: syn::Expr = value.parse()?;
            polarity = Some(polarity_from_expr(&expr)?);
            Ok(())
        } else if meta.path.is_ident("unit") {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            unit = lit.value();
            Ok(())
        } else {
            Err(meta.error(format!(
                "unknown metric attribute `{}` (expected name, polarity, unit)",
                meta.path
                    .get_ident()
                    .map(|i| i.to_string())
                    .unwrap_or_default()
            )))
        }
    })?;
    let name = name.ok_or_else(|| {
        syn::Error::new_spanned(attr, "metric attribute is missing `name = \"...\"`")
    })?;
    let polarity = polarity.unwrap_or_else(|| {
        quote! { ::ktstr::test_support::Polarity::Unknown }
    });
    let tokens = quote! {
        ::ktstr::test_support::MetricHint {
            name: #name,
            polarity: #polarity,
            unit: #unit,
        }
    };
    Ok((name, tokens))
}

/// Does this `#[default_check(...)]` expression already spell
/// `MetricCheck::` somewhere in its function path? Returns true for
/// `MetricCheck::min(...)` and `::ktstr::test_support::MetricCheck::min(...)`;
/// false for bare `min(...)`. Used to skip the macro's implicit
/// `::ktstr::test_support::MetricCheck::` prepend when the user has
/// already written the prefix, so `MetricCheck::MetricCheck::min(...)` can't
/// happen.
///
/// Only inspects the callee path of an `Expr::Call`; non-call
/// expressions (rare but legal: a free function returning `MetricCheck`,
/// or a `const` value) fall back to the prepend path, matching the
/// pre-bugfix behavior for anything that isn't a plain constructor
/// call. A future refactor could lift this to also handle
/// `MethodCall` / `Path`, but the MetricCheck API today is constructor
/// calls only — adding more shapes is a no-op until a new constructor
/// form lands.
fn expr_has_check_prefix(expr: &syn::Expr) -> bool {
    let syn::Expr::Call(call) = expr else {
        return false;
    };
    let syn::Expr::Path(expr_path) = &*call.func else {
        return false;
    };
    expr_path
        .path
        .segments
        .iter()
        .any(|seg| seg.ident == "MetricCheck")
}

/// Translate the user-facing `polarity = ...` expression to a
/// fully-qualified `Polarity` variant. Accepts the four enum
/// variants in bare-ident form (`HigherBetter`, `LowerBetter`,
/// `Unknown`) or as `TargetValue(<float>)`.
fn polarity_from_expr(expr: &syn::Expr) -> syn::Result<proc_macro2::TokenStream> {
    match expr {
        syn::Expr::Path(ep) => {
            let ident = ep.path.get_ident().ok_or_else(|| {
                syn::Error::new_spanned(
                    expr,
                    "expected `HigherBetter`, `LowerBetter`, `TargetValue(..)`, or `Unknown`",
                )
            })?;
            match ident.to_string().as_str() {
                "HigherBetter" => Ok(quote! { ::ktstr::test_support::Polarity::HigherBetter }),
                "LowerBetter" => Ok(quote! { ::ktstr::test_support::Polarity::LowerBetter }),
                "Unknown" => Ok(quote! { ::ktstr::test_support::Polarity::Unknown }),
                "TargetValue" => Err(syn::Error::new_spanned(
                    expr,
                    "TargetValue requires a float argument: `TargetValue(42.0)`",
                )),
                other => Err(syn::Error::new_spanned(
                    expr,
                    format!("unknown polarity `{other}`"),
                )),
            }
        }
        syn::Expr::Call(call) => {
            let ident = match &*call.func {
                syn::Expr::Path(ep) => ep.path.get_ident().ok_or_else(|| {
                    syn::Error::new_spanned(expr, "expected `TargetValue(<float>)`")
                })?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        expr,
                        "expected `TargetValue(<float>)`",
                    ));
                }
            };
            if ident != "TargetValue" {
                return Err(syn::Error::new_spanned(
                    expr,
                    format!(
                        "unknown polarity `{ident}(...)` (only `TargetValue` takes an argument)"
                    ),
                ));
            }
            if call.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    expr,
                    "TargetValue takes exactly one float literal argument",
                ));
            }
            let arg = &call.args[0];
            let lit = match arg {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Float(lf),
                    ..
                }) => lf,
                _ => {
                    return Err(syn::Error::new_spanned(
                        arg,
                        "TargetValue argument must be a float literal (e.g. 42.0)",
                    ));
                }
            };
            Ok(quote! { ::ktstr::test_support::Polarity::TargetValue(#lit) })
        }
        _ => Err(syn::Error::new_spanned(
            expr,
            "polarity must be HigherBetter, LowerBetter, TargetValue(<float>), or Unknown",
        )),
    }
}

// ============================================================================
// #[derive(Claim)] — pointwise-claim accessor generator
// ----------------------------------------------------------------------------
//
// Emits an extension trait `<StructName>Claim` with one
// `claim_<field_name>` method per public scalar field (returning
// `ClaimBuilder<'_, FieldType>`), plus container-typed accessors:
//
//   * `BTreeSet<T>` fields → `claim_<field_name>(&mut self, s: &Struct)
//     -> SetClaim<'_, T>`, dispatched through `Verdict::claim_set`.
//   * `Vec<T>` fields → `claim_<field_name>(&mut self, s: &Struct)
//     -> SeqClaim<'_, T>`, dispatched through `Verdict::claim_seq`.
//   * `BTreeMap<K, V>` / `HashMap<K, V>` fields → SKIPPED (no claim
//     surface for maps in v1; users reach for `claim!(verdict,
//     map.len())` or `claim!(verdict, map.contains_key(&k))` against
//     the explicit expression). Maps that need first-class support
//     can be added in a follow-up — the derive emits no method, so
//     the user's call site fails to compile rather than silently
//     dispatching to a wrong type.
//
// Per-field opt-out via `#[claim(skip)]`. Non-pub fields are skipped
// automatically (the claim surface is for the test author's view of
// the struct; private fields aren't part of that view).
//
// The generated trait is named `<StructName>Claim` with the same
// visibility as the input struct. The single
// `impl <StructName>Claim for ::ktstr::assert::Verdict` lives in the
// same expansion so callers only need `use ::ktstr::prelude::*` (which
// re-exports the trait) to bring the methods into scope.
//
// Label source: every method body is `verdict.claim(stringify!(field),
// ...)` — the label is the field's source-text identifier. Renaming
// the field updates both the method name AND the rendered failure
// label in lock-step; a stale call site that referenced the old
// method name fails to compile. This is the compile-mechanical
// drift-free axis that the design rejected manual-string labels in
// favor of.

/// Detect whether a `syn::Type` is a path whose last segment matches
/// `name` (e.g. `BTreeSet<T>` → `is_path_named(ty, "BTreeSet") == true`).
/// Used to dispatch container fields onto `claim_set` / `claim_seq`
/// without forcing the caller to import `BTreeSet` / `Vec` from a
/// specific module path. Misses `std::collections::BTreeSet<T>` if the
/// caller uses an alias — acceptable in v1 because the project's
/// stats structs all use the canonical `BTreeSet` / `Vec` names.
fn is_path_named(ty: &syn::Type, name: &str) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(ident) = path_last_segment_ident(&tp.path)
    {
        return ident == name;
    }
    false
}

/// Inner element type from a `Container<T>` syn::Type. Returns `None`
/// when the path has no angle-bracketed argument or the argument is
/// not a type (e.g. lifetime-only). Used by [`derive_claim_inner`] to
/// thread the element type through the emitted accessor's return
/// type.
fn first_type_arg(ty: &syn::Type) -> Option<&syn::Type> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
        && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
    {
        for arg in &args.args {
            if let syn::GenericArgument::Type(inner) = arg {
                return Some(inner);
            }
        }
    }
    None
}

/// Detect a `#[claim(skip)]` attribute on a field. Returns true when
/// any `#[claim(skip)]` is present — the field gets no claim accessor
/// in the emitted trait.
fn field_has_claim_skip(field: &syn::Field) -> bool {
    for attr in &field.attrs {
        if !attr.path().is_ident("claim") {
            continue;
        }
        let mut skip = false;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
            }
            Ok(())
        });
        if skip {
            return true;
        }
    }
    false
}

/// Generate per-field claim accessors on a stats struct.
///
/// See the section comment above this fn for the dispatch rules and
/// label invariant. Reject non-struct inputs and tuple-struct inputs
/// — the claim API is keyed on field names, which tuple structs do
/// not have.
#[proc_macro_derive(Claim, attributes(claim))]
pub fn derive_claim(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_claim_inner(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn derive_claim_inner(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = &input.ident;
    let struct_vis = &input.vis;
    let trait_name = format_ident!("{}Claim", struct_name);

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    struct_name,
                    "Claim cannot be derived for tuple structs (claim labels need field names)",
                ));
            }
            Fields::Unit => {
                return Err(syn::Error::new_spanned(
                    struct_name,
                    "Claim cannot be derived for unit structs (no fields to claim against)",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                struct_name,
                "Claim can only be derived for structs",
            ));
        }
    };

    let mut trait_methods: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut impl_methods: Vec<proc_macro2::TokenStream> = Vec::new();

    for field in fields {
        // Private fields are not part of the claim surface — the
        // generated accessors must be callable from outside the
        // defining crate. A private field would force the impl
        // body to dereference an inaccessible field path. Skip
        // silently rather than erroring; the field is invisible
        // to the API surface either way.
        if !matches!(field.vis, syn::Visibility::Public(_)) {
            continue;
        }
        if field_has_claim_skip(field) {
            continue;
        }
        let Some(field_name) = field.ident.as_ref() else {
            continue;
        };
        let method_name = format_ident!("claim_{}", field_name);
        let field_ty = &field.ty;

        // Container dispatch: BTreeSet<T> / Vec<T> route through
        // claim_set / claim_seq with element type T. Map types are
        // skipped (no method emitted).
        if is_path_named(field_ty, "BTreeSet") {
            let Some(elem) = first_type_arg(field_ty) else {
                continue;
            };
            trait_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SetClaim<'a, #elem>;
            });
            impl_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SetClaim<'a, #elem> {
                    verdict.claim_set(stringify!(#field_name), &self.#field_name)
                }
            });
            continue;
        }
        if is_path_named(field_ty, "Vec") {
            let Some(elem) = first_type_arg(field_ty) else {
                continue;
            };
            trait_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SeqClaim<'a, #elem>;
            });
            impl_methods.push(quote! {
                fn #method_name<'a>(
                    &'a self,
                    verdict: &'a mut ::ktstr::assert::Verdict,
                ) -> ::ktstr::assert::SeqClaim<'a, #elem> {
                    verdict.claim_seq(stringify!(#field_name), &self.#field_name)
                }
            });
            continue;
        }
        if is_path_named(field_ty, "BTreeMap") || is_path_named(field_ty, "HashMap") {
            // Skip map fields — no claim surface in v1.
            continue;
        }

        // Scalar field. Emit a method returning `ClaimBuilder<'_,
        // FieldType>` that copies/clones the value. Cloning compiles
        // for both `Copy` and `Clone` types; primitive fields lower
        // to a single move at -O.
        trait_methods.push(quote! {
            fn #method_name<'a>(
                &'a self,
                verdict: &'a mut ::ktstr::assert::Verdict,
            ) -> ::ktstr::assert::ClaimBuilder<'a, #field_ty>;
        });
        impl_methods.push(quote! {
            fn #method_name<'a>(
                &'a self,
                verdict: &'a mut ::ktstr::assert::Verdict,
            ) -> ::ktstr::assert::ClaimBuilder<'a, #field_ty> {
                verdict.claim(stringify!(#field_name), ::core::clone::Clone::clone(&self.#field_name))
            }
        });
    }

    let doc = format!(
        "Pointwise-claim accessors generated by `#[derive(Claim)]` on \
         [`{name}`]. One `claim_<field>` method per public field, taking \
         `&mut Verdict` as the accumulator; container fields (`BTreeSet`/`Vec`) \
         route through `SetClaim`/`SeqClaim`. Method dispatch keys on the \
         stats struct's type, so identical field names across distinct stats \
         structs do not collide. Brought into scope via `use ktstr::prelude::*`.",
        name = struct_name,
    );

    Ok(quote! {
        #[doc = #doc]
        #struct_vis trait #trait_name {
            #(#trait_methods)*
        }

        impl #trait_name for #struct_name {
            #(#impl_methods)*
        }
    })
}

/// Convert JSON-like Rust tokens into a `&'static str` at compile time.
///
/// Accepts a superset of JSON syntax using Rust token trees:
/// - Objects: `{ "key": value, ... }`
/// - Arrays: `[value, ...]`
/// - Strings: `"hello"`
/// - Numbers: `42`, `3.14`, `-1`
/// - Booleans: `true`, `false`
/// - Null: `null`
/// - Trailing commas are stripped
///
/// ```rust,ignore
/// const CFG: &str = ktstr::json!({
///     "layers": [{
///         "name": "batch",
///         "kind": { "Grouped": { "cpus_range": [0, 4] } },
///     }],
/// });
/// ```
#[proc_macro]
pub fn json(input: TokenStream) -> TokenStream {
    let mut out = String::new();
    tokens_to_json(&mut out, proc_macro2::TokenStream::from(input));
    let lit = syn::LitStr::new(&out, proc_macro2::Span::call_site());
    TokenStream::from(quote! { #lit })
}

fn tokens_to_json(out: &mut String, tokens: proc_macro2::TokenStream) {
    for tt in tokens {
        match tt {
            proc_macro2::TokenTree::Group(g) => match g.delimiter() {
                proc_macro2::Delimiter::Brace => {
                    out.push('{');
                    emit_comma_separated(out, g.stream(), true);
                    out.push('}');
                }
                proc_macro2::Delimiter::Bracket => {
                    out.push('[');
                    emit_comma_separated(out, g.stream(), false);
                    out.push(']');
                }
                proc_macro2::Delimiter::Parenthesis => {
                    out.push('(');
                    tokens_to_json(out, g.stream());
                    out.push(')');
                }
                proc_macro2::Delimiter::None => {
                    tokens_to_json(out, g.stream());
                }
            },
            proc_macro2::TokenTree::Literal(lit) => {
                out.push_str(&lit.to_string());
            }
            proc_macro2::TokenTree::Ident(id) => {
                let s = id.to_string();
                match s.as_str() {
                    "true" | "false" | "null" => out.push_str(&s),
                    _ => {
                        out.push('"');
                        out.push_str(&s);
                        out.push('"');
                    }
                }
            }
            proc_macro2::TokenTree::Punct(p) => {
                let ch = p.as_char();
                if ch == '-' {
                    out.push('-');
                } else if ch == ':' {
                    out.push(':');
                } else if ch == ',' {
                    out.push(',');
                } else {
                    out.push(ch);
                }
            }
        }
    }
}

fn emit_comma_separated(out: &mut String, tokens: proc_macro2::TokenStream, _is_object: bool) {
    let items = split_on_commas(tokens);
    let mut first = true;
    for item in &items {
        if item.is_empty() {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        tokens_to_json(out, item.clone());
    }
}

fn split_on_commas(tokens: proc_macro2::TokenStream) -> Vec<proc_macro2::TokenStream> {
    let mut result = Vec::new();
    let mut current = proc_macro2::TokenStream::new();
    for tt in tokens {
        match &tt {
            proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => {
                result.push(current);
                current = proc_macro2::TokenStream::new();
            }
            _ => {
                current.extend(std::iter::once(tt));
            }
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn camel_to_screaming_snake_acronym_run() {
        assert_eq!(camel_to_screaming_snake("HTTPServer"), "HTTP_SERVER");
    }

    #[test]
    fn camel_to_screaming_snake_single_word() {
        assert_eq!(camel_to_screaming_snake("Llc"), "LLC");
    }

    #[test]
    fn camel_to_screaming_snake_all_caps_passthrough() {
        assert_eq!(camel_to_screaming_snake("LLC"), "LLC");
    }

    #[test]
    fn option_tokens_some_int() {
        let opt: Option<u32> = Some(42);
        let ts = option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { Some(42u32) }.to_string());
    }

    #[test]
    fn option_tokens_none_int() {
        let opt: Option<u32> = None;
        let ts = option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { None }.to_string());
    }

    #[test]
    fn option_tokens_some_bool() {
        let opt: Option<bool> = Some(true);
        let ts = option_tokens(&opt);
        assert_eq!(ts.to_string(), quote! { Some(true) }.to_string());
    }
}
