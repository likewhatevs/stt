use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Data, DeriveInput, Fields, ItemFn, Meta, MetaNameValue, parse::Parser, parse_macro_input,
};

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
/// [`ktstr::assert::Assert`] (verification thresholds). A few are
/// worth calling out because their names differ from the underlying
/// field or because they have nontrivial defaults:
///
///   - `llcs = N` — number of LLCs (default: inherited from
///     scheduler, or 1). `sockets = N` is a deprecated alias kept
///     for backward compatibility with pre-topology-rename tests;
///     prefer `llcs` in new code.
///   - `cores = N` (default: inherited from scheduler, or 2)
///   - `threads = N` (default: inherited from scheduler, or 1)
///   - `numa_nodes = N` (default: inherited from scheduler, or 1)
///   - `memory_mb = N` (default: 2048)
///   - `duration_s = N` — scenario run duration in seconds; maps
///     onto `KtstrTestEntry::duration`
///   - `watchdog_timeout_s = N` — watchdog fire threshold in
///     seconds; maps onto `KtstrTestEntry::watchdog_timeout`
///   - `scheduler = PATH` — path to a `const Scheduler` (default
///     `Scheduler::EEVDF`, which runs without an scx scheduler)
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
    let mut scheduler: Option<syn::Path> = None;
    let mut payload: Option<syn::Path> = None;
    let mut payload_set = false;
    let mut workloads: Vec<syn::Path> = Vec::new();
    let mut workloads_set = false;
    let mut auto_repro = true;
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
    let mut required_flags: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut excluded_flags: Vec<proc_macro2::TokenStream> = Vec::new();
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
    let mut performance_mode: bool = false;
    let mut duration_s: u64 = 2;
    let mut workers_per_cgroup: u32 = 2;
    let mut bpf_map_write: Option<syn::Path> = None;
    let mut expect_err: bool = false;
    let mut host_only: bool = false;

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
                    "auto_repro" | "not_starved" | "isolation" | "performance_mode"
                    | "requires_smt" | "expect_err" | "fail_on_stall" | "host_only" => {
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
                            "auto_repro" => auto_repro = lit_bool.value(),
                            "not_starved" => not_starved = Some(lit_bool.value()),
                            "isolation" => isolation = Some(lit_bool.value()),
                            "performance_mode" => performance_mode = lit_bool.value(),
                            "requires_smt" => {
                                requires_smt = lit_bool.value();
                                requires_smt_set = true;
                            }
                            "expect_err" => expect_err = lit_bool.value(),
                            "fail_on_stall" => fail_on_stall = Some(lit_bool.value()),
                            "host_only" => host_only = lit_bool.value(),
                            _ => unreachable!(),
                        }
                    }
                    "sockets"
                    | "llcs"
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
                    | "min_sockets"
                    | "min_numa_nodes"
                    | "min_llcs"
                    | "min_cpus"
                    | "max_llcs"
                    | "max_numa_nodes"
                    | "max_cpus"
                    | "max_p99_wake_latency_ns" => {
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
                            "sockets" | "llcs" => {
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
                                    .unwrap_or_else(|e| panic!("{e}"))
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
                            "watchdog_timeout_s" => {
                                watchdog_timeout_s = lit_int
                                    .base10_parse::<u64>()
                                    .unwrap_or_else(|e| panic!("{e}"))
                            }
                            "duration_s" => {
                                duration_s = lit_int
                                    .base10_parse::<u64>()
                                    .unwrap_or_else(|e| panic!("{e}"))
                            }
                            "workers_per_cgroup" => {
                                workers_per_cgroup = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"))
                            }
                            "max_local_dsq_depth" => {
                                max_local_dsq_depth = Some(
                                    lit_int
                                        .base10_parse::<u32>()
                                        .unwrap_or_else(|e| panic!("{e}")),
                                )
                            }
                            "min_sockets" | "min_numa_nodes" => {
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
                    "required_flags" | "excluded_flags" => {
                        let arr = match value {
                            syn::Expr::Array(ea) => ea,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    format!("expected array for {ident}"),
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        let target = match ident.as_str() {
                            "required_flags" => &mut required_flags,
                            "excluded_flags" => &mut excluded_flags,
                            _ => unreachable!(),
                        };
                        for elem in &arr.elems {
                            match elem {
                                syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(ls),
                                    ..
                                }) => {
                                    let val = ls.value();
                                    target.push(quote! { #val });
                                }
                                syn::Expr::Path(_) => {
                                    target.push(quote! { #elem });
                                }
                                _ => {
                                    return syn::Error::new_spanned(
                                        elem,
                                        format!(
                                            "expected string literal or path expression in {ident}"
                                        ),
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
                            format!("unknown attribute `{ident}`, expected: llcs, sockets, cores, threads, numa_nodes, memory_mb, scheduler, payload, workloads, auto_repro, not_starved, isolation, max_gap_ms, max_spread_pct, max_throughput_cv, min_work_rate, max_p99_wake_latency_ns, max_wake_latency_cv, min_iteration_rate, max_migration_ratio, max_imbalance_ratio, max_local_dsq_depth, fail_on_stall, sustained_samples, max_fallback_rate, max_keep_last_rate, min_page_locality, max_cross_node_migration_ratio, max_slow_tier_ratio, extra_sched_args, required_flags, excluded_flags, min_numa_nodes, min_sockets, min_llcs, requires_smt, min_cpus, max_llcs, max_numa_nodes, max_cpus, watchdog_timeout_s, performance_mode, duration_s, workers_per_cgroup, bpf_map_write, expect_err, host_only"),
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
    // `KtstrTestEntry` is `&'static Payload`; callers pass either a
    // `{NAME}_PAYLOAD` wrapper emitted by `#[derive(Scheduler)]` or
    // `Payload::EEVDF` directly. The default is the kernel-default
    // placeholder.
    let scheduler_tokens = match &scheduler {
        Some(p) => {
            quote! { &#p }
        }
        None => {
            quote! { &::ktstr::test_support::Payload::EEVDF }
        }
    };

    // Build topology tokens. Each dimension independently inherits from
    // the scheduler payload's topology when not explicitly set.
    // `Payload::topology()` is a `const fn` that returns the inner
    // scheduler's `Topology` for scheduler-kind payloads and
    // `Topology::DEFAULT_FOR_PAYLOAD` for binary-kind, so the field
    // access below remains valid inside a `const` initializer.
    let llcs_tokens = if llcs_set {
        let l = llcs;
        quote! { #l }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology().llcs }
    } else {
        let l = llcs;
        quote! { #l }
    };
    let cores_tokens = if cores_set {
        let c = cores;
        quote! { #c }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology().cores_per_llc }
    } else {
        let c = cores;
        quote! { #c }
    };
    let threads_tokens = if threads_set {
        let t = threads;
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology().threads_per_core }
    } else {
        let t = threads;
        quote! { #t }
    };
    let numa_nodes_tokens = if numa_nodes_set {
        let n = numa_nodes;
        quote! { #n }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology().numa_nodes }
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

    let test_body = if expect_err {
        quote! {
            let result = ::ktstr::test_support::run_ktstr_test(&#entry_name);
            assert!(
                result.is_err(),
                "expected test to fail but it passed",
            );
        }
    } else {
        quote! {
            let _result = ::ktstr::test_support::run_ktstr_test(&#entry_name).unwrap();
        }
    };

    // Build constraint tokens. Each field independently inherits from
    // the scheduler's constraints when not explicitly set, following
    // the same pattern as topology inheritance.
    let min_numa_nodes_tokens = if min_numa_nodes_set {
        let v = min_numa_nodes;
        quote! { #v }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints().min_numa_nodes }
    } else {
        let v = min_numa_nodes;
        quote! { #v }
    };
    let max_numa_nodes_tokens = if max_numa_nodes_set {
        let t = option_tokens(&max_numa_nodes);
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints().max_numa_nodes }
    } else {
        let t = option_tokens(&max_numa_nodes);
        quote! { #t }
    };
    let min_llcs_tokens = if min_llcs_set {
        let v = min_llcs;
        quote! { #v }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints().min_llcs }
    } else {
        let v = min_llcs;
        quote! { #v }
    };
    let max_llcs_tokens = if max_llcs_set {
        let t = option_tokens(&max_llcs);
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints().max_llcs }
    } else {
        let t = option_tokens(&max_llcs);
        quote! { #t }
    };
    let requires_smt_tokens = if requires_smt_set {
        quote! { #requires_smt }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints().requires_smt }
    } else {
        quote! { #requires_smt }
    };
    let min_cpus_tokens = if min_cpus_set {
        let v = min_cpus;
        quote! { #v }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints().min_cpus }
    } else {
        let v = min_cpus;
        quote! { #v }
    };
    let max_cpus_tokens = if max_cpus_set {
        let t = option_tokens(&max_cpus);
        quote! { #t }
    } else if let Some(ref p) = scheduler {
        quote! { #p.constraints().max_cpus }
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
            memory_mb: #memory_mb,
            scheduler: #scheduler_tokens,
            payload: #payload_tokens,
            workloads: #workloads_tokens,
            auto_repro: #auto_repro,
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
            extra_sched_args: &[#(#extra_sched_args),*],
            required_flags: &[#(#required_flags),*],
            excluded_flags: &[#(#excluded_flags),*],
            watchdog_timeout: ::std::time::Duration::from_secs(#watchdog_timeout_s),
            bpf_map_write: #bpf_map_write_tokens,
            performance_mode: #performance_mode,
            duration: ::std::time::Duration::from_secs(#duration_s),
            workers_per_cgroup: #workers_per_cgroup,
            expect_err: #expect_err,
            host_only: #host_only,
        };

        #[test]
        fn #orig_name() {
            #test_body
        }
    };

    expanded.into()
}

/// Convert a CamelCase identifier to kebab-case.
///
/// Handles acronyms (consecutive uppercase): a separator is inserted
/// before the last letter of a run when followed by lowercase.
///
/// `Llc` -> `"llc"`, `RejectPin` -> `"reject-pin"`, `NoCtrl` -> `"no-ctrl"`,
/// `LLC` -> `"llc"`, `HTTPServer` -> `"http-server"`.
fn camel_to_kebab(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, &ch) in chars.iter().enumerate() {
        if ch.is_uppercase() && i > 0 {
            let prev_upper = chars[i - 1].is_uppercase();
            let next_lower = chars.get(i + 1).is_some_and(|c| c.is_lowercase());
            // Insert separator when:
            // - previous char is lowercase (standard CamelCase boundary), OR
            // - previous char is uppercase AND next char is lowercase
            //   (end of acronym run: "HTTPServer" -> "http-server")
            if !prev_upper || next_lower {
                out.push('-');
            }
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
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

/// Derive macro that generates a `Scheduler` const, `FlagDecl` statics,
/// and associated name constants from an annotated enum.
///
/// # Scheduler attributes (`#[scheduler(...)]`)
///
/// | Attribute | Required | Description |
/// |---|---|---|
/// | `name = "..."` | yes | Scheduler name passed to `Scheduler::new()` |
/// | `binary = "..."` | no | Binary name for `SchedulerSpec::Discover(...)`. Omit for EEVDF. |
/// | `topology(N, L, C, T)` | no | Default VM topology `(numa_nodes, llcs, cores, threads)`. Defaults to `(1, 1, 2, 1)`. |
/// | `cgroup_parent = "..."` | no | Cgroup parent path. Must begin with `/` (e.g. `"/ktstr"`). |
/// | `sched_args = [...]` | no | Default scheduler CLI args. |
/// | `sysctls = [Sysctl::new("key", "value"), ...]` | no | Guest sysctls applied before the scheduler starts. |
/// | `kargs = ["arg1", "arg2"]` | no | Extra kernel command-line args appended when booting the VM. |
/// | `min_numa_nodes = N` | no | Minimum NUMA nodes for gauntlet filtering. |
/// | `max_numa_nodes = N` | no | Maximum NUMA nodes for gauntlet filtering. |
/// | `min_llcs = N` | no | Minimum LLCs for gauntlet filtering. |
/// | `max_llcs = N` | no | Maximum LLCs for gauntlet filtering. |
/// | `min_cpus = N` | no | Minimum total CPUs for gauntlet filtering. |
/// | `max_cpus = N` | no | Maximum total CPUs for gauntlet filtering. |
/// | `requires_smt = bool` | no | Require SMT (threads_per_core > 1). |
/// | `config_file = "..."` | no | Host-side path to an opaque config file passed to the scheduler via `--config`. |
///
/// # Flag attributes (`#[flag(...)]`)
///
/// | Attribute | Description |
/// |---|---|
/// | `args = ["--flag-a", "--flag-b"]` | CLI args passed when this flag is active |
/// | `requires = [OtherVariant]` | Variants that must also be active |
///
/// # Generated items
///
/// Given `enum MitosisFlag { Llc, Steal }`:
///
/// - Per-variant `static FlagDecl` entries
/// - A `static &[&FlagDecl]` flags array
/// - `const MITOSIS: Scheduler` (see naming below)
/// - `impl MitosisFlag { pub const LLC: &str = "llc"; pub const STEAL: &str = "steal"; }`
///
/// The associated constants enable typed flag references:
/// `required_flags = [MitosisFlag::LLC]` in `#[ktstr_test]`.
///
/// # Const name derivation
///
/// The generated `Scheduler` const name is derived from the enum name:
/// 1. Strip trailing `"Flag"` or `"Flags"` suffix (if present)
/// 2. Convert to `SCREAMING_SNAKE_CASE`
///
/// Examples: `MitosisFlag` -> `MITOSIS`, `EevdfFlags` -> `EEVDF`,
/// `MySchedFlag` -> `MY_SCHED`, `Mitosis` -> `MITOSIS`.
///
/// # Variant naming
///
/// Variant identifiers are converted to kebab-case for the flag name.
/// Consecutive uppercase letters are treated as acronyms:
/// `Llc` -> `"llc"`, `RejectPin` -> `"reject-pin"`, `LLC` -> `"llc"`.
///
/// # Example
///
/// ```rust,ignore
/// use ktstr::prelude::*;
///
/// #[derive(Scheduler)]
/// #[scheduler(name = "mitosis", binary = "scx_mitosis", topology(1, 2, 4, 1),
///             cgroup_parent = "/ktstr", sched_args = ["--exit-dump-len", "1048576"])]
/// #[allow(dead_code)]
/// enum MitosisFlag {
///     #[flag(args = ["--enable-llc-awareness"])]
///     Llc,
///     #[flag(args = ["--enable-work-stealing"], requires = [Llc])]
///     Steal,
/// }
/// ```
#[proc_macro_derive(Scheduler, attributes(scheduler, flag))]
pub fn derive_scheduler(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_scheduler_inner(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn derive_scheduler_inner(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let enum_name = &input.ident;

    // Parse #[scheduler(...)] attributes
    let mut sched_name: Option<String> = None;
    let mut sched_binary: Option<String> = None;
    let mut sched_topology: Option<(u32, u32, u32, u32)> = None;
    let mut sched_cgroup_parent: Option<String> = None;
    let mut sched_args: Vec<String> = Vec::new();
    let mut sched_sysctls: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut sched_sysctls_set = false;
    let mut sched_kargs: Vec<String> = Vec::new();
    let mut sched_kargs_set = false;
    let mut sched_min_numa_nodes: Option<u32> = None;
    let mut sched_max_numa_nodes: Option<u32> = None;
    let mut sched_min_llcs: Option<u32> = None;
    let mut sched_max_llcs: Option<u32> = None;
    let mut sched_min_cpus: Option<u32> = None;
    let mut sched_max_cpus: Option<u32> = None;
    let mut sched_requires_smt: Option<bool> = None;
    let mut sched_config_file: Option<String> = None;

    for attr in &input.attrs {
        if !attr.path().is_ident("scheduler") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                sched_name = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("binary") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                sched_binary = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("topology") {
                let content;
                syn::parenthesized!(content in meta.input);
                let numa_nodes_lit: syn::LitInt = content.parse()?;
                let _: syn::Token![,] = content.parse()?;
                let llcs_lit: syn::LitInt = content.parse()?;
                let _: syn::Token![,] = content.parse()?;
                let cores: syn::LitInt = content.parse()?;
                let _: syn::Token![,] = content.parse()?;
                let threads: syn::LitInt = content.parse()?;
                sched_topology = Some((
                    numa_nodes_lit.base10_parse()?,
                    llcs_lit.base10_parse()?,
                    cores.base10_parse()?,
                    threads.base10_parse()?,
                ));
                Ok(())
            } else if meta.path.is_ident("cgroup_parent") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                sched_cgroup_parent = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("sched_args") {
                let value = meta.value()?;
                let arr: syn::ExprArray = value.parse()?;
                for elem in &arr.elems {
                    match elem {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(ls),
                            ..
                        }) => sched_args.push(ls.value()),
                        _ => {
                            return Err(syn::Error::new_spanned(
                                elem,
                                "expected string literal in sched_args",
                            ));
                        }
                    }
                }
                Ok(())
            } else if meta.path.is_ident("sysctls") {
                let value = meta.value()?;
                let arr: syn::ExprArray = value.parse()?;
                sched_sysctls_set = true;
                for elem in &arr.elems {
                    sched_sysctls.push(elem.to_token_stream());
                }
                Ok(())
            } else if meta.path.is_ident("kargs") {
                let value = meta.value()?;
                let arr: syn::ExprArray = value.parse()?;
                sched_kargs_set = true;
                for elem in &arr.elems {
                    match elem {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(ls),
                            ..
                        }) => sched_kargs.push(ls.value()),
                        _ => {
                            return Err(syn::Error::new_spanned(
                                elem,
                                "expected string literal in kargs",
                            ));
                        }
                    }
                }
                Ok(())
            } else if meta.path.is_ident("min_numa_nodes") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                sched_min_numa_nodes = Some(lit.base10_parse()?);
                Ok(())
            } else if meta.path.is_ident("max_numa_nodes") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                sched_max_numa_nodes = Some(lit.base10_parse()?);
                Ok(())
            } else if meta.path.is_ident("min_llcs") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                sched_min_llcs = Some(lit.base10_parse()?);
                Ok(())
            } else if meta.path.is_ident("max_llcs") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                sched_max_llcs = Some(lit.base10_parse()?);
                Ok(())
            } else if meta.path.is_ident("min_cpus") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                sched_min_cpus = Some(lit.base10_parse()?);
                Ok(())
            } else if meta.path.is_ident("max_cpus") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                sched_max_cpus = Some(lit.base10_parse()?);
                Ok(())
            } else if meta.path.is_ident("requires_smt") {
                let value = meta.value()?;
                let lit: syn::LitBool = value.parse()?;
                sched_requires_smt = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("config_file") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                sched_config_file = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error(format!(
                    "unknown scheduler attribute `{}`",
                    meta.path
                        .get_ident()
                        .map(|i| i.to_string())
                        .unwrap_or_default()
                )))
            }
        })?;
    }

    let sched_name = sched_name
        .ok_or_else(|| syn::Error::new_spanned(enum_name, "missing `name` in #[scheduler(...)]"))?;

    // Validate topology values at compile time.
    if let Some((n, l, c, t)) = sched_topology {
        if n == 0 || l == 0 || c == 0 || t == 0 {
            return Err(syn::Error::new_spanned(
                enum_name,
                "topology values must all be > 0",
            ));
        }
        if l % n != 0 {
            return Err(syn::Error::new_spanned(
                enum_name,
                format!("topology: llcs ({l}) must be divisible by numa_nodes ({n})"),
            ));
        }
    }

    // Extract enum variants
    let variants = match &input.data {
        Data::Enum(data) => &data.variants,
        _ => {
            return Err(syn::Error::new_spanned(
                enum_name,
                "Scheduler can only be derived for enums",
            ));
        }
    };

    // Validate all variants are unit variants
    for v in variants {
        if !matches!(v.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                v,
                "Scheduler derive requires unit variants",
            ));
        }
    }

    // Collect variant info
    struct FlagInfo {
        ident: syn::Ident,
        kebab_name: String,
        screaming_snake: String,
        args: Vec<String>,
        requires: Vec<syn::Ident>,
    }

    let mut flag_infos: Vec<FlagInfo> = Vec::new();

    for v in variants {
        let ident = v.ident.clone();
        let kebab_name = camel_to_kebab(&ident.to_string());
        let screaming_snake = camel_to_screaming_snake(&ident.to_string());
        let mut args: Vec<String> = Vec::new();
        let mut requires: Vec<syn::Ident> = Vec::new();

        for attr in &v.attrs {
            if !attr.path().is_ident("flag") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("args") {
                    let value = meta.value()?;
                    let arr: syn::ExprArray = value.parse()?;
                    for elem in &arr.elems {
                        match elem {
                            syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(ls),
                                ..
                            }) => args.push(ls.value()),
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    elem,
                                    "expected string literal in flag args",
                                ));
                            }
                        }
                    }
                    Ok(())
                } else if meta.path.is_ident("requires") {
                    let value = meta.value()?;
                    let arr: syn::ExprArray = value.parse()?;
                    for elem in &arr.elems {
                        match elem {
                            syn::Expr::Path(ep) => {
                                let req_ident = ep
                                    .path
                                    .get_ident()
                                    .ok_or_else(|| {
                                        syn::Error::new_spanned(
                                            ep,
                                            "expected variant identifier in requires",
                                        )
                                    })?
                                    .clone();
                                requires.push(req_ident);
                            }
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    elem,
                                    "expected variant identifier in requires",
                                ));
                            }
                        }
                    }
                    Ok(())
                } else {
                    Err(meta.error(format!(
                        "unknown flag attribute `{}`",
                        meta.path
                            .get_ident()
                            .map(|i| i.to_string())
                            .unwrap_or_default()
                    )))
                }
            })?;
        }

        flag_infos.push(FlagInfo {
            ident,
            kebab_name,
            screaming_snake,
            args,
            requires,
        });
    }

    // Validate requires references exist
    let variant_names: Vec<String> = flag_infos.iter().map(|f| f.ident.to_string()).collect();
    for fi in &flag_infos {
        for req in &fi.requires {
            if !variant_names.contains(&req.to_string()) {
                return Err(syn::Error::new_spanned(
                    req,
                    format!(
                        "unknown variant `{}` in requires (expected one of: {})",
                        req,
                        variant_names.join(", ")
                    ),
                ));
            }
        }
    }

    // Generate static FlagDecl names
    let enum_upper = camel_to_screaming_snake(&enum_name.to_string());
    let decl_idents: Vec<syn::Ident> = flag_infos
        .iter()
        .map(|fi| format_ident!("__{}_DECL_{}", enum_upper, fi.screaming_snake))
        .collect();

    // Generate static FlagDecl entries
    let mut decl_statics = Vec::new();
    for (i, fi) in flag_infos.iter().enumerate() {
        let decl_ident = &decl_idents[i];
        let name_str = &fi.kebab_name;
        let args = &fi.args;

        let requires_tokens: Vec<proc_macro2::TokenStream> = fi
            .requires
            .iter()
            .map(|req_ident| {
                let req_idx = flag_infos
                    .iter()
                    .position(|f| f.ident == *req_ident)
                    .unwrap();
                let req_decl_ident = &decl_idents[req_idx];
                quote! { &#req_decl_ident }
            })
            .collect();

        decl_statics.push(quote! {
            static #decl_ident: ::ktstr::scenario::flags::FlagDecl = ::ktstr::scenario::flags::FlagDecl {
                name: #name_str,
                args: &[#(#args),*],
                requires: &[#(#requires_tokens),*],
            };
        });
    }

    // Generate the flags array
    let flags_array_ident = format_ident!("__{}_FLAGS", enum_upper);
    let decl_refs: Vec<proc_macro2::TokenStream> =
        decl_idents.iter().map(|di| quote! { &#di }).collect();

    // Generate associated name constants
    let name_consts: Vec<proc_macro2::TokenStream> = flag_infos
        .iter()
        .map(|fi| {
            let const_ident = format_ident!("{}", fi.screaming_snake);
            let name_str = &fi.kebab_name;
            quote! {
                pub const #const_ident: &'static str = #name_str;
            }
        })
        .collect();

    // Derive the const name from enum name: strip "Flag"/"Flags" suffix, uppercase
    let enum_str = enum_name.to_string();
    let base_name = enum_str
        .strip_suffix("Flags")
        .or_else(|| enum_str.strip_suffix("Flag"))
        .unwrap_or(&enum_str);
    if base_name.is_empty() {
        return Err(syn::Error::new(
            enum_name.span(),
            "enum name cannot be just \"Flag\" or \"Flags\"",
        ));
    }
    let const_name = format_ident!("{}", camel_to_screaming_snake(base_name));

    // Build the Scheduler const with builder chain
    let sched_name_str = &sched_name;
    let mut builder_chain = quote! {
        ::ktstr::test_support::Scheduler::new(#sched_name_str)
    };

    if let Some(ref binary) = sched_binary {
        builder_chain = quote! {
            #builder_chain.binary(::ktstr::test_support::SchedulerSpec::Discover(#binary))
        };
    }

    builder_chain = quote! {
        #builder_chain.flags(#flags_array_ident)
    };

    if let Some((n, s, c, t)) = sched_topology {
        builder_chain = quote! {
            #builder_chain.topology(#n, #s, #c, #t)
        };
    }

    if let Some(ref parent) = sched_cgroup_parent {
        builder_chain = quote! {
            #builder_chain.cgroup_parent(#parent)
        };
    }

    if !sched_args.is_empty() {
        builder_chain = quote! {
            #builder_chain.sched_args(&[#(#sched_args),*])
        };
    }

    if sched_sysctls_set {
        let entries = &sched_sysctls;
        builder_chain = quote! {
            #builder_chain.sysctls(&[#(#entries),*])
        };
    }

    if sched_kargs_set {
        builder_chain = quote! {
            #builder_chain.kargs(&[#(#sched_kargs),*])
        };
    }

    // Chain individual constraint builder calls for each explicitly set
    // attribute. Unset fields inherit from TopologyConstraints::DEFAULT.
    if let Some(v) = sched_min_numa_nodes {
        builder_chain = quote! { #builder_chain.min_numa_nodes(#v) };
    }
    if let Some(v) = sched_max_numa_nodes {
        builder_chain = quote! { #builder_chain.max_numa_nodes(#v) };
    }
    if let Some(v) = sched_min_llcs {
        builder_chain = quote! { #builder_chain.min_llcs(#v) };
    }
    if let Some(v) = sched_max_llcs {
        builder_chain = quote! { #builder_chain.max_llcs(#v) };
    }
    if let Some(v) = sched_requires_smt {
        builder_chain = quote! { #builder_chain.requires_smt(#v) };
    }
    if let Some(v) = sched_min_cpus {
        builder_chain = quote! { #builder_chain.min_cpus(#v) };
    }
    if let Some(v) = sched_max_cpus {
        builder_chain = quote! { #builder_chain.max_cpus(#v) };
    }
    if let Some(ref v) = sched_config_file {
        builder_chain = quote! { #builder_chain.config_file(#v) };
    }

    // Generate the ctor function name for --ktstr-list-flags interception.
    let list_flags_ctor = format_ident!("__ktstr_list_flags_{}", enum_upper.to_lowercase());

    // Additionally emit a `Payload` const wrapping the `Scheduler`
    // so the scheduler is usable in the entry's scheduler slot once
    // the `&Payload` migration lands (WO-162-K), and in compositions
    // that need a `Payload` reference without calling `.run()`.
    // Scheduler-kind Payloads in `workloads = [..]` would fail at
    // `ctx.payload(p).run()` — those slots are binary-only — so the
    // example is deliberately not about `workloads`.
    //
    // Naming: `#[derive(Payload)]` strips the `Payload` suffix from
    // the struct name; the scheduler side appends `_PAYLOAD` so the
    // two surfaces stay symmetric and there's no collision with the
    // Scheduler const.
    let payload_const_name = format_ident!("{}_PAYLOAD", const_name);
    let sched_name_for_payload = sched_name_str;

    let expanded = quote! {
        #(#decl_statics)*

        static #flags_array_ident: &[&::ktstr::scenario::flags::FlagDecl] = &[#(#decl_refs),*];

        const #const_name: ::ktstr::test_support::Scheduler = #builder_chain;

        /// Payload wrapper around the generated `Scheduler` const so
        /// the scheduler is usable wherever a `&'static Payload` is
        /// expected without calling `.run()` — chiefly the entry's
        /// scheduler slot after the `&Payload` migration (WO-162-K),
        /// plus any composition site that takes a `&'static Payload`
        /// reference. Scheduler-kind Payloads are rejected at
        /// `ctx.payload(p).run()`, so this wrapper is intentionally
        /// not meant for `workloads = [..]` (which is binary-only).
        const #payload_const_name: ::ktstr::test_support::Payload =
            ::ktstr::test_support::Payload {
                name: #sched_name_for_payload,
                kind: ::ktstr::test_support::PayloadKind::Scheduler(&#const_name),
                output: ::ktstr::test_support::OutputFormat::ExitCode,
                default_args: &[],
                default_checks: &[],
                metrics: &[],
            };

        impl #enum_name {
            #(#name_consts)*
        }

        /// Intercept `--ktstr-list-flags` before `main()` runs.
        /// Serializes this scheduler's flag declarations as JSON to
        /// stdout and exits, avoiding BPF program loading.
        #[::ktstr::__private::ctor::ctor(crate_path = ::ktstr::__private::ctor)]
        fn #list_flags_ctor() {
            if !::std::env::args().any(|a| a == "--ktstr-list-flags") {
                return;
            }
            let decls: ::std::vec::Vec<::ktstr::scenario::flags::FlagDeclJson> =
                #flags_array_ident
                    .iter()
                    .map(|d| ::ktstr::scenario::flags::FlagDeclJson::from_decl(d))
                    .collect();
            let json = ::ktstr::__private::serde_json::to_string(&decls).expect("serialize flags");
            println!("{json}");
            ::std::process::exit(0);
        }
    };

    Ok(expanded)
}

/// Derive macro that generates a `Payload` const from an annotated
/// struct for a userspace binary workload (stress-ng, fio, and
/// similar tools test authors compose under a scheduler).
///
/// # Required struct-level attributes (`#[payload(...)]`)
///
/// - `binary = "..."` — the binary name resolved by the guest's
///   include-files infrastructure (required). Becomes
///   [`PayloadKind::Binary(name)`](ktstr::test_support::PayloadKind::Binary).
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
/// - `#[default_check(...)]` — one [`Check`](ktstr::test_support::Check)
///   construction expression (e.g. `min("iops", 1000.0)`,
///   `exit_code_eq(0)`). May repeat; entries accumulate in source
///   order. Both `min(...)` and `Check::min(...)` are accepted: the
///   macro prepends `::ktstr::test_support::Check::` when the
///   expression doesn't already spell `Check::` on its callee path,
///   so bare constructors work without an import and qualified
///   constructors read naturally in modules that already have
///   `Check` in scope.
/// - `#[metric(name = "...", polarity = ..., unit = "...")]` —
///   kwarg form. `polarity` is one of `HigherBetter`, `LowerBetter`,
///   `TargetValue(f64)`, `Unknown`. May repeat; entries accumulate.
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

    // Reject non-struct inputs; the flag-variant grammar is specific
    // to `Scheduler` (enums) and a struct-only payload keeps the
    // attribute space unambiguous.
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
    // `#[metric(...)]` attrs in source order so the emitted slices
    // match the declaration.
    let mut default_args: Vec<String> = Vec::new();
    let mut default_checks: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut metrics: Vec<proc_macro2::TokenStream> = Vec::new();

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
            // Single Check-constructing expression. Two forms accepted:
            //   - bare: `min("iops", 1000.0)` — the macro prepends
            //     `::ktstr::test_support::Check::` so users don't have
            //     to import `Check` in every module that derives.
            //   - qualified: `Check::min("iops", 1000.0)` — the user
            //     wrote `Check::` themselves; emit the expression
            //     unchanged so the user's own path resolution wins
            //     (and a double `Check::Check::` prefix can't happen).
            let expr: syn::Expr = attr.parse_args().map_err(|e| {
                syn::Error::new(
                    e.span(),
                    "default_check must be a Check constructor expression (e.g. min(\"iops\", 1000.0))",
                )
            })?;
            if expr_has_check_prefix(&expr) {
                default_checks.push(quote! { #expr });
            } else {
                default_checks.push(quote! { ::ktstr::test_support::Check::#expr });
            }
        } else if attr.path().is_ident("metric") {
            // Kwarg form: name = "...", polarity = ..., unit = "...".
            let parsed = parse_metric_attr(attr)?;
            metrics.push(parsed);
        }
    }

    // Derive the const name: strip "Payload" suffix and uppercase.
    // The suffix strip matches the derive(Scheduler) convention
    // (strip "Flag"/"Flags") so the two macros feel consistent to
    // test authors declaring both.
    let struct_str = struct_name.to_string();
    let base = struct_str.strip_suffix("Payload").unwrap_or(&struct_str);
    if base.is_empty() {
        return Err(syn::Error::new(
            struct_name.span(),
            "struct name cannot be just \"Payload\"",
        ));
    }
    let const_name = format_ident!("{}", camel_to_screaming_snake(base));

    let expanded = quote! {
        #struct_vis const #const_name: ::ktstr::test_support::Payload = ::ktstr::test_support::Payload {
            name: #payload_name,
            kind: ::ktstr::test_support::PayloadKind::Binary(#binary),
            output: #output_tokens,
            default_args: &[#(#default_args),*],
            default_checks: &[#(#default_checks),*],
            metrics: &[#(#metrics),*],
        };
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
/// attribute into a `MetricHint { ... }` token stream.
///
/// `polarity` accepts bare idents `HigherBetter`, `LowerBetter`,
/// `Unknown`, and the call form `TargetValue(<float literal>)`. The
/// float literal is stamped into a `Polarity::TargetValue(lit)` so
/// the generated const is const-evaluable.
fn parse_metric_attr(attr: &syn::Attribute) -> syn::Result<proc_macro2::TokenStream> {
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
    Ok(quote! {
        ::ktstr::test_support::MetricHint {
            name: #name,
            polarity: #polarity,
            unit: #unit,
        }
    })
}

/// Does this `#[default_check(...)]` expression already spell
/// `Check::` somewhere in its function path? Returns true for
/// `Check::min(...)` and `::ktstr::test_support::Check::min(...)`;
/// false for bare `min(...)`. Used to skip the macro's implicit
/// `::ktstr::test_support::Check::` prepend when the user has
/// already written the prefix, so `Check::Check::min(...)` can't
/// happen.
///
/// Only inspects the callee path of an `Expr::Call`; non-call
/// expressions (rare but legal: a free function returning `Check`,
/// or a `const` value) fall back to the prepend path, matching the
/// pre-bugfix behavior for anything that isn't a plain constructor
/// call. A future refactor could lift this to also handle
/// `MethodCall` / `Path`, but the Check API today is constructor
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
        .any(|seg| seg.ident == "Check")
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
