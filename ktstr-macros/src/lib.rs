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
///   - `llcs = N` / `sockets = N` (default: inherited from scheduler, or 1)
///   - `cores = N` (default: inherited from scheduler, or 2)
///   - `threads = N` (default: inherited from scheduler, or 1)
///   - `numa_nodes = N` (default: inherited from scheduler, or 1)
///   - `memory_mb = N` (default: 2048)
///   - `duration_s = N` — maps onto `KtstrTestEntry::duration`
///   - `watchdog_timeout_s = N` — maps onto `KtstrTestEntry::watchdog_timeout`
///   - `scheduler = PATH` — path to a `const Scheduler` (default
///     `Scheduler::EEVDF`, which runs without an scx scheduler)
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
    let mut auto_repro = true;
    let mut not_starved: Option<bool> = None;
    let mut isolation: Option<bool> = None;
    let mut max_gap_ms: Option<u64> = None;
    let mut max_spread_pct: Option<f64> = None;
    let mut max_imbalance_ratio: Option<f64> = None;
    let mut max_local_dsq_depth: Option<u32> = None;
    let mut fail_on_stall: Option<bool> = None;
    let mut sustained_samples: Option<usize> = None;
    let mut replicas: u32 = 1;
    let mut max_throughput_cv: Option<f64> = None;
    let mut min_work_rate: Option<f64> = None;
    let mut max_fallback_rate: Option<f64> = None;
    let mut max_keep_last_rate: Option<f64> = None;
    let mut max_p99_wake_latency_ns: Option<u64> = None;
    let mut max_wake_latency_cv: Option<f64> = None;
    let mut min_iteration_rate: Option<f64> = None;
    let mut max_migration_ratio: Option<f64> = None;
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
                    | "replicas"
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
                            "replicas" => {
                                replicas = lit_int
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
                    | "max_migration_ratio" => {
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
                            format!("unknown attribute `{ident}`, expected: llcs, sockets, cores, threads, numa_nodes, memory_mb, replicas, scheduler, auto_repro, not_starved, isolation, max_gap_ms, max_spread_pct, max_throughput_cv, min_work_rate, max_p99_wake_latency_ns, max_wake_latency_cv, min_iteration_rate, max_migration_ratio, max_imbalance_ratio, max_local_dsq_depth, fail_on_stall, sustained_samples, max_fallback_rate, max_keep_last_rate, extra_sched_args, required_flags, excluded_flags, min_numa_nodes, min_sockets, min_llcs, requires_smt, min_cpus, max_llcs, max_numa_nodes, max_cpus, watchdog_timeout_s, performance_mode, duration_s, workers_per_cgroup, bpf_map_write, expect_err"),
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

    // Reject zero values at compile time (only for explicitly set values).
    if llcs_set && llcs == 0 {
        return syn::Error::new(proc_macro2::Span::call_site(), "llcs must be > 0")
            .to_compile_error()
            .into();
    }
    if cores_set && cores == 0 {
        return syn::Error::new(proc_macro2::Span::call_site(), "cores must be > 0")
            .to_compile_error()
            .into();
    }
    if threads_set && threads == 0 {
        return syn::Error::new(proc_macro2::Span::call_site(), "threads must be > 0")
            .to_compile_error()
            .into();
    }
    if numa_nodes_set && numa_nodes == 0 {
        return syn::Error::new(proc_macro2::Span::call_site(), "numa_nodes must be > 0")
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
        return syn::Error::new(proc_macro2::Span::call_site(), "memory_mb must be > 0")
            .to_compile_error()
            .into();
    }
    if replicas == 0 {
        return syn::Error::new(proc_macro2::Span::call_site(), "replicas must be > 0")
            .to_compile_error()
            .into();
    }
    // Validate explicitly set constraint values. When a field is
    // inherited from the scheduler, the proc macro doesn't know the
    // value so cross-field validation is deferred to runtime.
    if max_llcs_set && max_llcs == Some(0) {
        return syn::Error::new(proc_macro2::Span::call_site(), "max_llcs must be > 0")
            .to_compile_error()
            .into();
    }
    if max_numa_nodes_set && max_numa_nodes == Some(0) {
        return syn::Error::new(proc_macro2::Span::call_site(), "max_numa_nodes must be > 0")
            .to_compile_error()
            .into();
    }
    if max_cpus_set && max_cpus == Some(0) {
        return syn::Error::new(proc_macro2::Span::call_site(), "max_cpus must be > 0")
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

    // Build the scheduler reference token
    let scheduler_tokens = match &scheduler {
        Some(p) => {
            quote! { &#p }
        }
        None => {
            quote! { &::ktstr::test_support::Scheduler::EEVDF }
        }
    };

    // Build topology tokens. Each dimension independently inherits from
    // the scheduler's topology when not explicitly set. Scheduler
    // topology fields are const, so field access is valid in static
    // initializers.
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

    let bpf_map_write_tokens = match &bpf_map_write {
        Some(p) => quote! { Some(&#p) },
        None => quote! { None },
    };

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

        #[::ktstr::__linkme::distributed_slice(::ktstr::test_support::KTSTR_TESTS)]
        #[linkme(crate = ::ktstr::__linkme)]
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
            auto_repro: #auto_repro,
            replicas: #replicas,
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
/// | `binary = "..."` | no | Binary name for `SchedulerSpec::Name(...)`. Omit for EEVDF. |
/// | `topology(N, L, C, T)` | no | Default VM topology `(numa_nodes, llcs, cores, threads)`. Defaults to `(1, 1, 2, 1)`. |
/// | `cgroup_parent = "..."` | no | Cgroup parent path. |
/// | `sched_args = [...]` | no | Default scheduler CLI args. |
/// | `min_numa_nodes = N` | no | Minimum NUMA nodes for gauntlet filtering. |
/// | `max_numa_nodes = N` | no | Maximum NUMA nodes for gauntlet filtering. |
/// | `min_llcs = N` | no | Minimum LLCs for gauntlet filtering. |
/// | `max_llcs = N` | no | Maximum LLCs for gauntlet filtering. |
/// | `min_cpus = N` | no | Minimum total CPUs for gauntlet filtering. |
/// | `max_cpus = N` | no | Maximum total CPUs for gauntlet filtering. |
/// | `requires_smt = bool` | no | Require SMT (threads_per_core > 1). |
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
    let mut sched_min_numa_nodes: Option<u32> = None;
    let mut sched_max_numa_nodes: Option<u32> = None;
    let mut sched_min_llcs: Option<u32> = None;
    let mut sched_max_llcs: Option<u32> = None;
    let mut sched_min_cpus: Option<u32> = None;
    let mut sched_max_cpus: Option<u32> = None;
    let mut sched_requires_smt: Option<bool> = None;

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
            #builder_chain.binary(::ktstr::test_support::SchedulerSpec::Name(#binary))
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

    // Generate the ctor function name for --ktstr-list-flags interception.
    let list_flags_ctor = format_ident!("__ktstr_list_flags_{}", enum_upper.to_lowercase());

    let expanded = quote! {
        #(#decl_statics)*

        static #flags_array_ident: &[&::ktstr::scenario::flags::FlagDecl] = &[#(#decl_refs),*];

        const #const_name: ::ktstr::test_support::Scheduler = #builder_chain;

        impl #enum_name {
            #(#name_consts)*
        }

        /// Intercept `--ktstr-list-flags` before `main()` runs.
        /// Serializes this scheduler's flag declarations as JSON to
        /// stdout and exits, avoiding BPF program loading.
        #[::ktstr::__ctor::ctor(crate_path = ::ktstr::__ctor)]
        fn #list_flags_ctor() {
            if !::std::env::args().any(|a| a == "--ktstr-list-flags") {
                return;
            }
            let decls: ::std::vec::Vec<::ktstr::scenario::flags::FlagDeclJson> =
                #flags_array_ident
                    .iter()
                    .map(|d| ::ktstr::scenario::flags::FlagDeclJson::from_decl(d))
                    .collect();
            let json = ::ktstr::__serde_json::to_string(&decls).expect("serialize flags");
            println!("{json}");
            ::std::process::exit(0);
        }
    };

    Ok(expanded)
}
