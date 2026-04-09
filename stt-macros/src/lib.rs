use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, Meta, MetaNameValue, parse::Parser, parse_macro_input};

/// Default topology and memory for stt_test-annotated functions.
const DEFAULT_SOCKETS: u32 = 1;
const DEFAULT_CORES: u32 = 2;
const DEFAULT_THREADS: u32 = 1;
const DEFAULT_MEMORY_MB: u32 = 2048;

/// Attribute macro that registers a function as an stt integration test.
///
/// The annotated function must have signature `fn(&stt::scenario::Ctx) ->
/// anyhow::Result<stt::assert::AssertResult>`. The macro:
///
/// 1. Renames the original function to `__stt_inner_{name}`.
/// 2. Registers it in the `STT_TESTS` distributed slice via linkme.
/// 3. Emits a `#[test]` wrapper that boots a VM and runs the function
///    inside it.
///
/// Optional attributes (all with defaults):
///   - `sockets = N` (default: inherited from scheduler, or 1)
///   - `cores = N` (default: inherited from scheduler, or 2)
///   - `threads = N` (default: inherited from scheduler, or 1)
///   - `memory_mb = N` (default: 2048)
///   - `performance_mode = bool` (default: false) -- vCPU pinning, hugepages
///   - `duration_s = N`, `workers_per_cgroup = N` -- workload overrides
///   - `scheduler = PATH` -- scheduler constant reference
///   - `max_gap_ms`, `max_spread_pct`, `max_imbalance_ratio` -- assertion thresholds
///   - `max_p99_wake_latency_ns`, `max_wake_latency_cv`, `min_iteration_rate` -- benchmarking
///   - `required_flags`, `excluded_flags` -- flag profile filtering
///   - See SttTestEntry and Assert for the full field list.
#[proc_macro_attribute]
pub fn stt_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let orig_name = &input.sig.ident;
    let inner_name = format_ident!("__stt_inner_{}", orig_name);
    let entry_name = format_ident!("__STT_ENTRY_{}", orig_name.to_string().to_uppercase());
    let name_str = orig_name.to_string();

    // Parse attributes
    let mut sockets = DEFAULT_SOCKETS;
    let mut cores = DEFAULT_CORES;
    let mut threads = DEFAULT_THREADS;
    let mut sockets_set = false;
    let mut cores_set = false;
    let mut threads_set = false;
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
    let mut required_flags: Vec<String> = Vec::new();
    let mut excluded_flags: Vec<String> = Vec::new();
    let mut min_sockets: u32 = 1;
    let mut min_llcs: u32 = 1;
    let mut requires_smt: bool = false;
    let mut min_cpus: u32 = 1;
    let mut watchdog_timeout_s: u64 = 4;
    let mut performance_mode: bool = false;
    let mut duration_s: u64 = 2;
    let mut workers_per_cgroup: u32 = 2;
    let mut bpf_map_write: Option<syn::Path> = None;
    let mut expect_err: bool = false;

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
                    | "requires_smt" | "expect_err" => {
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
                            "requires_smt" => requires_smt = lit_bool.value(),
                            "expect_err" => expect_err = lit_bool.value(),
                            _ => unreachable!(),
                        }
                    }
                    "sockets"
                    | "cores"
                    | "threads"
                    | "memory_mb"
                    | "replicas"
                    | "sustained_samples"
                    | "max_gap_ms"
                    | "watchdog_timeout_s"
                    | "duration_s"
                    | "workers_per_cgroup"
                    | "max_local_dsq_depth"
                    | "min_sockets"
                    | "min_llcs"
                    | "min_cpus"
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
                            "sockets" => {
                                sockets = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"));
                                sockets_set = true;
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
                            "min_sockets" => {
                                min_sockets = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"))
                            }
                            "min_llcs" => {
                                min_llcs = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"))
                            }
                            "min_cpus" => {
                                min_cpus = lit_int
                                    .base10_parse::<u32>()
                                    .unwrap_or_else(|e| panic!("{e}"))
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
                    "fail_on_stall" => {
                        let lit_bool = match value {
                            syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Bool(lb),
                                ..
                            }) => lb,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    "expected bool literal for fail_on_stall",
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        fail_on_stall = Some(lit_bool.value());
                    }
                    "extra_sched_args" | "required_flags" | "excluded_flags" => {
                        let arr = match value {
                            syn::Expr::Array(ea) => ea,
                            _ => {
                                return syn::Error::new_spanned(
                                    value,
                                    format!("expected array of string literals for {ident}"),
                                )
                                .to_compile_error()
                                .into();
                            }
                        };
                        let target = match ident.as_str() {
                            "extra_sched_args" => &mut extra_sched_args,
                            "required_flags" => &mut required_flags,
                            "excluded_flags" => &mut excluded_flags,
                            _ => unreachable!(),
                        };
                        for elem in &arr.elems {
                            match elem {
                                syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(ls),
                                    ..
                                }) => target.push(ls.value()),
                                _ => {
                                    return syn::Error::new_spanned(
                                        elem,
                                        format!("expected string literal in {ident}"),
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
                            format!("unknown attribute `{ident}`, expected: sockets, cores, threads, memory_mb, replicas, scheduler, auto_repro, not_starved, isolation, max_gap_ms, max_spread_pct, max_throughput_cv, min_work_rate, max_p99_wake_latency_ns, max_wake_latency_cv, min_iteration_rate, max_migration_ratio, max_imbalance_ratio, max_local_dsq_depth, fail_on_stall, sustained_samples, max_fallback_rate, max_keep_last_rate, extra_sched_args, required_flags, excluded_flags, min_sockets, min_llcs, requires_smt, min_cpus, watchdog_timeout_s, performance_mode, duration_s, workers_per_cgroup, bpf_map_write, expect_err"),
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
    if sockets_set && sockets == 0 {
        return syn::Error::new(proc_macro2::Span::call_site(), "sockets must be > 0")
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

    // Build the scheduler reference token
    let scheduler_tokens = match &scheduler {
        Some(p) => {
            quote! { &#p }
        }
        None => {
            quote! { &::stt::test_support::Scheduler::EEVDF }
        }
    };

    // Build topology tokens. Each dimension independently inherits from
    // the scheduler's topology when not explicitly set. Scheduler
    // topology fields are const, so field access is valid in static
    // initializers.
    let sockets_tokens = if sockets_set {
        let s = sockets;
        quote! { #s }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology.sockets }
    } else {
        let s = sockets;
        quote! { #s }
    };
    let cores_tokens = if cores_set {
        let c = cores;
        quote! { #c }
    } else if let Some(ref p) = scheduler {
        quote! { #p.topology.cores_per_socket }
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
    let topology_tokens = quote! {
        ::stt::test_support::Topology {
            sockets: #sockets_tokens,
            cores_per_socket: #cores_tokens,
            threads_per_core: #threads_tokens,
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
    let not_starved_tokens = match not_starved {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let isolation_tokens = match isolation {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let gap_tokens = match max_gap_ms {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let spread_tokens = match max_spread_pct {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let imbalance_tokens = match max_imbalance_ratio {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let dsq_tokens = match max_local_dsq_depth {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let stall_tokens = match fail_on_stall {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let sustained_tokens = match sustained_samples {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let throughput_cv_tokens = match max_throughput_cv {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let work_rate_tokens = match min_work_rate {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let fallback_rate_tokens = match max_fallback_rate {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let keep_last_rate_tokens = match max_keep_last_rate {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let p99_wake_tokens = match max_p99_wake_latency_ns {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let wake_cv_tokens = match max_wake_latency_cv {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let iter_rate_tokens = match min_iteration_rate {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };
    let mig_ratio_tokens = match max_migration_ratio {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    };

    let bpf_map_write_tokens = match &bpf_map_write {
        Some(p) => quote! { Some(&#p) },
        None => quote! { None },
    };

    let test_body = if expect_err {
        quote! {
            let result = ::stt::test_support::run_stt_test(&#entry_name);
            assert!(
                result.is_err(),
                "expected test to fail but it passed",
            );
        }
    } else {
        quote! {
            ::stt::test_support::run_stt_test(&#entry_name).unwrap();
        }
    };

    let expanded = quote! {
        #(#attrs)*
        #vis #inner_sig #block

        #[::stt::__linkme::distributed_slice(::stt::test_support::STT_TESTS)]
        #[linkme(crate = ::stt::__linkme)]
        static #entry_name: ::stt::test_support::SttTestEntry = ::stt::test_support::SttTestEntry {
            name: #name_str,
            func: #inner_name,
            topology: #topology_tokens,
            constraints: ::stt::test_support::TopologyConstraints {
                min_sockets: #min_sockets,
                min_llcs: #min_llcs,
                requires_smt: #requires_smt,
                min_cpus: #min_cpus,
            },
            memory_mb: #memory_mb,
            scheduler: #scheduler_tokens,
            auto_repro: #auto_repro,
            replicas: #replicas,
            assert: ::stt::assert::Assert {
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
            watchdog_timeout_s: #watchdog_timeout_s,
            bpf_map_write: #bpf_map_write_tokens,
            performance_mode: #performance_mode,
            duration_s: #duration_s,
            workers_per_cgroup: #workers_per_cgroup,
            expect_err: #expect_err,
            host_only: false,
        };

        #[test]
        fn #orig_name() {
            #test_body
        }
    };

    expanded.into()
}
