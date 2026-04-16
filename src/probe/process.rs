use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::btf::{BtfFunc, RenderHint, STRUCT_FIELDS};
use super::stack::StackFunction;

use crate::bpf_skel::types;

/// Input for Phase B probe attachment (BPF fentry/fexit).
///
/// Sent via mpsc channel after the scheduler starts and BPF programs
/// are discoverable. Phase A (kprobes + trigger + kernel fexit) runs
/// before the scheduler; Phase B attaches fentry/fexit to the
/// scheduler's BPF struct_ops callbacks.
pub struct PhaseBInput {
    /// BPF functions and kernel callers discovered from the running
    /// scheduler. Includes both BPF callbacks (for fentry) and their
    /// kernel-side callers from `expand_bpf_to_kernel_callers` (for
    /// additional kprobes).
    pub functions: Vec<StackFunction>,
    /// Pre-opened BPF program FDs keyed by prog_id.
    pub bpf_prog_fds: std::collections::HashMap<u32, i32>,
    /// BTF-resolved function signatures for BPF callbacks and kernel callers.
    pub btf_funcs: Vec<BtfFunc>,
    /// Set to true when Phase B attachment is complete.
    pub done: Arc<AtomicBool>,
    /// Starting func_idx for Phase B functions. Must equal the number
    /// of functions in Phase A to avoid index collisions in the shared
    /// `func_meta_map` and `probe_data` maps.
    pub func_idx_offset: u32,
}

/// Ring buffer event type for the trigger (matches `EVENT_TRIGGER`
/// in `intf.h`).
const EVENT_TRIGGER: u32 = 2;

/// Pipeline diagnostics from a probe run.
///
/// Tracks how many functions/events survived each stage so users can
/// see WHERE data is being lost (filter, attach, capture, stitch).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProbeDiagnostics {
    /// Kernel functions resolved to IPs.
    pub kprobe_resolved: u32,
    /// Kernel functions that failed IP resolution.
    pub kprobe_resolve_failed: Vec<String>,
    /// Kprobes successfully attached.
    pub kprobe_attached: u32,
    /// Kprobes that failed to attach (name, error).
    pub kprobe_attach_failed: Vec<(String, String)>,
    /// BPF functions with valid prog IDs for fentry.
    pub fentry_candidates: u32,
    /// Fentry probes successfully attached.
    pub fentry_attached: u32,
    /// Fentry probes that failed (name, error).
    pub fentry_attach_failed: Vec<(String, String)>,
    /// Total keys in probe_data map at readout.
    pub probe_data_keys: u32,
    /// Keys with unmatched IPs (no func_meta entry).
    pub probe_data_unmatched_ips: u32,
    /// Events read from probe_data before stitching.
    pub events_before_stitch: u32,
    /// Events surviving tptr+time stitching.
    pub events_after_stitch: u32,
    /// Whether the trigger fired.
    pub trigger_fired: bool,
    /// Which trigger mechanism attached ("tp_btf").
    #[serde(default)]
    pub trigger_type: String,
    /// Error from tp_btf/sched_ext_exit attach failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_attach_error: Option<String>,
    /// BPF-side kprobe fire count (from BSS ktstr_probe_count).
    pub bpf_kprobe_fires: u64,
    /// BPF-side trigger fire count (from BSS ktstr_trigger_count).
    pub bpf_trigger_fires: u64,
    /// BPF-side func_meta_map misses (IP not found in map).
    pub bpf_meta_misses: u64,
    /// IPs that missed func_meta_map lookup (from BSS ktstr_miss_log).
    pub bpf_miss_ips: Vec<u64>,
}

/// Structured probe event captured by the BPF skeleton.
///
/// One per (function, task_ptr) combination. `fields` contains BTF-resolved
/// struct field values keyed as `"param:struct.field"` (from
/// [`build_field_keys`]). Events are sorted by `ts` and stitched by
/// `task_struct` pointer before output.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProbeEvent {
    pub func_idx: u32,
    pub task_ptr: u64,
    pub ts: u64,
    pub args: [u64; 6],
    pub fields: Vec<(String, u64)>, // (field_key, value) — decoded by caller
    pub kstack: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub str_val: Option<String>,
    /// Post-mutation field values captured by fexit.
    /// Same field keys as `fields`, paired by index.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exit_fields: Vec<(String, u64)>,
    /// Timestamp when fexit fired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_ts: Option<u64>,
}

/// Resolve a kernel function name to its address via /proc/kallsyms.
pub fn resolve_func_ip(name: &str) -> Option<u64> {
    let kallsyms = std::fs::read_to_string("/proc/kallsyms").ok()?;
    for line in kallsyms.lines() {
        let mut parts = line.split_whitespace();
        let addr = parts.next()?;
        let _ty = parts.next()?;
        let sym = parts.next()?;
        if sym == name {
            return u64::from_str_radix(addr, 16).ok();
        }
    }
    None
}

/// Populate a `func_meta` with field specs from BTF-resolved offsets.
/// Shared between kprobe and fentry paths.
fn populate_field_specs(meta: &mut types::func_meta, field_specs: &[super::btf::FieldSpec]) {
    let n = field_specs.len().min(16);
    // nr_field_specs must be max(field_idx)+1, not count of specs,
    // because the BPF program writes entry.fields[field_idx] and
    // the Rust side reads entry.fields[..nr_field_specs] positionally
    // against build_field_keys (which includes skipped fields).
    let max_fidx = field_specs
        .iter()
        .take(n)
        .map(|fs| fs.field_idx)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    meta.nr_field_specs = max_fidx.min(16);
    for fs in field_specs.iter().take(n) {
        let slot = fs.field_idx as usize;
        if slot < 16 {
            meta.specs[slot] = types::field_spec {
                param_idx: fs.param_idx,
                offset: fs.offset,
                size: fs.size,
                field_idx: fs.field_idx,
                ptr_offset: fs.ptr_offset,
            };
        }
    }
}

/// Build field key names for a function based on its BTF info.
///
/// Returns a vec mapping `field_idx` to an output key name. Format:
/// - Known struct param: `"p:task_struct.pid"`
/// - Auto-discovered BPF struct: `"ctx:task_ctx*.field_a"`
/// - Scalar param: `"flags:val.flags"`
///
/// Processes at most 6 params (fentry/kprobe register limit) and
/// at most 16 fields total (matching `MAX_FIELDS` in intf.h).
fn build_field_keys(btf_func: &BtfFunc) -> Vec<(String, RenderHint)> {
    let mut keys = Vec::new();
    let mut field_idx: u32 = 0;

    let max_params = btf_func.params.len().min(6);
    for param in &btf_func.params[..max_params] {
        if let Some(ref sname) = param.struct_name {
            if let Some((_, fields)) = STRUCT_FIELDS.iter().find(|(s, _)| *s == sname) {
                for (_, key) in *fields {
                    // Known struct fields use dedicated decoders in
                    // decode.rs — hint is irrelevant (Default/Hex).
                    keys.push((format!("{}:{}.{}", param.name, sname, key), RenderHint::Hex));
                    field_idx += 1;
                    if field_idx >= 16 {
                        break;
                    }
                }
            }
        } else if !param.auto_fields.is_empty() {
            let tname = param.type_name.as_deref().unwrap_or("void");
            for (fname, _, hint) in &param.auto_fields {
                keys.push((format!("{}:{}.{}", param.name, tname, fname), *hint));
                field_idx += 1;
                if field_idx >= 16 {
                    break;
                }
            }
        } else if !param.is_ptr {
            keys.push((
                format!("{}:val.{}", param.name, param.name),
                RenderHint::Hex,
            ));
            field_idx += 1;
        }
    }

    keys
}

/// Detect which param (if any) is a char * string.
/// Uses BTF type detection first, then name heuristic as fallback.
/// Returns 0xff if none found.
fn detect_str_param(btf_func: &BtfFunc) -> u8 {
    let max = btf_func.params.len().min(6);
    // BTF-based: check is_string_ptr flag set by parse_btf_functions.
    for (i, p) in btf_func.params[..max].iter().enumerate() {
        if p.is_string_ptr {
            return i as u8;
        }
    }
    // Name heuristic fallback.
    for (i, p) in btf_func.params[..max].iter().enumerate() {
        if !p.is_ptr || p.struct_name.is_some() {
            continue;
        }
        let n = p.name.as_str();
        if matches!(n, "fmt" | "msg" | "str" | "reason" | "buf" | "s")
            || n.contains("str")
            || n.contains("msg")
            || n.contains("fmt")
        {
            return i as u8;
        }
    }
    0xff
}

/// Pre-open BPF program FDs while the scheduler is alive.
///
/// Returns a map from `bpf_prog_id` to owned fd. Holding these FDs
/// keeps the BPF programs alive via kernel refcounting even after the
/// scheduler exits. Must be called before the test function runs
/// (which may crash the scheduler).
pub fn open_bpf_prog_fds(functions: &[StackFunction]) -> std::collections::HashMap<u32, i32> {
    let mut fds = std::collections::HashMap::new();
    for f in functions {
        if let Some(prog_id) = f.bpf_prog_id {
            let fd = unsafe { libbpf_rs::libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
            if fd >= 0 {
                fds.insert(prog_id, fd);
            }
        }
    }
    fds
}

/// Run the BPF probe skeleton for auto-repro.
///
/// Operates in two modes depending on `phase_b_rx`:
///
/// **Single-phase (`phase_b_rx = None`):** loads the kprobe skeleton
/// and fentry/fexit skeleton together, attaches all probes, then
/// polls until the trigger fires.
///
/// **Two-phase (`phase_b_rx = Some`):** Phase A attaches kprobes +
/// kernel fexit + the `tp_btf/sched_ext_exit` trigger before the
/// scheduler starts, signals `ready`, then polls the ring buffer
/// while waiting for Phase B input via the channel. When Phase B
/// input arrives, attaches fentry/fexit to BPF struct_ops callbacks
/// and additional kprobes for kernel callers. Signals `done` on the
/// `PhaseBInput` when Phase B attachment completes. If the trigger
/// fires before Phase B input arrives, fentry is skipped — the
/// crash happened before BPF programs could be probed.
///
/// The trigger fires on `sched_ext_exit` inside `scx_claim_exit()`
/// — exactly once per scheduler lifetime. If the tracepoint is
/// unavailable, auto-repro is skipped.
///
/// Returns accumulated func_names from both phases as the third
/// tuple element.
pub fn run_probe_skeleton(
    functions: &[StackFunction],
    btf_funcs: &[BtfFunc],
    stop: &AtomicBool,
    bpf_prog_fds: &std::collections::HashMap<u32, i32>,
    ready: &AtomicBool,
    phase_b_rx: Option<std::sync::mpsc::Receiver<PhaseBInput>>,
) -> (
    Option<Vec<ProbeEvent>>,
    ProbeDiagnostics,
    Vec<(u32, String)>,
) {
    use crate::bpf_skel::*;
    use libbpf_rs::skel::{OpenSkel, SkelBuilder};
    use libbpf_rs::{Link, MapCore, MapFlags, RingBufferBuilder};

    tracing::debug!(n = functions.len(), "run_probe_skeleton");

    let mut diag = ProbeDiagnostics::default();

    // Open skeleton
    let mut open_object = std::mem::MaybeUninit::uninit();
    let builder = ProbeSkelBuilder::default();
    let mut open_skel = match builder.open(&mut open_object) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(%e, "probe skeleton open failed");
            ready.store(true, Ordering::Release);
            return (None, diag, Vec::new());
        }
    };

    // Enable probes (must set before load — rodata is immutable after)
    if let Some(rodata) = open_skel.maps.rodata_data.as_mut() {
        rodata.ktstr_enabled = true;
    }

    // Load skeleton
    let skel = match open_skel.load() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(%e, "probe skeleton load failed");
            ready.store(true, Ordering::Release);
            return (None, diag, Vec::new());
        }
    };

    // Populate func_meta_map with function IPs and metadata
    let mut func_ips: Vec<(u32, u64, String)> = Vec::new(); // (idx, ip, display_name)
    let mut bpf_funcs: Vec<(u32, &StackFunction)> = Vec::new(); // BPF functions for fentry

    for (idx, func) in functions.iter().enumerate() {
        if func.is_bpf {
            bpf_funcs.push((idx as u32, func));
            continue;
        }
        let ip = match resolve_func_ip(&func.raw_name) {
            Some(ip) => ip,
            None => {
                tracing::warn!(func = %func.raw_name, "could not resolve function IP");
                diag.kprobe_resolve_failed.push(func.raw_name.clone());
                continue;
            }
        };

        let mut meta = types::func_meta {
            func_idx: idx as u32,
            ..Default::default()
        };

        // Populate field specs from BTF-resolved offsets.
        if let Some(btf_func) = btf_funcs.iter().find(|f| f.name == func.raw_name) {
            let field_specs = super::btf::resolve_field_specs(btf_func, None);
            populate_field_specs(&mut meta, &field_specs);
            // Detect char * params for string capture.
            meta.str_param_idx = detect_str_param(btf_func);
        }

        let key_bytes = ip.to_ne_bytes();
        let meta_bytes = unsafe {
            std::slice::from_raw_parts(
                &meta as *const _ as *const u8,
                std::mem::size_of::<types::func_meta>(),
            )
        };

        if let Err(e) = skel
            .maps
            .func_meta_map
            .update(&key_bytes, meta_bytes, MapFlags::ANY)
        {
            tracing::warn!(%e, func = %func.raw_name, "failed to update func_meta_map");
            continue;
        }

        tracing::debug!(func = %func.raw_name, ip, nr = meta.nr_field_specs, "kprobe meta");
        diag.kprobe_resolved += 1;
        func_ips.push((idx as u32, ip, func.display_name.clone()));
    }

    if func_ips.is_empty() && bpf_funcs.is_empty() && phase_b_rx.is_none() {
        tracing::warn!("no kprobe IPs resolved and no BPF functions for fentry");
        ready.store(true, Ordering::Release);
        return (None, diag, Vec::new());
    }
    if func_ips.is_empty() && (phase_b_rx.is_some() || !bpf_funcs.is_empty()) {
        tracing::debug!("no kernel functions resolved to IPs, proceeding with fentry only");
    }

    // Attach kprobes to each function for entry capture. Exit capture
    // for kernel functions uses fexit via the fentry skeleton (batched
    // separately below with fd=0 for vmlinux BTF).
    let mut links: Vec<(Link, String)> = Vec::new();
    for (idx, _ip, _name) in &func_ips {
        let raw = &functions[*idx as usize].raw_name;
        match skel.progs.ktstr_probe.attach_kprobe(false, raw) {
            Ok(link) => {
                links.push((link, raw.clone()));
            }
            Err(e) => {
                tracing::warn!(%e, func = raw, "kprobe attach failed");
                diag.kprobe_attach_failed.push((raw.clone(), e.to_string()));
            }
        }
    }
    diag.kprobe_attached = links.len() as u32;
    tracing::debug!(attached = links.len(), total = func_ips.len(), "kprobes");

    // Attach fentry+fexit for BPF callbacks and kernel functions.
    // Batched in groups of FENTRY_BATCH per skeleton load to reduce
    // verifier passes. BPF callbacks use prog FD + sentinel IP.
    // Kernel functions use fd=0 (vmlinux BTF) + real IP.
    const FENTRY_BATCH: usize = 4;
    let mut fentry_links: Vec<Link> = Vec::new();
    let mut fexit_links: Vec<Link> = Vec::new();

    struct FentryTarget<'a> {
        slot: usize,
        fd: i32,
        idx: u32,
        name: &'a str,
        ok: bool,
        is_kernel: bool,
    }

    // Build combined list of targets: BPF callbacks + kernel functions.
    let valid_bpf: Vec<_> = bpf_funcs
        .iter()
        .filter(|(_, f)| f.bpf_prog_id.is_some())
        .collect();
    diag.fentry_candidates = valid_bpf.len() as u32;

    // Kernel functions that were attached via kprobe also get fentry+fexit
    // for exit capture. fd=0 targets vmlinux BTF.
    struct KernelFentryTarget {
        idx: u32,
        name: String,
    }
    let kernel_fexit_targets: Vec<KernelFentryTarget> = func_ips
        .iter()
        .map(|(idx, _, name)| KernelFentryTarget {
            idx: *idx,
            name: name.clone(),
        })
        .collect();

    // Interleave: process BPF targets first, then kernel targets.
    // Each gets batched into the fentry skeleton in groups of 4.
    //
    // When phase_b_rx is Some (Phase A/B split), BPF callback fentry
    // attachment is deferred to Phase B after the scheduler starts.
    // Only kernel fexit (fd=0) and kprobes run in Phase A.

    // --- BPF callback batches (skip in Phase A when split is active) ---
    if phase_b_rx.is_none() {
        for chunk in valid_bpf.chunks(FENTRY_BATCH) {
            let mut targets: Vec<FentryTarget<'_>> = Vec::new();
            for (slot, (idx, func)) in chunk.iter().enumerate() {
                let prog_id = func.bpf_prog_id.unwrap();
                let fd = if let Some(&pre_fd) = bpf_prog_fds.get(&prog_id) {
                    let dup_fd = unsafe { libc::dup(pre_fd) };
                    if dup_fd < 0 {
                        tracing::warn!(prog_id, func = %func.display_name, "fentry: dup failed");
                        diag.fentry_attach_failed.push((
                            func.display_name.clone(),
                            format!("dup(pre_fd={pre_fd}) failed"),
                        ));
                        continue;
                    }
                    dup_fd
                } else {
                    let fd = unsafe { libbpf_rs::libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
                    if fd < 0 {
                        tracing::warn!(prog_id, func = %func.display_name, "fentry: failed to get fd");
                        diag.fentry_attach_failed.push((
                            func.display_name.clone(),
                            format!("bpf_prog_get_fd_by_id({prog_id}) returned {fd}"),
                        ));
                        continue;
                    }
                    fd
                };
                targets.push(FentryTarget {
                    slot,
                    fd,
                    idx: *idx,
                    name: &func.display_name,
                    ok: false,
                    is_kernel: false,
                });
            }
            if targets.is_empty() {
                continue;
            }

            use crate::bpf_skel::fentry::*;
            let mut fentry_open_obj = std::mem::MaybeUninit::uninit();
            let fentry_builder = FentryProbeSkelBuilder::default();
            let mut fentry_open = match fentry_builder.open(&mut fentry_open_obj) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(%e, "fentry skeleton open failed");
                    for t in &targets {
                        unsafe { libc::close(t.fd) };
                    }
                    continue;
                }
            };

            // Set rodata: func_idx and is_kernel per slot.
            if let Some(rodata) = fentry_open.maps.rodata_data.as_mut() {
                rodata.ktstr_enabled = true;
                for t in &targets {
                    let k = t.is_kernel as u8;
                    match t.slot {
                        0 => {
                            rodata.ktstr_fentry_func_idx_0 = t.idx;
                            rodata.ktstr_fentry_is_kernel_0 = k;
                        }
                        1 => {
                            rodata.ktstr_fentry_func_idx_1 = t.idx;
                            rodata.ktstr_fentry_is_kernel_1 = k;
                        }
                        2 => {
                            rodata.ktstr_fentry_func_idx_2 = t.idx;
                            rodata.ktstr_fentry_is_kernel_2 = k;
                        }
                        3 => {
                            rodata.ktstr_fentry_func_idx_3 = t.idx;
                            rodata.ktstr_fentry_is_kernel_3 = k;
                        }
                        _ => {}
                    }
                }
            }

            for t in targets.iter_mut() {
                // Set fentry attach target.
                let fentry_prog = match t.slot {
                    0 => &mut fentry_open.progs.ktstr_fentry_0,
                    1 => &mut fentry_open.progs.ktstr_fentry_1,
                    2 => &mut fentry_open.progs.ktstr_fentry_2,
                    3 => &mut fentry_open.progs.ktstr_fentry_3,
                    _ => continue,
                };
                match fentry_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                    Ok(()) => {
                        t.ok = true;
                        tracing::debug!(
                            slot = t.slot,
                            func = t.name,
                            "fentry: set_attach_target ok"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(slot = t.slot, func = t.name, %e, "fentry: set_attach_target failed");
                        diag.fentry_attach_failed
                            .push((t.name.to_string(), format!("set_attach_target: {e}")));
                        continue;
                    }
                }
                // Set fexit attach target on the same function.
                let fexit_prog = match t.slot {
                    0 => &mut fentry_open.progs.ktstr_fexit_0,
                    1 => &mut fentry_open.progs.ktstr_fexit_1,
                    2 => &mut fentry_open.progs.ktstr_fexit_2,
                    3 => &mut fentry_open.progs.ktstr_fexit_3,
                    _ => continue,
                };
                if let Err(e) = fexit_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                    tracing::debug!(slot = t.slot, func = t.name, %e, "fexit: set_attach_target failed (entry-only)");
                    // Disable autoload so the verifier doesn't reject the
                    // skeleton due to a stale placeholder target.
                    fexit_prog.set_autoload(false);
                }
            }

            if !targets.iter().any(|t| t.ok) {
                for t in &targets {
                    unsafe { libc::close(t.fd) };
                }
                continue;
            }

            // Disable autoload on unused or failed fentry/fexit slots so the
            // verifier doesn't reject the placeholder target.
            let used_slots: std::collections::HashSet<usize> =
                targets.iter().filter(|t| t.ok).map(|t| t.slot).collect();
            for slot in 0..FENTRY_BATCH {
                if !used_slots.contains(&slot) {
                    match slot {
                        0 => {
                            fentry_open.progs.ktstr_fentry_0.set_autoload(false);
                            fentry_open.progs.ktstr_fexit_0.set_autoload(false);
                        }
                        1 => {
                            fentry_open.progs.ktstr_fentry_1.set_autoload(false);
                            fentry_open.progs.ktstr_fexit_1.set_autoload(false);
                        }
                        2 => {
                            fentry_open.progs.ktstr_fentry_2.set_autoload(false);
                            fentry_open.progs.ktstr_fexit_2.set_autoload(false);
                        }
                        3 => {
                            fentry_open.progs.ktstr_fentry_3.set_autoload(false);
                            fentry_open.progs.ktstr_fexit_3.set_autoload(false);
                        }
                        _ => {}
                    }
                }
            }
            tracing::debug!(
                active = used_slots.len(),
                disabled = FENTRY_BATCH - used_slots.len(),
                "fentry: loading batch",
            );
            // Reuse the main skeleton's maps so fentry events land in the
            // same probe_data map that the Rust side reads.
            use std::os::unix::io::AsFd;
            if let Err(e) = fentry_open
                .maps
                .probe_data
                .reuse_fd(skel.maps.probe_data.as_fd())
            {
                tracing::warn!(%e, "fentry: probe_data reuse_fd failed");
            }
            if let Err(e) = fentry_open
                .maps
                .func_meta_map
                .reuse_fd(skel.maps.func_meta_map.as_fd())
            {
                tracing::warn!(%e, "fentry: func_meta_map reuse_fd failed");
            }

            let fentry_skel = match fentry_open.load() {
                Ok(s) => {
                    tracing::debug!("fentry: batch load success");
                    for t in &targets {
                        unsafe { libc::close(t.fd) };
                    }
                    s
                }
                Err(e) => {
                    tracing::warn!(%e, "fentry: batch load failed");
                    for t in &targets {
                        if t.ok {
                            diag.fentry_attach_failed
                                .push((t.name.to_string(), format!("batch load: {e}")));
                        }
                        unsafe { libc::close(t.fd) };
                    }
                    continue;
                }
            };

            // Populate func_meta and attach each slot.
            for t in &targets {
                if !t.ok {
                    continue;
                }

                let sentinel_ip = (t.idx as u64) | (1u64 << 63);
                let mut meta = crate::bpf_skel::types::func_meta {
                    func_idx: t.idx,
                    ..Default::default()
                };

                if let Some(btf_func) = btf_funcs.iter().find(|f| f.name == t.name) {
                    // Try vmlinux BTF first (for known struct params like
                    // task_struct and auto-discovered vmlinux fields),
                    // then BPF program BTF (for BPF-local types like task_ctx).
                    let mut field_specs = super::btf::resolve_field_specs(btf_func, None);
                    if field_specs.is_empty()
                        && let Some(prog_id) = functions
                            .iter()
                            .find(|f| f.display_name == t.name)
                            .and_then(|f| f.bpf_prog_id)
                    {
                        field_specs = super::btf::resolve_bpf_field_specs(btf_func, prog_id);
                    }
                    populate_field_specs(&mut meta, &field_specs);
                    meta.str_param_idx = detect_str_param(btf_func);
                }

                let key_bytes = sentinel_ip.to_ne_bytes();
                let meta_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &meta as *const _ as *const u8,
                        std::mem::size_of::<crate::bpf_skel::types::func_meta>(),
                    )
                };
                if let Err(e) =
                    skel.maps
                        .func_meta_map
                        .update(&key_bytes, meta_bytes, MapFlags::ANY)
                {
                    tracing::warn!(%e, func = t.name, "fentry: failed to update func_meta_map");
                    continue;
                }
                func_ips.push((t.idx, sentinel_ip, t.name.to_string()));

                let result = match t.slot {
                    0 => fentry_skel.progs.ktstr_fentry_0.attach_trace(),
                    1 => fentry_skel.progs.ktstr_fentry_1.attach_trace(),
                    2 => fentry_skel.progs.ktstr_fentry_2.attach_trace(),
                    3 => fentry_skel.progs.ktstr_fentry_3.attach_trace(),
                    _ => continue,
                };
                match result {
                    Ok(link) => {
                        tracing::debug!(func = t.name, "fentry attached");
                        fentry_links.push(link);
                    }
                    Err(e) => {
                        tracing::warn!(%e, func = t.name, "fentry attach failed");
                        diag.fentry_attach_failed
                            .push((t.name.to_string(), e.to_string()));
                    }
                }
                // Attach fexit for exit-side capture.
                let fexit_result = match t.slot {
                    0 => fentry_skel.progs.ktstr_fexit_0.attach_trace(),
                    1 => fentry_skel.progs.ktstr_fexit_1.attach_trace(),
                    2 => fentry_skel.progs.ktstr_fexit_2.attach_trace(),
                    3 => fentry_skel.progs.ktstr_fexit_3.attach_trace(),
                    _ => continue,
                };
                match fexit_result {
                    Ok(link) => {
                        tracing::debug!(func = t.name, "fexit attached");
                        fexit_links.push(link);
                    }
                    Err(e) => {
                        tracing::debug!(%e, func = t.name, "fexit attach failed (entry-only)");
                    }
                }
            }

            drop(fentry_skel);
        }
        diag.fentry_attached = fentry_links.len() as u32;
        if !valid_bpf.is_empty() {
            tracing::debug!(
                fentry = fentry_links.len(),
                fexit = fexit_links.len(),
                total = valid_bpf.len(),
                "BPF probes",
            );
        }
    } // end if phase_b_rx.is_none() — BPF callback batches

    // --- Kernel function fexit batches (fd=0 = vmlinux BTF) ---
    for chunk in kernel_fexit_targets.chunks(FENTRY_BATCH) {
        let mut targets: Vec<FentryTarget<'_>> = Vec::new();
        for (slot, kt) in chunk.iter().enumerate() {
            targets.push(FentryTarget {
                slot,
                fd: 0, // vmlinux BTF
                idx: kt.idx,
                name: &kt.name,
                ok: false,
                is_kernel: true,
            });
        }

        use crate::bpf_skel::fentry::*;
        let mut fentry_open_obj = std::mem::MaybeUninit::uninit();
        let fentry_builder = FentryProbeSkelBuilder::default();
        let mut fentry_open = match fentry_builder.open(&mut fentry_open_obj) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "kernel fexit skeleton open failed");
                continue;
            }
        };

        if let Some(rodata) = fentry_open.maps.rodata_data.as_mut() {
            rodata.ktstr_enabled = true;
            for t in &targets {
                let k = t.is_kernel as u8;
                match t.slot {
                    0 => {
                        rodata.ktstr_fentry_func_idx_0 = t.idx;
                        rodata.ktstr_fentry_is_kernel_0 = k;
                    }
                    1 => {
                        rodata.ktstr_fentry_func_idx_1 = t.idx;
                        rodata.ktstr_fentry_is_kernel_1 = k;
                    }
                    2 => {
                        rodata.ktstr_fentry_func_idx_2 = t.idx;
                        rodata.ktstr_fentry_is_kernel_2 = k;
                    }
                    3 => {
                        rodata.ktstr_fentry_func_idx_3 = t.idx;
                        rodata.ktstr_fentry_is_kernel_3 = k;
                    }
                    _ => {}
                }
            }
        }

        // For kernel fexit, we only need fexit programs — disable fentry
        // (entry capture is handled by the kprobe skeleton).
        for t in targets.iter_mut() {
            // Disable fentry for kernel functions (kprobe handles entry).
            let fentry_prog = match t.slot {
                0 => &mut fentry_open.progs.ktstr_fentry_0,
                1 => &mut fentry_open.progs.ktstr_fentry_1,
                2 => &mut fentry_open.progs.ktstr_fentry_2,
                3 => &mut fentry_open.progs.ktstr_fentry_3,
                _ => continue,
            };
            fentry_prog.set_autoload(false);

            // Set fexit attach target with fd=0 (vmlinux BTF).
            let fexit_prog = match t.slot {
                0 => &mut fentry_open.progs.ktstr_fexit_0,
                1 => &mut fentry_open.progs.ktstr_fexit_1,
                2 => &mut fentry_open.progs.ktstr_fexit_2,
                3 => &mut fentry_open.progs.ktstr_fexit_3,
                _ => continue,
            };
            match fexit_prog.set_attach_target(0, Some(t.name.to_string())) {
                Ok(()) => {
                    t.ok = true;
                    tracing::debug!(
                        slot = t.slot,
                        func = t.name,
                        "kernel fexit: set_attach_target ok"
                    );
                }
                Err(e) => {
                    tracing::debug!(slot = t.slot, func = t.name, %e, "kernel fexit: set_attach_target failed");
                    fexit_prog.set_autoload(false);
                }
            }
        }

        if !targets.iter().any(|t| t.ok) {
            continue;
        }

        // Disable fexit for unused slots.
        let used_slots: std::collections::HashSet<usize> =
            targets.iter().filter(|t| t.ok).map(|t| t.slot).collect();
        for slot in 0..FENTRY_BATCH {
            if !used_slots.contains(&slot) {
                match slot {
                    0 => fentry_open.progs.ktstr_fexit_0.set_autoload(false),
                    1 => fentry_open.progs.ktstr_fexit_1.set_autoload(false),
                    2 => fentry_open.progs.ktstr_fexit_2.set_autoload(false),
                    3 => fentry_open.progs.ktstr_fexit_3.set_autoload(false),
                    _ => {}
                }
            }
        }

        // Reuse probe_data and func_meta_map from the main skeleton.
        use std::os::unix::io::AsFd;
        if let Err(e) = fentry_open
            .maps
            .probe_data
            .reuse_fd(skel.maps.probe_data.as_fd())
        {
            tracing::warn!(%e, "kernel fexit: probe_data reuse_fd failed");
        }
        if let Err(e) = fentry_open
            .maps
            .func_meta_map
            .reuse_fd(skel.maps.func_meta_map.as_fd())
        {
            tracing::warn!(%e, "kernel fexit: func_meta_map reuse_fd failed");
        }

        let fentry_skel = match fentry_open.load() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "kernel fexit: batch load failed");
                continue;
            }
        };

        for t in &targets {
            if !t.ok {
                continue;
            }
            let result = match t.slot {
                0 => fentry_skel.progs.ktstr_fexit_0.attach_trace(),
                1 => fentry_skel.progs.ktstr_fexit_1.attach_trace(),
                2 => fentry_skel.progs.ktstr_fexit_2.attach_trace(),
                3 => fentry_skel.progs.ktstr_fexit_3.attach_trace(),
                _ => continue,
            };
            match result {
                Ok(link) => {
                    tracing::debug!(func = t.name, "kernel fexit attached");
                    fexit_links.push(link);
                }
                Err(e) => {
                    tracing::debug!(%e, func = t.name, "kernel fexit attach failed");
                }
            }
        }

        drop(fentry_skel);
    }
    if !kernel_fexit_targets.is_empty() {
        tracing::debug!(
            fexit = fexit_links.len(),
            total = kernel_fexit_targets.len(),
            "kernel fexit probes",
        );
    }

    // Attach trigger: tp_btf/sched_ext_exit fires inside
    // scx_claim_exit() in the context of the current task at exit time.
    match skel.progs.ktstr_trigger_tp.attach_trace() {
        Ok(link) => {
            tracing::debug!("trigger attached via tp_btf/sched_ext_exit");
            diag.trigger_type = "tp_btf".to_string();
            links.push((link, "tp_btf/sched_ext_exit".to_string()));
        }
        Err(e) => {
            let msg = format!("auto-repro requires kernel with sched_ext_exit tracepoint: {e}");
            tracing::error!(%msg, "trigger attach failed");
            diag.trigger_attach_error = Some(msg);
            ready.store(true, Ordering::Release);
            return (None, diag, Vec::new());
        }
    }

    // Set up ring buffer
    let events: std::sync::Arc<std::sync::Mutex<Vec<ProbeEvent>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let triggered = std::sync::Arc::new(AtomicBool::new(false));
    let triggered_clone = triggered.clone();

    // Ring buffer event layout matching probe_event in intf.h
    #[repr(C)]
    struct RbEvent {
        type_: u32,
        tid: u32,
        func_idx: u32,
        ts: u64,
        args: [u64; 6],
        fields: [u64; 16],
        nr_fields: u32,
        kstack: [u64; 32],
        kstack_sz: u32,
    }

    let mut rb_builder = RingBufferBuilder::new();
    if let Err(e) = rb_builder.add(&skel.maps.events, move |data: &[u8]| {
        if data.len() < std::mem::size_of::<RbEvent>() {
            return 0;
        }
        let raw: &RbEvent = unsafe { &*(data.as_ptr() as *const RbEvent) };

        if raw.type_ == EVENT_TRIGGER {
            triggered_clone.store(true, Ordering::Relaxed);

            let kstack_sz = (raw.kstack_sz as usize).min(32);
            let event = ProbeEvent {
                func_idx: 0,
                task_ptr: raw.args[0],
                ts: raw.ts,
                args: raw.args,
                fields: vec![],
                kstack: raw.kstack[..kstack_sz].to_vec(),
                str_val: None,
                exit_fields: vec![],
                exit_ts: None,
            };

            events_clone.lock().unwrap().push(event);
        }

        0
    }) {
        tracing::error!(%e, "failed to register ring buffer callback");
        ready.store(true, Ordering::Release);
        return (None, diag, Vec::new());
    }

    let rb = match rb_builder.build() {
        Ok(rb) => rb,
        Err(e) => {
            tracing::error!(%e, "failed to build ring buffer");
            ready.store(true, Ordering::Release);
            return (None, diag, Vec::new());
        }
    };

    // Enable is handled by the BPF program reading the volatile const.
    // Since we can't mutate rodata after load, the program starts enabled.
    // (ktstr_enabled defaults to false in BPF, but we always want probes
    // active once attached — remove the gate or set it before load.)

    tracing::debug!(
        funcs = func_ips.len(),
        links = links.len(),
        trigger_type = %diag.trigger_type,
        "polling for probe data",
    );

    // Signal Phase A probes attached (kprobes + kernel fexit +
    // trigger). When phase_b_rx is None, this means all probes.
    // When Some, BPF fentry is deferred to Phase B.
    ready.store(true, Ordering::Release);

    // Phase B: receive BPF fentry targets and attach them while
    // polling the ring buffer. The channel is consumed once; after
    // that phase_b_done prevents re-checking.
    let mut phase_b_rx = phase_b_rx;
    let mut phase_b_done = false;
    // Accumulates BTF from Phase B functions so the readout phase
    // can resolve field keys for both Phase A and Phase B functions.
    let mut phase_b_btf: Vec<BtfFunc> = Vec::new();

    // Poll until trigger fires or stop requested.  When stop is
    // signaled, iterate all probe_data entries instead of waiting
    // for the trigger.
    loop {
        let _ = rb.poll(Duration::from_millis(100));

        // Check for Phase B input while polling.
        if !phase_b_done && let Some(ref rx) = phase_b_rx {
            match rx.try_recv() {
                Ok(pb) => {
                    tracing::debug!(
                        bpf_funcs = pb.functions.len(),
                        "Phase B: attaching BPF fentry/fexit"
                    );
                    // Save Phase B BTF for readout field key resolution.
                    phase_b_btf = pb.btf_funcs.clone();
                    // Attach Phase B probes: kernel callers (kprobes
                    // + kernel fexit) and BPF callbacks (fentry/fexit).
                    attach_phase_b_fentry(
                        &skel,
                        &pb,
                        &mut func_ips,
                        &mut fentry_links,
                        &mut fexit_links,
                        &mut links,
                        &mut diag,
                    );
                    pb.done.store(true, Ordering::Release);
                    phase_b_done = true;
                    // Drop the receiver to release the channel.
                    phase_b_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No Phase B input yet; if trigger already fired,
                    // break immediately — the crash happened before
                    // Phase B could attach.
                    if triggered.load(Ordering::Relaxed) {
                        tracing::debug!("trigger fired during Phase B wait, skipping fentry");
                        phase_b_done = true;
                        phase_b_rx = None;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    tracing::debug!("Phase B channel disconnected");
                    phase_b_done = true;
                    phase_b_rx = None;
                }
            }
        }

        if triggered.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
            diag.trigger_fired = triggered.load(Ordering::Relaxed);

            // Read BPF-side diagnostic counters from BSS.
            if let Some(bss) = skel.maps.bss_data.as_ref() {
                diag.bpf_kprobe_fires = bss.ktstr_probe_count;
                diag.bpf_trigger_fires = bss.ktstr_trigger_count;
                diag.bpf_meta_misses = bss.ktstr_meta_miss;
                let n = (bss.ktstr_miss_log_idx as usize).min(bss.ktstr_miss_log.len());
                diag.bpf_miss_ips = bss.ktstr_miss_log[..n].to_vec();
            }

            let key_size = std::mem::size_of::<types::probe_key>();
            let mut probe_events = Vec::new();
            let mut total_keys = 0u32;
            let mut unmatched_ips = 0u32;

            for key_bytes in skel.maps.probe_data.keys() {
                if key_bytes.len() < key_size {
                    continue;
                }
                total_keys += 1;
                let key: &types::probe_key =
                    unsafe { &*(key_bytes.as_ptr() as *const types::probe_key) };

                // Find which function this IP belongs to.
                let func_entry = func_ips.iter().find(|(_, ip, _)| *ip == key.func_ip);
                let (func_idx, display_name) = match func_entry {
                    Some((idx, _, name)) => (*idx, name.as_str()),
                    None => {
                        unmatched_ips += 1;
                        continue;
                    }
                };

                if let Ok(Some(val_bytes)) = skel.maps.probe_data.lookup(&key_bytes, MapFlags::ANY)
                {
                    let entry: &types::probe_entry =
                        unsafe { &*(val_bytes.as_ptr() as *const types::probe_entry) };
                    if entry.ts == 0 {
                        continue;
                    }

                    let field_keys_hints: Vec<(String, RenderHint)> = btf_funcs
                        .iter()
                        .chain(phase_b_btf.iter())
                        .find(|f| f.name == display_name)
                        .map(build_field_keys)
                        .unwrap_or_default();

                    let nr = (entry.nr_fields as usize).min(16);
                    let fields: Vec<(String, u64)> = entry.fields[..nr]
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &val)| {
                            field_keys_hints.get(i).map(|(k, _)| (k.clone(), val))
                        })
                        .collect();

                    let str_val = if entry.has_str != 0 {
                        let s = &entry.str_val;
                        let bytes: Vec<u8> = s.iter().map(|&b| b as u8).collect();
                        let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                        let text = std::str::from_utf8(&bytes[..len]).unwrap_or("").to_string();
                        if text.is_empty() { None } else { Some(text) }
                    } else {
                        None
                    };

                    // Extract exit-side fields if fexit fired.
                    let (exit_fields, exit_ts) = if entry.has_exit != 0 {
                        let nr_exit = (entry.nr_exit_fields as usize).min(16);
                        let ef: Vec<(String, u64)> = entry.exit_fields[..nr_exit]
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &val)| {
                                field_keys_hints.get(i).map(|(k, _)| (k.clone(), val))
                            })
                            .collect();
                        (ef, Some(entry.exit_ts))
                    } else {
                        (Vec::new(), None)
                    };

                    probe_events.push(ProbeEvent {
                        func_idx,
                        task_ptr: key.task_ptr,
                        ts: entry.ts,
                        args: entry.args,
                        fields,
                        kstack: vec![],
                        str_val,
                        exit_fields,
                        exit_ts,
                    });
                }
            }

            probe_events.sort_by_key(|e| e.ts);

            diag.probe_data_keys = total_keys;
            diag.probe_data_unmatched_ips = unmatched_ips;
            diag.events_before_stitch = probe_events.len() as u32;

            tracing::debug!(
                events = probe_events.len(),
                total_keys,
                unmatched_ips,
                "probe_data readout",
            );

            if probe_events.is_empty() {
                return (None, diag, Vec::new());
            }

            // Stitch by task_struct pointer. Build a map of func_idx ->
            // task_struct param index from BPF_OP_CALLERS and BTF, then
            // filter events to those referencing the same task_struct
            // pointer as the causal task.
            //
            // The BPF trigger handler sets args[0] to
            // bpf_get_current_task() only for ops callback errors
            // (SCX_EXIT_ERROR, SCX_EXIT_ERROR_BPF) where current IS
            // the causal task. For all other exit kinds (stalls,
            // sysrq, unregistration), args[0] is 0 and probe output
            // is suppressed — no causal task means no useful chain.
            let task_param_idx: std::collections::HashMap<u32, usize> = func_ips
                .iter()
                .filter_map(|(idx, _, name)| {
                    // BPF_OP_CALLERS: (op_fragment, kernel_caller, task_arg_idx)
                    if let Some((_, _, tidx)) = super::stack::BPF_OP_CALLERS
                        .iter()
                        .find(|(_, caller, _)| *caller == name.as_str())
                    {
                        return Some((*idx, *tidx as usize));
                    }
                    // Fallback: BTF params with task_struct
                    let btf = btf_funcs
                        .iter()
                        .chain(phase_b_btf.iter())
                        .find(|f| f.name == *name)?;
                    let pos = btf
                        .params
                        .iter()
                        .position(|p| p.struct_name.as_deref() == Some("task_struct"))?;
                    Some((*idx, pos))
                })
                .collect();

            // Extract tptr and kstack from the trigger event in one
            // lock acquisition. When the trigger did not fire (stop-
            // signaled) or the exit kind lacks a causal task, probe
            // output is suppressed.
            let (target_tptr, trigger_kstack) = {
                let guard = events.lock().unwrap();
                let tptr = guard.last().map(|e| e.task_ptr).filter(|&p| p != 0);
                let kstack = guard.last().map(|e| e.kstack.clone()).unwrap_or_default();
                (tptr, kstack)
            };

            let Some(tptr) = target_tptr else {
                // No causal task (stall, sysrq, unregistration) —
                // suppress probe output rather than dumping unstitched noise.
                tracing::debug!("no causal tptr — suppressing probe output");
                return (None, diag, Vec::new());
            };

            let before = probe_events.len();
            probe_events.retain(|e| {
                if let Some(&pidx) = task_param_idx.get(&e.func_idx) {
                    e.args[pidx] == tptr
                } else {
                    e.task_ptr == tptr // no task_struct param — match on current
                }
            });
            tracing::debug!(
                tptr = format_args!("0x{tptr:x}"),
                kept = probe_events.len(),
                total = before,
                "stitched by task_struct arg",
            );

            diag.events_after_stitch = probe_events.len() as u32;

            // Attach trigger kstack if available.
            if let Some(last) = probe_events.last_mut() {
                last.kstack = trigger_kstack;
            }

            let fnames: Vec<(u32, String)> = func_ips
                .iter()
                .map(|(idx, _, name)| (*idx, name.clone()))
                .collect();
            return (Some(probe_events), diag, fnames);
        }
    }
}

/// Attach Phase B probes after the scheduler starts.
///
/// Handles both kernel callers (kprobes + kernel fexit) and BPF
/// callbacks (fentry/fexit). Uses `pb.func_idx_offset` for all
/// func_idx values to avoid collisions with Phase A indices.
///
/// Kprobes are attached via the Phase A kprobe skeleton (`skel`),
/// which stays alive throughout. BPF fentry/fexit use separate
/// skeleton loads that share `probe_data` and `func_meta_map` via
/// `reuse_fd`.
fn attach_phase_b_fentry(
    skel: &crate::bpf_skel::ProbeSkel<'_>,
    pb: &PhaseBInput,
    func_ips: &mut Vec<(u32, u64, String)>,
    fentry_links: &mut Vec<libbpf_rs::Link>,
    fexit_links: &mut Vec<libbpf_rs::Link>,
    links: &mut Vec<(libbpf_rs::Link, String)>,
    diag: &mut ProbeDiagnostics,
) {
    use libbpf_rs::MapCore;
    use libbpf_rs::MapFlags;
    use libbpf_rs::skel::{OpenSkel, SkelBuilder};

    const FENTRY_BATCH: usize = 4;

    struct FentryTarget<'a> {
        slot: usize,
        fd: i32,
        idx: u32,
        name: &'a str,
        ok: bool,
        is_kernel: bool,
    }

    let offset = pb.func_idx_offset;

    // --- Phase B kernel functions: kprobes + func_meta ---
    for (i, func) in pb.functions.iter().enumerate() {
        if func.is_bpf {
            continue;
        }
        let idx = offset + i as u32;
        let ip = match resolve_func_ip(&func.raw_name) {
            Some(ip) => ip,
            None => {
                tracing::warn!(func = %func.raw_name, "phase_b: could not resolve function IP");
                diag.kprobe_resolve_failed.push(func.raw_name.clone());
                continue;
            }
        };

        let mut meta = types::func_meta {
            func_idx: idx,
            ..Default::default()
        };

        if let Some(btf_func) = pb.btf_funcs.iter().find(|f| f.name == func.raw_name) {
            let field_specs = super::btf::resolve_field_specs(btf_func, None);
            populate_field_specs(&mut meta, &field_specs);
            meta.str_param_idx = detect_str_param(btf_func);
        }

        let key_bytes = ip.to_ne_bytes();
        let meta_bytes = unsafe {
            std::slice::from_raw_parts(
                &meta as *const _ as *const u8,
                std::mem::size_of::<types::func_meta>(),
            )
        };

        if let Err(e) = skel
            .maps
            .func_meta_map
            .update(&key_bytes, meta_bytes, MapFlags::ANY)
        {
            tracing::warn!(%e, func = %func.raw_name, "phase_b: failed to update func_meta_map");
            continue;
        }

        // Attach kprobe using Phase A's skeleton.
        match skel.progs.ktstr_probe.attach_kprobe(false, &func.raw_name) {
            Ok(link) => {
                links.push((link, func.raw_name.clone()));
                diag.kprobe_attached += 1;
            }
            Err(e) => {
                tracing::warn!(%e, func = %func.raw_name, "phase_b kprobe attach failed");
                diag.kprobe_attach_failed
                    .push((func.raw_name.clone(), e.to_string()));
            }
        }

        diag.kprobe_resolved += 1;
        func_ips.push((idx, ip, func.display_name.clone()));
    }

    // --- Phase B kernel function fexit batches (fd=0 = vmlinux BTF) ---
    struct KernelFexitTarget {
        idx: u32,
        name: String,
    }
    let kernel_fexit_targets: Vec<KernelFexitTarget> = pb
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.is_bpf)
        .map(|(i, f)| KernelFexitTarget {
            idx: offset + i as u32,
            name: f.display_name.clone(),
        })
        .collect();

    for chunk in kernel_fexit_targets.chunks(FENTRY_BATCH) {
        let mut targets: Vec<FentryTarget<'_>> = Vec::new();
        for (slot, kt) in chunk.iter().enumerate() {
            targets.push(FentryTarget {
                slot,
                fd: 0,
                idx: kt.idx,
                name: &kt.name,
                ok: false,
                is_kernel: true,
            });
        }

        use crate::bpf_skel::fentry::*;
        let mut fentry_open_obj = std::mem::MaybeUninit::uninit();
        let fentry_builder = FentryProbeSkelBuilder::default();
        let mut fentry_open = match fentry_builder.open(&mut fentry_open_obj) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "phase_b kernel fexit skeleton open failed");
                continue;
            }
        };

        if let Some(rodata) = fentry_open.maps.rodata_data.as_mut() {
            rodata.ktstr_enabled = true;
            for t in &targets {
                let k = t.is_kernel as u8;
                match t.slot {
                    0 => {
                        rodata.ktstr_fentry_func_idx_0 = t.idx;
                        rodata.ktstr_fentry_is_kernel_0 = k;
                    }
                    1 => {
                        rodata.ktstr_fentry_func_idx_1 = t.idx;
                        rodata.ktstr_fentry_is_kernel_1 = k;
                    }
                    2 => {
                        rodata.ktstr_fentry_func_idx_2 = t.idx;
                        rodata.ktstr_fentry_is_kernel_2 = k;
                    }
                    3 => {
                        rodata.ktstr_fentry_func_idx_3 = t.idx;
                        rodata.ktstr_fentry_is_kernel_3 = k;
                    }
                    _ => {}
                }
            }
        }

        for t in targets.iter_mut() {
            let fentry_prog = match t.slot {
                0 => &mut fentry_open.progs.ktstr_fentry_0,
                1 => &mut fentry_open.progs.ktstr_fentry_1,
                2 => &mut fentry_open.progs.ktstr_fentry_2,
                3 => &mut fentry_open.progs.ktstr_fentry_3,
                _ => continue,
            };
            fentry_prog.set_autoload(false);

            let fexit_prog = match t.slot {
                0 => &mut fentry_open.progs.ktstr_fexit_0,
                1 => &mut fentry_open.progs.ktstr_fexit_1,
                2 => &mut fentry_open.progs.ktstr_fexit_2,
                3 => &mut fentry_open.progs.ktstr_fexit_3,
                _ => continue,
            };
            match fexit_prog.set_attach_target(0, Some(t.name.to_string())) {
                Ok(()) => {
                    t.ok = true;
                }
                Err(e) => {
                    tracing::debug!(func = t.name, %e, "phase_b kernel fexit: set_attach_target failed");
                    fexit_prog.set_autoload(false);
                }
            }
        }

        if !targets.iter().any(|t| t.ok) {
            continue;
        }

        let used_slots: std::collections::HashSet<usize> =
            targets.iter().filter(|t| t.ok).map(|t| t.slot).collect();
        for slot in 0..FENTRY_BATCH {
            if !used_slots.contains(&slot) {
                match slot {
                    0 => fentry_open.progs.ktstr_fexit_0.set_autoload(false),
                    1 => fentry_open.progs.ktstr_fexit_1.set_autoload(false),
                    2 => fentry_open.progs.ktstr_fexit_2.set_autoload(false),
                    3 => fentry_open.progs.ktstr_fexit_3.set_autoload(false),
                    _ => {}
                }
            }
        }

        use std::os::unix::io::AsFd;
        if let Err(e) = fentry_open
            .maps
            .probe_data
            .reuse_fd(skel.maps.probe_data.as_fd())
        {
            tracing::warn!(%e, "phase_b kernel fexit: probe_data reuse_fd failed");
        }
        if let Err(e) = fentry_open
            .maps
            .func_meta_map
            .reuse_fd(skel.maps.func_meta_map.as_fd())
        {
            tracing::warn!(%e, "phase_b kernel fexit: func_meta_map reuse_fd failed");
        }

        let fentry_skel = match fentry_open.load() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "phase_b kernel fexit: batch load failed");
                continue;
            }
        };

        for t in &targets {
            if !t.ok {
                continue;
            }
            let result = match t.slot {
                0 => fentry_skel.progs.ktstr_fexit_0.attach_trace(),
                1 => fentry_skel.progs.ktstr_fexit_1.attach_trace(),
                2 => fentry_skel.progs.ktstr_fexit_2.attach_trace(),
                3 => fentry_skel.progs.ktstr_fexit_3.attach_trace(),
                _ => continue,
            };
            match result {
                Ok(link) => {
                    tracing::debug!(func = t.name, "phase_b kernel fexit attached");
                    fexit_links.push(link);
                }
                Err(e) => {
                    tracing::debug!(%e, func = t.name, "phase_b kernel fexit attach failed");
                }
            }
        }

        drop(fentry_skel);
    }

    // --- Phase B BPF callback fentry/fexit batches ---
    let valid_bpf: Vec<(u32, &StackFunction)> = pb
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| f.bpf_prog_id.is_some())
        .map(|(i, f)| (offset + i as u32, f))
        .collect();
    diag.fentry_candidates += valid_bpf.len() as u32;

    for chunk in valid_bpf.chunks(FENTRY_BATCH) {
        let mut targets: Vec<FentryTarget<'_>> = Vec::new();
        for (slot, (idx, func)) in chunk.iter().enumerate() {
            let prog_id = func.bpf_prog_id.unwrap();
            let fd = if let Some(&pre_fd) = pb.bpf_prog_fds.get(&prog_id) {
                let dup_fd = unsafe { libc::dup(pre_fd) };
                if dup_fd < 0 {
                    tracing::warn!(prog_id, func = %func.display_name, "phase_b fentry: dup failed");
                    diag.fentry_attach_failed.push((
                        func.display_name.clone(),
                        format!("dup(pre_fd={pre_fd}) failed"),
                    ));
                    continue;
                }
                dup_fd
            } else {
                let fd = unsafe { libbpf_rs::libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
                if fd < 0 {
                    tracing::warn!(prog_id, func = %func.display_name, "phase_b fentry: failed to get fd");
                    diag.fentry_attach_failed.push((
                        func.display_name.clone(),
                        format!("bpf_prog_get_fd_by_id({prog_id}) returned {fd}"),
                    ));
                    continue;
                }
                fd
            };
            targets.push(FentryTarget {
                slot,
                fd,
                idx: *idx,
                name: &func.display_name,
                ok: false,
                is_kernel: false,
            });
        }
        if targets.is_empty() {
            continue;
        }

        use crate::bpf_skel::fentry::*;
        let mut fentry_open_obj = std::mem::MaybeUninit::uninit();
        let fentry_builder = FentryProbeSkelBuilder::default();
        let mut fentry_open = match fentry_builder.open(&mut fentry_open_obj) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%e, "phase_b fentry skeleton open failed");
                for t in &targets {
                    unsafe { libc::close(t.fd) };
                }
                continue;
            }
        };

        if let Some(rodata) = fentry_open.maps.rodata_data.as_mut() {
            rodata.ktstr_enabled = true;
            for t in &targets {
                let k = t.is_kernel as u8;
                match t.slot {
                    0 => {
                        rodata.ktstr_fentry_func_idx_0 = t.idx;
                        rodata.ktstr_fentry_is_kernel_0 = k;
                    }
                    1 => {
                        rodata.ktstr_fentry_func_idx_1 = t.idx;
                        rodata.ktstr_fentry_is_kernel_1 = k;
                    }
                    2 => {
                        rodata.ktstr_fentry_func_idx_2 = t.idx;
                        rodata.ktstr_fentry_is_kernel_2 = k;
                    }
                    3 => {
                        rodata.ktstr_fentry_func_idx_3 = t.idx;
                        rodata.ktstr_fentry_is_kernel_3 = k;
                    }
                    _ => {}
                }
            }
        }

        for t in targets.iter_mut() {
            let fentry_prog = match t.slot {
                0 => &mut fentry_open.progs.ktstr_fentry_0,
                1 => &mut fentry_open.progs.ktstr_fentry_1,
                2 => &mut fentry_open.progs.ktstr_fentry_2,
                3 => &mut fentry_open.progs.ktstr_fentry_3,
                _ => continue,
            };
            match fentry_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                Ok(()) => {
                    t.ok = true;
                }
                Err(e) => {
                    tracing::warn!(slot = t.slot, func = t.name, %e, "phase_b fentry: set_attach_target failed");
                    diag.fentry_attach_failed
                        .push((t.name.to_string(), format!("set_attach_target: {e}")));
                    continue;
                }
            }
            let fexit_prog = match t.slot {
                0 => &mut fentry_open.progs.ktstr_fexit_0,
                1 => &mut fentry_open.progs.ktstr_fexit_1,
                2 => &mut fentry_open.progs.ktstr_fexit_2,
                3 => &mut fentry_open.progs.ktstr_fexit_3,
                _ => continue,
            };
            if let Err(e) = fexit_prog.set_attach_target(t.fd, Some(t.name.to_string())) {
                tracing::debug!(slot = t.slot, func = t.name, %e, "phase_b fexit: set_attach_target failed (entry-only)");
                fexit_prog.set_autoload(false);
            }
        }

        if !targets.iter().any(|t| t.ok) {
            for t in &targets {
                unsafe { libc::close(t.fd) };
            }
            continue;
        }

        let used_slots: std::collections::HashSet<usize> =
            targets.iter().filter(|t| t.ok).map(|t| t.slot).collect();
        for slot in 0..FENTRY_BATCH {
            if !used_slots.contains(&slot) {
                match slot {
                    0 => {
                        fentry_open.progs.ktstr_fentry_0.set_autoload(false);
                        fentry_open.progs.ktstr_fexit_0.set_autoload(false);
                    }
                    1 => {
                        fentry_open.progs.ktstr_fentry_1.set_autoload(false);
                        fentry_open.progs.ktstr_fexit_1.set_autoload(false);
                    }
                    2 => {
                        fentry_open.progs.ktstr_fentry_2.set_autoload(false);
                        fentry_open.progs.ktstr_fexit_2.set_autoload(false);
                    }
                    3 => {
                        fentry_open.progs.ktstr_fentry_3.set_autoload(false);
                        fentry_open.progs.ktstr_fexit_3.set_autoload(false);
                    }
                    _ => {}
                }
            }
        }

        use std::os::unix::io::AsFd;
        if let Err(e) = fentry_open
            .maps
            .probe_data
            .reuse_fd(skel.maps.probe_data.as_fd())
        {
            tracing::warn!(%e, "phase_b fentry: probe_data reuse_fd failed");
        }
        if let Err(e) = fentry_open
            .maps
            .func_meta_map
            .reuse_fd(skel.maps.func_meta_map.as_fd())
        {
            tracing::warn!(%e, "phase_b fentry: func_meta_map reuse_fd failed");
        }

        let fentry_skel = match fentry_open.load() {
            Ok(s) => {
                for t in &targets {
                    unsafe { libc::close(t.fd) };
                }
                s
            }
            Err(e) => {
                tracing::warn!(%e, "phase_b fentry: batch load failed");
                for t in &targets {
                    if t.ok {
                        diag.fentry_attach_failed
                            .push((t.name.to_string(), format!("batch load: {e}")));
                    }
                    unsafe { libc::close(t.fd) };
                }
                continue;
            }
        };

        for t in &targets {
            if !t.ok {
                continue;
            }

            let sentinel_ip = (t.idx as u64) | (1u64 << 63);
            let mut meta = crate::bpf_skel::types::func_meta {
                func_idx: t.idx,
                ..Default::default()
            };

            if let Some(btf_func) = pb.btf_funcs.iter().find(|f| f.name == t.name) {
                let mut field_specs = super::btf::resolve_field_specs(btf_func, None);
                if field_specs.is_empty()
                    && let Some(prog_id) = pb
                        .functions
                        .iter()
                        .find(|f| f.display_name == t.name)
                        .and_then(|f| f.bpf_prog_id)
                {
                    field_specs = super::btf::resolve_bpf_field_specs(btf_func, prog_id);
                }
                populate_field_specs(&mut meta, &field_specs);
                meta.str_param_idx = detect_str_param(btf_func);
            }

            let key_bytes = sentinel_ip.to_ne_bytes();
            let meta_bytes = unsafe {
                std::slice::from_raw_parts(
                    &meta as *const _ as *const u8,
                    std::mem::size_of::<crate::bpf_skel::types::func_meta>(),
                )
            };
            if let Err(e) = skel
                .maps
                .func_meta_map
                .update(&key_bytes, meta_bytes, MapFlags::ANY)
            {
                tracing::warn!(%e, func = t.name, "phase_b fentry: failed to update func_meta_map");
                continue;
            }
            func_ips.push((t.idx, sentinel_ip, t.name.to_string()));

            let result = match t.slot {
                0 => fentry_skel.progs.ktstr_fentry_0.attach_trace(),
                1 => fentry_skel.progs.ktstr_fentry_1.attach_trace(),
                2 => fentry_skel.progs.ktstr_fentry_2.attach_trace(),
                3 => fentry_skel.progs.ktstr_fentry_3.attach_trace(),
                _ => continue,
            };
            match result {
                Ok(link) => {
                    tracing::debug!(func = t.name, "phase_b fentry attached");
                    fentry_links.push(link);
                }
                Err(e) => {
                    tracing::warn!(%e, func = t.name, "phase_b fentry attach failed");
                    diag.fentry_attach_failed
                        .push((t.name.to_string(), e.to_string()));
                }
            }
            let fexit_result = match t.slot {
                0 => fentry_skel.progs.ktstr_fexit_0.attach_trace(),
                1 => fentry_skel.progs.ktstr_fexit_1.attach_trace(),
                2 => fentry_skel.progs.ktstr_fexit_2.attach_trace(),
                3 => fentry_skel.progs.ktstr_fexit_3.attach_trace(),
                _ => continue,
            };
            match fexit_result {
                Ok(link) => {
                    tracing::debug!(func = t.name, "phase_b fexit attached");
                    fexit_links.push(link);
                }
                Err(e) => {
                    tracing::debug!(%e, func = t.name, "phase_b fexit attach failed (entry-only)");
                }
            }
        }

        drop(fentry_skel);
    }

    diag.fentry_attached = fentry_links.len() as u32;
    tracing::debug!(
        fentry = fentry_links.len(),
        fexit = fexit_links.len(),
        bpf_targets = valid_bpf.len(),
        kernel_targets = kernel_fexit_targets.len(),
        "Phase B probes attached",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_field_keys_known_struct() {
        let func = super::BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "p".into(),
                struct_name: Some("task_struct".into()),
                is_ptr: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        assert!(
            keys.iter()
                .any(|(k, _)| k.contains("task_struct") && k.contains("pid"))
        );
        assert!(keys.iter().any(|(k, _)| k.contains("dsq_id")));
    }

    #[test]
    fn build_field_keys_scalar_param() {
        let func = super::BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "flags".into(),
                struct_name: None,
                is_ptr: false,
                ..Default::default()
            }],
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        assert!(keys.iter().any(|(k, _)| k.contains("flags:val.flags")));
    }

    #[test]
    fn build_field_keys_ptr_no_struct() {
        let func = super::BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "ctx".into(),
                struct_name: None,
                is_ptr: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        // Raw pointer with no struct info: no keys generated
        assert!(keys.is_empty());
    }

    #[test]
    fn build_field_keys_empty_params() {
        let func = super::BtfFunc {
            name: "empty".into(),
            params: vec![],
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        assert!(keys.is_empty());
    }

    #[test]
    fn resolve_func_ip_nonexistent() {
        assert!(resolve_func_ip("__nonexistent_kernel_function_xyz__").is_none());
    }

    #[test]
    fn build_field_keys_unknown_struct() {
        let func = super::BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "p".into(),
                struct_name: Some("unknown_struct_xyz".into()),
                is_ptr: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        assert!(keys.is_empty(), "unknown struct should produce no keys");
    }

    // -- detect_str_param --

    #[test]
    fn detect_str_param_btf_string_ptr() {
        let func = BtfFunc {
            name: "test".into(),
            params: vec![
                super::super::btf::BtfParam {
                    name: "p".into(),
                    struct_name: Some("task_struct".into()),
                    is_ptr: true,
                    ..Default::default()
                },
                super::super::btf::BtfParam {
                    name: "fmt".into(),
                    struct_name: None,
                    is_ptr: true,
                    is_string_ptr: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(detect_str_param(&func), 1);
    }

    #[test]
    fn detect_str_param_name_heuristic() {
        let func = BtfFunc {
            name: "test".into(),
            params: vec![
                super::super::btf::BtfParam {
                    name: "flags".into(),
                    struct_name: None,
                    is_ptr: false,
                    ..Default::default()
                },
                super::super::btf::BtfParam {
                    name: "msg".into(),
                    struct_name: None,
                    is_ptr: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(detect_str_param(&func), 1);
    }

    #[test]
    fn detect_str_param_none() {
        let func = BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "flags".into(),
                struct_name: None,
                is_ptr: false,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(detect_str_param(&func), 0xff);
    }

    #[test]
    fn detect_str_param_struct_ptr_not_string() {
        let func = BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "rq".into(),
                struct_name: Some("rq".into()),
                is_ptr: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(detect_str_param(&func), 0xff);
    }

    #[test]
    fn detect_str_param_name_contains_str() {
        let func = BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "my_str_ptr".into(),
                struct_name: None,
                is_ptr: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(detect_str_param(&func), 0);
    }

    // -- build_field_keys with auto_fields --

    #[test]
    fn build_field_keys_auto_fields() {
        let func = BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "ctx".into(),
                struct_name: None,
                is_ptr: true,
                auto_fields: vec![
                    ("field_a".into(), "->field_a".into(), RenderHint::Bool),
                    ("field_b".into(), "->field_b".into(), RenderHint::Signed),
                ],
                type_name: Some("task_ctx".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        assert_eq!(keys.len(), 2);
        assert!(keys[0].0.contains("task_ctx"));
        assert!(keys[0].0.contains("field_a"));
        assert_eq!(keys[0].1, RenderHint::Bool);
        assert!(keys[1].0.contains("field_b"));
        assert_eq!(keys[1].1, RenderHint::Signed);
    }

    // -- build_field_keys with cpumask fields --

    #[test]
    fn build_field_keys_includes_cpumask_words() {
        let func = BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "p".into(),
                struct_name: Some("task_struct".into()),
                is_ptr: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        assert!(
            keys.iter().any(|(k, _)| k.contains("cpumask_0")),
            "should have cpumask_0: {keys:?}",
        );
        assert!(
            keys.iter().any(|(k, _)| k.contains("cpumask_3")),
            "should have cpumask_3: {keys:?}",
        );
    }

    #[test]
    fn build_field_keys_max_six_params() {
        let params: Vec<_> = (0..8)
            .map(|i| super::super::btf::BtfParam {
                name: format!("p{i}"),
                struct_name: None,
                is_ptr: false,
                ..Default::default()
            })
            .collect();
        let func = super::BtfFunc {
            name: "many".into(),
            params,
            ..Default::default()
        };
        let keys = build_field_keys(&func);
        // Only first 6 params processed
        assert!(keys.len() <= 6);
        assert!(keys.iter().any(|(k, _)| k.contains("p5")));
        assert!(!keys.iter().any(|(k, _)| k.contains("p6")));
    }
}
