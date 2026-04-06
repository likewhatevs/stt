use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::btf::{BtfFunc, STRUCT_FIELDS};
use super::stack::StackFunction;

/// Structured probe event returned from the skeleton.
#[derive(Debug, Clone)]
pub struct ProbeEvent {
    pub func_idx: u32,
    pub tid: u32,
    pub ts: u64,
    pub args: [u64; 6],
    pub fields: Vec<(String, u64)>, // (field_key, value) — decoded by caller
    pub kstack: Vec<u64>,
}

/// Resolve a kernel function name to its address via /proc/kallsyms.
#[cfg_attr(feature = "integration", visibility::make(pub))]
fn resolve_func_ip(name: &str) -> Option<u64> {
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

/// Build field key names for a function based on its BTF info.
/// Returns a vec mapping field_idx to an output key name like "param:struct.field".
fn build_field_keys(btf_func: &BtfFunc) -> Vec<String> {
    let mut keys = Vec::new();
    let mut field_idx: u32 = 0;

    let max_params = btf_func.params.len().min(6);
    for param in &btf_func.params[..max_params] {
        if let Some(ref sname) = param.struct_name {
            if let Some((_, fields)) = STRUCT_FIELDS.iter().find(|(s, _)| *s == sname) {
                for (_, key) in *fields {
                    keys.push(format!("{}:{}.{}", param.name, sname, key));
                    field_idx += 1;
                    if field_idx >= 16 {
                        break;
                    }
                }
            }
        } else if !param.is_ptr {
            keys.push(format!("{}:val.{}", param.name, param.name));
            field_idx += 1;
        }
    }

    keys
}

/// Run the BPF skeleton probe. Attaches kprobes to the given functions,
/// waits for the trigger to fire, returns captured events.
pub fn run_probe_skeleton(
    functions: &[StackFunction],
    btf_funcs: &[BtfFunc],
    trigger: &str,
    stop: &AtomicBool,
) -> Option<Vec<ProbeEvent>> {
    use crate::bpf_skel::*;
    use libbpf_rs::skel::{OpenSkel, SkelBuilder};
    use libbpf_rs::{Link, MapCore, MapFlags, RingBufferBuilder};

    // Open skeleton
    let mut open_object = std::mem::MaybeUninit::uninit();
    let builder = ProbeSkelBuilder::default();
    let mut open_skel = match builder.open(&mut open_object) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(%e, "failed to open probe skeleton");
            return None;
        }
    };

    // Enable probes (must set before load — rodata is immutable after)
    if let Some(rodata) = open_skel.maps.rodata_data.as_mut() {
        rodata.stt_enabled = true;
    }

    // Load skeleton
    let skel = match open_skel.load() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(%e, "failed to load probe skeleton");
            return None;
        }
    };

    // Populate func_meta_map with function IPs and metadata
    let mut func_ips: Vec<(u32, u64, String)> = Vec::new(); // (idx, ip, display_name)

    for (idx, func) in functions.iter().enumerate() {
        if func.is_bpf {
            continue; // BPF fentry handled separately (future)
        }
        let ip = match resolve_func_ip(&func.raw_name) {
            Some(ip) => ip,
            None => {
                tracing::warn!(func = %func.raw_name, "could not resolve function IP");
                continue;
            }
        };

        let meta = types::func_meta {
            func_idx: idx as u32,
            ..Default::default()
        };

        // nr_field_specs stays 0: BPF-side field dereferencing
        // uses raw args. Full field capture needs CO-RE offset
        // metadata which varies per kernel.

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

        func_ips.push((idx as u32, ip, func.display_name.clone()));
    }

    if func_ips.is_empty() {
        tracing::error!("no functions resolved — nothing to probe");
        return None;
    }

    // Attach kprobes to each function using the raw kernel symbol name.
    let mut links: Vec<Link> = Vec::new();
    for (idx, _, _) in &func_ips {
        let raw = &functions[*idx as usize].raw_name;
        match skel.progs.stt_probe.attach_kprobe(false, raw) {
            Ok(link) => links.push(link),
            Err(e) => {
                tracing::warn!(%e, func = %raw, "failed to attach kprobe");
            }
        }
    }

    // Attach trigger
    match skel.progs.stt_trigger.attach_kprobe(false, trigger) {
        Ok(link) => links.push(link),
        Err(e) => {
            tracing::error!(%e, trigger, "failed to attach trigger kprobe");
            return None;
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
    rb_builder
        .add(&skel.maps.events, move |data: &[u8]| {
            if data.len() < std::mem::size_of::<RbEvent>() {
                return 0;
            }
            let raw: &RbEvent = unsafe { &*(data.as_ptr() as *const RbEvent) };

            if raw.type_ == 2 {
                // EVENT_TRIGGER
                triggered_clone.store(true, Ordering::Relaxed);

                let kstack_sz = (raw.kstack_sz as usize).min(32);
                let event = ProbeEvent {
                    func_idx: 0,
                    tid: raw.tid,
                    ts: raw.ts,
                    args: [0; 6],
                    fields: vec![],
                    kstack: raw.kstack[..kstack_sz].to_vec(),
                };

                events_clone.lock().unwrap().push(event);
            }

            0
        })
        .ok();

    let rb = match rb_builder.build() {
        Ok(rb) => rb,
        Err(e) => {
            tracing::error!(%e, "failed to build ring buffer");
            return None;
        }
    };

    // Enable is handled by the BPF program reading the volatile const.
    // Since we can't mutate rodata after load, the program starts enabled.
    // (stt_enabled defaults to false in BPF, but we always want probes
    // active once attached — remove the gate or set it before load.)

    tracing::info!(
        n_funcs = func_ips.len(),
        trigger,
        "skeleton probes attached, waiting for trigger"
    );

    // Poll until trigger fires or stop requested
    loop {
        let _ = rb.poll(Duration::from_millis(100));

        if triggered.load(Ordering::Relaxed) {
            tracing::info!("trigger fired");
            // Read probe_data map for all functions × current tid
            // (the trigger event tells us the tid)
            let guard = events.lock().unwrap();
            if let Some(trigger_event) = guard.last() {
                let tid = trigger_event.tid;
                drop(guard);

                let mut probe_events = Vec::new();
                for (idx, ip, name) in &func_ips {
                    let key = types::probe_key {
                        func_ip: *ip,
                        tid,
                        _pad: 0,
                    };
                    let key_bytes = unsafe {
                        std::slice::from_raw_parts(
                            &key as *const _ as *const u8,
                            std::mem::size_of::<types::probe_key>(),
                        )
                    };

                    if let Ok(val_bytes) = skel.maps.probe_data.lookup(key_bytes, MapFlags::ANY)
                        && let Some(val_bytes) = val_bytes
                    {
                        let entry: &types::probe_entry =
                            unsafe { &*(val_bytes.as_ptr() as *const types::probe_entry) };
                        if entry.ts == 0 {
                            continue; // never hit for this tid
                        }

                        // Build field keys from BTF info
                        let field_names: Vec<String> = btf_funcs
                            .iter()
                            .find(|f| f.name == *name)
                            .map(build_field_keys)
                            .unwrap_or_default();

                        let fields: Vec<(String, u64)> = entry.fields[..entry.nr_fields as usize]
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &val)| field_names.get(i).map(|k| (k.clone(), val)))
                            .collect();

                        probe_events.push(ProbeEvent {
                            func_idx: *idx,
                            tid,
                            ts: entry.ts,
                            args: entry.args,
                            fields,
                            kstack: vec![],
                        });
                    }
                }

                // Sort by timestamp
                probe_events.sort_by_key(|e| e.ts);

                // Add the trigger event's kstack to the last probe event (or create one)
                let trigger_kstack = events
                    .lock()
                    .unwrap()
                    .last()
                    .map(|e| e.kstack.clone())
                    .unwrap_or_default();
                if let Some(last) = probe_events.last_mut() {
                    last.kstack = trigger_kstack;
                }

                return Some(probe_events);
            }

            return None;
        }

        if stop.load(Ordering::Relaxed) {
            return None;
        }
    }
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
            }],
        };
        let keys = build_field_keys(&func);
        assert!(
            keys.iter()
                .any(|k| k.contains("task_struct") && k.contains("pid"))
        );
        assert!(keys.iter().any(|k| k.contains("dsq_id")));
    }

    #[test]
    fn build_field_keys_scalar_param() {
        let func = super::BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "flags".into(),
                struct_name: None,
                is_ptr: false,
            }],
        };
        let keys = build_field_keys(&func);
        assert!(keys.iter().any(|k| k.contains("flags:val.flags")));
    }

    #[test]
    fn build_field_keys_ptr_no_struct() {
        let func = super::BtfFunc {
            name: "test".into(),
            params: vec![super::super::btf::BtfParam {
                name: "ctx".into(),
                struct_name: None,
                is_ptr: true,
            }],
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
            }],
        };
        let keys = build_field_keys(&func);
        assert!(keys.is_empty(), "unknown struct should produce no keys");
    }

    #[test]
    fn build_field_keys_max_six_params() {
        let params: Vec<_> = (0..8)
            .map(|i| super::super::btf::BtfParam {
                name: format!("p{i}"),
                struct_name: None,
                is_ptr: false,
            })
            .collect();
        let func = super::BtfFunc {
            name: "many".into(),
            params,
        };
        let keys = build_field_keys(&func);
        // Only first 6 params processed
        assert!(keys.len() <= 6);
        assert!(keys.iter().any(|k| k.contains("p5")));
        assert!(!keys.iter().any(|k| k.contains("p6")));
    }
}
