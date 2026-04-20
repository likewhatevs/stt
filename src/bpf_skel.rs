//! Generated BPF skeleton bindings produced by libbpf-cargo during build.

include!(concat!(env!("OUT_DIR"), "/probe_skel.rs"));

pub mod fentry {
    include!(concat!(env!("OUT_DIR"), "/fentry_probe_skel.rs"));
}
