//! Shared test helpers for `monitor` submodules.
//!
//! Helpers that two or more sibling test modules need — e.g.
//! [`name_from_str`] used by both `bpf_map::tests` and
//! `dump::tests` — live here to avoid duplicate copies that drift.
//! `#[cfg(test)] pub(crate)` so descendants of `crate::monitor` can
//! reach in (siblings cannot see each other's private `tests`
//! modules; a shared parent module is the only path).

use super::bpf_map::BPF_OBJ_NAME_LEN;

/// Pack a `&str` into the inline name representation
/// (`name_bytes`, `name_len`) used by
/// [`super::bpf_map::BpfMapInfo`]. Truncates to
/// `BPF_OBJ_NAME_LEN` when the input exceeds that — matches the
/// kernel's own bookkeeping (`bpf_obj_name_cpy` in
/// `kernel/bpf/syscall.c` rejects names longer than the field, but
/// for tests we silently truncate so call sites can use whatever
/// length is convenient without precomputing the cap).
pub(crate) fn name_from_str(s: &str) -> ([u8; BPF_OBJ_NAME_LEN], u8) {
    let mut buf = [0u8; BPF_OBJ_NAME_LEN];
    let bytes = s.as_bytes();
    let n = bytes.len().min(BPF_OBJ_NAME_LEN);
    buf[..n].copy_from_slice(&bytes[..n]);
    (buf, n as u8)
}
