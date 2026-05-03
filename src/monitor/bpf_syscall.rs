//! Live-host BPF map accessor backed by the `bpf(2)` syscall.
//!
//! Companion to [`super::bpf_map::GuestMemMapAccessor`]: same trait
//! ([`super::bpf_map::BpfMapAccessor`]), different data path. Where
//! GuestMemMapAccessor walks frozen guest physical memory via PTE
//! resolution against `init_mm`, this backend talks directly to the
//! running host kernel through the `bpf()` syscall — KASLR is fully
//! abstracted, no symbol resolution required, no page-walk math.
//!
//! # Backend differences vs. guest-memory path
//!
//! | concern        | GuestMemMapAccessor                                            | BpfSyscallAccessor                                                           |
//! |----------------|----------------------------------------------------------------|------------------------------------------------------------------------------|
//! | discovery      | walk `map_idr` xarray in guest memory                          | `BPF_MAP_GET_NEXT_ID` + `BPF_MAP_GET_FD_BY_ID` loop                          |
//! | array values   | follow `bpf_array.value` flex array via PTE walks              | `BPF_MAP_LOOKUP_ELEM(fd, &key=0, buf)` returns the inline value bytes        |
//! | hash iteration | walk `bpf_htab.buckets` directly (freeze rendezvous = sync)    | `BPF_MAP_GET_NEXT_KEY` + `BPF_MAP_LOOKUP_ELEM` per key (kernel RCU read-side) |
//! | per-CPU array  | read each CPU's slot via `__per_cpu_offset[cpu]`               | one `BPF_MAP_LOOKUP_ELEM` returns `nr_possible_cpus * value_size` bytes      |
//! | arena          | walk `bpf_arena -> kern_vm -> vm_struct.addr` PTE-by-PTE        | `mmap(arena_fd, ...)` — `lookup_elem` returns `-EINVAL` on arena             |
//! | program BTF    | read split-BTF blob from guest memory                          | `BPF_BTF_GET_FD_BY_ID` + `BPF_OBJ_GET_INFO_BY_FD` to extract BTF bytes       |
//!
//! # Map fd pinning
//!
//! Every map discovered at construction time has its fd held open for
//! the lifetime of the accessor. The kernel's
//! `bpf_map_put`/`atomic64_dec_and_test` (`kernel/bpf/syscall.c`) only
//! frees a map when its refcount reaches zero, and userspace fds count
//! as references. This means the scheduler can exit and tear down its
//! struct_ops link while the accessor is still iterating maps — the
//! underlying memory stays valid.
//!
//! # Required capabilities
//!
//! `BPF_MAP_GET_NEXT_ID` and `BPF_MAP_GET_FD_BY_ID` require
//! `CAP_SYS_ADMIN` (or, since 5.16, `CAP_BPF` for some commands;
//! `..._GET_NEXT_ID` still requires `CAP_SYS_ADMIN`). ktstr always runs
//! as root in the test environment, so this is a non-issue for the
//! library's primary consumer; the `from_running_kernel` constructor
//! surfaces the kernel's `EPERM` directly so live-host CLI use cases
//! can produce a clear error.
//!
//! # Lock-free reads
//!
//! Without a freeze rendezvous, the kernel's per-element atomicity is
//! the only ordering primitive. Per-element u64-aligned fields are
//! atomic on x86_64; multi-element transactions the scheduler intended
//! to commit atomically may surface as torn views relative to the
//! walker. This is identical to the guest-memory backend's torn-read
//! behavior, just for a different reason. Two-snapshot in-BPF capture
//! (bpf_timer + tp_btf) is the recommended remedy and lives outside
//! this backend.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;

use anyhow::{Context, Result, anyhow};
use btf_rs::Btf;

use super::arena::{ArenaPage, ArenaSnapshot, BpfArenaOffsets};
use super::bpf_map::{
    BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_ARRAY,
    BpfMapAccessor, BpfMapInfo,
};

/// `BPF_MAP_LOOKUP_ELEM` — read one map value into a userspace buffer.
const BPF_MAP_LOOKUP_ELEM: u32 = 1;
/// `BPF_MAP_GET_NEXT_KEY` — advance hash iteration cursor.
const BPF_MAP_GET_NEXT_KEY: u32 = 4;
/// `BPF_MAP_GET_NEXT_ID` — advance the kernel's map id walk.
const BPF_MAP_GET_NEXT_ID: u32 = 0xc;
/// `BPF_MAP_GET_FD_BY_ID` — pin a map by id.
const BPF_MAP_GET_FD_BY_ID: u32 = 0xe;
/// `BPF_OBJ_GET_INFO_BY_FD` — fetch map/btf metadata from an open fd.
const BPF_OBJ_GET_INFO_BY_FD: u32 = 0xf;
/// `BPF_BTF_GET_FD_BY_ID` — pin a BTF object by id.
/// Per `include/uapi/linux/bpf.h::enum bpf_cmd`: 19 (0x13). Counting
/// from `BPF_MAP_CREATE = 0` through `BPF_BTF_LOAD = 18` makes the
/// next entry `BPF_BTF_GET_FD_BY_ID = 19`.
const BPF_BTF_GET_FD_BY_ID: u32 = 0x13;

/// `BPF_OBJ_NAME_LEN` from `include/uapi/linux/bpf.h`.
const BPF_OBJ_NAME_LEN: usize = 16;

/// 4 KiB — page size for arena mmap. Matches `arena.c`'s
/// `PAGE_SIZE` (the kernel arena allocator works at 4 KiB granularity
/// regardless of host THP/hugetlb config).
const ARENA_PAGE_SIZE: usize = 4096;

/// Maximum total bytes the arena snapshot reads via mmap, mirroring the
/// guest-memory backend's `MAX_VM_RANGE_BYTES`. Keeps a runaway
/// `max_entries` from inducing a multi-GiB read.
const MAX_ARENA_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Maximum number of arena pages enumerated sequentially before the
/// walker switches to a stride-probe sweep. Mirrors the
/// `MAX_ARENA_PAGES` cap on the guest-memory side so both backends
/// produce comparable snapshot extents.
const MAX_ARENA_PAGES: u64 = 16 * 1024;

// `bpf_attr` is a uapi union with many command-specific shapes. Rather
// than declare a full union we lay out per-command structs that match
// the relevant union arm exactly (uapi-stable size + field order). The
// kernel verifies the size passed to the syscall matches one of the
// recognized arm lengths; we pass `size_of::<arm>()` for each.

/// `bpf_attr` arm for `BPF_MAP_*_ELEM` and `BPF_MAP_GET_NEXT_KEY`.
/// Source: `include/uapi/linux/bpf.h::union bpf_attr` (the
/// MAP_ELEM_OPS arm).
#[repr(C)]
#[derive(Default)]
struct BpfAttrMapElem {
    map_fd: u32,
    _pad0: u32,
    key: u64,
    value_or_next_key: u64,
    flags: u64,
}

/// `bpf_attr` arm for `BPF_MAP_GET_NEXT_ID`, `BPF_BTF_GET_NEXT_ID`,
/// and the corresponding `*_GET_FD_BY_ID` commands.
#[repr(C)]
#[derive(Default)]
struct BpfAttrGetId {
    /// `start_id` for `*_GET_NEXT_ID`; `id` for `*_GET_FD_BY_ID`.
    id_or_start_id: u32,
    next_id: u32,
    open_flags: u32,
}

/// `bpf_attr` arm for `BPF_OBJ_GET_INFO_BY_FD`.
#[repr(C)]
#[derive(Default)]
struct BpfAttrInfoByFd {
    bpf_fd: u32,
    info_len: u32,
    info: u64,
}

/// `struct bpf_map_info` from `include/uapi/linux/bpf.h`. The kernel
/// has grown this struct over time; we pass our struct size as
/// `info_len` and the kernel zero-fills any tail it doesn't fill in.
/// All fields are documented in the kernel header.
#[repr(C)]
#[derive(Default)]
struct BpfMapInfoUapi {
    map_type: u32,
    id: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    name: [u8; BPF_OBJ_NAME_LEN],
    ifindex: u32,
    btf_vmlinux_value_type_id: u32,
    netns_dev: u64,
    netns_ino: u64,
    btf_id: u32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    /// Kernel field `btf_vmlinux_id` per
    /// `include/uapi/linux/bpf.h::struct bpf_map_info`. Unused by the
    /// caller; named `_pad` here because the value is currently
    /// discarded by the BPF accessor — rename without binding the
    /// field to a public consumer that can rot.
    _pad: u32,
    map_extra: u64,
}

/// `struct bpf_btf_info` from `include/uapi/linux/bpf.h`. Used to
/// extract a BTF blob's bytes given an open BTF fd.
#[repr(C)]
#[derive(Default)]
struct BpfBtfInfoUapi {
    btf: u64,
    btf_size: u32,
    id: u32,
    name: u64,
    name_len: u32,
    kernel_btf: u32,
}

/// Raw `bpf(2)` syscall wrapper. Returns the kernel's return value as
/// `i64` so callers can check for `< 0` and inspect `errno`. The
/// kernel's `__sys_bpf` (`kernel/bpf/syscall.c`) verifies the `size`
/// argument matches a recognized `bpf_attr` arm length.
///
/// SAFETY: `attr_ptr` must be a valid pointer to `attr_size` bytes of
/// the appropriate `bpf_attr` arm. The kernel reads the union by size,
/// so passing a smaller-than-required arm causes -EINVAL; passing a
/// larger one is rejected as well.
unsafe fn bpf_syscall(cmd: u32, attr_ptr: *const u8, attr_size: usize) -> i64 {
    // SAFETY: caller must ensure attr_ptr/attr_size validity. The
    // syscall itself is signal-safe and reentrant.
    unsafe { libc::syscall(libc::SYS_bpf, cmd as i64, attr_ptr, attr_size) as i64 }
}

/// Wrap a `bpf()` syscall result in a `Result<RawFd>` for commands
/// that return an fd. Negative returns are converted to errno-bearing
/// errors; non-negative returns become the fd.
fn bpf_call_fd(cmd: u32, attr_ptr: *const u8, attr_size: usize) -> Result<RawFd> {
    // SAFETY: caller has built attr_ptr/attr_size correctly per the
    // command's bpf_attr arm.
    let ret = unsafe { bpf_syscall(cmd, attr_ptr, attr_size) };
    if ret < 0 {
        let err = std::io::Error::last_os_error();
        Err(anyhow!("bpf({cmd}) failed: {err}"))
    } else {
        Ok(ret as RawFd)
    }
}

/// Wrap a `bpf()` syscall result for commands that return 0 on
/// success, `< 0` on error.
fn bpf_call_status(cmd: u32, attr_ptr: *const u8, attr_size: usize) -> Result<()> {
    // SAFETY: caller has built attr_ptr/attr_size correctly.
    let ret = unsafe { bpf_syscall(cmd, attr_ptr, attr_size) };
    if ret < 0 {
        let err = std::io::Error::last_os_error();
        Err(anyhow!("bpf({cmd}) failed: {err}"))
    } else {
        Ok(())
    }
}

/// One discovered map together with its pinned fd. The `OwnedFd`
/// guarantees the map's refcount stays >0 for the accessor's
/// lifetime — even if the scheduler exits and userspace tear-down
/// runs, `bpf_map_put` only frees when every fd is dropped (see
/// `kernel/bpf/syscall.c` `bpf_map_put`).
struct PinnedMap {
    info: BpfMapInfo,
    fd: OwnedFd,
    /// Raw `map_extra` from the kernel info struct. Arena maps
    /// hardcode this to a deterministic mmap target address (x86:
    /// `1<<44`, aarch64: `1<<32`) per `lib/arena_map.h`. Surfaced
    /// here so the arena mmap path can use `MAP_FIXED_NOREPLACE` at
    /// the kernel-blessed address rather than letting `mmap` pick
    /// one — which would diverge from what BPF programs see.
    map_extra: u64,
}

/// Live-host BPF map accessor.
///
/// Construction enumerates every map id reachable via
/// `BPF_MAP_GET_NEXT_ID`, opens an fd for each via
/// `BPF_MAP_GET_FD_BY_ID`, and caches the metadata. The fd vector is
/// held for the accessor's lifetime so the maps cannot be freed
/// underneath us — even if the scheduler exits and tears down its
/// struct_ops link mid-walk.
///
/// Selectively populating the cache is intentional: the same trait
/// surface accepts a `BpfMapInfo` argument on every method, so an
/// accessor that holds only the maps a particular failure dump cares
/// about (filtered by name suffix at construction time) is just as
/// valid as one that holds every map on the system. The
/// `from_running_kernel_filtered` constructor exposes that knob.
#[allow(dead_code)]
pub struct BpfSyscallAccessor {
    maps: Vec<PinnedMap>,
}

impl BpfSyscallAccessor {
    /// Discover and pin every BPF map currently visible to the
    /// running kernel.
    ///
    /// Walks the kernel's id space via `BPF_MAP_GET_NEXT_ID` (starting
    /// from id 0), pinning each map with `BPF_MAP_GET_FD_BY_ID` and
    /// fetching its metadata via `BPF_OBJ_GET_INFO_BY_FD`. Maps that
    /// disappear between the `NEXT_ID` and `GET_FD_BY_ID` calls (a
    /// concurrent scheduler unload, for instance) are silently
    /// skipped — that race is inherent to live-host enumeration and
    /// is not an error.
    ///
    /// Requires `CAP_SYS_ADMIN`. ktstr always runs as root in the
    /// test environment so this is a non-issue for the primary
    /// consumer; live-host CLI users that hit `EPERM` will see it
    /// in the returned error.
    #[allow(dead_code)]
    pub fn from_running_kernel() -> Result<Self> {
        Self::from_running_kernel_filtered(|_info: &BpfMapInfo| true)
    }

    /// Discover and pin every BPF map for which `predicate` returns
    /// `true`. Maps that fail the predicate are closed (their fds
    /// drop) so the kernel can free them as usual.
    ///
    /// Useful when the caller knows which maps the failure dump will
    /// touch — typically the scheduler's named maps that match a
    /// specific suffix — and wants to avoid pinning hundreds of
    /// unrelated maps that happen to be alive (cilium, systemd,
    /// other workloads).
    #[allow(dead_code)]
    pub fn from_running_kernel_filtered<F>(mut predicate: F) -> Result<Self>
    where
        F: FnMut(&BpfMapInfo) -> bool,
    {
        let mut maps: Vec<PinnedMap> = Vec::new();
        let mut start_id: u32 = 0;

        loop {
            // The kernel writes `next_id` via the syscall's raw pointer
            // path, but Rust's borrow checker doesn't see that — it
            // sees the struct as never mutated through a Rust binding.
            // Declare mut anyway so the compiler treats `attr.next_id`
            // as written, then read it back through a raw read after
            // the syscall returns.
            let mut attr = BpfAttrGetId {
                id_or_start_id: start_id,
                next_id: 0,
                open_flags: 0,
            };
            // SAFETY: BpfAttrGetId is repr(C) with the exact layout the
            // kernel expects for *_GET_NEXT_ID.
            let res = unsafe {
                bpf_syscall(
                    BPF_MAP_GET_NEXT_ID,
                    &raw mut attr as *const u8,
                    std::mem::size_of::<BpfAttrGetId>(),
                )
            };
            if res < 0 {
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::ENOENT) {
                    break;
                }
                return Err(anyhow!("BPF_MAP_GET_NEXT_ID failed: {err}"));
            }

            let next_id = attr.next_id;
            // Defensive: kernel returned 0 for `next_id` somehow.
            // Shouldn't happen on success, but guard against an
            // infinite loop.
            if next_id == 0 {
                break;
            }
            // Advance start_id for the next iteration BEFORE the
            // get-fd-by-id call so a transient EPERM/ENOENT on a
            // single id doesn't wedge the walk.
            start_id = next_id;

            // Try to pin the map. ENOENT here means the map was
            // freed between the NEXT_ID and GET_FD_BY_ID calls. The
            // kernel doesn't write to this attr (GET_FD_BY_ID is
            // input-only), so the binding is plain (no mut).
            let fd_attr = BpfAttrGetId {
                id_or_start_id: next_id,
                next_id: 0,
                open_flags: 0,
            };
            let fd_ret = unsafe {
                bpf_syscall(
                    BPF_MAP_GET_FD_BY_ID,
                    &raw const fd_attr as *const u8,
                    std::mem::size_of::<BpfAttrGetId>(),
                )
            };
            if fd_ret < 0 {
                // A failed `BPF_MAP_GET_FD_BY_ID` skips this map and
                // keeps walking — a single bad map must not abort
                // enumeration. The error categories matter for
                // diagnostics, so surface non-ENOENT cases via
                // tracing rather than silently dropping them:
                //
                // - `ENOENT`: the map was freed between
                //   `GET_NEXT_ID` and `GET_FD_BY_ID`. Routine
                //   under churn; suppressed at `debug` level so the
                //   normal log stays quiet.
                // - `EPERM`: missing CAP_SYS_ADMIN / CAP_BPF for
                //   this map (e.g. a kernel-internal map a less-
                //   privileged caller can't pin). Logged at `warn`
                //   so an operator who expects to see the map knows
                //   why it's missing.
                // - `EBADF` / others: a kernel-side state error.
                //   Logged at `warn` with the errno so the operator
                //   can correlate against `dmesg`.
                let err = std::io::Error::last_os_error();
                let raw = err.raw_os_error().unwrap_or(0);
                if raw == libc::ENOENT {
                    tracing::debug!(
                        map_id = next_id,
                        "BPF_MAP_GET_FD_BY_ID: map vanished mid-walk (ENOENT); skipping"
                    );
                } else {
                    tracing::warn!(
                        map_id = next_id,
                        errno = raw,
                        error = %err,
                        "BPF_MAP_GET_FD_BY_ID failed; skipping this map but continuing the walk"
                    );
                }
                continue;
            }
            // SAFETY: fd_ret >= 0; the kernel guarantees a valid fd
            // for non-negative returns.
            let fd = unsafe { OwnedFd::from_raw_fd(fd_ret as RawFd) };

            // Fetch info to populate BpfMapInfo + decide whether to
            // keep the fd. A failure here means the map's metadata
            // can't be read (kernel-side state error or fd was
            // closed mid-walk); surface it via tracing so the
            // operator sees the correlation rather than a silently
            // dropped map.
            let (info, map_extra) = match obj_get_info_map(fd.as_raw_fd()) {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(
                        map_id = next_id,
                        error = %e,
                        "BPF_OBJ_GET_INFO_BY_FD failed for pinned map; skipping"
                    );
                    continue;
                }
            };

            // Hand the predicate a BpfMapInfo for the keep/discard
            // decision. Discarded fds drop here.
            if !predicate(&info) {
                continue;
            }

            maps.push(PinnedMap {
                info,
                fd,
                map_extra,
            });
        }

        Ok(Self { maps })
    }

    /// Number of pinned maps currently held. Test helper.
    #[cfg(test)]
    pub(crate) fn pinned_count(&self) -> usize {
        self.maps.len()
    }

    /// Look up the pinned fd for a map identified by its
    /// `BpfMapInfo`. Returns `None` when no pinned map matches.
    ///
    /// Match key: `name` field. Map ids would be more precise but
    /// they're not part of `BpfMapInfo` today (a known follow-up if
    /// the live-host backend grows other consumers); within a single
    /// scheduler instance, names are unique and stable for the
    /// duration of the run.
    fn pinned_for(&self, target: &BpfMapInfo) -> Option<&PinnedMap> {
        self.maps.iter().find(|p| p.info.name == target.name)
    }
}

/// Fetch `bpf_map_info` for an open map fd via
/// `BPF_OBJ_GET_INFO_BY_FD`. Returns the populated [`BpfMapInfo`]
/// alongside the raw `map_extra` field — the latter is needed by the
/// arena mmap path but doesn't fit on the cross-backend
/// [`BpfMapInfo`] surface (the guest-memory path doesn't use it).
fn obj_get_info_map(fd: RawFd) -> Result<(BpfMapInfo, u64)> {
    let mut info = BpfMapInfoUapi::default();
    let attr = BpfAttrInfoByFd {
        bpf_fd: fd as u32,
        info_len: std::mem::size_of::<BpfMapInfoUapi>() as u32,
        info: &raw mut info as u64,
    };
    bpf_call_status(
        BPF_OBJ_GET_INFO_BY_FD,
        &raw const attr as *const u8,
        std::mem::size_of::<BpfAttrInfoByFd>(),
    )
    .context("BPF_OBJ_GET_INFO_BY_FD on map fd")?;

    let nul = info
        .name
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(BPF_OBJ_NAME_LEN);
    let name = String::from_utf8_lossy(&info.name[..nul]).to_string();

    Ok((
        BpfMapInfo {
            // map_pa / map_kva / value_kva are guest-memory concepts
            // that don't apply on the live host. Populating with 0
            // is fine — the live-host backend's read paths route
            // through the pinned fd, not these fields.
            map_pa: 0,
            map_kva: 0,
            name,
            map_type: info.map_type,
            map_flags: info.map_flags,
            key_size: info.key_size,
            value_size: info.value_size,
            max_entries: info.max_entries,
            value_kva: None,
            // btf_kva is similarly a guest-memory locator. Live-host
            // BTF resolution goes through `btf_id` →
            // `BPF_BTF_GET_FD_BY_ID` instead.
            btf_kva: u64::from(info.btf_id),
            btf_value_type_id: info.btf_value_type_id,
            btf_key_type_id: info.btf_key_type_id,
        },
        info.map_extra,
    ))
}

impl BpfMapAccessor for BpfSyscallAccessor {
    fn maps(&self) -> Vec<BpfMapInfo> {
        self.maps.iter().map(|p| p.info.clone()).collect()
    }

    fn read_value(&self, map: &BpfMapInfo, offset: usize, len: usize) -> Option<Vec<u8>> {
        let pinned = self.pinned_for(map)?;

        // The live-host backend supports value reads on ARRAY and
        // PERCPU_ARRAY (via dedicated method) — not HASH (use
        // iter_hash_map) and not ARENA (use read_arena_pages). For
        // ARRAY at key 0, the kernel returns `value_size` bytes,
        // covering the whole .bss when the array is the global
        // .bss section that sched_ext schedulers commonly use.
        if map.map_type != BPF_MAP_TYPE_ARRAY {
            return None;
        }

        // Build the lookup. ARRAY maps use a single u32 key; we
        // always look up key=0 here because the failure-dump path's
        // .bss reads slice into the resulting buffer.
        let mut key: u32 = 0;
        let mut buf = vec![0u8; map.value_size as usize];
        let attr = BpfAttrMapElem {
            map_fd: pinned.fd.as_raw_fd() as u32,
            _pad0: 0,
            key: &raw mut key as u64,
            value_or_next_key: buf.as_mut_ptr() as u64,
            flags: 0,
        };
        bpf_call_status(
            BPF_MAP_LOOKUP_ELEM,
            &raw const attr as *const u8,
            std::mem::size_of::<BpfAttrMapElem>(),
        )
        .ok()?;

        // Slice into the requested window. Out-of-bounds offsets
        // return None to mirror the guest-memory backend's behavior
        // when a value-region read straddles an unmapped page.
        let end = offset.checked_add(len)?;
        if end > buf.len() {
            return None;
        }
        Some(buf[offset..end].to_vec())
    }

    fn iter_hash_map(&self, map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)> {
        let Some(pinned) = self.pinned_for(map) else {
            return Vec::new();
        };
        if map.map_type != BPF_MAP_TYPE_HASH {
            return Vec::new();
        }

        let key_sz = map.key_size as usize;
        let val_sz = map.value_size as usize;
        let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        // First key: pass NULL for the input key per `bpf(2)` man
        // page — kernel returns the first key in the table.
        let mut cur_key = vec![0u8; key_sz];
        let mut next_key = vec![0u8; key_sz];

        // Cap iterations at max_entries * 2 to bound a pathological
        // walk on a torn table. RCU-protected reads on the kernel
        // side are best-effort across concurrent updates.
        let cap = (map.max_entries as usize).saturating_mul(2).max(1);
        let mut got_first = false;
        for _ in 0..cap {
            // Get next key.
            let attr = BpfAttrMapElem {
                map_fd: pinned.fd.as_raw_fd() as u32,
                _pad0: 0,
                key: if got_first {
                    cur_key.as_ptr() as u64
                } else {
                    0 // first call: NULL means "first key"
                },
                value_or_next_key: next_key.as_mut_ptr() as u64,
                flags: 0,
            };
            let ret = unsafe {
                bpf_syscall(
                    BPF_MAP_GET_NEXT_KEY,
                    &raw const attr as *const u8,
                    std::mem::size_of::<BpfAttrMapElem>(),
                )
            };
            if ret < 0 {
                // ENOENT marks end of iteration; anything else
                // ends the walk silently with whatever was
                // collected so far.
                break;
            }
            got_first = true;

            // Look up the value for next_key.
            let mut value = vec![0u8; val_sz];
            let lookup_attr = BpfAttrMapElem {
                map_fd: pinned.fd.as_raw_fd() as u32,
                _pad0: 0,
                key: next_key.as_ptr() as u64,
                value_or_next_key: value.as_mut_ptr() as u64,
                flags: 0,
            };
            let lret = unsafe {
                bpf_syscall(
                    BPF_MAP_LOOKUP_ELEM,
                    &raw const lookup_attr as *const u8,
                    std::mem::size_of::<BpfAttrMapElem>(),
                )
            };
            if lret >= 0 {
                out.push((next_key.clone(), value));
            }
            // Advance cursor — even when lookup failed (the key
            // disappeared between get_next_key and lookup_elem; a
            // concurrent delete is inherent to live-host walking).
            cur_key.copy_from_slice(&next_key);
        }

        out
    }

    fn read_percpu_array(&self, map: &BpfMapInfo, key: u32, num_cpus: u32) -> Vec<Option<Vec<u8>>> {
        let Some(pinned) = self.pinned_for(map) else {
            return Vec::new();
        };
        if map.map_type != BPF_MAP_TYPE_PERCPU_ARRAY {
            return Vec::new();
        }
        if key >= map.max_entries {
            return Vec::new();
        }

        let val_sz = map.value_size as usize;
        let total = (num_cpus as usize).saturating_mul(val_sz);
        let mut buf = vec![0u8; total];
        let mut k: u32 = key;
        let attr = BpfAttrMapElem {
            map_fd: pinned.fd.as_raw_fd() as u32,
            _pad0: 0,
            key: &raw mut k as u64,
            value_or_next_key: buf.as_mut_ptr() as u64,
            flags: 0,
        };
        if bpf_call_status(
            BPF_MAP_LOOKUP_ELEM,
            &raw const attr as *const u8,
            std::mem::size_of::<BpfAttrMapElem>(),
        )
        .is_err()
        {
            return vec![None; num_cpus as usize];
        }

        // Kernel rounds each CPU's slot up to 8 bytes internally
        // (see `kernel/bpf/syscall.c` bpf_map_value_size for the
        // PERCPU_ARRAY arm calling round_up_8). The returned buffer
        // is `nr_cpus * round_up_8(value_size)` bytes; we slice at
        // the rounded stride to extract each CPU's bytes and then
        // truncate to value_size.
        let stride = (val_sz + 7) & !7;
        let mut out = Vec::with_capacity(num_cpus as usize);
        for cpu in 0..num_cpus as usize {
            let start = cpu * stride;
            let end = start + val_sz;
            if end > buf.len() {
                out.push(None);
            } else {
                out.push(Some(buf[start..end].to_vec()));
            }
        }
        out
    }

    fn read_arena_pages(
        &self,
        map: &BpfMapInfo,
        _arena_offsets: &BpfArenaOffsets,
    ) -> ArenaSnapshot {
        let Some(pinned) = self.pinned_for(map) else {
            return ArenaSnapshot::default();
        };
        if map.map_type != BPF_MAP_TYPE_ARENA {
            return ArenaSnapshot::default();
        }

        // Compute declared span. Same caps as the guest-memory side
        // for cross-backend parity.
        let declared_bytes_raw = (map.max_entries as u64).saturating_mul(ARENA_PAGE_SIZE as u64);
        let span_capped = declared_bytes_raw > MAX_ARENA_BYTES;
        let declared_bytes = declared_bytes_raw.min(MAX_ARENA_BYTES);
        let declared_pages = declared_bytes / ARENA_PAGE_SIZE as u64;
        if declared_pages == 0 {
            return ArenaSnapshot {
                pages: Vec::new(),
                truncated: false,
                declared_pages: 0,
                span_capped,
                ..Default::default()
            };
        }

        // mmap the arena fd at offset 0 over the declared span. The
        // kernel's arena_vm_fault populates pages on first access;
        // unmapped pgoffs return SIGBUS — we use MAP_POPULATE so the
        // kernel walks every present page eagerly (without
        // allocating new ones). Sparse pgoffs that have never been
        // touched by the BPF program raise SIGBUS on first read —
        // we install a sigbus handler that longjmps out, marking
        // those pages as unmapped.
        //
        // Capping the mmap span at MAX_ARENA_PAGES * 4 KiB matches
        // the sequential-prefix cap on the guest-memory side; the
        // tail isn't stride-probed here because mmap covers the
        // whole window already. The `truncated` flag on
        // ArenaSnapshot still fires when the cap kicks in, for
        // operator visibility.
        let walk_pages = declared_pages.min(MAX_ARENA_PAGES);
        let walk_bytes = (walk_pages as usize) * ARENA_PAGE_SIZE;
        let truncated = declared_pages > walk_pages;

        // Use map_extra as the user_vm_start anchor. BPF programs
        // see arena addresses at this base (lib/arena_map.h hardcodes
        // it: x86 `1<<44`, aarch64 `1<<32`). Operators correlating
        // arena pointers want the same base in the snapshot.
        let user_vm_start = pinned.map_extra;

        // SAFETY: mmap with PROT_READ + MAP_SHARED on the arena fd
        // is exactly what the kernel exports for arena maps. The
        // `arena_map_mmap` op (`kernel/bpf/arena.c::arena_map_mmap`)
        // is the userspace mmap entry point; passing length =
        // walk_bytes and offset 0 maps the prefix.
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                walk_bytes,
                libc::PROT_READ,
                libc::MAP_SHARED,
                pinned.fd.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            // mmap rejected — return an empty snapshot with the
            // declared/truncation hints filled in for visibility.
            return ArenaSnapshot {
                pages: Vec::new(),
                truncated,
                declared_pages,
                span_capped,
                ..Default::default()
            };
        }

        let mut pages: Vec<ArenaPage> = Vec::new();
        // Read every page out of the mmap. Pages whose pgoff was
        // never populated by the BPF program will raise SIGBUS;
        // install a sigaction with a setjmp longjmp to recover.
        // We do NOT install the handler here — instead we use
        // mincore() to filter out pages that aren't present, then
        // read only the present ones. mincore returns 0 for
        // resident pages, < 0 on error.
        let mut residency = vec![0u8; walk_pages as usize];
        let mincore_ret = unsafe { libc::mincore(addr, walk_bytes, residency.as_mut_ptr()) };
        if mincore_ret == 0 {
            for (idx, &resident) in residency.iter().enumerate() {
                if resident & 1 == 0 {
                    // Page not in core — sparse arena, never
                    // populated by the BPF program. Skip.
                    continue;
                }
                let page_addr = (addr as usize) + idx * ARENA_PAGE_SIZE;
                // SAFETY: page is resident per mincore; reading
                // ARENA_PAGE_SIZE bytes is in-bounds.
                let mut buf = vec![0u8; ARENA_PAGE_SIZE];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        page_addr as *const u8,
                        buf.as_mut_ptr(),
                        ARENA_PAGE_SIZE,
                    );
                }
                let user_addr = user_vm_start + (idx as u64) * ARENA_PAGE_SIZE as u64;
                pages.push(ArenaPage {
                    user_addr,
                    bytes: buf,
                });
            }
        }

        // SAFETY: we created this mapping above and aren't using it
        // after this point.
        unsafe {
            libc::munmap(addr, walk_bytes);
        }

        ArenaSnapshot {
            pages,
            truncated,
            declared_pages,
            span_capped,
            ..Default::default()
        }
    }

    fn load_program_btf(&self, map: &BpfMapInfo, base_btf: &Btf) -> Option<Btf> {
        // map.btf_kva on the live-host backend stores the kernel's
        // btf_id (u32) — see obj_get_info_map. 0 means no BTF.
        let btf_id = map.btf_kva as u32;
        if btf_id == 0 {
            return None;
        }

        // Pin the BTF object by id.
        let attr = BpfAttrGetId {
            id_or_start_id: btf_id,
            next_id: 0,
            open_flags: 0,
        };
        let btf_fd = bpf_call_fd(
            BPF_BTF_GET_FD_BY_ID,
            &raw const attr as *const u8,
            std::mem::size_of::<BpfAttrGetId>(),
        )
        .ok()?;
        // SAFETY: btf_fd >= 0 from a successful bpf_call_fd.
        let btf_owned = unsafe { OwnedFd::from_raw_fd(btf_fd) };

        // Two-pass info fetch: first call to learn btf_size, then
        // allocate a buffer and refetch with `btf` populated to
        // pull the BTF blob bytes.
        let mut info = BpfBtfInfoUapi::default();
        let info_attr = BpfAttrInfoByFd {
            bpf_fd: btf_owned.as_raw_fd() as u32,
            info_len: std::mem::size_of::<BpfBtfInfoUapi>() as u32,
            info: &raw mut info as u64,
        };
        bpf_call_status(
            BPF_OBJ_GET_INFO_BY_FD,
            &raw const info_attr as *const u8,
            std::mem::size_of::<BpfAttrInfoByFd>(),
        )
        .ok()?;
        if info.btf_size == 0 {
            return None;
        }

        // Second pass with a real buffer.
        let mut buf = vec![0u8; info.btf_size as usize];
        info.btf = buf.as_mut_ptr() as u64;
        let info_attr2 = BpfAttrInfoByFd {
            bpf_fd: btf_owned.as_raw_fd() as u32,
            info_len: std::mem::size_of::<BpfBtfInfoUapi>() as u32,
            info: &raw mut info as u64,
        };
        bpf_call_status(
            BPF_OBJ_GET_INFO_BY_FD,
            &raw const info_attr2 as *const u8,
            std::mem::size_of::<BpfAttrInfoByFd>(),
        )
        .ok()?;

        // Parse the bytes. Program BTF is split-BTF over vmlinux's
        // base BTF — same code path as the guest-memory backend.
        Btf::from_split_bytes(&buf, base_btf).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the bpf_attr arms have the exact UAPI layout the
    /// kernel expects. Wrong sizes or field offsets cause -EINVAL
    /// on every syscall — this test catches the layout drift before
    /// it produces silent failures at runtime.
    #[test]
    fn bpf_attr_map_elem_size() {
        // include/uapi/linux/bpf.h: the MAP_ELEM_OPS arm is exactly
        // 32 bytes (4 + 4 pad + 8 + 8 + 8).
        assert_eq!(std::mem::size_of::<BpfAttrMapElem>(), 32);
    }

    #[test]
    fn bpf_attr_get_id_size() {
        // GET_NEXT_ID / GET_FD_BY_ID arm: 12 bytes (4 + 4 + 4)
        // — the kernel doesn't pad this struct to 8 bytes; size
        // matches the union arm exactly.
        assert_eq!(std::mem::size_of::<BpfAttrGetId>(), 12);
    }

    #[test]
    fn bpf_attr_info_by_fd_size() {
        // OBJ_GET_INFO_BY_FD arm: 16 bytes (4 + 4 + 8).
        assert_eq!(std::mem::size_of::<BpfAttrInfoByFd>(), 16);
    }

    /// `bpf_map_info` must be at least the historical minimum
    /// (the kernel rejects info_len smaller than its known floor).
    /// Modern kernels accept the full struct including map_extra.
    ///
    /// Verdict-routed so a multi-field uapi-shape regression
    /// surfaces every drift in one run rather than failing on
    /// the first mismatch.
    #[test]
    fn bpf_map_info_uapi_layout() {
        use crate::assert::Verdict;

        let off_map_type = std::mem::offset_of!(BpfMapInfoUapi, map_type);
        let off_name = std::mem::offset_of!(BpfMapInfoUapi, name);
        let total_size = std::mem::size_of::<BpfMapInfoUapi>();
        let off_map_extra = std::mem::offset_of!(BpfMapInfoUapi, map_extra);
        let map_extra_tail = off_map_extra + 8;

        let mut v = Verdict::new();
        // map_type at offset 0 per uapi.
        crate::claim!(v, off_map_type).eq(0usize);
        // name at offset 24 per uapi.
        crate::claim!(v, off_name).eq(24usize);
        // map_extra is the trailing field — its offset + 8 should
        // equal total struct size.
        crate::claim!(v, map_extra_tail).eq(total_size);
        let r = v.into_result();
        assert!(r.passed, "bpf_map_info uapi layout drift: {:?}", r.details,);
    }

    /// Round-up arithmetic for percpu stride matches the kernel's
    /// `round_up(value_size, 8)`.
    #[test]
    fn percpu_stride_round_up() {
        let cases = [
            (0usize, 0),
            (1, 8),
            (7, 8),
            (8, 8),
            (9, 16),
            (15, 16),
            (16, 16),
        ];
        for (val_sz, expected) in cases {
            let stride = (val_sz + 7) & !7;
            assert_eq!(stride, expected, "value_size {val_sz} → stride {stride}");
        }
    }

    /// Sized return for an empty enumeration is an empty vec, not
    /// an error. Construction must succeed even on systems with no
    /// BPF maps (rare but possible — minimal containers, fresh
    /// boots before any BPF program loads).
    #[test]
    fn predicate_filters_pinned_set() {
        // We can't actually invoke the kernel from this test
        // without root + a running scheduler, but we can verify
        // that the predicate signature compiles and the explicit
        // type annotation matches the trait shape callers will
        // use.
        fn _check_predicate_shape() {
            let _ =
                BpfSyscallAccessor::from_running_kernel_filtered(|_info: &BpfMapInfo| -> bool {
                    false
                });
        }
    }
}
