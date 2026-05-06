//! Per-request-type handler implementations for virtio-block.
//!
//! Houses the four `handle_*_impl` free-associated functions on
//! `VirtioBlk` that service `T_IN` (read), `T_OUT` (write),
//! `T_FLUSH`, and `T_GET_ID`, plus the `cfg(test)` `&self` wrappers
//! the test harness calls. Split out of `device.rs` for module
//! locality: the handlers are pure per-request logic with no
//! MMIO/FSM/lifecycle interaction, and grouping them keeps the
//! pre-throttle dispatch table (in `drain.rs`) one hop away from
//! the implementation.
//!
//! Every handler returns `(status_byte, used_len)` and does NOT
//! itself write the status byte or call `add_used`; the caller
//! (`drain_bracket_impl`) gates `add_used` on a successful status
//! write via `publish_completion` to avoid the silent-data-corruption
//! gap from a stale blk-mq tag status (see the parent module's
//! "Why" doc).

use std::fs::File;
// `FileExt` provides `read_at`/`write_at`, used only by the
// `handle_read_impl` / `handle_write_impl` cfg(test) variants below;
// `handle_flush_impl` uses `File::sync_data` from std and does not
// need this trait in the lib build.
#[cfg(test)]
use std::os::unix::fs::FileExt;

// `GuestAddress` is consumed only by the `cfg(test)` `&self` wrapper
// signatures below; clippy --lib doesn't see those, so the import
// looks unused without the `cfg(test)` gate.
#[cfg(test)]
use vm_memory::GuestAddress;
use vm_memory::{Bytes, GuestMemoryMmap};

use virtio_bindings::virtio_blk::{VIRTIO_BLK_ID_BYTES, VIRTIO_BLK_S_IOERR, VIRTIO_BLK_S_OK};

use super::{ChainDescriptor, VIRTIO_BLK_SERIAL, VirtioBlk, VirtioBlkCounters};
// `VIRTIO_BLK_SECTOR_SIZE` is consumed only by the cfg(test)
// `handle_read_impl` / `handle_write_impl` per-segment variants; the
// production vectored handlers in `device.rs` use the constant
// directly from the `super` namespace.
#[cfg(test)]
use super::VIRTIO_BLK_SECTOR_SIZE;

impl VirtioBlk {
    /// Service `VIRTIO_BLK_T_IN` (read). Reads bytes from the
    /// backing file at `sector * 512` into the device-writable
    /// guest segments (scatter). Returns `(status_byte, used_len)`;
    /// the CALLER is responsible for writing `status_byte` to the
    /// status descriptor and calling `add_used` only when the
    /// status write succeeded — publishing a completion the guest
    /// can't observe is worse than dropping the chain.
    ///
    /// `checked_mul` is defense-in-depth against a sector value
    /// large enough to overflow `sector * 512` as u64. The
    /// downstream out-of-range check (`base_offset + total_data <=
    /// capacity_bytes`) would also catch most overflow cases on a
    /// reasonable capacity, but a checked multiply costs nothing
    /// and removes any worry about wrap-then-underflow corner
    /// cases when computing the post-multiply offset.
    ///
    /// Free function (not `&self`-method) so the caller can pass
    /// disjoint field borrows individually — `&self.backing`,
    /// `&self.counters`, and `self.capacity_sectors` (Copy) — and
    /// hold a concurrent `&mut self.queues[..]` borrow for
    /// `add_used`. A `&self`-method would have to borrow the whole
    /// receiver and conflict with the queue mutation in
    /// `process_requests`.
    ///
    /// `too_many_arguments` allow: deliberate disjoint-borrow
    /// shape — every parameter is a separate `&self` field that
    /// must be passed by reference so the caller can hold a
    /// concurrent mutable borrow of the queues vec.
    // TEST-ONLY: no production caller exists;
    // [`Self::handle_read_vectored_impl`] is canonical. Retained for
    // per-segment unit-test coverage — the cfg(test) `handle_read`
    // wrapper is the sole reach point.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn handle_read_impl(
        backing: &File,
        capacity_bytes: u64,
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        data_len: u64,
        scratch: &mut Vec<u8>,
    ) -> (u8, u32) {
        // Bytes the device wrote into the guest's data segments
        // (data + any zero-padded short-read tail). Drives the
        // virtio-spec used.elem.len in the caller's `add_used`.
        let mut bytes_to_guest: u32 = 0;
        // Bytes actually returned by `read_at` (i.e. bytes truly
        // read from the backing file). Drives the `bytes_read`
        // counter — the zero-pad tail on a short read is delivered
        // to the guest but not "read" from any source, so the
        // counter excludes it.
        let mut bytes_from_backing: u64 = 0;
        let Some(base_offset) = sector.checked_mul(VIRTIO_BLK_SECTOR_SIZE as u64) else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        // Read past EOF always returns S_IOERR. Capacity is fixed
        // at construction; auto-grow is not a v0 behaviour. A read
        // whose byte range extends past `capacity_bytes` fails
        // entirely — no partial-success short-read model — and
        // bumps `io_errors`. `capacity_bytes` is computed once in
        // `with_options` and threaded down — no per-request multiply.
        if base_offset
            .checked_add(data_len)
            .is_none_or(|end| end > capacity_bytes)
        {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }

        // Zero-length data segment: the empty-slice path is
        // intentional. The for-loop body runs unconditionally so
        // direction-mismatch checks (`!is_write_only`) still
        // apply; `read_at` against a zero-length slice is `Ok(0)`,
        // so `bytes_to_guest`/`cur_offset` are unchanged and the
        // chain proceeds to `S_OK` once all segments are walked.
        // A guest that submits a zero-length data descriptor has
        // issued a weird-but-legal request, not a malformed one —
        // qemu and firecracker behave the same way. This is an
        // explicit design choice, not an accidental fall-through.
        let mut cur_offset = base_offset;
        for seg in data_segments {
            if !seg.is_write_only {
                // Spec violation — a read request's data SGs must
                // be device-writable. Defense-in-depth: the outer
                // gate in process_requests already rejected this
                // chain before throttle. Kept in case a future
                // caller reaches handle_read_impl directly.
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, bytes_to_guest + 1);
            }
            // Reuse the device-owned scratch buffer. `resize(len, 0)`
            // zero-fills the buffer, then `read_at` overwrites bytes
            // [0..n] via pread64. The zero-fill is only paid on this
            // legacy path — production T_IN goes through
            // [`Self::handle_read_vectored_impl`], which writes
            // directly into guest memory via `preadv`. This handler
            // is retained for the cfg(test) `&self` wrapper and as a
            // fallback for any future caller that needs the
            // per-segment loop. A safe fill is preferable to an
            // uninit `set_len` window on a path where the
            // zero-fill cost is irrelevant.
            let len = seg.len as usize;
            scratch.resize(len, 0);
            match backing.read_at(&mut scratch[..], cur_offset) {
                Ok(n) => {
                    // Short reads leave bytes [n..len] at their
                    // initial zero from `resize` — sparse-file
                    // semantics fall out of the safe init.
                    if mem.write_slice(&scratch[..], seg.addr).is_err() {
                        counters.record_io_error();
                        return (VIRTIO_BLK_S_IOERR as u8, bytes_to_guest + 1);
                    }
                    // Counter: bytes ACTUALLY read from backing,
                    // excluding the zero-padded short-read tail
                    // (those bytes were delivered to the guest but
                    // were not sourced from any read).
                    bytes_from_backing += n as u64;
                    // used_len: bytes the device WROTE INTO the
                    // guest buffer = full seg.len (data + any
                    // zero-pad tail). virtio-v1.2 §2.7.7.2 defines
                    // used.elem.len as bytes written to the
                    // device-writable portion, so the zero-pad
                    // counts here even though it doesn't count for
                    // the bytes_read counter.
                    bytes_to_guest += seg.len;
                    cur_offset += seg.len as u64;
                }
                Err(e) => {
                    tracing::warn!(sector, %e, "virtio-blk read error");
                    counters.record_io_error();
                    return (VIRTIO_BLK_S_IOERR as u8, bytes_to_guest + 1);
                }
            }
        }
        counters.record_read(bytes_from_backing);
        // used_len: data bytes written to guest + 1 status byte.
        (VIRTIO_BLK_S_OK as u8, bytes_to_guest + 1)
    }

    /// Service `VIRTIO_BLK_T_OUT` (write). Reads bytes from the
    /// device-readable guest segments (gather) and writes them to
    /// the backing file at `sector * 512`. Returns
    /// `(status_byte, used_len)`; caller writes the status byte
    /// to the status descriptor and gates `add_used` on a
    /// successful status write. `checked_mul` matches
    /// `handle_read_impl` — same overflow concern.
    ///
    /// `too_many_arguments` allow: same disjoint-borrow shape as
    /// [`Self::handle_read_impl`].
    // TEST-ONLY: no production caller exists;
    // [`Self::handle_write_vectored_impl`] is canonical. Retained for
    // per-segment unit-test coverage — the cfg(test) `handle_write`
    // wrapper is the sole reach point.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn handle_write_impl(
        backing: &File,
        capacity_bytes: u64,
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        data_len: u64,
        scratch: &mut Vec<u8>,
    ) -> (u8, u32) {
        let Some(base_offset) = sector.checked_mul(VIRTIO_BLK_SECTOR_SIZE as u64) else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        // Write past EOF always returns S_IOERR. The disk is a
        // fixed-capacity virtio-blk device; auto-growing the
        // backing file would silently change the reported
        // config-space `capacity_sectors` and the guest partition
        // table would not see the new sectors without a
        // capacity-change notification path. Out-of-range writes
        // are a guest-side bug or a malicious request — fail
        // closed. `capacity_bytes` is computed once in
        // `with_options` and threaded down — no per-request multiply.
        if base_offset
            .checked_add(data_len)
            .is_none_or(|end| end > capacity_bytes)
        {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }

        let mut cur_offset = base_offset;
        let mut total_written: u32 = 0;
        for seg in data_segments {
            if seg.is_write_only {
                // Spec violation — a write request's data SGs must
                // be device-readable. Defense-in-depth: the outer
                // gate in process_requests already rejected this
                // chain before throttle. Kept in case a future
                // caller reaches handle_write_impl directly.
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, 1);
            }
            // Reuse the device-owned scratch buffer. `resize(len, 0)`
            // zero-fills the buffer, then `mem.read_slice` overwrites
            // every byte from guest memory. The zero-fill is only
            // paid on this legacy path — production T_OUT goes
            // through [`Self::handle_write_vectored_impl`], which
            // gathers directly from guest memory via `pwritev`. This
            // handler is retained for the cfg(test) `&self` wrapper
            // and as a fallback for any future caller that needs
            // the per-segment loop. A safe fill is preferable to an
            // uninit `set_len` window on a path where the
            // zero-fill cost is irrelevant.
            let len = seg.len as usize;
            scratch.resize(len, 0);
            if mem.read_slice(&mut scratch[..], seg.addr).is_err() {
                counters.record_io_error();
                return (VIRTIO_BLK_S_IOERR as u8, 1);
            }
            match backing.write_at(&scratch[..], cur_offset) {
                Ok(n) if (n as u32) == seg.len => {
                    total_written += seg.len;
                    cur_offset += seg.len as u64;
                }
                // Both partial write (`Ok(n)` with `n < seg.len`) and
                // hard error (`Err(_)`) collapse to S_IOERR + an
                // `io_errors` bump. From the guest's perspective the
                // request was not fulfilled in full, which is the same
                // failure signal — and counting partial writes as
                // io_errors keeps failure dumps honest. Note this
                // differs from the unsupported-type path, which sets
                // S_UNSUPP without bumping any counter (see
                // `classify_pre_throttle`). A future change that
                // wants to retry partial writes internally must not
                // silently suppress the `io_errors` increment when
                // the retry eventually fails — that signal is what
                // surfaces backing-store distress in failure dumps.
                Ok(_) | Err(_) => {
                    counters.record_io_error();
                    return (VIRTIO_BLK_S_IOERR as u8, 1);
                }
            }
        }
        counters.record_write(total_written as u64);
        // used_len: 1 (status byte only — write data is not written
        // back into guest mem).
        (VIRTIO_BLK_S_OK as u8, 1)
    }

    /// Service `VIRTIO_BLK_T_FLUSH`. `fdatasync(2)` on the backing.
    /// Returns `(status_byte, used_len)`; caller writes the status
    /// byte and gates `add_used` on a successful status write.
    pub(crate) fn handle_flush_impl(backing: &File, counters: &VirtioBlkCounters) -> (u8, u32) {
        let status = match backing.sync_data() {
            Ok(()) => {
                counters.record_flush();
                VIRTIO_BLK_S_OK as u8
            }
            Err(e) => {
                tracing::warn!(%e, "virtio-blk flush error");
                counters.record_io_error();
                VIRTIO_BLK_S_IOERR as u8
            }
        };
        (status, 1)
    }

    /// Service `VIRTIO_BLK_T_GET_ID` (virtio-v1.2 §5.2.6.4). Writes
    /// the device's 20-byte serial string into the FIRST data
    /// descriptor and returns `(status_byte, used_len)` where
    /// `used_len = VIRTIO_BLK_ID_BYTES + 1` on success (20 data
    /// bytes + 1 status byte). Caller publishes the status byte and
    /// gates `add_used` on a successful status write.
    ///
    /// The kernel driver `virtblk_get_id`
    /// (drivers/block/virtio_blk.c) maps a single 20-byte buffer
    /// via `blk_rq_map_kern(req, id_str, VIRTIO_BLK_ID_BYTES,
    /// GFP_KERNEL)`, so a well-formed chain has exactly one data
    /// descriptor of length >= 20. Multi-descriptor chains are
    /// theoretically legal under the spec but never produced by
    /// the kernel driver; we honor the kernel's contract by
    /// writing into the first descriptor only — matching
    /// firecracker's `process_get_device_id` and libkrun's
    /// `worker.rs` arm. If the first data descriptor is shorter
    /// than 20 bytes the request is rejected with `S_IOERR`
    /// (firecracker, cloud-hypervisor, libkrun all reject;
    /// QEMU truncates instead — we diverge intentionally because
    /// a guest that hands us a too-small buffer is already buggy
    /// and partial-data is a silent footgun).
    ///
    /// The data descriptor's direction has already been validated
    /// by the outer `direction_violation` gate in
    /// `process_requests` (T_GET_ID requires write-only); the
    /// per-segment direction check below is defense-in-depth for
    /// callers that bypass the gate.
    ///
    /// Free function (not `&self`-method) so the caller can pass
    /// disjoint field borrows individually — matching
    /// `handle_read_impl` / `handle_write_impl` for the same
    /// borrow-checker reason (`process_requests` holds
    /// `&mut self.queues[..]`).
    pub(crate) fn handle_get_id_impl(
        counters: &VirtioBlkCounters,
        mem: &GuestMemoryMmap,
        data_segments: &[ChainDescriptor],
    ) -> (u8, u32) {
        // First data descriptor receives the serial. The empty
        // case is filtered upstream by the zero-data gate, so
        // `first()` is always Some at production reach.
        // Defense-in-depth: still handle the empty slice by
        // returning S_IOERR rather than panicking on
        // `data_segments[0]` indexing.
        let Some(first) = data_segments.first() else {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        };
        if !first.is_write_only {
            // Spec violation — GET_ID's data SG must be
            // device-writable. Defense-in-depth: the outer gate in
            // process_requests already rejected this chain before
            // throttle. Kept in case a future caller reaches
            // handle_get_id_impl directly.
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }
        if first.len < VIRTIO_BLK_ID_BYTES {
            // Buffer too small — kernel driver always passes
            // exactly VIRTIO_BLK_ID_BYTES (20). Reject rather than
            // truncate: matches firecracker / cloud-hypervisor /
            // libkrun. A truncated serial would surface as a
            // garbled `/sys/block/<dev>/serial` value, which is
            // worse than an explicit IOERR (the guest's
            // `serial_show` maps -EIO to an empty string).
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }
        if mem.write_slice(&VIRTIO_BLK_SERIAL[..], first.addr).is_err() {
            counters.record_io_error();
            return (VIRTIO_BLK_S_IOERR as u8, 1);
        }
        // used_len: 20 data bytes written + 1 status byte. Symmetric
        // with handle_read_impl's `total_read + 1` accounting.
        (VIRTIO_BLK_S_OK as u8, VIRTIO_BLK_ID_BYTES + 1)
    }

    /// Test-only `&self` proxies for the request handlers. The
    /// production `process_requests` invokes the free-function
    /// associated forms (`Self::handle_*_impl`) so that the
    /// `&mut self.queues[..]` borrow in the request loop doesn't
    /// conflict with `&self`. Tests prefer the method form for
    /// brevity.
    ///
    /// Wrappers also write the status byte themselves before
    /// returning — the production caller (`process_requests`) does
    /// this as part of its publish-completion step, so test
    /// helpers replicate it for convenience.
    #[cfg(test)]
    pub(crate) fn handle_read(
        &self,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();
        let mut scratch = Vec::new();
        let s = self.worker.state();
        let (status, used_len) = Self::handle_read_impl(
            &s.backing,
            s.capacity_bytes,
            s.counters.as_ref(),
            mem,
            sector,
            data_segments,
            data_len,
            &mut scratch,
        );
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    #[cfg(test)]
    pub(crate) fn handle_write(
        &self,
        mem: &GuestMemoryMmap,
        sector: u64,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let data_len: u64 = data_segments.iter().map(|d| d.len as u64).sum();
        let mut scratch = Vec::new();
        let s = self.worker.state();
        let (status, used_len) = Self::handle_write_impl(
            &s.backing,
            s.capacity_bytes,
            s.counters.as_ref(),
            mem,
            sector,
            data_segments,
            data_len,
            &mut scratch,
        );
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    #[cfg(test)]
    pub(crate) fn handle_flush(
        &self,
        mem: &GuestMemoryMmap,
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let s = self.worker.state();
        let (status, used_len) = Self::handle_flush_impl(&s.backing, s.counters.as_ref());
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }

    #[cfg(test)]
    pub(crate) fn handle_get_id(
        &self,
        mem: &GuestMemoryMmap,
        data_segments: &[ChainDescriptor],
        status_addr: GuestAddress,
    ) -> (u8, u32) {
        let s = self.worker.state();
        let (status, used_len) = Self::handle_get_id_impl(s.counters.as_ref(), mem, data_segments);
        mem.write_slice(&[status], status_addr)
            .expect("write status in test wrapper");
        (status, used_len)
    }
}
