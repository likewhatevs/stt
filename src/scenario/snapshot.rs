//! Diagnostic snapshot capture and traversal.
//!
//! Test scenarios use [`Op::Snapshot`](crate::scenario::ops::Op::Snapshot)
//! to request a host-side diagnostic capture mid-run. The capture
//! result — a [`FailureDumpReport`] — is keyed by the `name` argument
//! and stored on the scenario's [`SnapshotBridge`], where downstream
//! test code reaches it via [`Snapshot`] for typed traversal of
//! BTF-rendered map values, per-CPU entries, and scalar variables.
//!
//! # Lifecycle
//!
//! 1. **Wire-up.** Before [`execute_steps`](crate::scenario::ops::execute_steps)
//!    runs, host orchestration installs a [`SnapshotBridge`] in the
//!    current thread via [`SnapshotBridge::set_thread_local`]. The
//!    bridge owns the storage map and a callable that performs the
//!    capture.
//!
//! 2. **Capture.** When the executor reaches `Op::Snapshot { name }`,
//!    it invokes [`SnapshotBridge::capture`] with the name. The
//!    closure performs the freeze rendezvous (request/reply with
//!    the freeze coordinator), builds a [`FailureDumpReport`], and
//!    returns it; the bridge stores it under the name.
//!
//! 3. **Inspection.** After the scenario completes, the test author
//!    pulls captured reports out via [`SnapshotBridge::drain`] and
//!    constructs [`Snapshot`] views to assert against rendered
//!    values:
//!    `snapshot.var("nr_cpus_onln").as_u64()? > 0`,
//!    `snapshot.map("scx_per_task")?.find(|e| e.get("tid").as_i64()? == pid)?`.
//!
//! # On-demand vs error-trigger captures
//!
//! `Op::Snapshot` requests are orthogonal to the error-class freeze
//! path. The freeze coordinator's existing state machine for
//! `SCX_EXIT_ERROR` triggers (Idle → TookEarly → Done) governs the
//! *unsolicited* capture pipeline; on-demand captures funnel
//! through a separate request/reply channel and never touch the
//! error-trigger state. The coordinator services on-demand requests
//! even after Done so post-failure scenarios can still snapshot
//! state for context. The serialisation rule: at most one capture in
//! flight at a time — the on-demand path waits for the previous
//! capture's vCPUs to fully return to `parked == false` before
//! issuing the next freeze request, mirroring the rendezvous
//! invariants the error-trigger path already obeys.
//!
//! # Guest → host wire: ioeventfd doorbell (locked)
//!
//! The guest-driven capture trigger uses an in-kernel ioeventfd
//! doorbell, NOT a synchronous MMIO `BusDevice` arm. Per user
//! direction:
//!
//! 1. Host registers an ioeventfd at a dedicated MMIO GPA inside
//!    the existing MMIO gap (e.g. `MMIO_GAP_START + 0x3000`) via
//!    `KVM_IOEVENTFD`. The fd is owned by the freeze coordinator
//!    and polled alongside its existing wake sources.
//! 2. Guest [`Op::Snapshot`](crate::scenario::ops::Op::Snapshot)
//!    handler `mmap`s `/dev/mem` to reach the doorbell GPA (same
//!    pattern the SHM ring already uses) and writes the tag value
//!    plus a serial counter into a small per-call slot, then
//!    writes the doorbell. KVM dispatches the write in-kernel and
//!    raises the eventfd; the vCPU thread does NOT exit to
//!    userspace for the doorbell write itself.
//! 3. The freeze coordinator wakes on `eventfd_signal`, reads the
//!    tag from the slot, runs `freeze_and_capture`, builds the
//!    [`FailureDumpReport`], and stores it on the bridge keyed by
//!    that tag. Reply to the guest is implicit — the
//!    [`SnapshotBridge::capture`] callback installed in the
//!    executor's thread-local blocks on a per-request reply
//!    eventfd / completion channel paired with the doorbell.
//!
//! This shape keeps the capture trigger off the vCPU userspace
//! exit path (cleaner — no MMIO `BusDevice` round-trip) and is
//! extensible to higher-rate triggers without redesigning the
//! wire. The [`SnapshotBridge`] surface defined below is the
//! integration point; `ioeventfd` is the wake mechanism that
//! drives the `CaptureCallback` from the guest side. The guest
//! [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
//! registration uses the same doorbell at scenario setup
//! (separate tag namespace) so symbol resolution + DR1-3 arm
//! happen on the host without a vCPU userspace exit.
//!
//! # No-bridge fallback
//!
//! When `Op::Snapshot` runs in a context with no installed bridge
//! (e.g. unit tests that exercise the executor without spinning up
//! a VM), the op is a no-op with a `tracing::warn!`. Existing
//! scenarios that do not declare snapshot ops keep working
//! unchanged.
//!
//! # Field accessor traversal
//!
//! [`SnapshotMap`], [`SnapshotEntry`], and [`SnapshotField`] form a
//! lazy borrow chain over the report. Dotted-path lookups (e.g.
//! `entry.get("ctx.weight.value")`) walk
//! [`RenderedValue::Struct`] members by name and follow
//! [`RenderedValue::Ptr`] dereferences transparently — the test
//! author writes the dotted path the BTF source would suggest;
//! pointer chasing is invisible.
//!
//! Missing fields land in [`SnapshotField::Missing`] with an
//! actionable error string identifying the path component that
//! could not be resolved AND the available alternatives at that
//! level. Terminal accessors (`as_u64`, `as_i64`, `as_bool`,
//! `as_str`) return `Result<T, SnapshotError>` so an absent /
//! type-mismatched field bubbles up as a recoverable error rather
//! than panicking.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::monitor::btf_render::{RenderedMember, RenderedValue};
use crate::monitor::dump::{
    FailureDumpEntry, FailureDumpMap, FailureDumpPercpuEntry, FailureDumpPercpuHashEntry,
    FailureDumpReport,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Reason a snapshot accessor or terminal read could not resolve.
///
/// Returned by every fallible accessor (`Snapshot::map`,
/// `SnapshotEntry::get`, `SnapshotField::as_u64`, …) so a missing
/// field, type mismatch, or absent map surfaces as a structured
/// error the test author can `?`-propagate. Each variant carries
/// the path / alternatives needed to fix the call site without
/// re-running the test.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SnapshotError {
    /// No map matched the requested name. `available` enumerates
    /// the captured map names so a typo surfaces in test output.
    MapNotFound {
        requested: String,
        available: Vec<String>,
    },
    /// No top-level global variable matched the requested name in
    /// any `*.bss` / `*.data` / `*.rodata` global-section map.
    /// `available` lists the union of every section's top-level
    /// member names.
    VarNotFound {
        requested: String,
        available: Vec<String>,
    },
    /// A path component did not match any
    /// [`RenderedValue::Struct`] member at that depth. `walked` is
    /// the prefix that resolved successfully; `component` is the
    /// failing segment; `available` lists the struct's actual
    /// member names.
    FieldNotFound {
        path: String,
        walked: String,
        component: String,
        available: Vec<String>,
    },
    /// A path component reached a non-Struct value where a struct
    /// was expected (e.g. descending into a `Uint` leaf). `kind`
    /// names the actual variant for diagnostics.
    NotAStruct {
        path: String,
        walked: String,
        component: String,
        kind: &'static str,
    },
    /// A typed accessor (`as_u64` etc.) was called on a rendered
    /// shape it cannot decode (e.g. `as_str` on a `Struct`).
    /// `requested` names the requested scalar type;
    /// `actual` names the rendered variant.
    TypeMismatch {
        requested: &'static str,
        actual: &'static str,
        path: String,
    },
    /// A map index was out of range for the underlying entry list.
    IndexOutOfRange {
        map: String,
        index: usize,
        len: usize,
    },
    /// A per-CPU slot was out of range or unmapped.
    PerCpuSlot {
        map: String,
        cpu: usize,
        len: usize,
        unmapped: bool,
    },
    /// A predicate-based lookup (`find`, `max_by`) found no match.
    NoMatch { map: String, op: &'static str },
    /// A path string contained an empty component (e.g. `"a..b"`).
    EmptyPathComponent { path: String },
    /// `EntryAccessor::get` was called on a per-CPU entry without
    /// narrowing to a CPU first via [`SnapshotMap::cpu`].
    PerCpuNotNarrowed { map: String },
    /// Hash entry has no rendered key/value side (BTF type id was
    /// missing at capture time, leaving the hex bytes only).
    NoRendered { map: String, side: &'static str },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::MapNotFound {
                requested,
                available,
            } => {
                write!(
                    f,
                    "snapshot has no map '{requested}' (captured maps: {available:?})"
                )
            }
            SnapshotError::VarNotFound {
                requested,
                available,
            } => {
                write!(
                    f,
                    "snapshot has no global variable '{requested}' in any \
                     *.bss/*.data/*.rodata map (available globals: {available:?})"
                )
            }
            SnapshotError::FieldNotFound {
                path,
                walked,
                component,
                available,
            } => {
                write!(
                    f,
                    "path '{path}': component '{component}' (after walking '{walked}') \
                     not found (members at this depth: {available:?})"
                )
            }
            SnapshotError::NotAStruct {
                path,
                walked,
                component,
                kind,
            } => {
                write!(
                    f,
                    "path '{path}': component '{component}' (after walking '{walked}') \
                     expected a Struct, got {kind}"
                )
            }
            SnapshotError::TypeMismatch {
                requested,
                actual,
                path,
            } => {
                write!(
                    f,
                    "path '{path}': cannot read as {requested} — actual rendered \
                     variant is {actual}"
                )
            }
            SnapshotError::IndexOutOfRange { map, index, len } => {
                write!(f, "map '{map}': index {index} out of range (length {len})")
            }
            SnapshotError::PerCpuSlot {
                map,
                cpu,
                len,
                unmapped,
            } => {
                if *unmapped {
                    write!(f, "map '{map}': cpu {cpu} per-CPU slot is unmapped (None)")
                } else {
                    write!(
                        f,
                        "map '{map}': cpu {cpu} out of range (have {len} per-CPU slots)"
                    )
                }
            }
            SnapshotError::NoMatch { map, op } => {
                write!(f, "map '{map}': {op} matched no entries")
            }
            SnapshotError::EmptyPathComponent { path } => {
                write!(f, "path '{path}' has an empty component (consecutive '.')")
            }
            SnapshotError::PerCpuNotNarrowed { map } => {
                write!(
                    f,
                    "map '{map}': per-CPU entry without a CPU narrow — call .cpu(N) first"
                )
            }
            SnapshotError::NoRendered { map, side } => {
                write!(
                    f,
                    "map '{map}': {side} has no rendered structure (no BTF type at capture time)"
                )
            }
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Result alias for snapshot accessors.
pub type SnapshotResult<T> = std::result::Result<T, SnapshotError>;

// ---------------------------------------------------------------------------
// Bridge: request/reply channel between executor and host capture
// ---------------------------------------------------------------------------

/// Closure type the bridge invokes to capture a snapshot.
///
/// Returns `None` when the capture pipeline could not produce a
/// report (rendezvous timed out, capture prerequisites missing, no
/// host-side wiring).
///
/// **Wire shape (locked: ioeventfd doorbell).** The production
/// implementation writes the tag into a small per-call slot inside
/// the SHM region, performs an `mmap`'d `u32` write to the
/// doorbell GPA inside the MMIO gap (KVM dispatches via
/// `KVM_IOEVENTFD` without a userspace exit), then blocks on a
/// per-request reply completion (an eventfd / mpsc receiver paired
/// with the doorbell registration). The freeze coordinator's
/// epoll loop wakes on the doorbell eventfd, reads the tag, runs
/// `freeze_and_capture`, and signals the reply completion with
/// the resulting `Option<FailureDumpReport>`.
///
/// On-demand captures are orthogonal to the error-trigger
/// `freeze_state` machine — the request handler in the coordinator
/// must not transition `freeze_state` from Idle, and must service
/// requests even when `freeze_state == Done`. The
/// rendezvous-serialisation invariant is the only constraint: each
/// request waits for `all parked == false` from the previous
/// capture before issuing.
pub type CaptureCallback = Arc<dyn Fn(&str) -> Option<FailureDumpReport> + Send + Sync + 'static>;

/// Closure type the bridge invokes to register a hardware-watchpoint
/// snapshot.
///
/// Implementations resolve the symbol path through the freeze
/// coordinator's BTF + kallsyms pipeline, allocate a free DR
/// register (DR1-3 on x86_64; DR0 is reserved for the existing
/// `*scx_root->exit_kind` trigger), arm the watchpoint via
/// `KVM_SET_GUEST_DEBUG`, and wire the corresponding
/// `KVM_EXIT_DEBUG` dispatch to a capture tagged with the
/// supplied symbol path.
///
/// **Guest → host wire.** Registration shares the on-demand
/// ioeventfd doorbell described on [`CaptureCallback`]: the guest
/// `Op::WatchSnapshot` arm writes the symbol string into the
/// shared slot and rings the doorbell at scenario-setup time. The
/// freeze coordinator does symbol resolution + DR allocation +
/// `KVM_SET_GUEST_DEBUG` on receipt, then signals the reply
/// completion with `Ok(())` or `Err(reason)`. Once armed, the
/// capture tagged with the symbol fires on every guest write
/// without any further userspace round-trip — `KVM_EXIT_DEBUG`
/// dispatches into the freeze coordinator directly, mirroring the
/// existing DR0 path the error-class trigger already uses.
///
/// Returns `Err(reason)` when:
///   - The symbol path does not resolve (BTF lookup miss,
///     kallsyms miss, per-CPU offset unavailable).
///   - The resolved KVA is not 4-byte aligned (DR_LEN_4
///     requirement per Intel SDM Vol. 3B Chapter 17).
///   - All three available DR registers (DR1-3) are already
///     allocated.
///   - `KVM_SET_GUEST_DEBUG` rejected the arm (host kernel
///     limitation).
pub type WatchRegisterCallback =
    Arc<dyn Fn(&str) -> std::result::Result<(), String> + Send + Sync + 'static>;

/// Shared state owning the capture closure plus the captured-report
/// map.
///
/// Cloneable via the wrapped `Arc`s. The host installs an instance
/// in the executor's thread-local via [`Self::set_thread_local`]
/// before [`execute_steps`](crate::scenario::ops::execute_steps)
/// runs; the executor's `Op::Snapshot` arm calls
/// [`Self::capture`] with the op's name.
/// Maximum number of [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
/// ops a single scenario may register. Tied to the available
/// hardware debug registers on x86_64: DR0 is reserved for the
/// existing `*scx_root->exit_kind` watchpoint that drives the
/// error-class freeze trigger; DR1, DR2, DR3 are the three slots
/// the on-demand watch path may use. Per Intel SDM Vol. 3B
/// Chapter 17.
pub const MAX_WATCH_SNAPSHOTS: usize = 3;

#[derive(Clone)]
#[must_use = "dropping a SnapshotBridge discards the capture pipeline"]
pub struct SnapshotBridge {
    capture: CaptureCallback,
    register_watch: Option<WatchRegisterCallback>,
    snapshots: Arc<Mutex<HashMap<String, FailureDumpReport>>>,
    watch_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl std::fmt::Debug for SnapshotBridge {
    /// Debug print does NOT show captured reports (their full
    /// rendering can be hundreds of KB) — only the count and the
    /// presence of callbacks.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotBridge")
            .field("snapshots", &self.len())
            .field("watch_count", &self.watch_count())
            .field("capture", &"<callback>")
            .field(
                "register_watch",
                &if self.register_watch.is_some() {
                    "<callback>"
                } else {
                    "<none>"
                },
            )
            .finish()
    }
}

impl SnapshotBridge {
    /// Build a bridge from a capture callback. The callback may
    /// freeze the VM, build the report, or return `None` when
    /// capture is unavailable. No watch-register callback —
    /// `Op::WatchSnapshot` returns "not supported" when the host
    /// did not wire one. Use [`Self::with_watch_register`] to
    /// install one.
    pub fn new(capture: CaptureCallback) -> Self {
        Self {
            capture,
            register_watch: None,
            snapshots: Arc::new(Mutex::new(HashMap::new())),
            watch_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Install a watch-register callback so [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
    /// ops can attach hardware-watchpoint snapshots. The callback
    /// is responsible for symbol resolution, DR allocation, and
    /// `KVM_SET_GUEST_DEBUG` arming.
    pub fn with_watch_register(mut self, register: WatchRegisterCallback) -> Self {
        self.register_watch = Some(register);
        self
    }

    /// Register a hardware-watchpoint snapshot for `symbol`.
    ///
    /// Enforces the per-scenario [`MAX_WATCH_SNAPSHOTS`] cap before
    /// invoking the host's watch-register callback. Returns
    /// `Err(reason)` when:
    /// - The cap has been reached (DR0 reserved + DR1-3 allocated).
    /// - No watch-register callback was installed via
    ///   [`Self::with_watch_register`].
    /// - The host's callback rejected the request (symbol unresolved,
    ///   alignment violation, ioctl failure).
    pub fn register_watch(&self, symbol: &str) -> std::result::Result<(), String> {
        let prev = self
            .watch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if prev >= MAX_WATCH_SNAPSHOTS {
            // Roll back to keep the count accurate.
            self.watch_count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return Err(format!(
                "Op::WatchSnapshot cap exceeded: scenario already registered \
                 {MAX_WATCH_SNAPSHOTS} watchpoints (DR1-3 occupied; DR0 reserved \
                 for the error-class exit_kind trigger). Drop a watch or use \
                 Op::Snapshot for a time-driven capture instead."
            ));
        }
        let Some(register) = self.register_watch.as_ref() else {
            self.watch_count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return Err(format!(
                "Op::WatchSnapshot('{symbol}'): no watch-register callback installed \
                 on this SnapshotBridge — the host wires one via \
                 SnapshotBridge::with_watch_register before execute_steps; \
                 in-guest / no-VM scenarios cannot register hardware watchpoints"
            ));
        };
        if let Err(reason) = register(symbol) {
            self.watch_count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return Err(reason);
        }
        Ok(())
    }

    /// Number of watchpoint snapshots currently registered.
    pub fn watch_count(&self) -> usize {
        self.watch_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Drive the capture closure and store the result under `name`.
    /// Returns `true` when a report was captured and stored;
    /// `false` when the closure returned `None`.
    pub fn capture(&self, name: &str) -> bool {
        let Some(report) = (self.capture)(name) else {
            tracing::warn!(
                name,
                "SnapshotBridge::capture: capture callback returned None — snapshot unavailable"
            );
            return false;
        };
        self.store(name, report);
        true
    }

    /// Store a pre-built [`FailureDumpReport`] under `name`,
    /// bypassing the capture callback. Used by the host-side freeze
    /// coordinator after it runs `freeze_and_capture(false)` and
    /// wants to publish the resulting report on the bridge for the
    /// test author to drain post-VM-exit.
    pub fn store(&self, name: &str, report: FailureDumpReport) {
        let mut map = self.snapshots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = map.insert(name.to_string(), report) {
            tracing::warn!(
                name,
                schema = %existing.schema,
                "SnapshotBridge::store: name already had a stored report; overwriting prior capture"
            );
        }
    }

    /// Snapshot count for diagnostic logging.
    pub fn len(&self) -> usize {
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// True when no snapshots have been captured.
    pub fn is_empty(&self) -> bool {
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Take ownership of the captured snapshots, leaving the bridge
    /// empty.
    pub fn drain(&self) -> HashMap<String, FailureDumpReport> {
        let mut map = self.snapshots.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *map)
    }

    /// Install this bridge as the active bridge for the calling
    /// thread. The bridge stays installed for the lifetime of the
    /// returned [`BridgeGuard`]; on drop the prior bridge (or
    /// `None`) is restored.
    ///
    /// Thread-local because [`execute_steps`](crate::scenario::ops::execute_steps)
    /// runs on the calling thread and `Op::Snapshot` only makes
    /// sense in that exact thread's call stack — installing a
    /// bridge process-wide would race against parallel test
    /// threads.
    pub fn set_thread_local(self) -> BridgeGuard {
        let prev = ACTIVE_BRIDGE.with(|c| c.borrow_mut().replace(self));
        BridgeGuard { prev }
    }
}

thread_local! {
    static ACTIVE_BRIDGE: std::cell::RefCell<Option<SnapshotBridge>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard returned by [`SnapshotBridge::set_thread_local`].
/// Restores the prior thread-local bridge on drop so a nested
/// scenario inside an outer one cannot leak its bridge into the
/// outer scope.
#[must_use = "BridgeGuard restores the prior bridge on drop; bind it"]
pub struct BridgeGuard {
    prev: Option<SnapshotBridge>,
}

impl Drop for BridgeGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ACTIVE_BRIDGE.with(|c| {
            *c.borrow_mut() = prev;
        });
    }
}

/// Run `f` with the active bridge if one is installed. When no
/// bridge is installed, returns `None` without invoking `f` — the
/// caller's responsibility to fall through to its own no-bridge
/// path.
pub fn with_active_bridge<R>(f: impl FnOnce(&SnapshotBridge) -> R) -> Option<R> {
    ACTIVE_BRIDGE.with(|c| c.borrow().as_ref().map(f))
}

// ---------------------------------------------------------------------------
// Snapshot view over a captured FailureDumpReport
// ---------------------------------------------------------------------------

/// Borrowed view over a captured [`FailureDumpReport`] for typed
/// traversal of BTF-rendered map values, per-CPU entries, and
/// scalar variables.
///
/// Constructed from a [`FailureDumpReport`] reference (typically
/// obtained via [`SnapshotBridge::drain`]); the view is cheap to
/// build — it does not copy the underlying report. Accessor
/// methods all return further borrowed views that walk the report
/// in place.
#[derive(Debug)]
#[must_use = "Snapshot is a borrowed view; bind or chain accessors"]
#[non_exhaustive]
pub struct Snapshot<'a> {
    report: &'a FailureDumpReport,
}

impl<'a> Snapshot<'a> {
    /// Build a borrowed view over `report`.
    pub fn new(report: &'a FailureDumpReport) -> Self {
        Self { report }
    }

    /// Underlying [`FailureDumpReport`] borrowed back to the caller.
    pub fn report(&self) -> &'a FailureDumpReport {
        self.report
    }

    /// Look up a BPF map by exact name. Returns
    /// [`SnapshotError::MapNotFound`] (with the captured map names
    /// in `available`) when no match is found.
    pub fn map(&self, name: &str) -> SnapshotResult<SnapshotMap<'a>> {
        for m in &self.report.maps {
            if m.name == name {
                return Ok(SnapshotMap { map: m, cpu: None });
            }
        }
        Err(SnapshotError::MapNotFound {
            requested: name.to_string(),
            available: self.report.maps.iter().map(|m| m.name.clone()).collect(),
        })
    }

    /// Walk the BTF-rendered fields of every `*.bss` / `*.data` /
    /// `*.rodata` global-section map for a top-level variable
    /// named `name`. Convenience for `.var("nr_cpus_onln")` style
    /// scalar reads without naming the section explicitly.
    ///
    /// Returns [`SnapshotField::Value`] on the first hit;
    /// [`SnapshotField::Missing`] with [`SnapshotError::VarNotFound`]
    /// (and the union of every global-section map's top-level
    /// member names in `available`) when no hit found.
    pub fn var(&self, name: &str) -> SnapshotField<'a> {
        for m in &self.report.maps {
            if !is_global_section_map(&m.name) {
                continue;
            }
            if let Some(v) = m.value.as_ref()
                && let Some(found) = lookup_member(v, name)
            {
                return SnapshotField::Value(found);
            }
        }
        let mut available: Vec<String> = Vec::new();
        for m in &self.report.maps {
            if !is_global_section_map(&m.name) {
                continue;
            }
            if let Some(RenderedValue::Struct { members, .. }) = m.value.as_ref() {
                for member in members {
                    available.push(member.name.clone());
                }
            }
        }
        available.sort();
        available.dedup();
        SnapshotField::Missing(SnapshotError::VarNotFound {
            requested: name.to_string(),
            available,
        })
    }

    /// Number of maps captured in the report.
    pub fn map_count(&self) -> usize {
        self.report.maps.len()
    }
}

/// True when a map name matches the libbpf-composed
/// `<obj>.<section>` naming for a global-section map.
fn is_global_section_map(name: &str) -> bool {
    name.ends_with(".bss") || name.ends_with(".data") || name.ends_with(".rodata")
}

// ---------------------------------------------------------------------------
// SnapshotMap
// ---------------------------------------------------------------------------

/// One map's view, possibly narrowed to a specific per-CPU slot via
/// [`Self::cpu`]. Returned by [`Snapshot::map`].
#[derive(Debug)]
#[must_use = "SnapshotMap is a borrowed view; chain accessors"]
#[non_exhaustive]
pub struct SnapshotMap<'a> {
    map: &'a FailureDumpMap,
    /// When `Some(cpu)`, subsequent [`Self::at`] /
    /// [`Self::find`] calls walk only the per-CPU slot for that
    /// CPU; `None` walks the natural (non-per-CPU) entry list.
    cpu: Option<usize>,
}

impl<'a> SnapshotMap<'a> {
    /// Map name as captured.
    pub fn name(&self) -> &'a str {
        &self.map.name
    }

    /// Underlying [`FailureDumpMap`].
    pub fn raw(&self) -> &'a FailureDumpMap {
        self.map
    }

    /// Narrow this map view to a specific per-CPU slot. On a
    /// non-per-CPU map this is recorded but ignored when the
    /// underlying entries are not per-CPU. Use on
    /// `BPF_MAP_TYPE_PERCPU_ARRAY` / `BPF_MAP_TYPE_PERCPU_HASH` /
    /// `BPF_MAP_TYPE_LRU_PERCPU_HASH`.
    pub fn cpu(self, n: usize) -> SnapshotMap<'a> {
        SnapshotMap {
            map: self.map,
            cpu: Some(n),
        }
    }

    /// Get an entry by ordinal index.
    ///
    /// For HASH-style entry lists, returns the `n`-th
    /// [`FailureDumpEntry`] in the captured order. For per-CPU
    /// array maps narrowed via [`Self::cpu`], returns the entry
    /// at key `n` with its per-CPU slot pre-resolved. For ARRAY
    /// maps with a single value, `n == 0` returns the value.
    pub fn at(&self, n: usize) -> SnapshotEntry<'a> {
        let resolved = self.entry_at(n);
        match resolved {
            Ok(e) => e,
            Err(err) => SnapshotEntry::Missing(err),
        }
    }

    /// Find the first entry matching `predicate`. Returns
    /// [`SnapshotEntry::Missing`] with [`SnapshotError::NoMatch`]
    /// when no entry matches.
    pub fn find(&self, predicate: impl Fn(&SnapshotEntry<'a>) -> bool) -> SnapshotEntry<'a> {
        for entry in self.iter_entries() {
            if predicate(&entry) {
                return entry;
            }
        }
        SnapshotEntry::Missing(SnapshotError::NoMatch {
            map: self.map.name.clone(),
            op: "find",
        })
    }

    /// Collect every entry matching `predicate` into a Vec.
    pub fn filter(&self, predicate: impl Fn(&SnapshotEntry<'a>) -> bool) -> Vec<SnapshotEntry<'a>> {
        self.iter_entries().filter(|e| predicate(e)).collect()
    }

    /// Find the entry whose `key_fn` produces the maximum u64.
    /// Returns [`SnapshotEntry::Missing`] when the map has no
    /// entries.
    pub fn max_by(&self, key_fn: impl Fn(&SnapshotEntry<'a>) -> u64) -> SnapshotEntry<'a> {
        let mut best: Option<(u64, SnapshotEntry<'a>)> = None;
        for entry in self.iter_entries() {
            let k = key_fn(&entry);
            let beats = best.as_ref().is_none_or(|(prev, _)| k > *prev);
            if beats {
                best = Some((k, entry));
            }
        }
        match best {
            Some((_, e)) => e,
            None => SnapshotEntry::Missing(SnapshotError::NoMatch {
                map: self.map.name.clone(),
                op: "max_by",
            }),
        }
    }

    /// Iterator over every entry under this view. Used by
    /// [`Self::find`] / [`Self::filter`] / [`Self::max_by`].
    fn iter_entries(&self) -> Box<dyn Iterator<Item = SnapshotEntry<'a>> + 'a> {
        if !self.map.percpu_entries.is_empty() {
            let cpu = self.cpu;
            let map = self.map;
            return Box::new(
                map.percpu_entries
                    .iter()
                    .map(move |e| resolve_percpu_entry(map, e, cpu)),
            );
        }
        if !self.map.percpu_hash_entries.is_empty() {
            let cpu = self.cpu;
            let map = self.map;
            return Box::new(
                map.percpu_hash_entries
                    .iter()
                    .map(move |e| resolve_percpu_hash_entry(map, e, cpu)),
            );
        }
        if !self.map.entries.is_empty() {
            return Box::new(self.map.entries.iter().map(SnapshotEntry::Hash));
        }
        if let Some(v) = self.map.value.as_ref() {
            return Box::new(std::iter::once(SnapshotEntry::Value(v)));
        }
        Box::new(std::iter::empty())
    }

    /// Internal entry-by-index resolver returning a structured
    /// error for the surrounding [`Self::at`] arm.
    fn entry_at(&self, n: usize) -> SnapshotResult<SnapshotEntry<'a>> {
        if !self.map.percpu_entries.is_empty() {
            return resolve_percpu_entry_at(self.map, n, self.cpu);
        }
        if !self.map.percpu_hash_entries.is_empty() {
            return resolve_percpu_hash_entry_at(self.map, n, self.cpu);
        }
        if !self.map.entries.is_empty() {
            if n < self.map.entries.len() {
                return Ok(SnapshotEntry::Hash(&self.map.entries[n]));
            }
            return Err(SnapshotError::IndexOutOfRange {
                map: self.map.name.clone(),
                index: n,
                len: self.map.entries.len(),
            });
        }
        if let Some(v) = self.map.value.as_ref() {
            if n == 0 {
                return Ok(SnapshotEntry::Value(v));
            }
            return Err(SnapshotError::IndexOutOfRange {
                map: self.map.name.clone(),
                index: n,
                len: 1,
            });
        }
        Err(SnapshotError::IndexOutOfRange {
            map: self.map.name.clone(),
            index: n,
            len: 0,
        })
    }
}

fn resolve_percpu_entry_at<'a>(
    map: &'a FailureDumpMap,
    n: usize,
    cpu: Option<usize>,
) -> SnapshotResult<SnapshotEntry<'a>> {
    if n >= map.percpu_entries.len() {
        return Err(SnapshotError::IndexOutOfRange {
            map: map.name.clone(),
            index: n,
            len: map.percpu_entries.len(),
        });
    }
    Ok(resolve_percpu_entry(map, &map.percpu_entries[n], cpu))
}

fn resolve_percpu_entry<'a>(
    map: &'a FailureDumpMap,
    entry: &'a FailureDumpPercpuEntry,
    cpu: Option<usize>,
) -> SnapshotEntry<'a> {
    let Some(c) = cpu else {
        return SnapshotEntry::Percpu(entry);
    };
    if c >= entry.per_cpu.len() {
        return SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: false,
        });
    }
    match entry.per_cpu[c].as_ref() {
        Some(v) => SnapshotEntry::Value(v),
        None => SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: true,
        }),
    }
}

fn resolve_percpu_hash_entry_at<'a>(
    map: &'a FailureDumpMap,
    n: usize,
    cpu: Option<usize>,
) -> SnapshotResult<SnapshotEntry<'a>> {
    if n >= map.percpu_hash_entries.len() {
        return Err(SnapshotError::IndexOutOfRange {
            map: map.name.clone(),
            index: n,
            len: map.percpu_hash_entries.len(),
        });
    }
    Ok(resolve_percpu_hash_entry(
        map,
        &map.percpu_hash_entries[n],
        cpu,
    ))
}

fn resolve_percpu_hash_entry<'a>(
    map: &'a FailureDumpMap,
    entry: &'a FailureDumpPercpuHashEntry,
    cpu: Option<usize>,
) -> SnapshotEntry<'a> {
    let Some(c) = cpu else {
        return SnapshotEntry::PercpuHash(entry);
    };
    if c >= entry.per_cpu.len() {
        return SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: false,
        });
    }
    match entry.per_cpu[c].as_ref() {
        Some(v) => SnapshotEntry::Value(v),
        None => SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: true,
        }),
    }
}

// ---------------------------------------------------------------------------
// SnapshotEntry
// ---------------------------------------------------------------------------

/// One entry's view — either a HASH (key, value) pair, a per-CPU
/// array entry, a per-CPU hash entry, a single rendered value, or
/// a missing-entry marker.
#[derive(Debug)]
#[must_use = "SnapshotEntry is a borrowed view; chain accessors"]
#[non_exhaustive]
pub enum SnapshotEntry<'a> {
    /// HASH map entry — `(key, value)` pair.
    Hash(&'a FailureDumpEntry),
    /// PERCPU_ARRAY entry — outer u32 key, inner per-CPU vec.
    Percpu(&'a FailureDumpPercpuEntry),
    /// PERCPU_HASH entry — rendered key, inner per-CPU vec.
    PercpuHash(&'a FailureDumpPercpuHashEntry),
    /// Single rendered value (ARRAY map's `value` field, or a
    /// per-CPU slot resolved via [`SnapshotMap::cpu`]).
    Value(&'a RenderedValue),
    /// No entry matched.
    Missing(SnapshotError),
}

impl<'a> SnapshotEntry<'a> {
    /// True when the lookup succeeded.
    pub fn is_present(&self) -> bool {
        !matches!(self, SnapshotEntry::Missing(_))
    }

    /// Walk into the entry's value side along a dotted path. Each
    /// path component names a [`RenderedValue::Struct`] member;
    /// pointer dereferences are followed transparently. Returns
    /// [`SnapshotField::Missing`] with an actionable error
    /// when the path cannot be resolved.
    pub fn get(&self, path: &str) -> SnapshotField<'a> {
        let value = match self {
            SnapshotEntry::Hash(e) => e.value.as_ref(),
            SnapshotEntry::Percpu(_) | SnapshotEntry::PercpuHash(_) => {
                let map_name = match self {
                    SnapshotEntry::Percpu(_) => "<percpu-array>".to_string(),
                    SnapshotEntry::PercpuHash(_) => "<percpu-hash>".to_string(),
                    _ => String::new(),
                };
                return SnapshotField::Missing(SnapshotError::PerCpuNotNarrowed { map: map_name });
            }
            SnapshotEntry::Value(v) => Some(*v),
            SnapshotEntry::Missing(err) => {
                return SnapshotField::Missing(err.clone());
            }
        };
        let Some(v) = value else {
            return SnapshotField::Missing(SnapshotError::NoRendered {
                map: "<entry>".to_string(),
                side: "value",
            });
        };
        walk_dotted_path(v, path)
    }

    /// Look up the entry's KEY side along a dotted path. Mirror
    /// of [`Self::get`] but operates on the key's rendered
    /// structure (HASH / PERCPU_HASH only).
    pub fn key(&self, path: &str) -> SnapshotField<'a> {
        match self {
            SnapshotEntry::Hash(e) => match e.key.as_ref() {
                Some(v) => walk_dotted_path(v, path),
                None => SnapshotField::Missing(SnapshotError::NoRendered {
                    map: "<entry>".to_string(),
                    side: "key",
                }),
            },
            SnapshotEntry::PercpuHash(e) => match e.key.as_ref() {
                Some(v) => walk_dotted_path(v, path),
                None => SnapshotField::Missing(SnapshotError::NoRendered {
                    map: "<entry>".to_string(),
                    side: "key",
                }),
            },
            SnapshotEntry::Percpu(e) => {
                if path.is_empty() {
                    SnapshotField::PercpuKey { key: e.key }
                } else {
                    SnapshotField::Missing(SnapshotError::TypeMismatch {
                        requested: "Struct",
                        actual: "Uint(percpu key)",
                        path: path.to_string(),
                    })
                }
            }
            SnapshotEntry::Value(_) => SnapshotField::Missing(SnapshotError::TypeMismatch {
                requested: "key",
                actual: "single Value (no key)",
                path: path.to_string(),
            }),
            SnapshotEntry::Missing(err) => SnapshotField::Missing(err.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotField — terminal traversal value
// ---------------------------------------------------------------------------

/// One field's view at the leaf of a dotted-path walk.
///
/// Returned by [`Snapshot::var`], [`SnapshotEntry::get`], and
/// [`SnapshotEntry::key`]. Terminal `as_*` accessors return
/// [`SnapshotResult`] so a missing or type-mismatched field
/// surfaces as a recoverable error rather than a panic.
#[derive(Debug)]
#[must_use = "SnapshotField is a borrowed view; call as_u64 / as_i64 / etc. to extract"]
#[non_exhaustive]
pub enum SnapshotField<'a> {
    /// Resolved rendered value at the leaf of the path walk.
    Value(&'a RenderedValue),
    /// Dedicated per-CPU array key shape (u32, no struct).
    PercpuKey { key: u32 },
    /// Path could not be resolved.
    Missing(SnapshotError),
}

impl<'a> SnapshotField<'a> {
    /// Walk into a sub-field. Composable with
    /// [`SnapshotEntry::get`].
    pub fn get(&self, path: &str) -> SnapshotField<'a> {
        match self {
            SnapshotField::Value(v) => walk_dotted_path(v, path),
            SnapshotField::PercpuKey { .. } => {
                SnapshotField::Missing(SnapshotError::TypeMismatch {
                    requested: "Struct",
                    actual: "Uint(percpu key)",
                    path: path.to_string(),
                })
            }
            SnapshotField::Missing(err) => SnapshotField::Missing(err.clone()),
        }
    }

    /// True when the field resolved successfully.
    pub fn is_present(&self) -> bool {
        !matches!(self, SnapshotField::Missing(_))
    }

    /// Read as `u64`. Accepts [`RenderedValue::Uint`],
    /// [`RenderedValue::Int`] (errors on negative),
    /// [`RenderedValue::Bool`] (0/1), [`RenderedValue::Char`]
    /// (raw byte), [`RenderedValue::Enum`] (raw enum integer),
    /// [`RenderedValue::Ptr`] (pointer value), and the
    /// percpu-array u32 key.
    pub fn as_u64(&self) -> SnapshotResult<u64> {
        match self {
            SnapshotField::Value(v) => render_to_u64(v),
            SnapshotField::PercpuKey { key } => Ok(u64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `i64`.
    pub fn as_i64(&self) -> SnapshotResult<i64> {
        match self {
            SnapshotField::Value(v) => render_to_i64(v),
            SnapshotField::PercpuKey { key } => Ok(i64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `bool`. [`RenderedValue::Bool`] direct, ints / enums
    /// non-zero is true.
    pub fn as_bool(&self) -> SnapshotResult<bool> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Bool { value } => Ok(*value),
                RenderedValue::Int { value, .. } => Ok(*value != 0),
                RenderedValue::Uint { value, .. } => Ok(*value != 0),
                RenderedValue::Char { value } => Ok(*value != 0),
                RenderedValue::Enum { value, .. } => Ok(*value != 0),
                RenderedValue::Ptr { value, .. } => Ok(*value != 0),
                other => Err(SnapshotError::TypeMismatch {
                    requested: "bool",
                    actual: describe_kind(other),
                    path: String::new(),
                }),
            },
            SnapshotField::PercpuKey { key } => Ok(*key != 0),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `f64`.
    pub fn as_f64(&self) -> SnapshotResult<f64> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Float { value, .. } => Ok(*value),
                RenderedValue::Int { value, .. } => Ok(*value as f64),
                RenderedValue::Uint { value, .. } => Ok(*value as f64),
                RenderedValue::Enum { value, .. } => Ok(*value as f64),
                other => Err(SnapshotError::TypeMismatch {
                    requested: "f64",
                    actual: describe_kind(other),
                    path: String::new(),
                }),
            },
            SnapshotField::PercpuKey { key } => Ok(f64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read the variant string for an [`RenderedValue::Enum`] with
    /// a resolved variant name.
    pub fn as_str(&self) -> SnapshotResult<&'a str> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Enum {
                    variant: Some(name),
                    ..
                } => Ok(name.as_str()),
                other => Err(SnapshotError::TypeMismatch {
                    requested: "str (enum variant name)",
                    actual: describe_kind(other),
                    path: String::new(),
                }),
            },
            SnapshotField::PercpuKey { .. } => Err(SnapshotError::TypeMismatch {
                requested: "str",
                actual: "Uint(percpu key)",
                path: String::new(),
            }),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Underlying rendered value if present.
    pub fn rendered(&self) -> Option<&'a RenderedValue> {
        match self {
            SnapshotField::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Error reference when the field is missing; `None`
    /// otherwise.
    pub fn error(&self) -> Option<&SnapshotError> {
        match self {
            SnapshotField::Missing(err) => Some(err),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Dotted-path walker
// ---------------------------------------------------------------------------

/// Walk a [`RenderedValue`] tree along a dotted path. Each
/// component matches a [`RenderedMember::name`] inside a
/// [`RenderedValue::Struct`]; [`RenderedValue::Ptr`] dereferences
/// are followed transparently. An empty path returns the root.
pub fn walk_dotted_path<'a>(root: &'a RenderedValue, path: &str) -> SnapshotField<'a> {
    if path.is_empty() {
        return SnapshotField::Value(root);
    }
    let mut cursor: &RenderedValue = root;
    let mut walked = String::new();
    for component in path.split('.') {
        if component.is_empty() {
            return SnapshotField::Missing(SnapshotError::EmptyPathComponent {
                path: path.to_string(),
            });
        }
        cursor = peel_pointer(cursor);
        let RenderedValue::Struct { members, .. } = cursor else {
            return SnapshotField::Missing(SnapshotError::NotAStruct {
                path: path.to_string(),
                walked: walked.clone(),
                component: component.to_string(),
                kind: describe_kind(cursor),
            });
        };
        let next = members.iter().find(|m| m.name == component);
        let Some(member) = next else {
            let names: Vec<String> = members.iter().map(|m| m.name.clone()).collect();
            return SnapshotField::Missing(SnapshotError::FieldNotFound {
                path: path.to_string(),
                walked: walked.clone(),
                component: component.to_string(),
                available: names,
            });
        };
        cursor = &member.value;
        if !walked.is_empty() {
            walked.push('.');
        }
        walked.push_str(component);
    }
    SnapshotField::Value(cursor)
}

/// Look up a single top-level member by exact name. Used by
/// [`Snapshot::var`].
fn lookup_member<'a>(value: &'a RenderedValue, name: &str) -> Option<&'a RenderedValue> {
    let v = peel_pointer(value);
    let RenderedValue::Struct { members, .. } = v else {
        return None;
    };
    members
        .iter()
        .find(|m: &&RenderedMember| m.name == name)
        .map(|m| &m.value)
}

/// Peel through any [`RenderedValue::Ptr`] layers whose `deref`
/// is `Some`. Stops at the first non-pointer (or a pointer
/// without a chased deref).
fn peel_pointer(mut v: &RenderedValue) -> &RenderedValue {
    let mut steps = 0;
    while let RenderedValue::Ptr {
        deref: Some(inner), ..
    } = v
    {
        v = inner.as_ref();
        steps += 1;
        if steps > 16 {
            break;
        }
    }
    v
}

/// Human-readable variant name used in error messages.
fn describe_kind(v: &RenderedValue) -> &'static str {
    match v {
        RenderedValue::Int { .. } => "Int",
        RenderedValue::Uint { .. } => "Uint",
        RenderedValue::Bool { .. } => "Bool",
        RenderedValue::Char { .. } => "Char",
        RenderedValue::Float { .. } => "Float",
        RenderedValue::Enum { .. } => "Enum",
        RenderedValue::Struct { .. } => "Struct",
        RenderedValue::Array { .. } => "Array",
        RenderedValue::CpuList { .. } => "CpuList",
        RenderedValue::Ptr { .. } => "Ptr",
        RenderedValue::Bytes { .. } => "Bytes",
        RenderedValue::Truncated { .. } => "Truncated",
        RenderedValue::Unsupported { .. } => "Unsupported",
    }
}

/// Shared u64 coercion used by [`SnapshotField::as_u64`].
fn render_to_u64(v: &RenderedValue) -> SnapshotResult<u64> {
    match v {
        RenderedValue::Uint { value, .. } => Ok(*value),
        RenderedValue::Int { value, .. } => {
            if *value < 0 {
                Err(SnapshotError::TypeMismatch {
                    requested: "u64",
                    actual: "Int(negative)",
                    path: String::new(),
                })
            } else {
                Ok(*value as u64)
            }
        }
        RenderedValue::Bool { value } => Ok(u64::from(*value)),
        RenderedValue::Char { value } => Ok(u64::from(*value)),
        RenderedValue::Enum { value, .. } => {
            if *value < 0 {
                Err(SnapshotError::TypeMismatch {
                    requested: "u64",
                    actual: "Enum(negative)",
                    path: String::new(),
                })
            } else {
                Ok(*value as u64)
            }
        }
        RenderedValue::Ptr { value, .. } => Ok(*value),
        other => Err(SnapshotError::TypeMismatch {
            requested: "u64",
            actual: describe_kind(other),
            path: String::new(),
        }),
    }
}

/// Shared i64 coercion used by [`SnapshotField::as_i64`].
fn render_to_i64(v: &RenderedValue) -> SnapshotResult<i64> {
    match v {
        RenderedValue::Int { value, .. } => Ok(*value),
        RenderedValue::Uint { value, .. } => {
            if *value > i64::MAX as u64 {
                Err(SnapshotError::TypeMismatch {
                    requested: "i64",
                    actual: "Uint(>i64::MAX)",
                    path: String::new(),
                })
            } else {
                Ok(*value as i64)
            }
        }
        RenderedValue::Bool { value } => Ok(i64::from(*value)),
        RenderedValue::Char { value } => Ok(i64::from(*value)),
        RenderedValue::Enum { value, .. } => Ok(*value),
        other => Err(SnapshotError::TypeMismatch {
            requested: "i64",
            actual: describe_kind(other),
            path: String::new(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::dump::SCHEMA_SINGLE;

    /// Build a synthetic [`FailureDumpReport`] used by every
    /// accessor unit test below.
    fn synthetic_report() -> FailureDumpReport {
        let bss_value = RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![
                RenderedMember {
                    name: "nr_cpus_onln".into(),
                    value: RenderedValue::Uint { bits: 32, value: 4 },
                },
                RenderedMember {
                    name: "stall".into(),
                    value: RenderedValue::Uint { bits: 8, value: 1 },
                },
                RenderedMember {
                    name: "balance_factor".into(),
                    value: RenderedValue::Float {
                        bits: 64,
                        value: 1.5,
                    },
                },
                RenderedMember {
                    name: "ctx".into(),
                    value: RenderedValue::Struct {
                        type_name: Some("scx_ctx".into()),
                        members: vec![
                            RenderedMember {
                                name: "weight".into(),
                                value: RenderedValue::Uint {
                                    bits: 32,
                                    value: 1024,
                                },
                            },
                            RenderedMember {
                                name: "policy".into(),
                                value: RenderedValue::Enum {
                                    bits: 32,
                                    value: 1,
                                    variant: Some("SCHED_NORMAL".into()),
                                },
                            },
                        ],
                    },
                },
                RenderedMember {
                    name: "leader".into(),
                    value: RenderedValue::Ptr {
                        value: 0xffff_8000_0000_1000,
                        deref: Some(Box::new(RenderedValue::Struct {
                            type_name: Some("task_struct".into()),
                            members: vec![RenderedMember {
                                name: "pid".into(),
                                value: RenderedValue::Int {
                                    bits: 32,
                                    value: 1234,
                                },
                            }],
                        })),
                        deref_skipped_reason: None,
                    },
                },
            ],
        };
        let bss_map = FailureDumpMap {
            name: "bpf.bss".into(),
            map_type: 2,
            value_size: 32,
            max_entries: 1,
            value: Some(bss_value),
            entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let hash_map = FailureDumpMap {
            name: "scx_per_task".into(),
            map_type: 1,
            value_size: 8,
            max_entries: 16,
            value: None,
            entries: vec![
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 100,
                    }),
                    key_hex: "64000000".into(),
                    value: Some(RenderedValue::Struct {
                        type_name: Some("task_ctx".into()),
                        members: vec![
                            RenderedMember {
                                name: "tid".into(),
                                value: RenderedValue::Int {
                                    bits: 32,
                                    value: 100,
                                },
                            },
                            RenderedMember {
                                name: "runtime_ns".into(),
                                value: RenderedValue::Uint {
                                    bits: 64,
                                    value: 5_000_000,
                                },
                            },
                        ],
                    }),
                    value_hex: "0064000000000000".into(),
                    payload: None,
                },
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 200,
                    }),
                    key_hex: "c8000000".into(),
                    value: Some(RenderedValue::Struct {
                        type_name: Some("task_ctx".into()),
                        members: vec![
                            RenderedMember {
                                name: "tid".into(),
                                value: RenderedValue::Int {
                                    bits: 32,
                                    value: 200,
                                },
                            },
                            RenderedMember {
                                name: "runtime_ns".into(),
                                value: RenderedValue::Uint {
                                    bits: 64,
                                    value: 9_000_000,
                                },
                            },
                        ],
                    }),
                    value_hex: "00c8000000000000".into(),
                    payload: None,
                },
            ],
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let percpu_map = FailureDumpMap {
            name: "scx_pcpu".into(),
            map_type: 6,
            value_size: 8,
            max_entries: 1,
            value: None,
            entries: Vec::new(),
            percpu_entries: vec![FailureDumpPercpuEntry {
                key: 0,
                per_cpu: vec![
                    Some(RenderedValue::Uint {
                        bits: 64,
                        value: 11,
                    }),
                    Some(RenderedValue::Uint {
                        bits: 64,
                        value: 22,
                    }),
                    None,
                    Some(RenderedValue::Uint {
                        bits: 64,
                        value: 44,
                    }),
                ],
            }],
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![bss_map, hash_map, percpu_map],
            ..Default::default()
        }
    }

    #[test]
    fn snapshot_var_walks_into_bss_struct() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        assert_eq!(snap.var("nr_cpus_onln").as_u64().unwrap(), 4);
        assert!(snap.var("stall").as_bool().unwrap());
        assert!((snap.var("balance_factor").as_f64().unwrap() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn snapshot_var_dotted_path_walks_nested_struct() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        assert_eq!(snap.var("ctx").get("weight").as_u64().unwrap(), 1024);
        assert_eq!(
            snap.var("ctx").get("policy").as_str().unwrap(),
            "SCHED_NORMAL"
        );
        assert_eq!(snap.var("ctx").get("policy").as_i64().unwrap(), 1);
    }

    #[test]
    fn dotted_path_follows_ptr_deref_transparently() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        assert_eq!(snap.var("leader").get("pid").as_i64().unwrap(), 1234);
    }

    #[test]
    fn missing_var_lists_available_globals() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("absent");
        let err = f.error().expect("missing field carries an error");
        match err {
            SnapshotError::VarNotFound {
                requested,
                available,
            } => {
                assert_eq!(requested, "absent");
                assert!(available.contains(&"nr_cpus_onln".to_string()));
                assert!(available.contains(&"ctx".to_string()));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        assert!(f.as_u64().is_err());
        assert!(f.as_i64().is_err());
        assert!(f.as_bool().is_err());
    }

    #[test]
    fn missing_field_in_struct_lists_available_members() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("nonexistent");
        let err = f.error().expect("missing field carries an error");
        match err {
            SnapshotError::FieldNotFound {
                component,
                available,
                ..
            } => {
                assert_eq!(component, "nonexistent");
                assert!(available.contains(&"weight".to_string()));
                assert!(available.contains(&"policy".to_string()));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn missing_map_lists_available_maps() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let err = snap.map("does_not_exist").unwrap_err();
        match err {
            SnapshotError::MapNotFound {
                requested,
                available,
            } => {
                assert_eq!(requested, "does_not_exist");
                assert!(available.contains(&"bpf.bss".to_string()));
                assert!(available.contains(&"scx_per_task".to_string()));
                assert!(available.contains(&"scx_pcpu".to_string()));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn empty_path_component_returns_error() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("weight..value");
        match f.error().expect("missing carries error") {
            SnapshotError::EmptyPathComponent { path } => {
                assert_eq!(path, "weight..value");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn wrong_kind_at_path_step_explains() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("weight").get("inner");
        match f.error().expect("missing carries error") {
            SnapshotError::NotAStruct { kind, .. } => {
                assert_eq!(*kind, "Uint");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn map_at_returns_hash_entry() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_per_task").unwrap().at(0);
        assert!(entry.is_present());
        assert_eq!(entry.get("tid").as_i64().unwrap(), 100);
        assert_eq!(entry.get("runtime_ns").as_u64().unwrap(), 5_000_000);
    }

    #[test]
    fn map_at_out_of_range_carries_index_and_len() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_per_task").unwrap().at(99);
        match entry {
            SnapshotEntry::Missing(SnapshotError::IndexOutOfRange { index, len, .. }) => {
                assert_eq!(index, 99);
                assert_eq!(len, 2);
            }
            other => panic!("unexpected entry: present={}", other.is_present()),
        }
    }

    #[test]
    fn map_find_returns_first_match() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let entry = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 200);
        assert!(entry.is_present());
        assert_eq!(entry.get("runtime_ns").as_u64().unwrap(), 9_000_000);
    }

    #[test]
    fn map_find_no_match_carries_op_name() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let entry = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 999);
        match entry {
            SnapshotEntry::Missing(SnapshotError::NoMatch { op, .. }) => {
                assert_eq!(op, "find");
            }
            other => panic!("expected NoMatch, got present={}", other.is_present()),
        }
    }

    #[test]
    fn map_filter_collects_matches() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let matches = map.filter(|e| e.get("runtime_ns").as_u64().unwrap_or(0) > 0);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn map_max_by_picks_largest() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let busiest = map.max_by(|e| e.get("runtime_ns").as_u64().unwrap_or(0));
        assert!(busiest.is_present());
        assert_eq!(busiest.get("tid").as_i64().unwrap(), 200);
    }

    #[test]
    fn percpu_array_cpu_narrow_reads_per_cpu_slot() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().cpu(1).at(0);
        assert!(entry.is_present());
        assert_eq!(entry.get("").as_u64().unwrap(), 22);
    }

    #[test]
    fn percpu_array_unmapped_cpu_returns_unmapped_error() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().cpu(2).at(0);
        match entry {
            SnapshotEntry::Missing(SnapshotError::PerCpuSlot { cpu, unmapped, .. }) => {
                assert_eq!(cpu, 2);
                assert!(unmapped);
            }
            _ => panic!("expected unmapped PerCpuSlot"),
        }
    }

    #[test]
    fn percpu_array_out_of_range_cpu_returns_oor_error() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().cpu(99).at(0);
        match entry {
            SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
                cpu, unmapped, len, ..
            }) => {
                assert_eq!(cpu, 99);
                assert!(!unmapped);
                assert_eq!(len, 4);
            }
            _ => panic!("expected out-of-range PerCpuSlot"),
        }
    }

    #[test]
    fn percpu_array_get_without_narrow_explains() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().at(0);
        let f = entry.get("anything");
        match f.error().expect("missing") {
            SnapshotError::PerCpuNotNarrowed { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn snapshot_bridge_capture_stores_under_name() {
        let report = synthetic_report();
        let cb: CaptureCallback = Arc::new(move |_name| Some(report.clone()));
        let bridge = SnapshotBridge::new(cb);
        assert!(bridge.is_empty());
        assert!(bridge.capture("test_name"));
        assert_eq!(bridge.len(), 1);
        let drained = bridge.drain();
        assert!(drained.contains_key("test_name"));
        assert_eq!(drained["test_name"].maps.len(), 3);
    }

    #[test]
    fn snapshot_bridge_capture_failure_returns_false() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        assert!(!bridge.capture("oops"));
        assert!(bridge.is_empty());
    }

    #[test]
    fn snapshot_bridge_register_watch_without_callback_errors() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        let err = bridge
            .register_watch("kernel.foo")
            .expect_err("no watch register installed");
        assert!(err.contains("no watch-register callback installed"));
        // Cap rollback: failed register must not consume a slot.
        assert_eq!(bridge.watch_count(), 0);
    }

    #[test]
    fn snapshot_bridge_register_watch_enforces_max_3() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let reg: WatchRegisterCallback = Arc::new(|_symbol| Ok(()));
        let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
        assert!(bridge.register_watch("kernel.a").is_ok());
        assert!(bridge.register_watch("kernel.b").is_ok());
        assert!(bridge.register_watch("kernel.c").is_ok());
        assert_eq!(bridge.watch_count(), MAX_WATCH_SNAPSHOTS);
        let err = bridge
            .register_watch("kernel.d")
            .expect_err("4th watch must be rejected");
        assert!(err.contains("cap exceeded"));
        // Cap rollback: rejection does not consume a slot.
        assert_eq!(bridge.watch_count(), MAX_WATCH_SNAPSHOTS);
    }

    #[test]
    fn snapshot_bridge_register_watch_propagates_callback_error() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let reg: WatchRegisterCallback =
            Arc::new(|symbol| Err(format!("symbol '{symbol}' did not resolve")));
        let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
        let err = bridge
            .register_watch("kernel.nonexistent")
            .expect_err("callback errored");
        assert!(err.contains("kernel.nonexistent"));
        // Failed register must not consume a slot.
        assert_eq!(bridge.watch_count(), 0);
    }

    #[test]
    fn snapshot_bridge_thread_local_install_and_restore() {
        assert!(with_active_bridge(|_| ()).is_none());
        let report = synthetic_report();
        let cb: CaptureCallback = Arc::new(move |_| Some(report.clone()));
        let bridge = SnapshotBridge::new(cb);
        let bridge_clone = bridge.clone();
        {
            let _g = bridge.set_thread_local();
            let captured = with_active_bridge(|b| b.capture("nested"));
            assert_eq!(captured, Some(true));
        }
        assert!(with_active_bridge(|_| ()).is_none());
        assert_eq!(bridge_clone.len(), 1);
    }

    #[test]
    fn snapshot_bridge_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        assert_send_sync(&bridge);
    }

    #[test]
    fn enum_variant_round_trips() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let policy = snap.var("ctx").get("policy");
        assert_eq!(policy.as_i64().unwrap(), 1);
        assert_eq!(policy.as_u64().unwrap(), 1);
        assert_eq!(policy.as_str().unwrap(), "SCHED_NORMAL");
    }

    #[test]
    fn rendered_passthrough_returns_raw_value() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("weight");
        let rendered = f.rendered().expect("weight is a Value");
        match rendered {
            RenderedValue::Uint { bits, value } => {
                assert_eq!(*bits, 32);
                assert_eq!(*value, 1024);
            }
            other => panic!("unexpected rendered shape: {other:?}"),
        }
    }

    #[test]
    fn snapshot_error_display_includes_path_and_alternatives() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let err = snap.var("ctx").get("nope").error().unwrap().to_string();
        assert!(err.contains("nope"));
        assert!(err.contains("weight"));
    }

    #[test]
    fn nested_field_get_composes() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        // Two-segment path equivalent to one chained get.
        let one = snap.var("ctx.weight");
        let chained = snap.var("ctx").get("weight");
        assert!(one.error().is_none());
        assert_eq!(chained.as_u64().unwrap(), 1024);
        // Snapshot::var does not split — `ctx.weight` is treated
        // as one global variable name by var() (and not present).
        assert!(one.error().is_some());
    }

    #[test]
    fn type_mismatch_carries_actual_kind() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        // weight is a Uint — try to read it as bool variant
        // string. Any Value that is not Enum-with-name lands in
        // TypeMismatch.
        let result = snap.var("ctx").get("weight").as_str();
        match result {
            Err(SnapshotError::TypeMismatch {
                requested, actual, ..
            }) => {
                assert_eq!(requested, "str (enum variant name)");
                assert_eq!(actual, "Uint");
            }
            _ => panic!("expected TypeMismatch"),
        }
    }
}
