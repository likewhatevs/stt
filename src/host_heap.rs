//! Heap-state snapshot for the running ktstr binary.
//!
//! [`HostHeapState`] is a thin snapshot of the process's jemalloc
//! allocator state at capture time: active / allocated / resident /
//! mapped bytes plus the arena count. It rides along inside
//! [`HostContext::heap_state`](crate::host_context::HostContext::heap_state)
//! so a sidecar reader can correlate scheduler-test outcomes with
//! the ktstr tool's own memory footprint — e.g. distinguish a
//! legitimate regression from one where the runner itself OOM-pressured
//! the host.
//!
//! # jemalloc is always linked
//!
//! `tikv-jemalloc-ctl` declares a non-optional dependency on
//! `tikv-jemalloc-sys`, which builds and links libjemalloc
//! unconditionally. So even consumers that do NOT install
//! `tikv_jemallocator::Jemalloc` as `#[global_allocator]` carry
//! libjemalloc in their binary, and every `mallctl` call from this
//! module resolves to libjemalloc's implementation rather than a
//! libc stub. `mallctl` reads succeed regardless of which allocator
//! `#[global_allocator]` resolves to.
//!
//! What differs is the *meaning* of the numbers:
//!
//! - When jemalloc IS `#[global_allocator]` (every binary in this
//!   workspace — see `src/bin/*.rs`), every heap allocation flows
//!   through jemalloc and `stats.allocated` / `stats.active` report
//!   real application usage in the tens-to-hundreds of MiB range.
//! - When jemalloc is linked but is NOT `#[global_allocator]`
//!   (downstream consumers using ktstr as a library without opting
//!   into jemallocator), jemalloc still initializes its arenas but
//!   the application never allocates through it. `stats.allocated`
//!   and `stats.active` return `Some(0)` in that case.
//!   `arenas.narenas` is still populated (jemalloc computes it as
//!   `4 * ncpus` at init time) and `stats.resident` / `stats.mapped`
//!   reflect jemalloc's own metadata footprint — small but non-zero.
//!
//! [`collect`] collapses the "jemalloc linked but unused" shape
//! (`allocated_bytes == Some(0) && active_bytes == Some(0)`) to
//! `None` at the [`HostContext::heap_state`](crate::host_context::HostContext::heap_state)
//! call site, so sidecars from non-jemallocator consumers do not
//! carry misleading mostly-zero rows. The `jemalloc-used` signal
//! (non-zero allocated AND active) is what warrants sidecar space.
//!
//! # `stats` feature is required for stats reads
//!
//! libjemalloc only tracks `stats.*` counters when the C library is
//! built with `--enable-stats`; without it the mallctl reads still
//! succeed but return zero. The `stats` feature on both
//! `tikv-jemalloc-ctl` and `tikv-jemallocator` in `Cargo.toml` forces
//! the C build flag — `host_heap::collect` depends on this.
//! `arenas.narenas` is independent of `--enable-stats`; it reports
//! correctly either way.
//!
//! # Epoch discipline
//!
//! `stats.*` reads return cached values; the cache refreshes when
//! the `epoch` mallctl is advanced. [`collect`] advances the epoch
//! exactly once before issuing reads so each snapshot reflects
//! post-advance state. Callers that invoke [`collect`] back-to-back
//! see fresh reads every time because each call advances the epoch
//! again.
//!
//! Side effect of `epoch::advance()`: libjemalloc flushes per-thread
//! stat caches into the shared counters under a mallctl-internal
//! lock. The operation is thread-safe per jemalloc's mallctl
//! contract — concurrent `collect` calls from multiple ktstr
//! threads are defined-behavior (though pointless; each caller sees
//! its own refreshed snapshot and the last writer wins in the
//! caches).

/// Heap-state snapshot for the running process's jemalloc allocator.
///
/// Every field is `Option<u64>` (or `Option<usize>` for the arena
/// count) so a partial read lands what succeeded and consumers can
/// distinguish "jemalloc reported X" from "jemalloc did not report
/// this field". The `Default` impl lands every field as `None`,
/// matching the non-jemalloc fallback path and serving as the
/// fixture for test call sites that want the empty shape.
///
/// # Constructing instances in tests
///
/// Like [`HostContext`](crate::host_context::HostContext), this
/// struct is `#[non_exhaustive]`: cross-crate consumers cannot build
/// one with any struct-expression form. Start from
/// [`HostHeapState::test_fixture`] (populated baseline) or
/// [`HostHeapState::default`] (all-`None`) and mutate fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct HostHeapState {
    /// `stats.active` — bytes in active pages allocated by the
    /// application. A multiple of the page size and `>=`
    /// [`Self::allocated_bytes`]. Populated whenever libjemalloc
    /// was built with `--enable-stats` (the `stats` feature on
    /// `tikv-jemalloc-ctl` forces this). `Some(0)` when jemalloc
    /// is linked but is not `#[global_allocator]` — the whole
    /// [`HostHeapState`] collapses to `None` at the HostContext
    /// call site in that case (see module doc).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_bytes: Option<u64>,
    /// `stats.allocated` — total bytes allocated by the
    /// application (sum of live allocations, excluding allocator
    /// metadata and padding). `Some(0)` when jemalloc is linked
    /// but not installed as `#[global_allocator]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocated_bytes: Option<u64>,
    /// `stats.resident` — bytes in physically resident data pages
    /// mapped by the allocator. Overestimates by including
    /// demand-zeroed pages that have not been touched; jemalloc
    /// documents this. A multiple of the page size and `>=`
    /// [`Self::active_bytes`]. Reflects jemalloc's own metadata
    /// footprint even when jemalloc is not `#[global_allocator]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident_bytes: Option<u64>,
    /// `stats.mapped` — bytes in active extents mapped by the
    /// allocator. Excludes inactive extents even those with
    /// unused dirty pages, so there is no strict ordering between
    /// this and [`Self::resident_bytes`]. A multiple of the page
    /// size and `>=` [`Self::active_bytes`]. Reflects jemalloc's
    /// own metadata footprint even when jemalloc is not
    /// `#[global_allocator]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapped_bytes: Option<u64>,
    /// `arenas.narenas` — current limit on the number of arenas.
    /// Initialized at jemalloc startup (typically `4 * ncpus` on a
    /// multi-core Linux host) and updated as the allocator grows
    /// new arenas. Populated whenever libjemalloc is linked into
    /// the binary, including on consumers that use ktstr as a
    /// library without opting into jemallocator as
    /// `#[global_allocator]` (see the module doc). `None` only on
    /// the rare mallctl-error path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub narenas: Option<usize>,
}

impl HostHeapState {
    /// Populated fixture for unit tests. Values are plausible for a
    /// small ktstr run on a 16-CPU host: a few hundred KiB
    /// allocated, rounded up to page-size multiples for active /
    /// resident / mapped, and `narenas = 64` (jemalloc's
    /// `4 * ncpus` default on a 16-CPU box).
    ///
    /// Call sites mutate the fields they care about:
    ///
    /// ```
    /// use ktstr::prelude::HostHeapState;
    /// let mut h = HostHeapState::test_fixture();
    /// h.allocated_bytes = Some(0);
    /// ```
    pub fn test_fixture() -> HostHeapState {
        HostHeapState {
            active_bytes: Some(1 << 20),
            allocated_bytes: Some(512 * 1024),
            resident_bytes: Some(2 << 20),
            mapped_bytes: Some(4 << 20),
            narenas: Some(64),
        }
    }

    /// Render as a human-readable multi-line block. Each field is
    /// one `key: value` line; absent fields render `(unknown)` so
    /// operators see which reads failed. The block ends with a
    /// newline. Matches [`HostContext::format_human`](crate::host_context::HostContext::format_human)'s
    /// shape — pair the two in `cargo ktstr show-host` for a
    /// single-block host summary.
    pub fn format_human(&self) -> String {
        use std::fmt::Write;
        // Destructuring bind forces every field of HostHeapState to
        // appear by name here; adding a field will break the build
        // until it's rendered.
        let HostHeapState {
            active_bytes,
            allocated_bytes,
            resident_bytes,
            mapped_bytes,
            narenas,
        } = self;
        fn row<T: std::fmt::Display>(out: &mut String, key: &str, value: Option<&T>) {
            match value {
                Some(v) => {
                    let _ = writeln!(out, "{key}: {v}");
                }
                None => {
                    let _ = writeln!(out, "{key}: (unknown)");
                }
            }
        }
        let mut out = String::new();
        row(&mut out, "allocated_bytes", allocated_bytes.as_ref());
        row(&mut out, "active_bytes", active_bytes.as_ref());
        row(&mut out, "resident_bytes", resident_bytes.as_ref());
        row(&mut out, "mapped_bytes", mapped_bytes.as_ref());
        row(&mut out, "narenas", narenas.as_ref());
        out
    }

    /// Render a field-by-field diff as `key: before → after` lines.
    /// Omits unchanged fields; an empty return means the two
    /// snapshots are identical. `None` renders as `(unknown)` so a
    /// `None → Some(..)` transition is visible.
    pub fn diff(&self, other: &HostHeapState) -> String {
        use std::fmt::Write;
        let HostHeapState {
            active_bytes: a_active,
            allocated_bytes: a_allocated,
            resident_bytes: a_resident,
            mapped_bytes: a_mapped,
            narenas: a_narenas,
        } = self;
        let HostHeapState {
            active_bytes: b_active,
            allocated_bytes: b_allocated,
            resident_bytes: b_resident,
            mapped_bytes: b_mapped,
            narenas: b_narenas,
        } = other;
        let mut out = String::new();
        fn row_opt<T: std::fmt::Display + PartialEq>(
            out: &mut String,
            key: &str,
            a: Option<&T>,
            b: Option<&T>,
        ) {
            if a == b {
                return;
            }
            let render = |v: Option<&T>| match v {
                Some(x) => format!("{x}"),
                None => "(unknown)".to_string(),
            };
            let _ = writeln!(out, "{key}: {} → {}", render(a), render(b));
        }
        row_opt(&mut out, "allocated_bytes", a_allocated.as_ref(), b_allocated.as_ref());
        row_opt(&mut out, "active_bytes", a_active.as_ref(), b_active.as_ref());
        row_opt(&mut out, "resident_bytes", a_resident.as_ref(), b_resident.as_ref());
        row_opt(&mut out, "mapped_bytes", a_mapped.as_ref(), b_mapped.as_ref());
        row_opt(&mut out, "narenas", a_narenas.as_ref(), b_narenas.as_ref());
        out
    }
}

/// Capture the running process's jemalloc heap state.
///
/// Advances the jemalloc `epoch` exactly once so cached `stats.*`
/// values refresh (this is a jemalloc-internal operation —
/// libjemalloc flushes per-thread stat caches into the shared
/// counters under its mallctl lock, and the operation is thread-safe
/// per jemalloc's mallctl contract), then reads five mallctl values.
/// Any individual read error lands that field as `None`; an
/// `epoch::advance()` error short-circuits the whole function to
/// [`HostHeapState::default`] because without a refreshed epoch the
/// stats reads would return values from an arbitrary prior snapshot.
///
/// Since libjemalloc is linked unconditionally via
/// `tikv-jemalloc-sys` (see module doc), `epoch::advance()` and the
/// subsequent mallctl reads always succeed on a well-formed build.
/// The `is_err()` branch below is a defensive guard against future
/// jemalloc versions changing the error surface, not an expected
/// fallback.
///
/// When jemalloc is linked but is not `#[global_allocator]`, the
/// reads succeed and return small-or-zero values —
/// [`HostContext::heap_state`](crate::host_context::HostContext)
/// detects that shape and stores `None` so the sidecar does not
/// carry an empty row. When jemalloc IS `#[global_allocator]`
/// (every binary target in this workspace), every field reflects
/// real runner memory usage.
///
/// # Cost
///
/// One `mallctl("epoch", ...)` call plus five
/// `mallctl("stats.*"/"arenas.narenas", ...)` reads. Each is a
/// `memcpy` from a cached value after a short tree walk inside
/// jemalloc — microseconds total. Safe to call on every sidecar
/// write.
pub fn collect() -> HostHeapState {
    // epoch advance refreshes jemalloc's stat cache. libjemalloc is
    // always linked (tikv-jemalloc-sys is a hard dep of
    // tikv-jemalloc-ctl), so this only fails on an unexpected
    // jemalloc-internal error path. Defensive fall-through to the
    // all-None default keeps `collect` infallible.
    if tikv_jemalloc_ctl::epoch::advance().is_err() {
        return HostHeapState::default();
    }
    // Each read is independent — a single error on `stats.allocated`
    // does not poison `arenas.narenas`. `arenas.narenas` is
    // initialized at jemalloc startup (typically `4 * ncpus`) and
    // always readable on a libjemalloc-linked build regardless of
    // `--enable-stats`. `Option::from(r.ok())` lands Err as None.
    let active_bytes = tikv_jemalloc_ctl::stats::active::read()
        .ok()
        .map(|v| v as u64);
    let allocated_bytes = tikv_jemalloc_ctl::stats::allocated::read()
        .ok()
        .map(|v| v as u64);
    let resident_bytes = tikv_jemalloc_ctl::stats::resident::read()
        .ok()
        .map(|v| v as u64);
    let mapped_bytes = tikv_jemalloc_ctl::stats::mapped::read()
        .ok()
        .map(|v| v as u64);
    let narenas = tikv_jemalloc_ctl::arenas::narenas::read()
        .ok()
        .map(|v| v as usize);
    HostHeapState {
        active_bytes,
        allocated_bytes,
        resident_bytes,
        mapped_bytes,
        narenas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lands_all_none() {
        let h = HostHeapState::default();
        assert!(h.active_bytes.is_none());
        assert!(h.allocated_bytes.is_none());
        assert!(h.resident_bytes.is_none());
        assert!(h.mapped_bytes.is_none());
        assert!(h.narenas.is_none());
    }

    #[test]
    fn test_fixture_populates_every_field() {
        let h = HostHeapState::test_fixture();
        assert!(h.active_bytes.is_some());
        assert!(h.allocated_bytes.is_some());
        assert!(h.resident_bytes.is_some());
        assert!(h.mapped_bytes.is_some());
        assert!(h.narenas.is_some());
    }

    #[test]
    fn format_human_lists_every_field() {
        let out = HostHeapState::test_fixture().format_human();
        assert!(out.contains("allocated_bytes:"));
        assert!(out.contains("active_bytes:"));
        assert!(out.contains("resident_bytes:"));
        assert!(out.contains("mapped_bytes:"));
        assert!(out.contains("narenas:"));
        assert!(out.ends_with('\n'));
    }

    /// Snapshot-style pin of the label sequence `format_human`
    /// emits. Mirrors
    /// `host_context::tests::format_human_field_order_is_stable` —
    /// `HostContext::format_human` embeds this block indented under
    /// the `heap_state:` parent label, and downstream diff tools +
    /// operator-eye scanning depend on a stable
    /// `allocated → active → resident → mapped → narenas`
    /// top-to-bottom ordering. A silent reorder from a future edit
    /// that shuffles the `row(...)` calls inside `format_human`
    /// would slip past the order-blind `.contains(...)` checks in
    /// the sibling tests. This test fails the moment the sequence
    /// drifts; updating it forces the author to acknowledge the
    /// reorder and double-check the HostContext host_state sub-block
    /// still reads coherently.
    #[test]
    fn format_human_field_order_is_stable() {
        let out = HostHeapState::default().format_human();
        let labels: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split(':').next())
            .filter(|s| !s.starts_with(' '))
            .collect();
        assert_eq!(
            labels,
            vec![
                "allocated_bytes",
                "active_bytes",
                "resident_bytes",
                "mapped_bytes",
                "narenas",
            ],
            "format_human field order drifted — if intentional, update \
             the expected vector and verify the HostContext heap_state \
             sub-block still reads in the expected top-to-bottom order",
        );
    }

    #[test]
    fn format_human_renders_none_as_unknown() {
        let out = HostHeapState::default().format_human();
        // Every line should end with `: (unknown)`.
        for line in out.lines() {
            assert!(
                line.ends_with(": (unknown)"),
                "expected unknown, got {line:?}"
            );
        }
    }

    #[test]
    fn diff_is_empty_on_equal_snapshots() {
        let a = HostHeapState::test_fixture();
        let b = HostHeapState::test_fixture();
        assert_eq!(a.diff(&b), "");
    }

    #[test]
    fn diff_reports_only_changed_fields() {
        let a = HostHeapState::test_fixture();
        let mut b = a.clone();
        b.allocated_bytes = Some(9 * 1024 * 1024);
        let d = a.diff(&b);
        assert!(d.contains("allocated_bytes:"));
        assert!(!d.contains("active_bytes:"));
        assert!(!d.contains("resident_bytes:"));
        assert!(!d.contains("mapped_bytes:"));
        assert!(!d.contains("narenas:"));
        assert!(d.contains("→"));
    }

    #[test]
    fn diff_renders_none_transitions() {
        let a = HostHeapState::default();
        let b = HostHeapState::test_fixture();
        let d = a.diff(&b);
        // Every field changed, every line should carry the unknown→x arrow.
        assert!(d.contains("allocated_bytes: (unknown) →"));
        assert!(d.contains("narenas: (unknown) →"));
    }

    #[test]
    fn diff_renders_some_to_none_transitions() {
        // Symmetric case to `diff_renders_none_transitions`: a full
        // fixture diffed against `default()` must surface each field
        // as `x → (unknown)`, not be silently absorbed. Without this
        // test a one-sided `(unknown) → x` match could mask a
        // formatting bug in the reverse direction (e.g. the renderer
        // inadvertently suppressing `Some → None` as unchanged).
        let a = HostHeapState::test_fixture();
        let b = HostHeapState::default();
        let d = a.diff(&b);
        assert!(
            d.contains("allocated_bytes:") && d.contains("→ (unknown)"),
            "expected allocated_bytes → (unknown), got:\n{d}",
        );
        assert!(d.contains("active_bytes:"));
        assert!(d.contains("resident_bytes:"));
        assert!(d.contains("mapped_bytes:"));
        assert!(d.contains("narenas:"));
    }

    #[test]
    fn serde_round_trip_preserves_fields() {
        let h = HostHeapState::test_fixture();
        let s = serde_json::to_string(&h).expect("serialize");
        let back: HostHeapState = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, h);
    }

    #[test]
    fn serde_skips_none_fields() {
        let h = HostHeapState::default();
        let s = serde_json::to_string(&h).expect("serialize");
        // Every field is None and skip_serializing_if drops all of
        // them, so the JSON object is empty.
        assert_eq!(s, "{}");
    }

    #[test]
    fn serde_accepts_missing_fields_via_defaults() {
        // Older sidecar with no heap_state fields still
        // deserializes to a valid (all-None) snapshot.
        let back: HostHeapState = serde_json::from_str("{}").expect("deserialize");
        assert_eq!(back, HostHeapState::default());
    }

    /// Under the library-crate test harness, `tikv-jemallocator` is
    /// NOT installed as `#[global_allocator]` — the ktstr library
    /// itself declares no allocator so downstream consumers can
    /// pick their own. So even though libjemalloc is linked (hard
    /// dep of `tikv-jemalloc-ctl`) and `collect()` returns a
    /// populated struct with real mallctl values, `stats.allocated`
    /// and `stats.active` are both zero because the application
    /// (the test binary running under libc's malloc) never
    /// allocates through libjemalloc.
    ///
    /// Under this shape, the jemalloc invariants `active >=
    /// allocated`, `resident >= active`, `mapped >= active` all
    /// hold trivially (`0 >= 0`, small >= 0, small >= 0). They do
    /// NOT validate jemalloc behavior — they are tautologies.
    /// Real invariant coverage lives in each ktstr binary
    /// (ktstr, cargo-ktstr, jemalloc-probe) whose `main.rs` installs
    /// `tikv_jemallocator::Jemalloc` as `#[global_allocator]`; a
    /// live production run of any of those binaries exercises the
    /// non-trivial invariants. Documenting rather than
    /// feature-gating because a lib-crate integration test with its
    /// own `#[global_allocator]` would add its own binary target
    /// and is heavier than the coverage warrants.
    ///
    /// What this test DOES pin, non-trivially:
    /// - libjemalloc-linked-build contract: `narenas` is always
    ///   populated after `epoch::advance()` because arena count is
    ///   a jemalloc-init-time constant, independent of whether
    ///   jemalloc served any allocations.
    /// - `collect()` infallibility on a libjemalloc build — the
    ///   epoch::advance defensive guard does not fire.
    #[test]
    fn collect_returns_populated_snapshot_under_jemallocator() {
        let h = collect();
        // libjemalloc is linked unconditionally, so every field
        // populates. `narenas` in particular is a jemalloc-init
        // constant (4*ncpus by default) and always non-zero on a
        // multi-core host.
        assert!(h.narenas.is_some(), "narenas must populate on a libjemalloc build");
        assert!(
            h.narenas.unwrap() > 0,
            "narenas must be > 0; jemalloc computes 4*ncpus at init",
        );
        // stats reads also populate (they're not None) because
        // `--enable-stats` is forced by the `stats` feature in
        // Cargo.toml. Their VALUES depend on whether jemalloc is
        // `#[global_allocator]` — see the doc comment on this test.
        assert!(h.allocated_bytes.is_some());
        assert!(h.active_bytes.is_some());
        assert!(h.resident_bytes.is_some());
        assert!(h.mapped_bytes.is_some());
    }
}
