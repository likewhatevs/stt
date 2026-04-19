//! UTC calendar math and run-ID generation.
//!
//! Sidecar writes stamp each result with an ISO-8601 timestamp and a
//! monotonic run ID. Both values need to be stable across builds and
//! platforms so downstream analysis can group variants without relying
//! on clock skew or thread scheduling. This module is pure: no I/O, no
//! locale handling, no crate-external dependencies beyond `std` and
//! `crate::GIT_HASH`.
//!
//! [`now_iso8601`] formats the current UTC time in the fixed
//! `YYYY-MM-DDTHH:MM:SSZ` shape. [`days_to_ymd`] and [`is_leap`] are
//! the helpers it uses to convert a UNIX-epoch day count into
//! `(year, month, day)` without pulling in `chrono`. [`generate_run_id`]
//! stamps each run with `{GIT_HASH}-{counter}`; the counter is a
//! process-local atomic so concurrent gauntlet variants can't collide.

/// ISO 8601 timestamp.
pub(crate) fn now_iso8601() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

pub(crate) fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let mut y = 1970;
    let mut remaining = days;
    loop {
        let leap = is_leap(y);
        let year_days = if leap { 366 } else { 365 };
        if remaining < year_days {
            break;
        }
        remaining -= year_days;
        y += 1;
    }
    let leap = is_leap(y);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 1u64;
    for &md in &month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        mo += 1;
    }
    (y, mo, remaining + 1)
}

pub(crate) fn is_leap(y: u64) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

/// Generate a run ID from git hash + monotonic counter.
pub(crate) fn generate_run_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{n}", crate::GIT_HASH)
}
