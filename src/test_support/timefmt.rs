//! UTC calendar math and run-ID generation.
//!
//! Sidecar writes stamp each result with an ISO-8601 timestamp and a
//! monotonic run ID. Both values need to be stable across builds and
//! platforms so downstream analysis can group variants without relying
//! on clock skew or thread scheduling. This module is pure: no I/O, no
//! locale handling, no crate-external dependencies beyond `std`.
//!
//! [`now_iso8601`] formats the current UTC time in the fixed
//! `YYYY-MM-DDTHH:MM:SSZ` shape. [`days_to_ymd`] and [`is_leap`] are
//! the helpers it uses to convert a UNIX-epoch day count into
//! `(year, month, day)` without pulling in `chrono`.
//! [`run_id_timestamp`] returns a compact `YYYYMMDDTHHMMSSZ` stamp
//! captured once per process in a `OnceLock` so every sidecar and
//! every run-directory name written from one `cargo ktstr test`
//! invocation share a stable key. [`generate_run_id`] composes
//! `{run_id_timestamp}-{counter}`; the counter is a process-local
//! atomic so concurrent gauntlet variants can't collide on the same
//! run-ID value.

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

/// Compact process-start UTC timestamp in `YYYYMMDDTHHMMSSZ` form.
///
/// Captured once per process via a `OnceLock` so every sidecar
/// written from one `cargo ktstr test` invocation — and the
/// enclosing run directory name — share a single stable key. The
/// compact form (no dashes, no colons) keeps the string safe for
/// use as a filename/directory segment on every target filesystem
/// and sorts lexicographically in chronological order.
///
/// Two consumers today:
/// - [`crate::test_support::sidecar_dir`] uses it as the second
///   segment of the run directory name (`{kernel}-{timestamp}`).
/// - [`generate_run_id`] prepends it to a monotonic counter so
///   every sidecar's `run_id` field carries the same
///   per-invocation prefix.
///
/// A regression that re-sampled the clock on every call would
/// break both consumers: sidecars written mid-run would scatter
/// across multiple dirs, and `run_id`s within one run would no
/// longer share a stable prefix. The `OnceLock` pin is the
/// single-sample guarantee.
pub(crate) fn run_id_timestamp() -> &'static str {
    use std::sync::OnceLock;
    static STAMP: OnceLock<String> = OnceLock::new();
    STAMP.get_or_init(|| {
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
        format!("{y:04}{mo:02}{d:02}T{h:02}{m:02}{s:02}Z")
    })
}

/// Generate a run ID as `{run_id_timestamp}-{counter}`. Timestamp
/// is shared across the process; counter monotonically
/// increments per call so concurrent gauntlet variants can't
/// collide.
pub(crate) fn generate_run_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{n}", run_id_timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- days_to_ymd / is_leap --

    #[test]
    fn days_to_ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        let (y, m, d) = days_to_ymd(18628);
        assert_eq!((y, m, d), (2021, 1, 1));
    }

    #[test]
    fn days_to_ymd_leap_day() {
        let (y, m, d) = days_to_ymd(11016);
        assert_eq!((y, m, d), (2000, 2, 29));
    }

    #[test]
    fn days_to_ymd_2024_jan_1() {
        // 2024-01-01 = 19723 days since epoch.
        assert_eq!(days_to_ymd(19723), (2024, 1, 1));
    }

    #[test]
    fn days_to_ymd_2024_leap_day() {
        // 2024-02-29 = 19723 + 31 + 28 = 19782.
        assert_eq!(days_to_ymd(19782), (2024, 2, 29));
    }

    #[test]
    fn days_to_ymd_2023_end_of_year() {
        // 2023-12-31 = 19722.
        assert_eq!(days_to_ymd(19722), (2023, 12, 31));
    }

    #[test]
    fn is_leap_years() {
        assert!(is_leap(2000));
        assert!(is_leap(2024));
        assert!(!is_leap(1900));
        assert!(!is_leap(2023));
    }

    // -- now_iso8601 --

    #[test]
    fn now_iso8601_format() {
        let ts = now_iso8601();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
        assert_eq!(ts.len(), 20);
    }

    // -- run_id_timestamp / generate_run_id --

    /// `run_id_timestamp` is the `YYYYMMDDTHHMMSSZ` compact form:
    /// 16 ASCII chars, no `-` / `:`, trailing `Z`. Pin the shape so
    /// a regression that switched back to the extended form
    /// (`YYYY-MM-DDTHH:MM:SSZ`) — which contains filesystem-hostile
    /// `:` — would surface here rather than as a directory-creation
    /// failure at run time.
    #[test]
    fn run_id_timestamp_compact_form() {
        let stamp = run_id_timestamp();
        assert_eq!(stamp.len(), 16, "compact form must be 16 chars: {stamp}");
        assert!(stamp.ends_with('Z'), "must end with Z: {stamp}");
        assert!(stamp.contains('T'), "must contain T separator: {stamp}");
        assert!(!stamp.contains(':'), "compact form must not contain ':': {stamp}");
        assert!(!stamp.contains('-'), "compact form must not contain '-': {stamp}");
        assert!(
            stamp[..8].chars().all(|c| c.is_ascii_digit()),
            "date prefix must be all digits: {stamp}",
        );
        assert!(
            stamp[9..15].chars().all(|c| c.is_ascii_digit()),
            "time segment must be all digits: {stamp}",
        );
    }

    /// `run_id_timestamp` is a `OnceLock`-backed process-local
    /// value — every call within a single process returns the
    /// same string. A regression that re-sampled the clock would
    /// cause sidecars written mid-run to scatter across multiple
    /// directories.
    #[test]
    fn run_id_timestamp_is_stable_across_calls() {
        let a = run_id_timestamp();
        let b = run_id_timestamp();
        assert_eq!(a, b, "OnceLock must return the same value across calls");
    }

    /// `generate_run_id` composes `{run_id_timestamp}-{counter}`.
    /// Pin the prefix contract so every sidecar produced in one
    /// process shares the same timestamp prefix even as the
    /// counter advances. The timestamp reference is captured
    /// BEFORE the two `generate_run_id` calls because `OnceLock`
    /// seals the value on first access — calling
    /// `run_id_timestamp` first guarantees we read the same value
    /// the `generate_run_id` calls will compose against.
    #[test]
    fn generate_run_id_prefixes_with_stable_timestamp() {
        let ts = run_id_timestamp();
        let id = generate_run_id();
        assert!(
            id.starts_with(ts),
            "run ID {id} must begin with timestamp prefix {ts}",
        );
        let rest = &id[ts.len()..];
        assert!(
            rest.starts_with('-'),
            "timestamp must be followed by '-': {id}",
        );
        assert!(
            rest[1..].chars().all(|c| c.is_ascii_digit()),
            "counter segment must be digits: {id}",
        );
    }

    #[test]
    fn generate_run_id_monotonic() {
        let id1 = generate_run_id();
        let id2 = generate_run_id();
        assert_ne!(id1, id2);
    }
}
