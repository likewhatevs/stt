//! CLI argument parsers shared between `ktstr` and `cargo-ktstr`.
//!
//! Holds the topology-string and disk-size parsers along with the
//! `--disk` help text. Lives outside `kernel_cmd` because the
//! parsers are dispatch-time helpers, not clap-attribute fixtures.

use anyhow::{Result, bail};

/// Parse a comma-separated topology string into its four dimensions:
/// `(numa_nodes, llcs, cores, threads)`. The canonical format is
/// `"numa_nodes,llcs,cores,threads"` — the same shape accepted by the
/// `ktstr shell --topology` and `cargo ktstr shell --topology` flags.
///
/// Validation:
/// - Exactly four comma-separated components are required.
/// - Each component must parse as `u32`. A parse failure names the
///   failing field explicitly (e.g. `"invalid llcs value: 'abc'"`)
///   so the user can see which dimension they mistyped without
///   counting commas.
/// - Every dimension must be at least 1 — a zero in any position
///   produces an unusable VM topology, so we reject it up front.
///
/// Consolidating the parse + validate in one helper eliminates the
/// identical 4-arm `parts[i].parse().map_err(...)` block that the two
/// binary entry points (`src/bin/ktstr.rs` Command::Shell and
/// `src/bin/cargo-ktstr.rs` `run_shell`) would otherwise drift on.
/// Error shape is `anyhow::Error`; callers that need a `String` (like
/// cargo-ktstr's `Result<(), String>` surface) bridge via
/// `.map_err(|e| format!("{e:#}"))` at the call site.
pub fn parse_topology_string(topology: &str) -> Result<(u32, u32, u32, u32)> {
    let parts: Vec<&str> = topology.split(',').collect();
    if parts.len() != 4 {
        bail!(
            "invalid topology '{topology}': expected 'numa_nodes,llcs,cores,threads' \
             (e.g. '1,2,4,1')"
        );
    }
    // Stable field order mirrors the 4-tuple return so a future
    // field-rename lands consistently in one place.
    let fields: [(&str, &str); 4] = [
        ("numa_nodes", parts[0]),
        ("llcs", parts[1]),
        ("cores", parts[2]),
        ("threads", parts[3]),
    ];
    let mut vals: [u32; 4] = [0; 4];
    for (i, (name, raw)) in fields.iter().enumerate() {
        vals[i] = raw
            .parse::<u32>()
            .map_err(|_| anyhow::anyhow!("invalid {name} value: '{raw}'"))?;
    }
    let [numa_nodes, llcs, cores, threads] = vals;
    if numa_nodes == 0 || llcs == 0 || cores == 0 || threads == 0 {
        bail!("invalid topology '{topology}': all values must be >= 1");
    }
    Ok((numa_nodes, llcs, cores, threads))
}

/// Parse a human-readable size string (e.g. `"256mib"`, `"10gib"`, `"1gib"`)
/// into a count of mebibytes (MiB), rounded down. Returns `Err` when the
/// suffix is unrecognized, the numeric portion fails to parse, the value
/// is not a positive integer multiple of one MiB, or the result exceeds
/// `u32::MAX` MiB (the [`crate::vmm::disk_config::DiskConfig::capacity_mb`]
/// capacity).
///
/// Accepted suffixes (case-insensitive): `b`, `kib`, `mib`, `gib`. All
/// IEC (powers of two): `kib`=2^10, `mib`=2^20, `gib`=2^30. SI variants
/// (`kb`/`mb`/`gb`) are intentionally NOT accepted; they're rejected by
/// a dedicated SI-suffix check at the top of the function — before any
/// number-parsing or MiB-alignment runs — so the diagnostic names the
/// IEC-only policy directly instead of leaking through as a misleading
/// "numeric portion not an unsigned integer" message after the suffix
/// strip eats the trailing `b`. IEC-only is unambiguous and consistent.
/// The bare suffix-less form is also rejected so units are never
/// implicit.
///
/// The output unit is MiB to match
/// [`crate::vmm::disk_config::DiskConfig::capacity_mb`] (despite the
/// field name, [`DiskConfig::capacity_bytes`] left-shifts by 20 — i.e.
/// the field is MiB, not SI MB). A future rename of that field would
/// land in this function in lockstep.
pub fn parse_disk_size_mib(s: &str) -> Result<u32> {
    let lower = s.trim().to_ascii_lowercase();
    if lower.is_empty() {
        bail!("invalid disk size '{s}': empty");
    }
    // Reject SI-suffix forms (kb/mb/gb) up front. The IEC-only
    // policy keeps the contract unambiguous: 1mib means exactly
    // 2^20 bytes, never 10^6. Without this short-circuit the
    // generic `b` (byte) suffix below would chew off the trailing
    // 'b' and then fail to parse e.g. "1k" as a u64, producing a
    // misleading "numeric portion not an unsigned integer" error
    // instead of the unit-list diagnostic the user needs.
    if lower.ends_with("kb") || lower.ends_with("mb") || lower.ends_with("gb") {
        bail!(
            "invalid disk size '{s}': SI suffixes (kb/mb/gb) are \
             not supported. Use one of b, kib, mib, gib \
             (case-insensitive)."
        );
    }
    let (num_str, suffix, unit_bytes): (&str, &str, u64) =
        if let Some(rest) = lower.strip_suffix("gib") {
            (rest, "gib", 1u64 << 30)
        } else if let Some(rest) = lower.strip_suffix("mib") {
            (rest, "mib", 1u64 << 20)
        } else if let Some(rest) = lower.strip_suffix("kib") {
            (rest, "kib", 1u64 << 10)
        } else if let Some(rest) = lower.strip_suffix('b') {
            (rest, "b", 1u64)
        } else {
            bail!(
                "invalid disk size '{s}': missing unit suffix. Use one of \
             b, kib, mib, gib (case-insensitive)."
            );
        };
    let n = num_str.trim().parse::<u64>().map_err(|_| {
        anyhow::anyhow!(
            "invalid disk size '{s}': numeric portion '{num_str}' before \
             '{suffix}' is not an unsigned integer"
        )
    })?;
    let bytes = n
        .checked_mul(unit_bytes)
        .ok_or_else(|| anyhow::anyhow!("invalid disk size '{s}': {n}{suffix} overflows u64"))?;
    if bytes == 0 {
        bail!("invalid disk size '{s}': must be > 0");
    }
    let mib = 1u64 << 20;
    if bytes % mib != 0 {
        bail!(
            "invalid disk size '{s}': {bytes} bytes is not a whole number \
             of mebibytes (MiB). Round to a multiple of 1 MiB (= 1048576 \
             bytes)."
        );
    }
    let mib_count = bytes / mib;
    if mib_count > u32::MAX as u64 {
        bail!(
            "invalid disk size '{s}': {mib_count} MiB exceeds u32::MAX \
             (DiskConfig.capacity_mb is u32)"
        );
    }
    Ok(mib_count as u32)
}

/// Help text for the `--disk <SIZE>` shell flag, shared between
/// `cargo ktstr shell` (`src/bin/cargo-ktstr.rs`) and
/// `ktstr shell` (`src/bin/ktstr.rs`) so a future tweak lands in
/// one place. Mirrors the [`super::CPU_CAP_HELP`] pattern.
pub const DISK_HELP: &str = "Attach a raw virtio-blk disk to /dev/vda. \
     Accepts a human-readable size with a unit suffix (case-insensitive): \
     b, kib, mib, gib. IEC-only — SI variants (kb/mb/gb) are rejected to \
     keep the contract unambiguous. The size must be a positive whole \
     number of MiB (e.g. 256mib, 1gib). Omit to boot without a disk.";

/// Parse the `--disk <SIZE>` CLI argument into an
/// [`Option<crate::vmm::disk_config::DiskConfig>`]. `None` input
/// returns `Ok(None)` (no disk attached); a `Some(s)` input runs
/// `s` through [`parse_disk_size_mib`] and wraps the result in a
/// `DiskConfig` whose remaining fields fall through to
/// [`crate::vmm::disk_config::DiskConfig::default`] (raw filesystem,
/// no throttle, read-write). Shared between `cargo ktstr shell` and
/// `ktstr shell` so both bins parse identically; a malformed size
/// surfaces here at CLI-argument time, never mid-VM-setup.
pub fn parse_disk_arg(s: Option<&str>) -> Result<Option<crate::vmm::disk_config::DiskConfig>> {
    match s {
        Some(raw) => {
            let mib = parse_disk_size_mib(raw)?;
            Ok(Some(crate::vmm::disk_config::DiskConfig {
                capacity_mb: mib,
                ..crate::vmm::disk_config::DiskConfig::default()
            }))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Happy path: a canonical `"n,l,c,t"` string round-trips to the
    /// four u32 dimensions in positional order. Pins the field order
    /// so a future refactor that reshuffles (numa_nodes/llcs/cores/
    /// threads) → something else can't silently swap one dimension
    /// for another without flipping this pin.
    #[test]
    fn parse_topology_string_happy_path() {
        let (n, l, c, t) = parse_topology_string("1,2,4,8").expect("valid");
        assert_eq!((n, l, c, t), (1, 2, 4, 8));
    }

    /// Wrong component count: fewer than 4 parts names the expected
    /// shape in the error so the user sees the canonical format.
    #[test]
    fn parse_topology_string_rejects_too_few_parts() {
        let err = parse_topology_string("1,2,4").expect_err("3 parts must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("invalid topology '1,2,4'"),
            "error must echo the bad input: {rendered}",
        );
        assert!(
            rendered.contains("numa_nodes,llcs,cores,threads"),
            "error must name the expected shape: {rendered}",
        );
    }

    /// Too MANY parts is rejected the same way. Pairs with the
    /// too-few case so the guard is symmetric.
    #[test]
    fn parse_topology_string_rejects_too_many_parts() {
        let err = parse_topology_string("1,2,4,8,16").expect_err("5 parts must fail");
        assert!(format!("{err:#}").contains("invalid topology"));
    }

    /// A non-numeric component fails with a message that names the
    /// offending FIELD, not just the bad token — a user who mistypes
    /// the second dimension sees `"invalid llcs value: 'abc'"` and
    /// knows immediately which dimension needs fixing. Pin all four
    /// position-to-name mappings so a field-order refactor surfaces
    /// here.
    #[test]
    fn parse_topology_string_names_failing_field() {
        for (pos, field) in [(0, "numa_nodes"), (1, "llcs"), (2, "cores"), (3, "threads")] {
            let mut parts = ["1"; 4];
            parts[pos] = "abc";
            let input = parts.join(",");
            let err = parse_topology_string(&input).expect_err("non-numeric must fail");
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains(&format!("invalid {field} value: 'abc'")),
                "pos {pos}: error must name the `{field}` field, got: {rendered}",
            );
        }
    }

    /// Zero in any position fails the `>= 1` guard with the
    /// "all values must be >= 1" phrasing. A zero topology would
    /// build a non-bootable VM, so rejecting it up-front is a
    /// correctness requirement, not a style choice.
    #[test]
    fn parse_topology_string_rejects_zero_dimensions() {
        for pos in 0..4 {
            let mut parts = ["1"; 4];
            parts[pos] = "0";
            let input = parts.join(",");
            let err = parse_topology_string(&input).expect_err("zero must fail");
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains(">= 1"),
                "pos {pos}: error must cite the >=1 rule: {rendered}",
            );
        }
    }

    /// Upper bound: u32::MAX in every position parses successfully.
    /// Pins the return-type decision (u32, not u16 / usize) so a
    /// future refactor that narrows the type surfaces here rather
    /// than truncating large-host topology strings.
    #[test]
    fn parse_topology_string_accepts_u32_max() {
        let big = u32::MAX;
        let input = format!("{big},{big},{big},{big}");
        let (n, l, c, t) = parse_topology_string(&input).expect("u32::MAX valid");
        assert_eq!((n, l, c, t), (big, big, big, big));
    }

    /// u32 overflow (value above u32::MAX) fails with the field
    /// name, not a generic parse error. Exercises the `parse::<u32>`
    /// failure path rather than only the non-numeric path.
    #[test]
    fn parse_topology_string_rejects_u32_overflow() {
        let too_big = (u32::MAX as u64) + 1;
        let input = format!("1,{too_big},4,1");
        let err = parse_topology_string(&input).expect_err("overflow must fail");
        assert!(
            format!("{err:#}").contains(&format!("invalid llcs value: '{too_big}'")),
            "overflow must surface field + bad token: {err:#}",
        );
    }

    /// IEC suffixes (`mib`, `gib`) round-trip to whole MiB counts. Pins
    /// the binary-base interpretation of the IEC family.
    #[test]
    fn parse_disk_size_mib_iec_suffixes() {
        assert_eq!(parse_disk_size_mib("256mib").unwrap(), 256);
        assert_eq!(parse_disk_size_mib("1gib").unwrap(), 1024);
        assert_eq!(parse_disk_size_mib("10GIB").unwrap(), 10 * 1024);
        assert_eq!(parse_disk_size_mib("1024kib").unwrap(), 1);
    }

    /// SI suffixes (`kb`, `mb`, `gb`) are rejected as unrecognized so
    /// the user sees the unit-list diagnostic instead of a confusing
    /// MiB-alignment failure. IEC-only is the unambiguous contract.
    #[test]
    fn parse_disk_size_mib_rejects_si_suffixes() {
        for input in ["1kb", "1mb", "1gb", "256MB", "10GB"] {
            let err = parse_disk_size_mib(input)
                .expect_err(&format!("SI suffix '{input}' must be rejected"));
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains("SI suffixes"),
                "expected SI-rejection diagnostic for {input:?}, got: {rendered}",
            );
        }
    }

    /// Bare `b` with a value that aligns to a MiB succeeds; a value
    /// off-by-one fails. Pins the byte-suffix path.
    #[test]
    fn parse_disk_size_mib_byte_suffix() {
        assert_eq!(parse_disk_size_mib("1048576b").unwrap(), 1);
        let err = parse_disk_size_mib("1048575b").expect_err("off-by-one byte must fail");
        assert!(format!("{err:#}").contains("not a whole number"));
    }

    /// Whitespace + mixed case in the input are tolerated by trim +
    /// to_lowercase.
    #[test]
    fn parse_disk_size_mib_normalizes_input() {
        assert_eq!(parse_disk_size_mib("  256MiB  ").unwrap(), 256);
        assert_eq!(parse_disk_size_mib("1GiB").unwrap(), 1024);
    }

    /// Missing suffix is rejected with a unit-list diagnostic so the
    /// user sees what's accepted.
    #[test]
    fn parse_disk_size_mib_rejects_missing_suffix() {
        let err = parse_disk_size_mib("256").expect_err("bare integer must fail");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("missing unit suffix"));
        assert!(rendered.contains("kib"));
        assert!(rendered.contains("mib"));
        assert!(rendered.contains("gib"));
    }

    /// Empty / whitespace-only input is rejected up front.
    #[test]
    fn parse_disk_size_mib_rejects_empty() {
        assert!(parse_disk_size_mib("").is_err());
        assert!(parse_disk_size_mib("   ").is_err());
    }

    /// Zero is rejected — a 0-byte disk is a configuration footgun
    /// (every IO IOERRs per `DiskConfig::with_options`).
    #[test]
    fn parse_disk_size_mib_rejects_zero() {
        let err = parse_disk_size_mib("0mib").expect_err("zero must fail");
        assert!(format!("{err:#}").contains("must be > 0"));
    }

    /// Non-numeric prefix is rejected.
    #[test]
    fn parse_disk_size_mib_rejects_garbage_number() {
        assert!(parse_disk_size_mib("abcmib").is_err());
        assert!(parse_disk_size_mib("-5mib").is_err());
        assert!(parse_disk_size_mib("3.5mib").is_err());
    }

    /// Unknown suffix is rejected.
    #[test]
    fn parse_disk_size_mib_rejects_unknown_suffix() {
        let err = parse_disk_size_mib("1tb").expect_err("tb is not currently accepted");
        let rendered = format!("{err:#}");
        // Last matching strip_suffix is "b", which leaves "1t" as the
        // numeric portion and surfaces the parse error there.
        assert!(rendered.contains("invalid disk size '1tb'"));
    }

    /// A value that overflows u32::MAX MiB is rejected (capacity_mb is u32).
    #[test]
    fn parse_disk_size_mib_rejects_u32_overflow() {
        // (u32::MAX + 1) MiB
        let too_big_mib = (u32::MAX as u64) + 1;
        let input = format!("{too_big_mib}mib");
        let err = parse_disk_size_mib(&input).expect_err("> u32::MAX MiB must fail");
        assert!(format!("{err:#}").contains("exceeds u32::MAX"));
    }

    /// A value whose byte product overflows u64 is rejected before
    /// the MiB conversion runs.
    #[test]
    fn parse_disk_size_mib_rejects_u64_overflow() {
        // u64::MAX gib is way past u64::MAX bytes.
        let input = format!("{}gib", u64::MAX);
        let err = parse_disk_size_mib(&input).expect_err("u64 overflow must fail");
        assert!(format!("{err:#}").contains("overflows u64"));
    }

    /// Absent `--disk` flag → `Ok(None)`. Pins the
    /// no-disk-attached default so a future refactor that flips the
    /// arm to a `Some(default())` placeholder fails the test
    /// instead of silently changing the boot shape (a disk where
    /// the user asked for none).
    #[test]
    fn parse_disk_arg_none_yields_no_disk() {
        let got = parse_disk_arg(None).expect("None input must not error");
        assert!(
            got.is_none(),
            "absent --disk must produce Ok(None), got: {got:?}",
        );
    }

    /// `--disk 256mib` → `Some(DiskConfig)` with `capacity_mb=256`
    /// and the remaining fields equal to `DiskConfig::default()`.
    /// Pins the size-only fast path (the only shape `parse_disk_arg`
    /// accepts today) and guards against drift in the spread of
    /// non-size fields — if a future change flips a default
    /// (read_only=true, throttle non-default), this test surfaces it
    /// at the CLI parse boundary rather than mid-VM-setup.
    #[test]
    fn parse_disk_arg_some_size_uses_default_other_fields() {
        let got = parse_disk_arg(Some("256mib"))
            .expect("256mib must parse")
            .expect("Some(...) input must yield Some(DiskConfig)");
        let expected = crate::vmm::disk_config::DiskConfig {
            capacity_mb: 256,
            ..crate::vmm::disk_config::DiskConfig::default()
        };
        assert_eq!(
            got, expected,
            "parse_disk_arg(\"256mib\") must equal DiskConfig::default() \
             with capacity_mb=256: got {got:?}, expected {expected:?}",
        );
    }

    /// Malformed size → `Err`. Pins that a CLI typo surfaces at
    /// argument time with a parse-error message, not mid-VM-setup
    /// or as a confusing zero-size disk.
    #[test]
    fn parse_disk_arg_garbage_propagates_size_error() {
        let err =
            parse_disk_arg(Some("garbage")).expect_err("malformed size must propagate parse error");
        let rendered = format!("{err:#}");
        // Every `parse_disk_size_mib` bail prefixes its message with
        // `invalid disk size '...'` (the input echoed back), so a
        // single-substring check is sufficient — every error path
        // satisfies it. A future message-format change that drops the
        // prefix would surface here instead of being silently absorbed.
        assert!(
            rendered.contains("invalid disk size"),
            "expected size-parse diagnostic in disk-arg error, got: {rendered}",
        );
    }
}
