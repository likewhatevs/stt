//! `cargo ktstr funify` — replace non-metric JSON values with
//! deterministic adjective-animal petnames.
//!
//! Reads JSON from `--input PATH`, stdin (when no path given), or
//! the literal `-` sentinel; routes the parsed value through
//! [`ktstr::fun::funify_json`] under either a deterministic
//! seed-derived [`ktstr::fun::Funifier::with_seed`] or a process-
//! ephemeral [`ktstr::fun::Funifier::ephemeral`]; serializes the
//! transformed value back to stdout (compact or pretty per
//! `--pretty`).

use std::path::PathBuf;

/// `cargo ktstr funify <input>` — read a JSON dump, replace every
/// non-metric value (per
/// [`ktstr::fun::Funifier::is_metric_passthrough`](ktstr::fun::Funifier::is_metric_passthrough))
/// with `adjective-animal` petnames, write the result to stdout.
///
/// `seed.is_some()` makes the mapping deterministic across
/// invocations of this binary so a user running `funify` twice on
/// the same dump gets identical fun names (and can correlate fun
/// names across two related dumps). `seed.is_none()` derives a
/// process-fresh ephemeral key — every invocation produces a
/// different fun name for the same input.
///
/// Errors are returned as `String` (matching the existing
/// `Result<(), String>` shape every other run_* helper uses), so
/// the dispatch site's `error: {e:#}` formatter handles them
/// uniformly.
pub(crate) fn run_funify(
    input: Option<PathBuf>,
    seed: Option<String>,
    pretty: bool,
) -> Result<(), String> {
    use std::io::Read;

    // Read input: stdin when no path is given OR when path is the
    // explicit "-" sentinel; otherwise read the file at `input`.
    let from_stdin = match input.as_deref() {
        None => true,
        Some(p) => p.as_os_str() == "-",
    };
    let raw = if from_stdin {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("read stdin: {e}"))?;
        s
    } else {
        let p = input.as_deref().unwrap();
        std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?
    };

    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse JSON input: {e}"))?;

    let funifier = match seed {
        Some(s) => ktstr::fun::Funifier::with_seed(&s),
        None => ktstr::fun::Funifier::ephemeral(),
    };
    let funified = ktstr::fun::funify_json(value, &funifier);

    let out = if pretty {
        serde_json::to_string_pretty(&funified)
    } else {
        serde_json::to_string(&funified)
    }
    .map_err(|e| format!("serialize funified JSON: {e}"))?;
    println!("{out}");
    Ok(())
}
