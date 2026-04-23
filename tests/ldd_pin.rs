//! Pin test for the `ktstr` binary's dynamic library dependency
//! count.
//!
//! Shells out to `ldd $ktstr_binary` and asserts the NON-vdso entry
//! count matches the expected baseline. Catches drift in BOTH
//! directions:
//!
//! - **Increase**: a new direct crate dep (or a transitive bump)
//!   introduces a `.so` link requirement — e.g. switching an
//!   internally-pure-Rust subsystem to a crate backed by
//!   `libfoo-sys`. New shared-library dependencies enlarge the
//!   deployment surface, add distro-version coupling, and may
//!   reduce reproducibility on hosts without the `.so`.
//! - **Decrease**: a previously-dynamic dep got statically linked
//!   (feature flag flip, upstream crate vendoring C source, a
//!   linker-option change). Not a defect per se, but it shifts
//!   the link model silently — a host that used to supply
//!   `libfoo.so.N` now has dead state on its filesystem, a
//!   future crash-dump / ldd-based debugging workflow loses the
//!   breadcrumb that the dep was ever dynamically linked, and the
//!   binary's statically-embedded C carries CVE responsibility
//!   that the system package manager no longer owns.
//!
//! In EITHER direction, a regression here is not always broken,
//! but it deserves an explicit acknowledgement: bump the expected
//! count and document the reason in the commit message.
//!
//! Scoped to glibc (target_env = "gnu") on Linux x86_64 / aarch64.
//! musl targets (target_env = "musl") statically link the C
//! runtime and produce a PIE with no `.so` deps, so `ldd` either
//! reports "not a dynamic executable" or an empty set — a
//! different count baseline than glibc's. Other platforms have
//! different link models entirely (macOS uses `otool -L`, Windows
//! has no single equivalent) that would need their own pin tests.
//!
//! # glibc version assumption
//!
//! The 3-entry baseline (libgcc_s.so.1, libm.so.6, libc.so.6)
//! assumes **glibc >= 2.34**. Starting with 2.34 (released
//! 2021-08-01), glibc consolidated `libpthread.so.0`,
//! `libdl.so.2`, `libutil.so.1`, and `libanl.so.1` into
//! `libc.so.6`; a binary that pulls in any of those legacy libs
//! on a pre-2.34 host surfaces as extra `ldd` entries and the
//! EXPECTED_DEPS count assertion fails loudly.
//!
//! The test deliberately does NOT range-check glibc at runtime —
//! the failure message already spells out the full ldd output,
//! which makes the "someone rebuilt CI on an older glibc" root
//! cause obvious without the complexity of parsing
//! `/lib/x86_64-linux-gnu/libc.so.6 --version` or conditional
//! baselines per glibc generation. Maintainers on CI runners
//! with glibc < 2.34 must either upgrade the runner or add a
//! conditional baseline here.

#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    any(target_arch = "x86_64", target_arch = "aarch64"),
))]
#[test]
fn ktstr_binary_dynamic_deps_pinned() {
    // The debug binary and the release binary carry the same link
    // graph — `cargo build --release` does not change which crates
    // are linked, only their optimization level. The test runs
    // against whichever binary profile `cargo test` / `cargo nextest`
    // builds (debug by default).
    let binary = env!("CARGO_BIN_EXE_ktstr");
    let output = std::process::Command::new("ldd")
        .arg(binary)
        .output()
        .expect("ldd must be available on the host");
    assert!(
        output.status.success(),
        "ldd failed on {binary}: {}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Skip `linux-vdso.so.*` — it is provided by the kernel and
    // has no filesystem backing; every dynamically-linked binary on
    // Linux carries it. Skip the dynamic loader entry too
    // (`/lib*/ld-linux-*.so.*`): it is also unconditional and its
    // path varies by distro (`/lib64/ld-linux-x86-64.so.2`,
    // `/lib/ld-linux-aarch64.so.1`, etc.).
    let deps: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.contains("linux-vdso.so"))
        .filter(|l| !l.contains("ld-linux-") && !l.contains("/ld.so"))
        .filter(|l| !l.trim().is_empty())
        .collect();

    // Expected dynamic dep set on a glibc-linked Linux host:
    //   libgcc_s.so.1 — Rust unwinding personality on Linux. Even
    //     under panic=abort, the linker pulls it in for stack-unwind
    //     metadata referenced by the standard library.
    //   libm.so.6    — transcendental math (exp/log/sqrt) used
    //     transitively by a few deps (rand, polars, candle).
    //   libc.so.6    — glibc itself.
    // Third-party jemalloc ships vendored and statically-linked
    // (see Cargo.toml comment on tikv-jemallocator). libbpf is
    // vendored via libbpf-sys. No additional .so links should
    // appear without an explicit dep bump.
    //
    // A regression that raises this count means a new direct or
    // transitive crate dep introduced a dynamic link requirement.
    // Confirm the new link is intentional, then bump this pin and
    // note the reason in the commit message.
    const EXPECTED_DEPS: usize = 3;

    assert_eq!(
        deps.len(),
        EXPECTED_DEPS,
        "ktstr binary dynamic-dep count drifted from {EXPECTED_DEPS}:\nldd output:\n{stdout}\n\
         Filtered deps ({} entries):\n{}",
        deps.len(),
        deps.join("\n"),
    );

    // Also pin the specific library basenames so a swap (e.g.
    // libgcc_s replaced by libunwind) is caught even if the count
    // happens to match.
    let expected_names = ["libgcc_s.so", "libm.so", "libc.so"];
    for name in &expected_names {
        assert!(
            deps.iter().any(|l| l.contains(name)),
            "expected dynamic dep {name:?} not found in ldd output:\n{stdout}",
        );
    }

    // Harden the libgcc_s check: the presence gate above proves
    // the NAME appears somewhere in the ldd output, but NOT that
    // the dynamic loader actually resolved it to a real file. An
    // ldd line shaped `libgcc_s.so.1 => not found (0x0)` would
    // satisfy `contains("libgcc_s.so")` yet crash at runtime on
    // the first unwind. Pin the `=>` resolution form AND verify
    // the resolved path exists as a regular file on disk — belt
    // and suspenders against a broken runner image.
    let libgcc_line = deps
        .iter()
        .find(|l| l.contains("libgcc_s.so"))
        .expect("libgcc_s.so line must be in the filtered deps — checked above");
    assert!(
        libgcc_line.contains("=>") && !libgcc_line.contains("not found"),
        "libgcc_s line does not show a successful `name => path` \
         resolution: {libgcc_line:?}\nfull ldd output:\n{stdout}",
    );
    // The resolved path is the token after `=>` and before the
    // trailing `(0x...)` address. Parse it defensively: if the
    // format ever drifts, fall back to treating the assertion as
    // a best-effort hardening step rather than failing the test
    // on a non-critical format diff.
    if let Some(after_arrow) = libgcc_line.split("=>").nth(1) {
        let resolved_path = after_arrow.trim().split_whitespace().next().unwrap_or("");
        if !resolved_path.is_empty() {
            let p = std::path::Path::new(resolved_path);
            assert!(
                p.exists() && p.is_file(),
                "libgcc_s resolved path {resolved_path:?} does not \
                 exist or is not a regular file. Runner image is \
                 likely corrupt or missing the libgcc runtime.\n\
                 full ldd output:\n{stdout}",
            );
        }
    }
}
