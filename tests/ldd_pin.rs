//! Pin test for the `ktstr` binary's dynamic library dependency
//! count.
//!
//! Shells out to `ldd $ktstr_binary` and asserts the NON-vdso entry
//! count matches the expected baseline. Catches silent dep inflation
//! from a new direct crate dep that introduces a `.so` link
//! requirement — e.g. switching an internally-pure-Rust subsystem to
//! a crate backed by `libfoo-sys`. A regression here is not always
//! broken, but it deserves an explicit acknowledgement: bump the
//! expected count and document the reason in the commit message.
//!
//! Scoped to glibc (target_env = "gnu") on Linux x86_64 / aarch64.
//! musl targets (target_env = "musl") statically link the C
//! runtime and produce a PIE with no `.so` deps, so `ldd` either
//! reports "not a dynamic executable" or an empty set — a
//! different count baseline than glibc's. Other platforms have
//! different link models entirely (macOS uses `otool -L`, Windows
//! has no single equivalent) that would need their own pin tests.

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
}
