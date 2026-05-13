// Mirror the parent ktstr crate's `src/kernel_path.rs` into OUT_DIR so
// `declare_scheduler!` can validate `kernels = [..]` entries through
// the same `KernelId::parse` + `KernelId::validate` path the verifier
// uses at runtime. Single source of truth — divergence between
// macro-time and runtime parsing would silently accept different
// strings on each axis.
//
// `kernel_path.rs` is std-only (see the file header in
// `../src/kernel_path.rs`) and exposes `KernelId::parse` + `validate`
// without any non-std dependency, so the macro crate can pull the file
// in directly with no extra Cargo metadata. The file's `#[cfg(test)]`
// block uses dev-deps (tempfile, proptest) that ktstr-macros does
// not list, so the build strips that block before writing to OUT_DIR
// — the ktstr crate already runs those tests against its own copy of
// the file, so duplicating the run inside ktstr-macros buys nothing.

use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    let src = PathBuf::from("../src/kernel_path.rs");
    println!("cargo:rerun-if-changed=../src/kernel_path.rs");
    println!("cargo:rerun-if-changed=build.rs");
    let contents = std::fs::read_to_string(&src).unwrap_or_else(|e| {
        panic!(
            "ktstr-macros build.rs: cannot read {} ({e}). The macro \
             crate mirrors `src/kernel_path.rs` from the parent ktstr \
             crate for `declare_scheduler!` kernel-string validation; \
             the file must be present at this relative path during \
             builds.",
            src.display(),
        );
    });

    // Strip the `#[cfg(test)] mod tests { .. }` block at the file
    // tail. The kernel_path.rs convention pins this block to the end
    // of the file, so a substring split on the `#[cfg(test)]` marker
    // is sufficient — no token-level parsing required. A future
    // regression that adds a second cfg(test) item earlier in the
    // file would still get split at the first marker; the file
    // header explicitly notes the cfg(test) convention.
    let trimmed = match contents.split_once("\n#[cfg(test)]") {
        Some((head, _tail)) => format!("{head}\n"),
        None => contents,
    };

    let dst = out_dir.join("kernel_path.rs");
    std::fs::write(&dst, trimmed)
        .unwrap_or_else(|e| panic!("ktstr-macros build.rs: write {} failed: {e}", dst.display()));
}
