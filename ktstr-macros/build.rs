// Verify that ktstr-macros/src/kernel_path.rs stays in sync with the
// parent ktstr crate's src/kernel_path.rs. During workspace builds the
// parent file is reachable at ../src/kernel_path.rs; during crates.io
// builds it is absent and the check is skipped (the bundled copy is
// authoritative).

fn main() {
    println!("cargo:rerun-if-changed=src/kernel_path.rs");
    println!("cargo:rerun-if-changed=build.rs");

    let parent = std::path::PathBuf::from("../src/kernel_path.rs");
    println!("cargo:rerun-if-changed=../src/kernel_path.rs");

    let parent_contents = match std::fs::read_to_string(&parent) {
        Ok(c) => c,
        Err(_) => return,
    };

    let stripped = match parent_contents.split_once("\n#[cfg(test)]") {
        Some((head, _)) => format!("{head}\n"),
        None => parent_contents,
    };

    let bundled = std::fs::read_to_string("src/kernel_path.rs")
        .unwrap_or_else(|e| panic!("ktstr-macros build.rs: cannot read src/kernel_path.rs: {e}"));

    if stripped != bundled {
        panic!(
            "ktstr-macros/src/kernel_path.rs is out of sync with \
             ../src/kernel_path.rs. Copy the non-test portion of the \
             parent file to update the bundled copy.",
        );
    }
}
