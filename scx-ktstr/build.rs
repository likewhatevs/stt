// Builds scx-ktstr's BPF object. The scheduler exercises the
// `sdt_alloc` arena allocator on top of the BPF arena so that ktstr's
// failure-dump arena renderer has real allocator-shaped state to
// capture. The two `.bpf.c` files that implement the allocator
// (`lib/sdt_alloc.bpf.c`, `lib/sdt_task.bpf.c`) and the one header
// they pull in via a quoted include (`lib/scxtest/scx_test.h`) are
// fetched from the upstream scx repo at build time rather than
// vendored into this tree.

use std::env;
use std::path::{Path, PathBuf};

// scx tag whose `lib/` API matches the `scx_*` symbols declared by
// the bundled `scx_utils-bpf_h/lib/sdt_task.h` that scx_cargo 1.1.0
// installs into OUT_DIR. Tags ≤ v1.0.11 ship the older `sdt_*` API
// (the rename to `scx_*` first appeared at v1.0.12, verified by
// reading `lib/sdt_task.bpf.c` at both tags); pinning to a pre-rename
// tag would link-fail against scx_cargo 1.1.0 headers. This constant
// is the single source of truth; bumping it triggers a re-fetch via
// the `.scx-tag` sentinel comparison in `scx_lib_complete`.
const SCX_TAG: &str = "v1.1.0";

/// Sentinel filename inside the cache dir that records which
/// `SCX_TAG` produced the extracted sources. `scx_lib_complete`
/// reads this and forces a re-fetch when the recorded tag differs
/// from the current `SCX_TAG`. Without this gate, bumping
/// `SCX_TAG` on a tree with a populated cache would silently keep
/// using sources from the previous tag.
const SCX_TAG_SENTINEL: &str = ".scx-tag";

// Files fetched from the upstream tarball into `OUT_DIR/scx-lib/`.
// Paths are relative to the repo root after stripping the GitHub
// archive's top-level prefix component.
const SCX_FETCH_FILES: &[&str] = &[
    "lib/sdt_alloc.bpf.c",
    "lib/sdt_task.bpf.c",
    "lib/scxtest/scx_test.h",
];

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let scx_lib = out_dir.join("scx-lib");

    fetch_scx_lib(&scx_lib);

    // Compile main.bpf.c plus the fetched allocator sources into a
    // single linked BPF object. `sdt_alloc.bpf.c` provides the
    // `scx_alloc_*` allocator surface and pulls in the arena map
    // definition via `<lib/arena_map.h>` (the actual file that
    // declares `arena __weak SEC(".maps")`); `sdt_task.bpf.c`
    // defines the task-storage map and the `scx_task_*` per-task
    // wrappers that main.bpf.c calls. `compile_link_gen` writes
    // one `bpf.bpf.o` from all three sources via libbpf-rs's
    // bpf_linker.
    //
    // Both `sdt_alloc.bpf.c` and `sdt_task.bpf.c` include
    // `"scxtest/scx_test.h"` with a quoted include, which the C
    // preprocessor resolves relative to the `.bpf.c` file's own
    // directory before searching -I paths. The tarball/clone fetch
    // lays each file out next to its sibling at
    // `OUT_DIR/scx-lib/lib/scxtest/scx_test.h` so the relative
    // resolution finds it; no extra -I is needed for that include.
    // The angle-bracket includes (`<lib/sdt_task.h>`,
    // `<lib/arena_map.h>`, `<scx/common.bpf.h>`, etc.) resolve via
    // the `scx_utils-bpf_h/` include path that `BpfBuilder::new()`
    // sets up.
    let sdt_alloc_path = scx_lib.join("lib/sdt_alloc.bpf.c");
    let sdt_task_path = scx_lib.join("lib/sdt_task.bpf.c");

    scx_cargo::BpfBuilder::new()
        .expect("BpfBuilder::new")
        .enable_skel("src/bpf/main.bpf.c", "bpf")
        .add_source(
            sdt_alloc_path
                .to_str()
                .expect("OUT_DIR/scx-lib/lib/sdt_alloc.bpf.c path must be UTF-8"),
        )
        .add_source(
            sdt_task_path
                .to_str()
                .expect("OUT_DIR/scx-lib/lib/sdt_task.bpf.c path must be UTF-8"),
        )
        .compile_link_gen()
        .expect("BpfBuilder::compile_link_gen");

    // Note: `compile_link_gen` already emits
    // `cargo:rerun-if-changed=src/bpf/main.bpf.c` (registered via
    // `enable_skel`'s `self.sources` set) plus a glob for
    // `src/bpf/*.[hc]`, so an explicit `rerun-if-changed` for
    // main.bpf.c here would be a duplicate.
}

/// Populate `dest` (typically `$OUT_DIR/scx-lib`) with the subset of
/// scx upstream sources `SCX_FETCH_FILES` enumerates. Returns when
/// every file in the list exists at its destination path.
///
/// Strategy mirrors the busybox fetch in `../build.rs`:
///   1. cache hit -> no work
///   2. download GitHub archive tarball, extract just the listed
///      files into a stage directory, then promote with one rename
///   3. fall back to a shallow `gix` clone of the tag if the
///      tarball download or extraction fails
///   4. panic with an actionable message if both paths fail
///
/// Partial state from a prior failed run is removed before each
/// fallback so a half-extracted tree cannot satisfy the cache check
/// on the next attempt.
fn fetch_scx_lib(dest: &Path) {
    if scx_lib_complete(dest) {
        return;
    }

    println!(
        "cargo:warning=fetching scx {SCX_TAG} sources for sdt_alloc \
         (first build only)..."
    );

    if dest.exists() {
        std::fs::remove_dir_all(dest).expect("remove stale OUT_DIR/scx-lib");
    }

    let tarball_url =
        format!("https://github.com/sched-ext/scx/archive/refs/tags/{SCX_TAG}.tar.gz");
    let tarball_err = fetch_via_tarball(&tarball_url, dest).err();

    if !scx_lib_complete(dest) {
        let tarball_err = tarball_err.unwrap_or_else(|| "unknown".to_string());
        // Intentionally do NOT include the upstream error string in the
        // `cargo:warning` line: a transient HTML error page or network-
        // stack diagnostic from the upstream side could embed arbitrary
        // text in a `cargo:warning`, which is otherwise local-build
        // output but is consumed by CI log scrapers. The full
        // `tarball_err` is preserved for the panic path below, which
        // only fires on terminal failure and surfaces to the operator's
        // own terminal, not to a `cargo:warning` line.
        println!("cargo:warning=tarball fetch failed, trying git clone...");

        if dest.exists() {
            std::fs::remove_dir_all(dest).expect("remove partial OUT_DIR/scx-lib before clone");
        }

        let clone_err = fetch_via_clone(dest).err();

        if !scx_lib_complete(dest) {
            let clone_err = clone_err
                .unwrap_or_else(|| "checkout missing one or more SCX_FETCH_FILES".to_string());
            // The panic surface is the right place for the full upstream
            // error strings — it runs only on terminal failure, prints
            // to the operator's terminal, and is needed for diagnosis.
            // Workarounds line below covers the two non-network paths
            // that a constrained operator can take to unblock the build.
            let scx_lib_dir = dest.display();
            panic!(
                "failed to obtain scx {SCX_TAG} sources for sdt_alloc.\n\
                 tarball ({tarball_url}): {tarball_err}\n\
                 git clone (sched-ext/scx@{SCX_TAG}): {clone_err}\n\
                 First build requires network access to GitHub.\n\
                 Workarounds: set HTTPS_PROXY for proxy environments, or\n\
                 manually place files at {scx_lib_dir}/lib/{{sdt_alloc.bpf.c, sdt_task.bpf.c, scxtest/scx_test.h}}."
            );
        }
    }
}

/// True iff every entry in `SCX_FETCH_FILES` already exists under
/// `dest` AND the recorded sentinel matches `SCX_TAG`. Used both as
/// the cache-hit predicate and as the success gate after each fetch
/// attempt.
///
/// The `SCX_TAG_SENTINEL` check is what makes a `SCX_TAG` bump
/// invalidate the on-disk cache: an old extraction without the
/// sentinel returns false, and an extraction with a stale sentinel
/// also returns false, so `fetch_scx_lib` falls into the
/// remove-and-refetch path.
fn scx_lib_complete(dest: &Path) -> bool {
    if !SCX_FETCH_FILES.iter().all(|p| dest.join(p).is_file()) {
        return false;
    }
    match std::fs::read_to_string(dest.join(SCX_TAG_SENTINEL)) {
        Ok(stamped) => stamped.trim() == SCX_TAG,
        Err(_) => false,
    }
}

/// Stamp the `SCX_TAG_SENTINEL` file inside `dest` with the current
/// `SCX_TAG`. Called by both fetch paths after a successful staged
/// rename, before the function returns `Ok`. Stamping after the
/// rename means a partially-written sentinel cannot satisfy
/// `scx_lib_complete` on the next run — if `write` fails the cache
/// stays incomplete and the next build re-fetches.
fn stamp_scx_tag(dest: &Path) -> Result<(), String> {
    std::fs::write(dest.join(SCX_TAG_SENTINEL), SCX_TAG)
        .map_err(|e| format!("write {SCX_TAG_SENTINEL}: {e}"))
}

/// Download the tagged GitHub archive tarball and extract just the
/// files in `SCX_FETCH_FILES` into a stage directory, then promote
/// the stage into `dest` with a single rename. Tarball entries carry
/// a top-level `<repo>-<tag>/` prefix component that is stripped here
/// without depending on the exact prefix shape.
fn fetch_via_tarball(url: &str, dest: &Path) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let resp = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("download: {e}"))?;

    let gz = flate2::read::GzDecoder::new(resp);
    let mut archive = tar::Archive::new(gz);

    let wanted: std::collections::HashSet<&str> = SCX_FETCH_FILES.iter().copied().collect();
    let mut found = std::collections::HashSet::<String>::new();

    let parent = dest
        .parent()
        .ok_or_else(|| "OUT_DIR/scx-lib has no parent".to_string())?;
    let stage = parent.join("scx-lib-stage");
    if stage.exists() {
        std::fs::remove_dir_all(&stage).map_err(|e| format!("remove stale stage dir: {e}"))?;
    }
    std::fs::create_dir_all(&stage).map_err(|e| format!("create stage dir: {e}"))?;

    for entry in archive
        .entries()
        .map_err(|e| format!("read tar entries: {e}"))?
    {
        let mut entry = entry.map_err(|e| format!("tar entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("tar entry path: {e}"))?
            .into_owned();
        // Strip the GitHub-injected top-level prefix component
        // (`scx-<tag-without-leading-v>/`).
        let mut comps = path.components();
        comps.next();
        let rel = comps.as_path();
        let rel_str = match rel.to_str() {
            Some(s) => s,
            None => continue,
        };
        if !wanted.contains(rel_str) {
            continue;
        }

        let target = stage.join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        entry
            .unpack(&target)
            .map_err(|e| format!("unpack {rel_str}: {e}"))?;
        found.insert(rel_str.to_string());
    }

    let missing: Vec<&&str> = SCX_FETCH_FILES
        .iter()
        .filter(|p| !found.contains(**p))
        .collect();
    if !missing.is_empty() {
        return Err(format!("tarball missing expected files: {missing:?}"));
    }

    std::fs::rename(&stage, dest).map_err(|e| format!("promote stage to scx-lib: {e}"))?;
    stamp_scx_tag(dest)?;
    Ok(())
}

/// Shallow-clone the tag and copy `SCX_FETCH_FILES` into `dest` via a
/// stage directory. Used as a fallback when the GitHub archive
/// tarball cannot be reached.
fn fetch_via_clone(dest: &Path) -> Result<(), String> {
    let parent = dest
        .parent()
        .ok_or_else(|| "OUT_DIR/scx-lib has no parent".to_string())?;
    let work = parent.join("scx-lib-clone");
    if work.exists() {
        std::fs::remove_dir_all(&work).map_err(|e| format!("remove stale clone dir: {e}"))?;
    }

    let url = "https://github.com/sched-ext/scx.git";
    let interrupt = std::sync::atomic::AtomicBool::new(false);

    let mut prep = gix::prepare_clone(url, &work)
        .map_err(|e| format!("prepare_clone: {e}"))?
        .with_shallow(gix::remote::fetch::Shallow::DepthAtRemote(
            1.try_into().expect("non-zero depth"),
        ))
        .with_ref_name(Some(SCX_TAG))
        .map_err(|e| format!("with_ref_name: {e}"))?;
    let (mut checkout, _) = prep
        .fetch_then_checkout(gix::progress::Discard, &interrupt)
        .map_err(|e| format!("fetch_then_checkout: {e}"))?;
    let (_repo, _) = checkout
        .main_worktree(gix::progress::Discard, &interrupt)
        .map_err(|e| format!("main_worktree: {e}"))?;

    let stage = parent.join("scx-lib-stage");
    if stage.exists() {
        std::fs::remove_dir_all(&stage).map_err(|e| format!("remove stale stage dir: {e}"))?;
    }
    std::fs::create_dir_all(&stage).map_err(|e| format!("create stage dir: {e}"))?;

    for rel in SCX_FETCH_FILES {
        let src = work.join(rel);
        if !src.is_file() {
            return Err(format!("clone missing expected file: {rel}"));
        }
        let dst = stage.join(rel);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        std::fs::copy(&src, &dst).map_err(|e| format!("copy {rel}: {e}"))?;
    }

    std::fs::rename(&stage, dest).map_err(|e| format!("promote stage to scx-lib: {e}"))?;
    stamp_scx_tag(dest)?;
    let _ = std::fs::remove_dir_all(&work);
    Ok(())
}
