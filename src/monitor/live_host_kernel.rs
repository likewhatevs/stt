//! Auto-discovery of the running host kernel's vmlinux, BTF, and
//! symbol table for the live-host introspection path.
//!
//! Companion to [`super::bpf_syscall::BpfSyscallAccessor`]. Where the
//! frozen-VM path resolves vmlinux/symbols from a kernel build tree
//! the freeze coordinator already controls, the live-host path has
//! to find them on whatever distro happens to be running. This
//! module centralizes the search.
//!
//! # What gets discovered
//!
//! | resource           | source                                                       |
//! |--------------------|--------------------------------------------------------------|
//! | kernel release     | `uname(2)` (libc::uname)                                     |
//! | vmlinux ELF        | `/lib/modules/$(uname -r)/build/vmlinux`, distro debug paths |
//! | BTF                | `/sys/kernel/btf/vmlinux` (always present with sched_ext)    |
//! | kernel symbols     | `/proc/kallsyms` (root-readable, falls back per-line)        |
//!
//! # Why a separate module
//!
//! The frozen-VM pipeline (`vmm/mod.rs::find_vmlinux`,
//! `kernel_path::resolve_btf`) already searches similar paths but
//! with different priorities — it expects the freeze coordinator to
//! have built or downloaded the kernel and knows where the build
//! tree lives. The live-host pipeline starts from "no idea where
//! anything is, just whatever the running kernel exposes" and works
//! outward. Reusing the existing search would conflate the two
//! semantics (e.g. the frozen-VM code prefers a kernel-source-tree
//! vmlinux over `/sys/kernel/btf/vmlinux`; the live-host code
//! prefers BTF first for parse-cost reasons since it skips the goblin
//! ELF section walk).
//!
//! # BTF preference order
//!
//! 1. `/sys/kernel/btf/vmlinux` — always present when sched_ext is
//!    enabled (CONFIG_DEBUG_INFO_BTF). Raw BTF blob; no ELF parse
//!    needed.
//! 2. `/lib/modules/$(uname -r)/build/vmlinux` — kernel build tree,
//!    typically only present when `linux-headers-*` or a vendor
//!    kbuild package is installed.
//! 3. `/usr/lib/debug/boot/vmlinux-$(uname -r)` — debian/ubuntu
//!    `linux-image-*-dbg` package.
//! 4. `/usr/lib/debug/lib/modules/$(uname -r)/vmlinux` — fedora /
//!    rhel kernel-debuginfo layout.
//! 5. ktstr's kernel cache — for ktstr-built kernels installed via
//!    `cargo ktstr kernel build`, the cache root holds vmlinux next
//!    to the boot image.
//!
//! Order chosen so the cheapest parse path (raw BTF) wins by
//! default, and the more expensive ELF-extraction paths only run
//! when the live-host caller specifically asks for the full ELF
//! (e.g. for symbol resolution that goes beyond /proc/kallsyms).

use std::ffi::CStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

/// Resolved live-host kernel environment.
///
/// Built once at the start of a live-host introspection run; the
/// dump pipeline holds it alongside the
/// [`super::bpf_syscall::BpfSyscallAccessor`] for the duration of
/// the dump.
#[derive(Debug, Clone)]
#[allow(dead_code)] // wired from #11 (debug capture mode); the freeze-VM
                    // pipeline doesn't use this struct.
pub struct LiveHostKernelEnv {
    /// Output of `uname -r` — the running kernel's release string
    /// (e.g. "6.16.0-1234-generic"). Used to interpolate paths
    /// like `/lib/modules/<release>/build/vmlinux`.
    pub release: String,
    /// Path to the vmlinux ELF (or raw BTF blob) the BTF parser
    /// will load. Always set; resolution order is documented on the
    /// module-level doc.
    pub btf_path: PathBuf,
    /// Path to a vmlinux ELF when one is reachable on this host.
    /// `None` when only `/sys/kernel/btf/vmlinux` is available
    /// (raw BTF, no ELF) — most common on stripped distro kernels
    /// without `linux-headers-*` / `linux-image-*-dbg` installed.
    pub vmlinux_elf_path: Option<PathBuf>,
    /// Path to `/proc/kallsyms` — fixed but kept here so callers
    /// that want to swap in a unit-test fixture (or `/proc/PID/maps`
    /// alternative) have a single override point.
    pub kallsyms_path: PathBuf,
}

impl LiveHostKernelEnv {
    /// Auto-discover every resource needed by the live-host
    /// introspection pipeline.
    ///
    /// Returns an error only when none of the BTF candidate paths
    /// resolve — without BTF the failure-dump renderer can't decode
    /// any field. Missing vmlinux ELF or unreadable `/proc/kallsyms`
    /// are NOT errors at this layer: callers that need ELF or
    /// symbols for their specific dump pass surface their own error
    /// when they reach for an unavailable resource.
    #[allow(dead_code)]
    pub fn discover() -> Result<Self> {
        let release = uname_release().context("uname(2) failed")?;
        let btf_path = locate_btf(&release)
            .ok_or_else(|| anyhow!("no BTF found (looked in /sys/kernel/btf/vmlinux and ELF paths)"))?;
        let vmlinux_elf_path = locate_vmlinux_elf(&release);
        Ok(Self {
            release,
            btf_path,
            vmlinux_elf_path,
            kallsyms_path: PathBuf::from("/proc/kallsyms"),
        })
    }
}

/// `uname(2)` syscall wrapper. Returns the running kernel's release
/// string (the field that `uname -r` prints).
///
/// SAFETY: libc::uname populates a `utsname` struct on the stack;
/// the release field is a NUL-terminated `c_char[65]` in glibc /
/// musl's definition.
#[allow(dead_code)]
pub fn uname_release() -> Result<String> {
    // SAFETY: libc::utsname is a POD; zero-init is valid input to
    // libc::uname which fills it.
    let mut uts: libc::utsname = unsafe { std::mem::zeroed() };
    // SAFETY: libc::uname is a thin wrapper over the syscall;
    // returns 0 on success, -1 on failure (very rare — only fails
    // on a kernel-side fault).
    let ret = unsafe { libc::uname(&mut uts as *mut libc::utsname) };
    if ret != 0 {
        return Err(anyhow!(
            "uname(2) failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: libc fills .release with a NUL-terminated string of
    // at most 65 bytes. CStr::from_ptr requires a valid NUL-
    // terminated pointer; libc guarantees this on success.
    let release = unsafe { CStr::from_ptr(uts.release.as_ptr()) }
        .to_str()
        .context("uname.release was not valid UTF-8")?
        .to_string();
    Ok(release)
}

/// Locate a BTF source for the running kernel. See module-level doc
/// for the search order.
///
/// `/sys/kernel/btf/vmlinux` is preferred — when present (kernel
/// built with `CONFIG_DEBUG_INFO_BTF`, mandatory for sched_ext on
/// modern distros), it provides a raw BTF blob ready for
/// `Btf::from_bytes`. ELF candidates are returned only when the
/// raw BTF is absent.
fn locate_btf(release: &str) -> Option<PathBuf> {
    // 1. /sys/kernel/btf/vmlinux — fastest path, always present
    //    on a sched_ext-capable kernel.
    let sysfs = Path::new("/sys/kernel/btf/vmlinux");
    if sysfs.is_file() {
        return Some(sysfs.to_path_buf());
    }
    // 2-5. Fall back to ELF candidates that ALSO carry BTF in their
    //      .BTF section — the BTF loader handles both formats.
    locate_vmlinux_elf(release)
}

/// Locate a vmlinux ELF for the running kernel.
///
/// Search order (descending priority):
/// - `/lib/modules/$(uname -r)/build/vmlinux`
/// - `/usr/lib/debug/boot/vmlinux-$(uname -r)` (debian/ubuntu dbg)
/// - `/usr/lib/debug/lib/modules/$(uname -r)/vmlinux` (fedora/rhel)
/// - ktstr kernel cache (when present, falls through last)
fn locate_vmlinux_elf(release: &str) -> Option<PathBuf> {
    let candidates = [
        format!("/lib/modules/{release}/build/vmlinux"),
        format!("/usr/lib/debug/boot/vmlinux-{release}"),
        format!("/usr/lib/debug/lib/modules/{release}/vmlinux"),
    ];
    for cand in &candidates {
        let p = Path::new(cand);
        if p.is_file() {
            return Some(p.to_path_buf());
        }
    }
    // ktstr kernel cache — defer to the cache module's resolver.
    // The cache root is computed by [`crate::cache::cache_root`] but
    // we only consult the per-release entry shape: cache_root /
    // <key> / vmlinux. Without a cache key we can't address a
    // specific build, so we fall through here and let the live-host
    // caller specify a kernel cache entry explicitly when they
    // know one matches.
    None
}

/// Parsed kernel symbol table from `/proc/kallsyms`.
///
/// Per-line lazy lookup is too slow for the live-host pipeline,
/// which resolves dozens of symbols (sched_class addresses, lock
/// slowpath entry points, scx_root, etc.) at a single dump time.
/// `KallsymsTable` parses once and holds an O(1) name→addr map.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KallsymsTable {
    by_name: std::collections::HashMap<String, u64>,
}

impl KallsymsTable {
    /// Read and parse `/proc/kallsyms` from the configured path on
    /// `env`. Returns an error when the file is unreadable —
    /// `/proc/kallsyms` is root-readable only, and unprivileged
    /// callers see a 0-filled file. The parser detects the all-
    /// zeros case and returns an empty map without erroring (so
    /// non-privileged unit tests still get a usable
    /// `KallsymsTable` even though it can't resolve anything).
    #[allow(dead_code)]
    pub fn load_from(env: &LiveHostKernelEnv) -> Result<Self> {
        Self::load_from_path(&env.kallsyms_path)
    }

    /// Read and parse a kallsyms file from an explicit path. Useful
    /// for unit tests and for the rare live-host caller that wants
    /// to point at a saved snapshot rather than the live
    /// `/proc/kallsyms`.
    #[allow(dead_code)]
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        Ok(Self::parse(&raw))
    }

    /// Parse kallsyms-format text (one `HEX TYPE NAME ...` line per
    /// symbol) into a name→address map.
    ///
    /// Skipped lines (silently, without affecting other symbols):
    /// - lines with fewer than 3 whitespace-separated tokens
    /// - lines whose first token is not a hex-parseable u64
    /// - lines whose address is 0 (the kallsyms-redacted view that
    ///   unprivileged readers see — addresses are zero-filled by
    ///   the kernel for non-CAP_SYSLOG callers)
    ///
    /// A returned table with `len() == 0` is valid: the caller can
    /// detect "kallsyms unreadable" via `is_empty()` and surface a
    /// permission diagnostic without this layer producing an error.
    pub fn parse(raw: &str) -> Self {
        let mut by_name = std::collections::HashMap::new();
        for line in raw.lines() {
            let mut parts = line.split_whitespace();
            let Some(addr) = parts.next() else { continue };
            let _ty = parts.next();
            let Some(sym) = parts.next() else { continue };
            let Ok(addr) = u64::from_str_radix(addr, 16) else {
                continue;
            };
            // Skip the redacted-view all-zeros entries. A genuine
            // 0-valued symbol address would be a kernel bug; the
            // expected case is "unprivileged kallsyms reader sees
            // every line zeroed out".
            if addr == 0 {
                continue;
            }
            by_name.insert(sym.to_string(), addr);
        }
        Self { by_name }
    }

    /// Look up a symbol by exact name. Returns the kernel virtual
    /// address (u64) or `None` when the name is not in the table.
    #[allow(dead_code)]
    pub fn resolve(&self, name: &str) -> Option<u64> {
        self.by_name.get(name).copied()
    }

    /// Total number of resolved symbols. Zero when /proc/kallsyms
    /// was readable but every line was redacted (unprivileged
    /// caller case).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// True when the table holds no usable symbols. Live-host
    /// callers that hit this should surface a "run as root"
    /// diagnostic.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// uname_release returns a non-empty string on any platform
    /// where libc::uname succeeds. Linux always succeeds — this
    /// test would only fail on a hypothetical hostile kernel that
    /// returned -1, which would be a test-environment bug.
    #[test]
    fn uname_release_returns_nonempty() {
        let release = uname_release().expect("uname succeeds on Linux");
        assert!(!release.is_empty());
        // Sanity: every kernel since the dawn of time has had at
        // least one dot in the release (major.minor or
        // major.minor.patch).
        assert!(
            release.contains('.'),
            "release {release:?} should look like X.Y or X.Y.Z"
        );
    }

    /// `KallsymsTable::parse` recovers every well-formed symbol from a
    /// representative kallsyms snippet. Mirrors the format the
    /// kernel actually produces (HEX TYPE NAME [MODULE]).
    #[test]
    fn kallsyms_parse_basic() {
        let raw = "\
ffffffff80100000 T _stext
ffffffff80101234 T scx_disable_workfn
ffffffff80105678 t local_static_function
ffffffff8000abcd D ext_sched_class
";
        let table = KallsymsTable::parse(raw);
        assert_eq!(table.resolve("_stext"), Some(0xffffffff80100000));
        assert_eq!(table.resolve("scx_disable_workfn"), Some(0xffffffff80101234));
        assert_eq!(
            table.resolve("local_static_function"),
            Some(0xffffffff80105678)
        );
        assert_eq!(table.resolve("ext_sched_class"), Some(0xffffffff8000abcd));
        assert_eq!(table.len(), 4);
        assert!(!table.is_empty());
    }

    /// Redacted-view kallsyms (every address zero, what an
    /// unprivileged reader sees) parses to an empty table. The
    /// table is `is_empty()` rather than failing — callers
    /// distinguish "unreadable" (load failure) from "redacted"
    /// (parsed-but-empty) themselves.
    #[test]
    fn kallsyms_parse_skips_zero_addresses() {
        let raw = "\
0000000000000000 T _stext
0000000000000000 T scx_disable_workfn
";
        let table = KallsymsTable::parse(raw);
        assert!(table.is_empty());
        assert_eq!(table.resolve("_stext"), None);
    }

    /// Malformed lines (too few fields, non-hex address) are
    /// skipped without affecting good lines on either side.
    #[test]
    fn kallsyms_parse_skips_malformed_lines() {
        let raw = "\
ffffffff80100000 T _stext
not-a-hex-address T garbage
short_line
ffffffff80105678 T good_symbol
";
        let table = KallsymsTable::parse(raw);
        assert_eq!(table.resolve("_stext"), Some(0xffffffff80100000));
        assert_eq!(table.resolve("good_symbol"), Some(0xffffffff80105678));
        assert_eq!(table.resolve("garbage"), None);
        assert_eq!(table.len(), 2);
    }

    /// `KallsymsTable::load_from_path` reads from a file path
    /// rather than the live `/proc/kallsyms`. Verifies the
    /// pluggable-path constructor used by tests.
    #[test]
    fn kallsyms_load_from_path() {
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut f = tmp.reopen().unwrap();
        writeln!(f, "ffffffff80100000 T _stext").unwrap();
        writeln!(f, "ffffffff80101234 T target_symbol").unwrap();
        drop(f);

        let table = KallsymsTable::load_from_path(tmp.path()).unwrap();
        assert_eq!(table.resolve("target_symbol"), Some(0xffffffff80101234));
    }

    /// `LiveHostKernelEnv::discover` works on any sched_ext-capable
    /// kernel — it just needs `/sys/kernel/btf/vmlinux` to exist.
    /// Skip the test when running on a host without it (e.g. a
    /// build container without sched_ext debug info).
    #[test]
    fn live_host_kernel_env_discover_smoke() {
        if !Path::new("/sys/kernel/btf/vmlinux").is_file() {
            // No way to verify discover() on this host; skip.
            return;
        }
        let env = LiveHostKernelEnv::discover().expect("BTF present, discover should succeed");
        assert!(!env.release.is_empty());
        assert!(env.btf_path.exists());
        // kallsyms_path is always /proc/kallsyms regardless of
        // whether the file is readable.
        assert_eq!(env.kallsyms_path, Path::new("/proc/kallsyms"));
    }

    /// `locate_btf` falls through to ELF candidates when sysfs is
    /// missing. We can't easily test the sysfs path here without
    /// a syscall mock; verify the ELF fallback shape by passing a
    /// release that maps to no real path.
    #[test]
    fn locate_btf_no_real_release_returns_none_or_sysfs() {
        let result = locate_btf("definitely-not-a-kernel-release-9.99");
        // Either /sys/kernel/btf/vmlinux exists (and we get that)
        // or no fallback path resolves (and we get None).
        match result {
            Some(p) => assert_eq!(p, Path::new("/sys/kernel/btf/vmlinux")),
            None => {}
        }
    }
}
