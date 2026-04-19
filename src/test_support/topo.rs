//! Topology-override plumbing for `#[ktstr_test]` dispatch.
//!
//! `TopoOverride` is the runtime replacement for the topology declared
//! on a `KtstrTestEntry`: gauntlet expansion and the `--ktstr-topo` CLI
//! flag both construct one to boot the VM with a different topology
//! than the entry statically specified. `parse_topo_string` is the
//! wire-format parser used by both paths.

/// Optional topology override for `run_ktstr_test`.
pub(crate) struct TopoOverride {
    pub numa_nodes: u32,
    pub llcs: u32,
    pub cores: u32,
    pub threads: u32,
    pub memory_mb: u32,
}

impl TopoOverride {
    /// Reject zero-valued fields. The proc macro validates the same
    /// constraints at compile time for attribute-built entries; this
    /// covers runtime-constructed overrides (topology gauntlet
    /// presets, CLI flags that bypass the macro).
    ///
    /// Unlike the auto-derived `entry.memory_mb` path, which floors
    /// to `max(256, cpus*64)`, an explicit override is used verbatim
    /// — so `memory_mb == 0` would instruct the VM builder to boot
    /// with zero memory, which is a silent configuration error.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.numa_nodes == 0 {
            anyhow::bail!(
                "TopoOverride.numa_nodes must be > 0 (a topology with zero \
                 NUMA nodes has nothing to attach LLCs or memory to; every \
                 downstream accessor would observe an empty node set)"
            );
        }
        if self.llcs == 0 {
            anyhow::bail!(
                "TopoOverride.llcs must be > 0 (a topology with zero LLCs \
                 has zero CPUs — `total_cpus = llcs * cores * threads` — \
                 so the VM would boot with no addressable processors)"
            );
        }
        if self.cores == 0 {
            anyhow::bail!(
                "TopoOverride.cores must be > 0 (a topology with zero cores \
                 per LLC has zero CPUs — `total_cpus = llcs * cores * \
                 threads` — so the VM would boot with no addressable \
                 processors)"
            );
        }
        if self.threads == 0 {
            anyhow::bail!(
                "TopoOverride.threads must be > 0 (a topology with zero \
                 threads per core has zero CPUs — `total_cpus = llcs * \
                 cores * threads` — so the VM would boot with no \
                 addressable processors)"
            );
        }
        if self.memory_mb == 0 {
            anyhow::bail!(
                "TopoOverride.memory_mb must be > 0 (a VM with zero memory \
                 cannot boot); no implicit floor is applied to override path"
            );
        }
        Ok(())
    }
}

/// Parse a topology string in "NnNlNcNt" or legacy "NsNcNt" format.
/// `n` = NUMA nodes, `l` = LLCs, `c` = cores/LLC, `t` = threads/core.
/// Returns `(numa_nodes, llcs, cores, threads)` or None on parse failure.
pub(crate) fn parse_topo_string(s: &str) -> Option<(u32, u32, u32, u32)> {
    // Try new format: NnNlNcNt (e.g. "1n2l4c2t")
    if let Some(result) = parse_topo_string_new(s) {
        return Some(result);
    }
    // Legacy format: NsNcNt (treated as 1 NUMA node, N LLCs, N cores, N threads)
    let s_pos = s.find('s')?;
    let c_pos = s.find('c')?;
    let t_pos = s.find('t')?;
    if s_pos >= c_pos || c_pos >= t_pos {
        return None;
    }
    // Reject trailing garbage after 't', same as the new-format path.
    if s[t_pos + 1..].chars().next().is_some() {
        return None;
    }
    let llcs: u32 = s[..s_pos].parse().ok()?;
    let cores: u32 = s[s_pos + 1..c_pos].parse().ok()?;
    let threads: u32 = s[c_pos + 1..t_pos].parse().ok()?;
    if llcs == 0 || cores == 0 || threads == 0 {
        return None;
    }
    Some((1, llcs, cores, threads))
}

/// Parse new-format topology string "NnNlNcNt" (e.g. "1n2l4c2t").
/// Also accepts the old "NsNlNcNt" format for backward compatibility
/// with sidecar JSON and cached nextest args.
/// Returns `(numa_nodes, llcs, cores, threads)`.
fn parse_topo_string_new(s: &str) -> Option<(u32, u32, u32, u32)> {
    // Try 'n' first (new format), fall back to 's' (old format).
    let first_pos = s.find('n').or_else(|| s.find('s'))?;
    let l_pos = s.find('l')?;
    let c_pos = s.find('c')?;
    let t_pos = s.find('t')?;
    if first_pos >= l_pos || l_pos >= c_pos || c_pos >= t_pos {
        return None;
    }
    // Reject trailing garbage after the 't' terminator. Silently
    // accepting e.g. "1n2l4c2tEXTRA" as (1,2,4,2) masks typos and
    // version-skew bugs in cached nextest args / sidecar filenames.
    if s[t_pos + 1..].chars().next().is_some() {
        return None;
    }
    let numa_nodes: u32 = s[..first_pos].parse().ok()?;
    let llcs: u32 = s[first_pos + 1..l_pos].parse().ok()?;
    let cores: u32 = s[l_pos + 1..c_pos].parse().ok()?;
    let threads: u32 = s[c_pos + 1..t_pos].parse().ok()?;
    if numa_nodes == 0 || llcs == 0 || cores == 0 || threads == 0 {
        return None;
    }
    Some((numa_nodes, llcs, cores, threads))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- parse_topo_string (legacy + new format) --

    #[test]
    fn parse_topo_valid() {
        assert_eq!(parse_topo_string("2s4c2t"), Some((1, 2, 4, 2)));
    }

    #[test]
    fn parse_topo_single_digits() {
        assert_eq!(parse_topo_string("1s1c1t"), Some((1, 1, 1, 1)));
    }

    #[test]
    fn parse_topo_large() {
        assert_eq!(parse_topo_string("14s9c2t"), Some((1, 14, 9, 2)));
    }

    #[test]
    fn parse_topo_zero_llcs_legacy() {
        // Legacy three-part format "NsNcNt" — the `s` token originally
        // meant "sockets" but, after the sockets→LLCs rename, the leading
        // field is the LLC count. A zero here must still be rejected.
        assert!(parse_topo_string("0s4c2t").is_none());
    }

    #[test]
    fn parse_topo_zero_cores() {
        assert!(parse_topo_string("2s0c2t").is_none());
    }

    #[test]
    fn parse_topo_zero_threads() {
        assert!(parse_topo_string("2s4c0t").is_none());
    }

    #[test]
    fn parse_topo_missing_suffix() {
        assert!(parse_topo_string("2s4c2").is_none());
    }

    #[test]
    fn parse_topo_empty() {
        assert!(parse_topo_string("").is_none());
    }

    #[test]
    fn parse_topo_garbage() {
        assert!(parse_topo_string("hello").is_none());
    }

    #[test]
    fn parse_topo_wrong_order() {
        assert!(parse_topo_string("2c4s2t").is_none());
    }

    // -- new format NnNlNcNt --

    #[test]
    fn parse_topo_new_format_basic() {
        assert_eq!(parse_topo_string("1n2l4c2t"), Some((1, 2, 4, 2)));
    }

    #[test]
    fn parse_topo_new_format_multi_numa() {
        assert_eq!(parse_topo_string("2n4l8c2t"), Some((2, 4, 8, 2)));
    }

    #[test]
    fn parse_topo_new_format_single() {
        assert_eq!(parse_topo_string("1n1l1c1t"), Some((1, 1, 1, 1)));
    }

    #[test]
    fn parse_topo_new_format_large() {
        assert_eq!(parse_topo_string("4n16l8c2t"), Some((4, 16, 8, 2)));
    }

    #[test]
    fn parse_topo_new_format_zero_numa() {
        assert!(parse_topo_string("0n2l4c2t").is_none());
    }

    #[test]
    fn parse_topo_new_format_zero_llcs() {
        assert!(parse_topo_string("1n0l4c2t").is_none());
    }

    #[test]
    fn parse_topo_new_format_zero_cores() {
        assert!(parse_topo_string("1n2l0c2t").is_none());
    }

    #[test]
    fn parse_topo_new_format_zero_threads() {
        assert!(parse_topo_string("1n2l4c0t").is_none());
    }

    #[test]
    fn parse_topo_new_format_wrong_order() {
        assert!(parse_topo_string("2l1n4c2t").is_none());
    }

    #[test]
    fn parse_topo_rejects_trailing_garbage() {
        // Regression for #27: trailing characters after the 't'
        // terminator previously parsed as the clean form, silently
        // dropping the trailing garbage. Now rejected.
        assert!(
            parse_topo_string("1n2l4c2tEXTRA").is_none(),
            "trailing garbage after 't' must cause None"
        );
        assert!(
            parse_topo_string("1n2l4c2tb").is_none(),
            "single trailing character must cause None"
        );
        assert!(
            parse_topo_string("1s2l4c2t_debug").is_none(),
            "trailing garbage after old-format 't' must cause None"
        );
    }

    #[test]
    fn parse_topo_legacy_rejects_trailing_garbage() {
        // Regression for #27 on the legacy NsNcNt path: trailing
        // characters after 't' must be rejected as firmly as the
        // new format.
        assert!(
            parse_topo_string("2s4c2tEXTRA").is_none(),
            "legacy-format trailing garbage after 't' must cause None"
        );
        assert!(
            parse_topo_string("2s4c2tx").is_none(),
            "legacy-format single trailing character must cause None"
        );
    }

    #[test]
    fn parse_topo_accepts_clean_terminator() {
        // Sanity: after #27, the clean form still parses.
        assert_eq!(parse_topo_string("1n2l4c2t"), Some((1, 2, 4, 2)));
        assert_eq!(parse_topo_string("1s2l4c2t"), Some((1, 2, 4, 2)));
    }

    // -- old four-part NsNlNcNt (back-compat with sidecar JSON) --

    #[test]
    fn parse_topo_old_four_part_basic() {
        assert_eq!(parse_topo_string("1s2l4c2t"), Some((1, 2, 4, 2)));
    }

    #[test]
    fn parse_topo_old_four_part_multi_numa() {
        assert_eq!(parse_topo_string("2s4l8c2t"), Some((2, 4, 8, 2)));
    }

    #[test]
    fn parse_topo_old_four_part_single() {
        assert_eq!(parse_topo_string("1s1l1c1t"), Some((1, 1, 1, 1)));
    }

    #[test]
    fn parse_topo_double_digit_threads() {
        assert_eq!(parse_topo_string("1s1c12t"), Some((1, 1, 1, 12)));
    }

    // -- TopoOverride --

    #[test]
    fn topo_override_fields() {
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: 2,
            cores: 4,
            threads: 2,
            memory_mb: 8192,
        };
        assert_eq!(t.numa_nodes, 1);
        assert_eq!(t.llcs, 2);
        assert_eq!(t.cores, 4);
        assert_eq!(t.threads, 2);
        assert_eq!(t.memory_mb, 8192);
    }

    #[test]
    fn topo_override_validate_accepts_nonzero() {
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: 2,
            cores: 4,
            threads: 2,
            memory_mb: 8192,
        };
        t.validate().unwrap();
    }

    #[test]
    fn topo_override_validate_rejects_zero_memory() {
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: 2,
            cores: 4,
            threads: 2,
            memory_mb: 0,
        };
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("memory_mb") && msg.contains("> 0"),
            "error must name memory_mb: {msg}"
        );
    }

    #[test]
    fn topo_override_validate_rejects_zero_numa_nodes() {
        let t = TopoOverride {
            numa_nodes: 0,
            llcs: 2,
            cores: 4,
            threads: 2,
            memory_mb: 8192,
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn topo_override_validate_rejects_zero_llcs() {
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: 0,
            cores: 4,
            threads: 2,
            memory_mb: 8192,
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn topo_override_validate_rejects_zero_cores() {
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: 2,
            cores: 0,
            threads: 2,
            memory_mb: 8192,
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn topo_override_validate_rejects_zero_threads() {
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: 2,
            cores: 4,
            threads: 0,
            memory_mb: 8192,
        };
        assert!(t.validate().is_err());
    }
}
