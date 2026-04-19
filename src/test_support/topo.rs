//! Topology-override plumbing for `#[ktstr_test]` dispatch.
//!
//! `TopoOverride` is the runtime replacement for the topology declared
//! on a `KtstrTestEntry`: gauntlet expansion and the `--ktstr-topo` CLI
//! flag both construct one to boot the VM with a different topology
//! than the entry statically specified. `parse_topo_string` is the
//! wire-format parser used by both paths.

/// Optional topology override for `run_ktstr_test`.
///
/// Field names intentionally drop the `_per_llc` / `_per_core` suffix
/// that `vmm::topology::Topology` uses. A `TopoOverride` is the wire
/// form carried in gauntlet preset tables and on the `--ktstr-topo`
/// CLI (e.g. `1n2l4c2t`), where the short axis names are readable and
/// the per-unit meaning is unambiguous. Convert to
/// [`crate::vmm::topology::Topology`] via the `From` impl below.
pub(crate) struct TopoOverride {
    pub numa_nodes: u32,
    pub llcs: u32,
    pub cores: u32,
    pub threads: u32,
    pub memory_mb: u32,
}

impl From<&TopoOverride> for crate::vmm::topology::Topology {
    /// Construct a VM-builder [`Topology`](crate::vmm::topology::Topology)
    /// from an override's four topology axes. `memory_mb` is discarded —
    /// VM memory lives on [`vmm::KtstrVm::builder().memory_deferred_min()`]
    /// which the dispatcher sets separately from the topology.
    fn from(t: &TopoOverride) -> Self {
        crate::vmm::topology::Topology::new(t.numa_nodes, t.llcs, t.cores, t.threads)
    }
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
        // Reject topologies whose total CPU count overflows u32.
        // Downstream `Topology::new(numa_nodes, llcs, cores, threads)`
        // and cpuset math multiply these three u32 dimensions
        // together; silent wrap produces a nonsense total_cpus that
        // would mis-size vCPU allocations and cgroup cpusets.
        let cpus_per_llc = self.cores.checked_mul(self.threads).ok_or_else(|| {
            anyhow::anyhow!(
                "TopoOverride: cores ({}) * threads ({}) overflows u32 — \
                 cpus_per_llc cannot be represented",
                self.cores,
                self.threads,
            )
        })?;
        self.llcs.checked_mul(cpus_per_llc).ok_or_else(|| {
            anyhow::anyhow!(
                "TopoOverride: llcs ({}) * cores ({}) * threads ({}) \
                 overflows u32 — total_cpus cannot be represented",
                self.llcs,
                self.cores,
                self.threads,
            )
        })?;
        Ok(())
    }
}

/// Recognise the pre-1.0 legacy topology forms `NsNcNt` and
/// `NsNlNcNt` strictly enough to avoid false positives on arbitrary
/// input. Returns `true` when `s` contains no `n` axis letter AND has
/// the legacy axis letters (`s`, `c`, `t`) in order, each preceded by
/// at least one digit. Inputs like `"sched"` or `"abcs"` — which
/// satisfy the older loose `contains('s') && !contains('n')` check —
/// fall through here because their `s` is not preceded by a digit or
/// the remaining axis letters are missing.
fn looks_like_legacy_topo(s: &str) -> bool {
    if s.contains('n') {
        return false;
    }
    let Some(s_pos) = s.find('s') else {
        return false;
    };
    let Some(c_pos) = s.find('c') else {
        return false;
    };
    let Some(t_pos) = s.find('t') else {
        return false;
    };
    if s_pos >= c_pos || c_pos >= t_pos {
        return false;
    }
    // Digit must precede each axis letter: `NsNcNt` / `NsNlNcNt`.
    // The span between `s` and `c` allows an optional `l` axis (LLCs)
    // with digits on both sides so the `NsNlNcNt` form (e.g.
    // `1s2l4c2t`) still trips the warn — the earlier
    // digit-only-between-s-and-c check missed it and the docstring
    // and warn message lied about the coverage.
    let prefix_is_digits =
        |slice: &str| !slice.is_empty() && slice.chars().all(|c| c.is_ascii_digit());
    let s_to_c = &s[s_pos + 1..c_pos];
    let s_to_c_ok = if let Some(l_pos) = s_to_c.find('l') {
        prefix_is_digits(&s_to_c[..l_pos]) && prefix_is_digits(&s_to_c[l_pos + 1..])
    } else {
        prefix_is_digits(s_to_c)
    };
    prefix_is_digits(&s[..s_pos]) && s_to_c_ok && prefix_is_digits(&s[c_pos + 1..t_pos])
}

/// Parse a topology string in "NnNlNcNt" form (e.g. `"1n2l4c2t"`).
/// `n` = NUMA nodes, `l` = LLCs per node, `c` = cores per LLC,
/// `t` = threads per core. Returns `(numa_nodes, llcs, cores, threads)`
/// or `None` on parse failure — nonzero fields, canonical axis order,
/// and no trailing characters after the `t` terminator.
///
/// The legacy `NsNcNt` and `NsNlNcNt` forms (pre-1.0, `s` axis
/// letter) are flagged with a [`tracing::warn!`] before returning
/// `None`, so stale cached args surface instead of silently failing.
pub(crate) fn parse_topo_string(s: &str) -> Option<(u32, u32, u32, u32)> {
    // Legacy-form heuristic: require the full axis structure with
    // the `s` axis letter (digit-s, digit-c, digit-t in order) and
    // no `n` axis letter anywhere. The earlier
    // `s.contains('s') && !s.contains('n')` tripped on any string
    // with an 's' and no 'n' (e.g. "sched", "abcs"); guarding on
    // the structural pattern keeps false positives off the tracing
    // stream when users pass arbitrary garbage.
    if looks_like_legacy_topo(s) {
        tracing::warn!(
            topo = s,
            "legacy NsNcNt / NsNlNcNt topology form detected; use NnNlNcNt (e.g. 1n2l4c2t)"
        );
    }
    let n_pos = s.find('n')?;
    let l_pos = s.find('l')?;
    let c_pos = s.find('c')?;
    let t_pos = s.find('t')?;
    if n_pos >= l_pos || l_pos >= c_pos || c_pos >= t_pos {
        return None;
    }
    // Reject trailing garbage after the 't' terminator. Silently
    // accepting e.g. "1n2l4c2tEXTRA" as (1,2,4,2) would mask typos
    // and version-skew bugs in cached nextest args and sidecar
    // filenames.
    if s[t_pos + 1..].chars().next().is_some() {
        return None;
    }
    let numa_nodes: u32 = s[..n_pos].parse().ok()?;
    let llcs: u32 = s[n_pos + 1..l_pos].parse().ok()?;
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

    // -- parse_topo_string NnNlNcNt --

    #[test]
    fn parse_topo_basic() {
        assert_eq!(parse_topo_string("1n2l4c2t"), Some((1, 2, 4, 2)));
    }

    #[test]
    fn parse_topo_multi_numa() {
        assert_eq!(parse_topo_string("2n4l8c2t"), Some((2, 4, 8, 2)));
    }

    #[test]
    fn parse_topo_single() {
        assert_eq!(parse_topo_string("1n1l1c1t"), Some((1, 1, 1, 1)));
    }

    #[test]
    fn parse_topo_large() {
        assert_eq!(parse_topo_string("4n16l8c2t"), Some((4, 16, 8, 2)));
    }

    #[test]
    fn parse_topo_double_digit_threads() {
        assert_eq!(parse_topo_string("1n1l1c12t"), Some((1, 1, 1, 12)));
    }

    #[test]
    fn parse_topo_zero_numa() {
        assert!(parse_topo_string("0n2l4c2t").is_none());
    }

    #[test]
    fn parse_topo_zero_llcs() {
        assert!(parse_topo_string("1n0l4c2t").is_none());
    }

    #[test]
    fn parse_topo_zero_cores() {
        assert!(parse_topo_string("1n2l0c2t").is_none());
    }

    #[test]
    fn parse_topo_zero_threads() {
        assert!(parse_topo_string("1n2l4c0t").is_none());
    }

    #[test]
    fn parse_topo_wrong_order() {
        assert!(parse_topo_string("2l1n4c2t").is_none());
    }

    #[test]
    fn parse_topo_missing_suffix() {
        assert!(parse_topo_string("1n2l4c2").is_none());
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
    fn parse_topo_legacy_form_rejected() {
        // The pre-1.0 "NsNcNt" / "NsNlNcNt" forms were dropped when the
        // format consolidated on the explicit NUMA axis. Ensure neither
        // reads as a typo-matching accidental success.
        assert!(parse_topo_string("2s4c2t").is_none());
        assert!(parse_topo_string("1s2l4c2t").is_none());
    }

    #[test]
    fn looks_like_legacy_topo_matches_both_legacy_forms() {
        // Direct unit test for the structural heuristic — both the
        // 3-axis NsNcNt and the 4-axis NsNlNcNt forms must trip the
        // warn. Earlier version only matched the 3-axis form because
        // the `l` in the middle of the span broke the digit-only
        // check between `s` and `c`.
        assert!(looks_like_legacy_topo("2s4c2t"));
        assert!(looks_like_legacy_topo("1s2l4c2t"));
    }

    #[test]
    fn looks_like_legacy_topo_rejects_false_positives() {
        // Structural heuristic must not fire on arbitrary input that
        // happens to contain an `s` without an `n` — the earlier
        // `contains('s') && !contains('n')` check tripped on these.
        assert!(!looks_like_legacy_topo("sched"));
        assert!(!looks_like_legacy_topo("abcs"));
        assert!(!looks_like_legacy_topo(""));
        assert!(!looks_like_legacy_topo("1s"));
        assert!(!looks_like_legacy_topo("1s2c"));
        // Contains `n` → NnNlNcNt form, not a legacy warning target.
        assert!(!looks_like_legacy_topo("1n2l4c2t"));
        // `l` span with empty sides should not pass.
        assert!(!looks_like_legacy_topo("1sl4c2t"));
        assert!(!looks_like_legacy_topo("1s2lc2t"));
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

    #[test]
    fn topo_override_validate_rejects_cpus_per_llc_overflow() {
        // cores * threads > u32::MAX must fail fast with an overflow
        // message, not silently wrap.
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: 1,
            cores: 0x1_0001,
            threads: 0x1_0000,
            memory_mb: 8192,
        };
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cpus_per_llc") && msg.contains("overflows u32"),
            "expected overflow error for cores*threads, got: {msg}"
        );
    }

    #[test]
    fn topo_override_validate_rejects_total_cpus_overflow() {
        // cores * threads fits in u32 (2 * 2 = 4), but
        // llcs * cpus_per_llc overflows (u32::MAX * 4).
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: u32::MAX,
            cores: 2,
            threads: 2,
            memory_mb: 8192,
        };
        let err = t.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("total_cpus") && msg.contains("overflows u32"),
            "expected overflow error for total_cpus, got: {msg}"
        );
    }

    #[test]
    fn topo_override_validate_accepts_max_non_overflowing() {
        // Boundary case: exactly u32::MAX total CPUs — no overflow,
        // but also no realistic hardware. Validator only gates on
        // "can u32 hold this?"; physical host constraints are
        // checked elsewhere (TopologyConstraints::accepts).
        let t = TopoOverride {
            numa_nodes: 1,
            llcs: u32::MAX,
            cores: 1,
            threads: 1,
            memory_mb: 8192,
        };
        t.validate().unwrap();
    }
}
