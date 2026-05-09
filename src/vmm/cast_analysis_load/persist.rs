use crate::monitor::cast_analysis::{AddrSpace, CastHit, CastMap};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use super::FwdIndexEntry;

const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct PersistedAddrSpace(u8);

impl From<AddrSpace> for PersistedAddrSpace {
    fn from(a: AddrSpace) -> Self {
        match a {
            AddrSpace::Arena => Self(0),
            AddrSpace::Kernel => Self(1),
        }
    }
}

impl PersistedAddrSpace {
    fn to_addr_space(self) -> Option<AddrSpace> {
        match self.0 {
            0 => Some(AddrSpace::Arena),
            1 => Some(AddrSpace::Kernel),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedCastHit {
    target_type_id: u32,
    addr_space: PersistedAddrSpace,
}

impl From<CastHit> for PersistedCastHit {
    fn from(h: CastHit) -> Self {
        Self {
            target_type_id: h.target_type_id,
            addr_space: h.addr_space.into(),
        }
    }
}

impl PersistedCastHit {
    fn to_cast_hit(self) -> Option<CastHit> {
        Some(CastHit {
            target_type_id: self.target_type_id,
            addr_space: self.addr_space.to_addr_space()?,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedFwdIndexEntry {
    btfs_idx: u32,
    type_id: u32,
}

impl From<&FwdIndexEntry> for PersistedFwdIndexEntry {
    fn from(e: &FwdIndexEntry) -> Self {
        Self {
            btfs_idx: e.btfs_idx as u32,
            type_id: e.type_id,
        }
    }
}

impl PersistedFwdIndexEntry {
    fn to_fwd_index_entry(self) -> FwdIndexEntry {
        FwdIndexEntry {
            btfs_idx: self.btfs_idx as usize,
            type_id: self.type_id,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedCastAnalysis {
    schema_version: u32,
    content_hash: u64,
    cast_entries: Vec<((u32, u32), PersistedCastHit)>,
    fwd_entries: Vec<(String, PersistedFwdIndexEntry)>,
    btf_count: u32,
}

fn cache_dir() -> Option<PathBuf> {
    crate::cache::resolve_cache_root_with_suffix("cast_analysis").ok()
}

fn cache_path(hash: u64) -> Option<PathBuf> {
    cache_dir().map(|d| d.join(format!("{hash:016x}.bin")))
}

pub(super) fn try_load(
    hash: u64,
    expected_btf_count: usize,
) -> Option<(CastMap, HashMap<String, FwdIndexEntry>)> {
    let path = cache_path(hash)?;
    let bytes = std::fs::read(&path).ok()?;
    let (persisted, _): (PersistedCastAnalysis, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).ok()?;

    if persisted.schema_version != SCHEMA_VERSION {
        return None;
    }
    if persisted.content_hash != hash {
        return None;
    }
    if persisted.btf_count as usize != expected_btf_count {
        tracing::debug!(
            expected = expected_btf_count,
            cached = persisted.btf_count,
            "cast_analysis: disk cache btf_count mismatch; treating as miss"
        );
        return None;
    }

    let mut cast_map = BTreeMap::new();
    for (key, hit) in persisted.cast_entries {
        cast_map.insert(key, hit.to_cast_hit()?);
    }

    let mut fwd_index = HashMap::new();
    for (name, entry) in persisted.fwd_entries {
        fwd_index.insert(name, entry.to_fwd_index_entry());
    }

    tracing::info!(
        casts = cast_map.len(),
        fwd = fwd_index.len(),
        path = %path.display(),
        "cast_analysis: loaded from disk cache"
    );
    Some((cast_map, fwd_index))
}

pub(super) fn try_save(
    hash: u64,
    cast_map: &CastMap,
    fwd_index: &HashMap<String, FwdIndexEntry>,
    btf_count: usize,
) {
    let Some(path) = cache_path(hash) else { return };

    let persisted = PersistedCastAnalysis {
        schema_version: SCHEMA_VERSION,
        content_hash: hash,
        cast_entries: cast_map
            .iter()
            .map(|(&k, &v)| (k, v.into()))
            .collect(),
        fwd_entries: fwd_index
            .iter()
            .map(|(k, v)| (k.clone(), v.into()))
            .collect(),
        btf_count: btf_count as u32,
    };

    let encoded = match bincode::serde::encode_to_vec(&persisted, bincode::config::standard()) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "cast_analysis: failed to encode for disk cache");
            return;
        }
    };

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let tmp = path.with_extension(format!("bin.tmp.{}", std::process::id()));
    if std::fs::write(&tmp, &encoded).is_ok() {
        if std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        } else {
            tracing::debug!(
                path = %path.display(),
                bytes = encoded.len(),
                "cast_analysis: saved to disk cache"
            );
        }
    }
}
