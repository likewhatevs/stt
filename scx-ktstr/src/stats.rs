// SPDX-License-Identifier: GPL-2.0-only
//! scx_stats userspace-protocol surface for the scx-ktstr fixture
//! scheduler.
//!
//! Mirrors the pattern used by stock scx_* schedulers (e.g.
//! `scx_bpfland::stats`): a `KtstrStats` struct deriving
//! `scx_stats_derive::Stats`, paired with a `server_data()` helper
//! that wires a `StatsServerData::open` callback to the Sender/
//! Receiver pair the main run loop owns. Each `stats` request the
//! socket receives drives one read of the BPF .bss counters
//! (`nr_dispatched`, `nr_enqueued`, `nr_select_cpu`) and one
//! response over the channel.
//!
//! The default socket path is `/var/run/scx/root/stats` (set by
//! `StatsServer::new`'s defaults); ktstr's host-side
//! `SchedStatsClient` reaches it through the in-guest stats relay
//! that bridges `/dev/vport0p2` to the same Unix socket.

use scx_stats::prelude::*;
use scx_stats_derive::Stats;
use serde::Deserialize;
use serde::Serialize;

/// Cumulative scx-ktstr scheduler counters. Wire shape matches the
/// scx_stats line-delimited JSON envelope: a successful "stats"
/// request returns
/// `{"errno":0,"args":{"resp":{"nr_dispatched":N,"nr_enqueued":N,"nr_select_cpu":N}}}\n`.
///
/// All three counters are monotonic and 64-bit (matching the BPF
/// `volatile u64` declarations in `main.bpf.c`). They are updated
/// atomically via `__sync_fetch_and_add` from the per-CPU ops
/// callbacks; userspace reads them under the BPF .bss accessor
/// without additional locking — the read is atomic with respect to
/// the increment because both sides operate on naturally aligned
/// 64-bit fields and the reader observes whatever value is current
/// at the read instant. Test code asserts directionally
/// (counter increased) rather than against exact values.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Stats)]
#[stat(top)]
pub struct KtstrStats {
    /// Cumulative count of successful `scx_bpf_dsq_move_to_local`
    /// calls in `ktstr_dispatch`. Increments after the move
    /// returns, so `--stall` and `--slow` skip-paths do not bump
    /// the counter.
    #[stat(desc = "Number of successful dispatches via SHARED_DSQ")]
    pub nr_dispatched: u64,
    /// Cumulative count of `ktstr_enqueue` invocations. Bumps on
    /// every callback regardless of which DSQ the task lands in
    /// (SHARED_DSQ vs. SCX_DSQ_LOCAL_ON | cpu under
    /// scattershot/degrade).
    #[stat(desc = "Number of enqueue callbacks observed")]
    pub nr_enqueued: u64,
    /// Cumulative count of `ktstr_select_cpu` invocations.
    #[stat(desc = "Number of select_cpu callbacks observed")]
    pub nr_select_cpu: u64,
}

/// Build the `StatsServerData` instance the scheduler hands to
/// `StatsServer::new`. The "top" verb dispatches each incoming
/// stats request through the channel pair the main run loop owns:
/// the request triggers a fresh BPF .bss read on the userspace
/// thread, which sends the new `KtstrStats` instance back over
/// the response channel.
///
/// No primer / delta computation: tests want raw cumulative
/// counters so the host can assert "increased" rather than
/// "delta during the last interval".
pub fn server_data() -> StatsServerData<(), KtstrStats> {
    let open: Box<dyn StatsOpener<(), KtstrStats>> = Box::new(move |_| {
        let read: Box<dyn StatsReader<(), KtstrStats>> =
            Box::new(move |_args, (req_ch, res_ch)| {
                req_ch.send(())?;
                let cur = res_ch.recv()?;
                cur.to_json()
            });
        Ok(read)
    });

    StatsServerData::new()
        .add_meta(KtstrStats::meta())
        .add_ops("top", StatsOps { open, close: None })
}
