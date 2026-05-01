//! Fun mode — context-hygiene for failure dumps. Funifies every
//! non-metric value by default so LLM context stays clean: strings
//! and integers under non-metric keys are replaced with
//! deterministic `adjective-animal` names or hashed numeric IDs,
//! while values under metric-allowlisted keys (counts, rates,
//! ratios, byte/duration units, structural enums) pass through
//! unchanged. The result lets an LLM reason about the structural
//! and relational shape of a dump without dragging real internal
//! identifiers into its context.
//!
//! This is a CONTEXT-HYGIENE feature, not a security feature. Real
//! pids, cpu ids, cgroup names, and process comms are not sensitive
//! per se — they are just noisy when fed to an LLM that does not
//! need them. Replacing them with `swift-otter`-style names lets
//! Claude reason about "swift-otter migrated from CPU 3 to CPU 7"
//! without learning anything internal about whatever pid that
//! actually was.
//!
//! # Polarity: metric allowlist
//!
//! The walker funifies **every** value by default and passes through
//! only the values whose containing key is a recognised metric
//! ([`Funifier::is_metric_passthrough`]). This is the inverse of v1's
//! identifier deny-list. A novel identifier-shaped field added to a
//! schema is hidden by default; only counts / rates / ratios /
//! byte-and-duration units / structural enums survive funification.
//! The suffix-based allowlist may over-match novel keys ending in
//! structural-enum suffixes (`_type`, `_kind`, `_state`, `_len`,
//! `_offset`) — schema-driven classification is a future direction
//! that would remove the heuristic's false positives.
//!
//! # Surfaces
//!
//!   - [`Funifier::petname_for`] turns a string identifier (cgroup
//!     name, process comm, scheduler name, ...) into a deterministic
//!     `adjective-animal` pair like `"swift-otter"`.
//!   - [`Funifier::numeric_id`] turns a u64 identifier (pid, tid, cpu,
//!     cgroup id, ...) into another u64 via an HMAC-keyed permutation.
//!     The mapping is deterministic per `(seed, category, n)` so
//!     cross-references inside a dump survive.
//!
//! Categories namespace the mapping: `petname_for("pid", "42")` and
//! `petname_for("cgroup", "42")` produce different fun names because
//! the category byte string is mixed into the keyed hash. The walker
//! uses each non-metric key's literal name as the namespace, so two
//! values under the same key collide deterministically (intentional —
//! cross-reference preservation) and two values under different keys
//! don't. Two pids with the same numeric value across two different
//! dumps map to the same fun name only when both dumps share a
//! `--seed`.
//!
//! Determinism contract: given a fixed seed, the same input always
//! produces the same fun output. With the default
//! [`Funifier::ephemeral`] constructor a fresh random key is
//! generated per process invocation; `--seed` on the CLI passes
//! through to [`Funifier::with_seed`] so a user can correlate fun
//! names across multiple `funify` runs of the same dump.

use std::hash::Hasher;

use sha2::{Digest, Sha256};
use siphasher::sip128::{Hasher128, SipHasher24};

/// Fixed pepper mixed into seed-derived keys so two users picking
/// the same `--seed` value get a different keyed mapping than each
/// other unless they also coordinate the pepper. Burned into the
/// binary on purpose — no need to make this configurable, the
/// determinism contract is "same seed within one binary" not "same
/// seed across the world".
const FUN_PEPPER: &[u8] = b"ktstr-fun-mode/v1";

/// All-vCPU fun-mode key + petname dictionary handle. Cheap to
/// clone (everything inside is `Copy` or `'static`); typically
/// constructed once per CLI invocation and reused for every
/// identifier in the dump.
#[derive(Clone, Debug)]
pub struct Funifier {
    /// 16-byte SipHash key. SipHash-2-4 is a keyed PRF; 128-bit key
    /// is enough for the LLM-context-hygiene goal (we are not
    /// defending against an attacker, only against accidental
    /// context pollution). Derived either from a CSPRNG
    /// ([`Self::ephemeral`]) or from an HMAC of a user-supplied
    /// seed plus [`FUN_PEPPER`] ([`Self::with_seed`]).
    key: [u8; 16],
}

impl Funifier {
    /// Construct a Funifier with a process-fresh random key. Two
    /// invocations in the same process give DIFFERENT mappings —
    /// callers who need cross-invocation determinism use
    /// [`Self::with_seed`] instead. Used by callers that just want
    /// "produce a fun version of this output" without any need to
    /// reproduce the mapping later.
    ///
    /// Reads from /dev/urandom via the standard `getrandom`
    /// syscall path (through `rand::thread_rng`).
    pub fn ephemeral() -> Self {
        // SHA-256 over (process pid, monotonic ns) for the
        // ephemeral key. Avoids depending on a specific rand-crate
        // trait import path (rand 0.10's RNG-core trait paths
        // shifted between minor versions); the inputs here are
        // already non-replayable across processes — pid is unique
        // per kernel concurrent-life, ns timestamp gives 64-bit
        // intra-process distinctness. SHA-256 then mixes those
        // into a 16-byte key with adequate avalanche for the
        // context-hygiene goal.
        let pid = std::process::id() as u64;
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mut h = Sha256::new();
        h.update(FUN_PEPPER);
        h.update([0u8]);
        h.update(b"ephemeral");
        h.update([0u8]);
        h.update(pid.to_le_bytes());
        h.update(ns.to_le_bytes());
        let digest = h.finalize();
        let mut key = [0u8; 16];
        key.copy_from_slice(&digest[..16]);
        Self { key }
    }

    /// Construct a Funifier whose mapping is fully determined by
    /// `seed`. Two invocations with the same `seed` (in the same
    /// binary build) produce identical fun names for the same
    /// inputs. Different seeds give independent mappings.
    ///
    /// Uses SHA-256 over the fixed [`FUN_PEPPER`] || seed bytes,
    /// truncated to 128 bits for SipHash. Not cryptographic but
    /// sufficient for the deterministic-mapping contract.
    pub fn with_seed(seed: &str) -> Self {
        let mut h = Sha256::new();
        h.update(FUN_PEPPER);
        h.update([0u8]);
        h.update(seed.as_bytes());
        let digest = h.finalize();
        let mut key = [0u8; 16];
        key.copy_from_slice(&digest[..16]);
        Self { key }
    }

    /// Internal: keyed 128-bit hash over (`category` || NUL ||
    /// `payload`). The NUL byte separator guarantees that
    /// `("pid", "42")` and `("pi", "d42")` yield distinct hashes
    /// even with concatenation (no length prefix needed because
    /// every category we use is a fixed-shape ASCII identifier
    /// that does not embed NUL).
    fn keyed_hash(&self, category: &[u8], payload: &[u8]) -> u128 {
        // SipHasher24::new_with_key takes 16 bytes — match.
        let mut h = SipHasher24::new_with_key(&self.key);
        h.write(category);
        h.write(&[0u8]);
        h.write(payload);
        h.finish128().as_u128()
    }

    /// Replace a string identifier with a deterministic
    /// `adjective-animal` pair. The 65 536 (adjective, animal)
    /// pairs the dictionary supports give a comfortable margin for
    /// dumps with hundreds of distinct identifiers per category —
    /// the birthday-paradox collision probability for 100 names
    /// drawn from 65k buckets is ~7%, for 50 names ~2%. A future
    /// extension could append a 4-digit suffix on collision; for
    /// v1 we accept the rare collision.
    ///
    /// Examples (with a fixed seed):
    /// ```ignore
    /// let f = Funifier::with_seed("demo");
    /// f.petname_for("comm", "ktstr_test_main");  // "swift-otter"
    /// f.petname_for("comm", "scx_simple");       // "fluffy-badger"
    /// ```
    pub fn petname_for(&self, category: &str, payload: &str) -> String {
        let h = self.keyed_hash(category.as_bytes(), payload.as_bytes());
        let adj_idx = (h & 0xff) as usize;
        let ani_idx = ((h >> 8) & 0xff) as usize;
        let adj = ADJECTIVES[adj_idx % ADJECTIVES.len()];
        let ani = ANIMALS[ani_idx % ANIMALS.len()];
        format!("{adj}-{ani}")
    }

    /// Replace a u64 identifier with another u64. The mapping is a
    /// deterministic permutation per (seed, category): the keyed
    /// hash mixes (category, n.to_le_bytes()), and we take the low
    /// 64 bits as the new identifier.
    ///
    /// This is NOT format-preserving encryption — we are not
    /// defending against an attacker who has the corpus and is
    /// trying to reverse the mapping. The user explicitly framed
    /// fun mode as "nothing is sensitive to begin with, but like,
    /// why risk it" / context hygiene for LLMs, NOT a security
    /// feature.
    ///
    /// Two distinct `(category, n)` inputs collide on the same
    /// output u64 with probability ~2^-64. Within a single
    /// category, n=0 always maps to 0 is NOT guaranteed; consumers
    /// that need a sentinel zero should call [`Self::is_sentinel_u64`]
    /// or carry the original value out-of-band.
    pub fn numeric_id(&self, category: &str, n: u64) -> u64 {
        let h = self.keyed_hash(category.as_bytes(), &n.to_le_bytes());
        // Take the low 64 bits. The high 64 bits are discarded —
        // SipHash's avalanche means either half is uniformly
        // distributed conditional on the input.
        h as u64
    }

    /// Replace an i64 identifier (e.g. a kernel pid_t which is
    /// signed). Same contract as [`Self::numeric_id`] but
    /// preserves the i64 zero (since dumps frequently use 0 or
    /// -1 as sentinels). Negative values are funified by their
    /// absolute value; the sign survives.
    pub fn numeric_id_i64(&self, category: &str, n: i64) -> i64 {
        if n == 0 {
            return 0;
        }
        let abs = n.unsigned_abs();
        // Mask to 63 bits so the result always fits in i64.
        let funified = (self.numeric_id(category, abs) & ((1u64 << 63) - 1)) as i64;
        if n < 0 { -funified } else { funified }
    }

    /// True when the given identifier is "obvious sentinel" — 0
    /// or "max" — and should be passed through unchanged. Lets
    /// downstream renderers preserve the failure-dump's "kthread"
    /// vs "pid 0" semantics without leaking real pids.
    pub fn is_sentinel_u64(n: u64) -> bool {
        n == 0 || n == u64::MAX
    }

    /// Allowlist gate for the funify walker: returns `true` when
    /// the JSON-object key holds a value that is a METRIC (count,
    /// rate, ratio, byte/duration unit, structural enum) and
    /// should pass through funification unchanged. Returns `false`
    /// for everything else — those values get funified.
    ///
    /// Inverted polarity vs. v1: previously a deny-list of known
    /// identifier keys (pid/cpu/cgroup/...) selected the funify
    /// path. The deny-list missed every novel identifier-shaped
    /// field as the schema grew. The allowlist makes the safe
    /// default "funify it" — any new or unrecognised field is
    /// hidden by default, only metrics whose values are
    /// numeric/categorical truth (and therefore safe to retain)
    /// pass through.
    ///
    /// Match strategy:
    ///   * lowercased-key whole-match against a fixed structural
    ///     vocabulary (schema/version/type/kind/status/...);
    ///   * suffix-match against unit/quantity vocabulary
    ///     (_count/_total/_per_sec/_ns/_bytes/_ratio/_pct/...);
    ///   * everything else returns false.
    ///
    /// Returns true when `key` names a metric value.
    pub fn is_metric_passthrough(key: &str) -> bool {
        let lc = key.to_ascii_lowercase();

        // Whole-key allowlist. Structural enums, schema markers,
        // top-level kernel/runqueue counters, and other named
        // metrics whose value is numeric/categorical truth.
        if matches!(
            lc.as_str(),
            "schema"
                | "version"
                | "type"
                | "kind"
                | "status"
                | "state"
                | "result"
                | "verdict"
                | "outcome"
                | "phase"
                | "policy"
                | "priority"
                | "nice"
                | "weight"
                | "capacity"
                | "size"
                | "len"
                | "length"
                | "depth"
                | "index"
                | "idx"
                | "level"
                | "tier"
                | "rank"
                | "slot"
                | "epoch"
                | "generation"
                | "nr_running"
                | "nr_queued"
                | "nr_failed"
                | "nr_switches"
                | "runqueue_depth"
                // NUMA event counters (vm_numa_event)
                | "numa_hit" | "numa_miss" | "numa_foreign" | "numa_interleave_hit" | "numa_local" | "numa_other"
                // SCX event counters (scx_exit_info)
                | "select_cpu_fallback" | "dispatch_local_dsq_offline" | "dispatch_keep_last" | "enq_skip_exiting" | "enq_skip_migration_disabled" | "reenq_immed" | "reenq_local_repeat" | "refill_slice_dfl" | "bypass_duration" | "bypass_dispatch" | "bypass_activate" | "insert_not_owned" | "sub_bypass_dispatch"
                // BPF prog runtime stats
                | "cnt" | "nsecs" | "misses" | "verified_insns"
                // Hardware perf counters
                | "cycles" | "instructions" | "cache_misses" | "branch_misses"
                // Per-rq SCX state
                | "flags" | "ops_qseq" | "kick_sync" | "nr_immed" | "rq_clock"
                // DSQ state
                | "nr" | "seq"
                // Task enrichment
                | "nr_threads" | "prio" | "static_prio" | "normal_prio" | "nvcsw" | "nivcsw" | "signal_nvcsw" | "signal_nivcsw"
                // VirtioBlkCounters disk metrics
                | "bytes_read" | "bytes_written" | "io_errors"
        ) {
            return true;
        }

        // Suffix allowlist. Quantity / rate / ratio / unit
        // vocabulary the failure-dump schemas use across
        // VirtioBlkCounters, FailureDumpReport, ctprof samples,
        // and the topology/cgroup-stats trees.
        const METRIC_SUFFIXES: &[&str] = &[
            "_count",
            "_total",
            "_completed",
            "_dropped",
            "_failed",
            "_skipped",
            "_throttled",
            "_read",
            "_written",
            "_errors",
            "_per_sec",
            "_per_ms",
            "_rate",
            "_hz",
            "_ratio",
            "_fraction",
            "_pct",
            "_percent",
            "_ns",
            "_us",
            "_ms",
            "_sec",
            "_seconds",
            "_bytes",
            "_kb",
            "_mb",
            "_gb",
            "_pages",
            "_min",
            "_max",
            "_mean",
            "_avg",
            "_stddev",
            "_p50",
            "_p90",
            "_p95",
            "_p99",
            "_capacity",
            "_size",
            "_depth",
            "_len",
            "_length",
            "_weight",
            "_nice",
            "_priority",
            "_index",
            "_idx",
            "_offset",
            "_generation",
            "_epoch",
            "_version",
            "_status",
            "_state",
            "_kind",
            "_type",
            "_phase",
            "_verdict",
            "_outcome",
        ];
        for suffix in METRIC_SUFFIXES {
            if lc.ends_with(suffix) {
                return true;
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// JSON walker
// ---------------------------------------------------------------------------

/// Recursively walk a `serde_json::Value` and funify every value
/// whose containing key is NOT in [`Funifier::is_metric_passthrough`].
/// Returns the funified value — input is consumed (cheaper than
/// cloning a deep tree).
///
/// Inverted polarity (metric allowlist): the default action is
/// "funify it" — a value passes through unchanged ONLY when its
/// containing key is a metric (count/rate/ratio/byte/duration/
/// structural enum). Any other field — pid, comm, cgroup_path,
/// scheduler name, version string, novel identifier-shaped key
/// the schema didn't have last week — gets replaced.
///
/// Funification rules at the leaves:
/// * **String** under a non-metric key — replaced via
///   [`Funifier::petname_for`] using the key name itself as the
///   namespace. Two distinct keys with the same string value get
///   different fun names; the same key + same value yields the
///   same fun name everywhere in the dump (cross-reference
///   preservation).
/// * **Integer** (u64 or i64) under a non-metric key — replaced
///   via [`Funifier::numeric_id`] / [`Funifier::numeric_id_i64`]
///   with the key name as namespace. Sentinel zero and `u64::MAX`
///   pass through unchanged ([`Funifier::is_sentinel_u64`]); the
///   i64 path also preserves zero per [`Funifier::numeric_id_i64`].
/// * **Float** — always passes through. Floats are quasi-
///   exclusively rates/ratios/durations in the dump schemas
///   (cpu_time_fraction, wakeups_per_sec, ...) and there is no
///   sensible fun mapping for IEEE-754 values; making the rule
///   uniform avoids hazarding the rate/ratio metrics that happen
///   to live under non-metric-keyed parents (e.g. inside an
///   anonymous-object array element).
/// * **Bool / null** — always pass through.
///
/// Recursive rules:
/// * **Object** — re-classify each key independently. Nested
///   objects do NOT inherit metric state across the boundary.
/// * **Array** — children inherit the parent key's
///   metric/non-metric verdict and (when non-metric) the parent
///   key's namespace. So `"pids": [1, 2, 3]` funifies each int
///   under namespace "pids" and `"counters": [...]` passes every
///   element through.
pub fn funify_json(value: serde_json::Value, f: &Funifier) -> serde_json::Value {
    funify_json_with_context(value, f, None)
}

/// `category` semantics:
/// * `Some(key)` — value sits under a NON-metric key whose name
///   is `key`; leaves get funified using `key` as the namespace.
/// * `None` — value sits at the root or under a metric key;
///   leaves pass through unchanged.
fn funify_json_with_context(
    value: serde_json::Value,
    f: &Funifier,
    category: Option<&str>,
) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                // Re-classify each key independently. Metric ⇒
                // descendants pass through (`None`); non-metric
                // ⇒ descendants funify under `k`'s namespace.
                let child_cat: Option<&str> = if Funifier::is_metric_passthrough(&k) {
                    None
                } else {
                    Some(k.as_str())
                };
                let funified = funify_json_with_context(v, f, child_cat);
                out.insert(k, funified);
            }
            Value::Object(out)
        }
        Value::Array(items) => {
            // Inherit the parent key's category verbatim. An
            // array under a metric key passes through; an array
            // under a non-metric key funifies each element using
            // the parent's name as namespace.
            let out: Vec<Value> = items
                .into_iter()
                .map(|v| funify_json_with_context(v, f, category))
                .collect();
            Value::Array(out)
        }
        Value::String(s) => {
            if let Some(cat) = category {
                Value::String(f.petname_for(cat, &s))
            } else {
                Value::String(s)
            }
        }
        Value::Number(num) => {
            // Floats always pass through (see module doc) — check
            // first so the u64/i64 cascade only runs for integer
            // numbers.
            if num.is_f64() {
                return Value::Number(num);
            }
            // Sentinel preservation applies universally — even
            // at a non-metric key, 0 and u64::MAX retain their
            // sentinel meaning (kthread pid 0, "no value"
            // u64::MAX) so failure-dump renderers downstream
            // don't have to special-case the funified bytes.
            if let Some(cat) = category {
                if let Some(u) = num.as_u64() {
                    if Funifier::is_sentinel_u64(u) {
                        return Value::Number(num);
                    }
                    Value::Number(serde_json::Number::from(f.numeric_id(cat, u)))
                } else if let Some(i) = num.as_i64() {
                    // numeric_id_i64 itself preserves zero.
                    Value::Number(serde_json::Number::from(f.numeric_id_i64(cat, i)))
                } else {
                    // Defensive: serde_json::Number variants are
                    // u64/i64/f64; the float case is handled above.
                    Value::Number(num)
                }
            } else {
                Value::Number(num)
            }
        }
        // Booleans, null pass through.
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Petname dictionary
// ---------------------------------------------------------------------------
//
// 256 adjectives + 256 animals = 65 536 distinct (adjective, animal)
// pairs. Words are common-language, public-domain, single-word
// (no spaces or hyphens) so the rendered name is always a clean
// `adjective-animal` token suitable for downstream tooling.
//
// Word lists curated for ktstr's costume-party direction:
// playful, recognizable, no edge-cases (no profanity, no political,
// no unusual spellings). The order is fixed for the lifetime of
// this v1 — adding new words to the END is safe; reordering would
// break the determinism contract for callers using a fixed seed.

const ADJECTIVES: &[&str] = &[
    "able", "agile", "airy", "amber", "ample", "amused", "ancient", "angry",
    "antsy", "apt", "ardent", "arid", "ashen", "auburn", "aware", "awesome",
    "balmy", "bashful", "beaded", "beamy", "bendy", "best", "big", "bitter",
    "black", "blameless", "blazing", "bleached", "blissful", "blithe", "blocky", "bloomy",
    "blue", "blunt", "bold", "bony", "bouncy", "brainy", "brassy", "brave",
    "breezy", "bright", "brisk", "bristly", "brittle", "broad", "bronze", "brown",
    "bubbly", "burly", "busy", "buttery", "calm", "candid", "casual", "cheery",
    "chilly", "chipper", "chubby", "chummy", "civic", "classy", "clean", "clear",
    "clever", "cloudy", "clumsy", "coiled", "cold", "comfy", "cool", "copper",
    "cosmic", "cozy", "crafty", "crimson", "crisp", "crystal", "curious", "dainty",
    "damp", "dapper", "daring", "dark", "dashing", "dazed", "deep", "deft",
    "delft", "dewy", "dim", "dimpled", "dingy", "dippy", "distant", "dizzy",
    "dopey", "dotted", "drafty", "dreamy", "dressy", "drowsy", "dry", "dual",
    "dulcet", "dusty", "eager", "early", "easy", "eclectic", "edgy", "eerie",
    "elastic", "elated", "elder", "electric", "elfin", "emerald", "empty", "endless",
    "ethereal", "even", "exact", "fabled", "faint", "fancy", "fawn", "fearless",
    "feisty", "ferny", "festive", "fey", "fierce", "fiery", "filmy", "fine",
    "fizzy", "flat", "fleet", "fleeting", "flighty", "flinty", "floaty", "floral",
    "flowy", "fluffy", "fluted", "foamy", "fond", "foppish", "frank", "fresh",
    "fretful", "frilly", "frisky", "frosty", "frugal", "fudgy", "funky", "furry",
    "fuzzy", "gallant", "game", "gawky", "gentle", "genuine", "ghostly", "giddy",
    "giggly", "glad", "glassy", "gleaming", "glib", "global", "glossy", "glowing",
    "glum", "golden", "good", "goopy", "gossamer", "graceful", "grainy", "grand",
    "grassy", "great", "green", "grim", "groovy", "grown", "grumpy", "gummy",
    "gusty", "hale", "halting", "handy", "happy", "hardy", "harmless", "hasty",
    "hazy", "heady", "hearty", "heavy", "helpful", "high", "hilly", "hippy",
    "hoarse", "hollow", "holy", "homely", "honest", "hooked", "hopeful", "hot",
    "humble", "hungry", "icy", "ideal", "iffy", "immense", "indigo", "inland",
    "inner", "ironic", "itchy", "ivory", "jade", "jaunty", "jazzy", "jelly",
    "jiffy", "jiggly", "jolly", "jovial", "joyful", "jumpy", "kelpy", "keen",
    "kind", "kindly", "kinetic", "knotty", "lacy", "ladylike", "lambent", "lanky",
    "lapis", "large", "late", "lavish", "lawful", "lazy", "leafy", "lean",
    "lemony", "lenient", "level", "lifelong", "light", "lily", "linen", "linked",
    "lithe", "little", "lively", "loamy", "lofty", "long", "loud", "lovely",
];

const ANIMALS: &[&str] = &[
    "aardvark", "albatross", "alligator", "alpaca", "ant", "antelope", "ape", "armadillo",
    "ass", "auk", "axolotl", "baboon", "badger", "bandicoot", "barnacle", "barracuda",
    "basilisk", "bat", "bear", "beaver", "bee", "beetle", "bison", "blackbird",
    "boar", "bobcat", "bonobo", "boomslang", "buffalo", "bulldog", "bullfrog", "bumblebee",
    "bushbaby", "butterfly", "buzzard", "camel", "canary", "capybara", "caracal", "cardinal",
    "caribou", "carp", "cat", "caterpillar", "catfish", "centaur", "centipede", "chameleon",
    "cheetah", "chickadee", "chicken", "chihuahua", "chinchilla", "chipmunk", "civet", "clam",
    "cobra", "cockatoo", "cod", "coral", "cougar", "cow", "coyote", "crab",
    "crane", "crayfish", "cricket", "crocodile", "crow", "cuckoo", "curlew", "cuttlefish",
    "dachshund", "dalmatian", "deer", "dingo", "dodo", "dog", "dolphin", "donkey",
    "dormouse", "dove", "dragon", "dragonfly", "drake", "duck", "dugong", "eagle",
    "eel", "egret", "elephant", "elk", "emu", "ermine", "falcon", "fawn",
    "ferret", "finch", "firefly", "fish", "flamingo", "flatfish", "flounder", "fly",
    "flycatcher", "fowl", "fox", "frog", "fulmar", "gannet", "gar", "gazelle",
    "gecko", "gerbil", "gibbon", "giraffe", "gnat", "gnu", "goat", "goldfish",
    "goose", "gopher", "gorilla", "goshawk", "grasshopper", "greyhound", "grouse", "guanaco",
    "gull", "guppy", "haddock", "hagfish", "halibut", "hamster", "hare", "harrier",
    "hawk", "hedgehog", "hen", "heron", "herring", "hippo", "hognose", "hornet",
    "horse", "hound", "hyena", "ibex", "ibis", "iguana", "impala", "jackal",
    "jackrabbit", "jaguar", "javelina", "jay", "jellyfish", "kangaroo", "katydid", "kestrel",
    "kingfisher", "kite", "kiwi", "koala", "kookaburra", "krill", "lamb", "lamprey",
    "langur", "lark", "lemming", "lemur", "leopard", "lion", "lizard", "llama",
    "lobster", "locust", "loon", "louse", "lynx", "macaque", "macaw", "mackerel",
    "magpie", "mallard", "mammoth", "manatee", "mandrill", "marlin", "marmoset", "marmot",
    "marten", "meerkat", "mink", "minnow", "mole", "molly", "mongoose", "monkey",
    "moose", "mosquito", "moth", "mouse", "mule", "muskrat", "narwhal", "newt",
    "nightingale", "ocelot", "octopus", "okapi", "opossum", "orangutan", "orca", "oriole",
    "ostrich", "otter", "owl", "ox", "oyster", "panda", "pangolin", "panther",
    "parakeet", "parrot", "partridge", "peacock", "pelican", "penguin", "perch", "petrel",
    "pheasant", "pig", "pigeon", "piglet", "pika", "pike", "pinscher", "piranha",
    "platypus", "polecat", "pony", "poodle", "porcupine", "porpoise", "possum", "prawn",
    "puffin", "puma", "puppy", "python", "quagga", "quail", "quetzal", "quokka",
    "rabbit", "raccoon", "ram", "rat", "raven", "reindeer", "rhino", "robin",
];

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Same seed → same fun name. Two Funifiers built with the
    /// same seed must agree on every input.
    #[test]
    fn petname_deterministic_per_seed() {
        let a = Funifier::with_seed("demo-seed");
        let b = Funifier::with_seed("demo-seed");
        assert_eq!(
            a.petname_for("comm", "ktstr_test"),
            b.petname_for("comm", "ktstr_test"),
        );
    }

    /// Different categories must produce different fun names for
    /// the SAME payload — pid 42 and cgroup 42 should not collapse
    /// to the same name.
    #[test]
    fn petname_namespaced_by_category() {
        let f = Funifier::with_seed("demo");
        let pid_name = f.petname_for("pid", "42");
        let cg_name = f.petname_for("cgroup", "42");
        // Could rarely collide by chance (1/65536); pin a specific
        // payload pair where the dictionary lookup differs.
        // The seed is fixed, so this is a stable assertion.
        assert_ne!(pid_name, cg_name, "category bytes must namespace the keyed hash");
    }

    /// Petname output is always two non-empty tokens joined by
    /// `-`. Pins the wire shape so a CLI consumer can tokenize.
    #[test]
    fn petname_format_is_adjective_dash_animal() {
        let f = Funifier::with_seed("demo");
        let name = f.petname_for("comm", "anything");
        let parts: Vec<&str> = name.split('-').collect();
        assert_eq!(parts.len(), 2, "expected exactly two segments: {name}");
        assert!(!parts[0].is_empty());
        assert!(!parts[1].is_empty());
        assert!(parts[0].chars().all(|c| c.is_ascii_lowercase()));
        assert!(parts[1].chars().all(|c| c.is_ascii_lowercase()));
    }

    /// Numeric id is deterministic per (seed, category, n).
    #[test]
    fn numeric_id_deterministic() {
        let f = Funifier::with_seed("demo");
        assert_eq!(f.numeric_id("pid", 42), f.numeric_id("pid", 42));
        assert_ne!(f.numeric_id("pid", 42), f.numeric_id("pid", 43));
        assert_ne!(f.numeric_id("pid", 42), f.numeric_id("cgroup", 42));
    }

    /// `numeric_id_i64` preserves zero verbatim (sentinel) and
    /// keeps sign across funification.
    #[test]
    fn numeric_id_i64_preserves_zero_and_sign() {
        let f = Funifier::with_seed("demo");
        assert_eq!(f.numeric_id_i64("pid", 0), 0);
        let pos = f.numeric_id_i64("pid", 42);
        let neg = f.numeric_id_i64("pid", -42);
        assert!(pos > 0);
        assert!(neg < 0);
        assert_eq!(pos, -neg, "abs value must match across signs");
    }

    /// Sentinel u64 values pass through is_sentinel_u64.
    #[test]
    fn is_sentinel_u64_table() {
        assert!(Funifier::is_sentinel_u64(0));
        assert!(Funifier::is_sentinel_u64(u64::MAX));
        assert!(!Funifier::is_sentinel_u64(1));
        assert!(!Funifier::is_sentinel_u64(42));
    }

    /// `is_metric_passthrough` allowlist hits — whole-key
    /// structural vocabulary plus suffix-based unit/quantity
    /// patterns. Pins the allowlist content so a future edit
    /// that drops an entry (and silently un-allowlists a metric)
    /// trips here.
    #[test]
    fn is_metric_passthrough_allowlist_hits() {
        // Whole-key structural vocabulary.
        assert!(Funifier::is_metric_passthrough("schema"));
        assert!(Funifier::is_metric_passthrough("version"));
        assert!(Funifier::is_metric_passthrough("type"));
        assert!(Funifier::is_metric_passthrough("kind"));
        assert!(Funifier::is_metric_passthrough("status"));
        assert!(Funifier::is_metric_passthrough("nr_running"));
        assert!(Funifier::is_metric_passthrough("nr_queued"));
        assert!(Funifier::is_metric_passthrough("runqueue_depth"));
        assert!(Funifier::is_metric_passthrough("nice"));
        assert!(Funifier::is_metric_passthrough("weight"));
        assert!(Funifier::is_metric_passthrough("priority"));

        // Suffix vocabulary — count / rate / unit / ratio.
        assert!(Funifier::is_metric_passthrough("reads_completed"));
        assert!(Funifier::is_metric_passthrough("io_errors_total"));
        assert!(Funifier::is_metric_passthrough("wakeups_per_sec"));
        assert!(Funifier::is_metric_passthrough("memory_max_bytes"));
        assert!(Funifier::is_metric_passthrough("cpu_max_quota_us"));
        assert!(Funifier::is_metric_passthrough("page_locality_ratio"));
        assert!(Funifier::is_metric_passthrough("cpu_time_fraction"));
        assert!(Funifier::is_metric_passthrough("idle_pct"));
        assert!(Funifier::is_metric_passthrough("queue_depth"));
        assert!(Funifier::is_metric_passthrough("buffer_size"));
        assert!(Funifier::is_metric_passthrough("thread_count"));
    }

    /// `is_metric_passthrough` allowlist misses — identifier-
    /// shaped keys that the inverted polarity now FUNIFIES (vs.
    /// v1, which passed through everything not in the
    /// identifier deny-list).
    #[test]
    fn is_metric_passthrough_allowlist_misses() {
        // Keys the v1 deny-list classified as identifiers — now
        // funified through the default-funify path.
        assert!(!Funifier::is_metric_passthrough("pid"));
        assert!(!Funifier::is_metric_passthrough("tid"));
        assert!(!Funifier::is_metric_passthrough("tgid"));
        assert!(!Funifier::is_metric_passthrough("ppid"));
        assert!(!Funifier::is_metric_passthrough("comm"));
        assert!(!Funifier::is_metric_passthrough("cpu"));
        assert!(!Funifier::is_metric_passthrough("cgroup"));
        assert!(!Funifier::is_metric_passthrough("dest_cpu"));
        assert!(!Funifier::is_metric_passthrough("running_pid"));
        assert!(!Funifier::is_metric_passthrough("scheduler"));

        // Known suffix-aliasing gaps. The current allowlist treats
        // `_type`, `_kind`, `_state`, `_len`, `_offset` (and other
        // structural-enum / quantity suffixes) as metric markers,
        // which is sound when the value is a structural enum or
        // numeric quantity but over-matches on identifier-shaped
        // keys whose tail happens to resemble one. The keys below
        // SHOULD funify and DO NOT under the suffix gate. No
        // assertions are added — they would fail today, and the
        // resolution is schema-driven classification rather than
        // encoding a known-bad expectation. Examples observed in
        // the failure-dump and capture schemas:
        //   - `task_type`, `node_type`   — cgroup / NUMA tags whose
        //                                  values are identifier-
        //                                  shaped enums
        //   - `parent_kind`              — task-relationship tag
        //   - `path_len`                 — ends in `_len`, but the
        //                                  sibling `path` carries
        //                                  the actual identifier
        //                                  string
        //   - `mount_offset`             — ends in `_offset`, but
        //                                  co-locates with a
        //                                  mount-point identifier
        // All of the above pass through is_metric_passthrough
        // today. Schema-driven classification (tagging each field's
        // intent at the type level) is a future direction that
        // would remove the suffix heuristic's false positives.

        // Novel identifier-shaped keys the v1 deny-list missed
        // entirely — now funified by default. The suffix heuristic
        // can over-match keys ending in structural-enum suffixes
        // (see the gap comment above); the cases below avoid those
        // suffixes and are reliably hidden.
        assert!(!Funifier::is_metric_passthrough("cgroup_path"));
        assert!(!Funifier::is_metric_passthrough("path"));
        assert!(!Funifier::is_metric_passthrough("hostname"));
        assert!(!Funifier::is_metric_passthrough("xyz"));
    }

    /// Every VirtioBlkCounters field name passes the metric
    /// allowlist. Pinning each name guards against fun mode
    /// silently hiding disk counters in failure dumps when a
    /// future allowlist edit drops a suffix or whole-key entry
    /// these counters depend on.
    #[test]
    fn virtio_blk_counter_names_are_metric_passthrough() {
        for name in [
            "reads_completed",
            "writes_completed",
            "flushes_completed",
            "bytes_read",
            "bytes_written",
            "throttled_count",
            "io_errors",
        ] {
            assert!(
                Funifier::is_metric_passthrough(name),
                "{name} must be metric",
            );
        }
    }

    /// funify_json funifies non-metric-keyed values and
    /// preserves metric-keyed values. The input mixes both
    /// classes plus an array of objects to exercise every
    /// walker path.
    #[test]
    fn funify_json_funifies_non_metric_keys_and_preserves_metrics() {
        let f = Funifier::with_seed("demo");
        let input = json!({
            "schema": "single",
            "version": "1.2.3",
            "comm": "ktstr_test",
            "pid": 42,
            "nr_running": 7,
            "scheduler": "scx_simple",
            "wakeups_per_sec": 500.0,
            "thread_count": 4,
            "cpus": [
                { "cpu": 1, "comm": "swapper" },
                { "cpu": 3, "comm": "ktstr_worker" }
            ]
        });
        let out = funify_json(input.clone(), &f);

        // Metric-keyed values pass through unchanged.
        assert_eq!(out["schema"], json!("single"));
        assert_eq!(out["version"], json!("1.2.3"));
        assert_eq!(out["nr_running"], json!(7));
        assert_eq!(out["wakeups_per_sec"], json!(500.0));
        assert_eq!(out["thread_count"], json!(4));

        // Non-metric-keyed values get funified.
        assert_ne!(out["comm"], input["comm"]);
        assert_ne!(out["pid"], input["pid"]);
        assert_ne!(out["scheduler"], input["scheduler"]);

        // String funification renders as "adjective-animal".
        let comm = out["comm"].as_str().unwrap();
        assert!(
            comm.contains('-'),
            "expected adjective-animal token, got {comm:?}",
        );

        // Array of objects: each object's keys are independently
        // re-classified. cpu and comm are non-metric so they get
        // funified per element.
        assert_ne!(out["cpus"][0]["comm"], input["cpus"][0]["comm"]);
        assert_ne!(out["cpus"][1]["comm"], input["cpus"][1]["comm"]);
        // cpu=1 is non-sentinel so funification swaps it.
        assert_ne!(out["cpus"][0]["cpu"], input["cpus"][0]["cpu"]);
        assert_ne!(out["cpus"][1]["cpu"], input["cpus"][1]["cpu"]);

        // Round-trip through serde_json::to_string succeeds.
        let s = serde_json::to_string(&out).expect("serialize");
        assert!(!s.is_empty());
    }

    /// Sentinel preservation under non-metric keys: `cpu: 0`
    /// stays 0, `pid: u64::MAX` stays u64::MAX. Sentinels carry
    /// kthread / "no value" semantics that downstream renderers
    /// must still see.
    #[test]
    fn funify_json_preserves_numeric_sentinels() {
        let f = Funifier::with_seed("demo");
        let input = json!({
            "cpu": 0,
            "pid": u64::MAX,
            "tid": 1,
        });
        let out = funify_json(input.clone(), &f);
        // Sentinel u64 zero preserved (cpu).
        assert_eq!(out["cpu"], json!(0));
        // Sentinel u64::MAX preserved (pid).
        assert_eq!(out["pid"], json!(u64::MAX));
        // Non-sentinel funified (tid=1).
        assert_ne!(out["tid"], json!(1));
    }

    /// Floats always pass through, regardless of whether the
    /// containing key is a metric. Non-metric float keys stay
    /// stable because there is no sensible fun mapping for
    /// IEEE-754 values and rates/ratios live everywhere in the
    /// schemas.
    #[test]
    fn funify_json_floats_pass_through_unconditionally() {
        let f = Funifier::with_seed("demo");
        let input = json!({
            "wakeups_per_sec": 500.5,
            "fairness_score": 0.75,
            "anonymous_float": 3.14,
        });
        let out = funify_json(input.clone(), &f);
        assert_eq!(out["wakeups_per_sec"], json!(500.5));
        assert_eq!(out["fairness_score"], json!(0.75));
        assert_eq!(out["anonymous_float"], json!(3.14));
    }

    /// Cross-reference preservation: two values that share both
    /// a key AND a payload yield the same funified output, so
    /// downstream tooling can correlate "same pid mentioned in
    /// two places" without leaking the real pid.
    #[test]
    fn funify_json_cross_reference_within_dump() {
        let f = Funifier::with_seed("demo");
        let input = json!({
            "running": [
                { "pid": 100 },
                { "pid": 100 },
                { "pid": 200 }
            ]
        });
        let out = funify_json(input, &f);
        let p0 = &out["running"][0]["pid"];
        let p1 = &out["running"][1]["pid"];
        let p2 = &out["running"][2]["pid"];
        assert_eq!(p0, p1, "same key + same value must funify identically");
        assert_ne!(p0, p2, "same key + different value must differ");
    }

    /// Array elements inherit the parent key's category. A
    /// non-metric parent key (e.g. `pids`) makes every array
    /// element funify under that key's namespace; a metric
    /// parent key passes every element through.
    #[test]
    fn funify_json_array_inherits_parent_category() {
        let f = Funifier::with_seed("demo");
        let input = json!({
            "pids": [1, 2, 3],
            "completed_per_sec": [10.0, 20.0, 30.0],
        });
        let out = funify_json(input.clone(), &f);
        // Non-metric parent → each element funified.
        for i in 0..3 {
            assert_ne!(out["pids"][i], input["pids"][i]);
        }
        // Metric parent → array passes through.
        assert_eq!(out["completed_per_sec"], input["completed_per_sec"]);
    }

    /// Two seeds produce different mappings for the same input.
    #[test]
    fn distinct_seeds_produce_distinct_mappings() {
        let a = Funifier::with_seed("seed-a");
        let b = Funifier::with_seed("seed-b");
        let na = a.petname_for("comm", "x");
        let nb = b.petname_for("comm", "x");
        // Could rarely collide by chance; assert at least one
        // category differs.
        let na2 = a.numeric_id("pid", 42);
        let nb2 = b.numeric_id("pid", 42);
        assert!(na != nb || na2 != nb2, "two seeds must differ on at least one mapping");
    }

    /// Ephemeral Funifier produces stable names within ITS OWN
    /// process life but two ephemeral instances differ.
    #[test]
    fn ephemeral_within_instance_stable_across_instances_random() {
        let a = Funifier::ephemeral();
        let n1 = a.petname_for("comm", "same");
        let n2 = a.petname_for("comm", "same");
        assert_eq!(n1, n2);
        // Two ephemerals nearly always differ. Compare two
        // different categories to keep the test stable in the
        // 1-in-65536 collision case.
        let b = Funifier::ephemeral();
        let a_bundle = (
            a.petname_for("comm", "same"),
            a.numeric_id("pid", 42),
            a.numeric_id("cgroup", 7),
        );
        let b_bundle = (
            b.petname_for("comm", "same"),
            b.numeric_id("pid", 42),
            b.numeric_id("cgroup", 7),
        );
        assert_ne!(a_bundle, b_bundle, "two ephemeral instances must differ");
    }

    /// Dictionary sizes — pinned so a future word-list edit that
    /// trims either array trips here before downstream callers
    /// see fewer fun names than expected.
    #[test]
    fn dictionary_sizes_pinned_at_256() {
        assert_eq!(ADJECTIVES.len(), 256, "adjective list must be 256 entries");
        assert_eq!(ANIMALS.len(), 256, "animal list must be 256 entries");
    }

    /// Every dictionary entry is non-empty lowercase ASCII (no
    /// spaces, hyphens, or special characters). Guards against a
    /// future word-list addition that breaks the
    /// "adjective-animal" tokenization invariant.
    #[test]
    fn dictionary_entries_are_lowercase_ascii_words() {
        for w in ADJECTIVES.iter().chain(ANIMALS.iter()) {
            assert!(!w.is_empty(), "empty word in dictionary");
            assert!(
                w.chars().all(|c| c.is_ascii_lowercase()),
                "non-lowercase-ASCII word in dictionary: {w:?}",
            );
        }
    }
}
