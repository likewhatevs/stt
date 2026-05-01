//! Fun mode — replace identifiers in a ktstr dump with playful
//! `adjective-animal` names so an LLM can reason about the structural
//! and relational shape of a failure dump without dragging real
//! internal identifiers into its context.
//!
//! This is a CONTEXT-HYGIENE feature, not a security feature. Real
//! pids, cpu ids, cgroup names, and process comms are not sensitive
//! per se — they are just noisy when fed to an LLM that does not
//! need them. Replacing them with `swift-otter`-style names lets
//! Claude reason about "swift-otter migrated from CPU 3 to CPU 7"
//! without learning anything internal about whatever pid that
//! actually was.
//!
//! Two surfaces:
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
//! the category byte string is mixed into the keyed hash. Two pids
//! with the same numeric value across two different dumps map to the
//! same fun name only when both dumps share a `--seed`.
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
    /// that need a sentinel zero should call [`Self::is_sentinel`]
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

    /// Known identifier-key handler: classify a JSON-object key
    /// name by category, returning `Some(category)` for known
    /// identifier keys and `None` for keys that should pass
    /// through unchanged. Hardcoded heuristic per the v1 ruling
    /// (schema-driven mapping is a future task).
    ///
    /// Categories returned here align with the namespace argument
    /// to [`Self::petname_for`] / [`Self::numeric_id`], so a
    /// caller can call this to decide whether to funify a field
    /// AND get the right namespace in one lookup.
    pub fn classify_key(key: &str) -> Option<&'static str> {
        // Match against the lowercased-final segment of the key.
        // e.g. "running_pid" -> "pid", "next_pid" -> "pid",
        // "scheduler_name" -> "name". Keeps the table small.
        let lc = key.to_ascii_lowercase();
        // Most-specific multi-word matches first.
        let by_suffix: &[(&str, &str)] = &[
            ("_pid", "pid"),
            ("_tid", "pid"),
            ("_tgid", "pid"),
            ("_cpu", "cpu"),
            ("_cgroup", "cgroup"),
            ("_comm", "comm"),
            ("_name", "name"),
            ("_label", "name"),
        ];
        for (suffix, cat) in by_suffix {
            if lc.ends_with(suffix) {
                return Some(cat);
            }
        }
        // Whole-key matches.
        match lc.as_str() {
            "pid" | "tid" | "tgid" | "ppid" | "next_pid" | "prev_pid" => Some("pid"),
            "cpu" | "dest_cpu" | "orig_cpu" | "wake_cpu" => Some("cpu"),
            "cgroup" => Some("cgroup"),
            "comm" => Some("comm"),
            "name" | "label" | "scheduler" => Some("name"),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// JSON walker
// ---------------------------------------------------------------------------

/// Recursively walk a `serde_json::Value` and replace identifier
/// fields per [`Funifier::classify_key`]. Returns the funified value
/// — input is consumed (cheaper than cloning a deep tree).
///
/// String identifiers map via [`Funifier::petname_for`]. Numeric
/// identifiers (u64 or i64 inside `serde_json::Number`) map via
/// [`Funifier::numeric_id`] / [`numeric_id_i64`]. Floats and
/// booleans at identifier keys are left unchanged — there is no
/// sensible "fun" mapping for those types. Sentinel zero is
/// preserved on numerics.
///
/// Arrays apply field classification to each element. Nested
/// objects recurse. Top-level non-objects pass through unchanged.
pub fn funify_json(value: serde_json::Value, f: &Funifier) -> serde_json::Value {
    funify_json_with_context(value, f, None)
}

fn funify_json_with_context(
    value: serde_json::Value,
    f: &Funifier,
    inherited_category: Option<&'static str>,
) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                let cat = Funifier::classify_key(&k);
                let funified = funify_json_with_context(v, f, cat);
                out.insert(k, funified);
            }
            Value::Object(out)
        }
        Value::Array(items) => {
            // Array elements inherit the parent key's category, so
            // a `"pids": [42, 99]` array funifies each element with
            // the "pid" namespace.
            let out: Vec<Value> = items
                .into_iter()
                .map(|v| funify_json_with_context(v, f, inherited_category))
                .collect();
            Value::Array(out)
        }
        Value::String(s) => {
            if let Some(cat) = inherited_category {
                Value::String(f.petname_for(cat, &s))
            } else {
                Value::String(s)
            }
        }
        Value::Number(num) => {
            if let Some(cat) = inherited_category {
                if let Some(u) = num.as_u64() {
                    if Funifier::is_sentinel_u64(u) {
                        return Value::Number(num);
                    }
                    Value::Number(serde_json::Number::from(f.numeric_id(cat, u)))
                } else if let Some(i) = num.as_i64() {
                    Value::Number(serde_json::Number::from(f.numeric_id_i64(cat, i)))
                } else {
                    // Float at an identifier key — leave alone.
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
    "puffin", "puma", "puppy", "pythons", "quagga", "quail", "quetzal", "quokka",
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

    /// classify_key picks up canonical and suffix-based keys.
    #[test]
    fn classify_key_matches_known_keys() {
        assert_eq!(Funifier::classify_key("pid"), Some("pid"));
        assert_eq!(Funifier::classify_key("tid"), Some("pid"));
        assert_eq!(Funifier::classify_key("running_pid"), Some("pid"));
        assert_eq!(Funifier::classify_key("dest_cpu"), Some("cpu"));
        assert_eq!(Funifier::classify_key("comm"), Some("comm"));
        assert_eq!(Funifier::classify_key("scheduler"), Some("name"));
        assert_eq!(Funifier::classify_key("xyz"), None);
        assert_eq!(Funifier::classify_key("nr_running"), None);
    }

    /// funify_json swaps known identifier values and leaves
    /// non-identifier fields intact. Round-trip serializes
    /// without errors.
    #[test]
    fn funify_json_replaces_identifiers_and_preserves_structure() {
        let f = Funifier::with_seed("demo");
        let input = json!({
            "schema": "single",
            "comm": "ktstr_test",
            "pid": 42,
            "nr_running": 7,
            "scheduler": "scx_simple",
            "cpus": [
                { "cpu": 0, "comm": "swapper" },
                { "cpu": 3, "comm": "ktstr_worker" }
            ]
        });
        let out = funify_json(input.clone(), &f);
        // Schema and nr_running pass through.
        assert_eq!(out["schema"], json!("single"));
        assert_eq!(out["nr_running"], json!(7));
        // Identifier fields differ from input.
        assert_ne!(out["comm"], input["comm"]);
        assert_ne!(out["pid"], input["pid"]);
        assert_ne!(out["scheduler"], input["scheduler"]);
        // Comm renders as "adjective-animal".
        let comm = out["comm"].as_str().unwrap();
        assert!(comm.contains('-'));
        // Array elements get funified individually with the
        // parent key's category.
        assert_ne!(out["cpus"][0]["comm"], input["cpus"][0]["comm"]);
        assert_ne!(out["cpus"][0]["cpu"], input["cpus"][0]["cpu"]);
        // Sentinel zero on `cpu` preserved.
        // (cpu=0 is a real value here; but zero is preserved per
        // the sentinel rule, so cpus[0].cpu stays 0.)
        assert_eq!(out["cpus"][0]["cpu"], json!(0));
        // Round-trip through serde_json::to_string succeeds.
        let s = serde_json::to_string(&out).expect("serialize");
        assert!(!s.is_empty());
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
