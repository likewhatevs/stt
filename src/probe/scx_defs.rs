//! sched_ext kernel constants.
//!
//! These mirror the definitions in include/linux/sched/ext.h and
//! tools/sched_ext/include/scx/common.bpf.h. If sched_ext changes
//! these values, update this file.

// ---- DSQ IDs (Dispatch Queue identifiers) ----
// Bits [63:62] encode the DSQ type:
//   11 = LOCAL_ON|cpu   (per-CPU local DSQ, lower 32 bits = CPU)
//   10 = builtin DSQ    (lower 32 bits = specific builtin)
//   0x = user DSQ       (full 64-bit user-defined ID)

/// Bits [63:62] for builtin DSQ identification.
pub const DSQ_TYPE_SHIFT: u32 = 62;
/// LOCAL_ON: dispatch to a specific CPU's local DSQ.
pub const DSQ_TYPE_LOCAL_ON: u64 = 3;
/// Builtin DSQ type (GLOBAL, LOCAL, INVALID, BYPASS).
pub const DSQ_TYPE_BUILTIN: u64 = 2;

/// Lower 32-bit values for builtin DSQs (when type == DSQ_TYPE_BUILTIN).
pub const DSQ_INVALID: u32 = 0;
pub const DSQ_GLOBAL: u32 = 1;
pub const DSQ_LOCAL: u32 = 2;
pub const DSQ_BYPASS: u32 = 3;

// ---- Enqueue flags (SCX_ENQ_*) ----

pub const ENQ_WAKEUP: u64 = 1 << 0;
pub const ENQ_HEAD: u64 = 1 << 1;
pub const ENQ_PREEMPT: u64 = 1 << 32;
pub const ENQ_REENQ: u64 = 1 << 40;
pub const ENQ_LAST: u64 = 1 << 41;
pub const ENQ_CLEAR_OPSS: u64 = 1 << 56;
pub const ENQ_DSQ_PRIQ: u64 = 1 << 57;
pub const ENQ_NESTED: u64 = 1 << 58;

/// All known enqueue flags with their names.
pub const ENQ_FLAG_NAMES: &[(u64, &str)] = &[
    (ENQ_WAKEUP, "WAKEUP"),
    (ENQ_HEAD, "HEAD"),
    (ENQ_PREEMPT, "PREEMPT"),
    (ENQ_REENQ, "REENQ"),
    (ENQ_LAST, "LAST"),
    (ENQ_CLEAR_OPSS, "CLEAR_OPSS"),
    (ENQ_DSQ_PRIQ, "DSQ_PRIQ"),
    (ENQ_NESTED, "NESTED"),
];

// ---- Exit kinds (SCX_EXIT_*) ----

pub const EXIT_NONE: u64 = 0;
pub const EXIT_DONE: u64 = 1;
pub const EXIT_UNREG: u64 = 64;
pub const EXIT_UNREG_BPF: u64 = 65;
pub const EXIT_UNREG_KERN: u64 = 66;
pub const EXIT_SYSRQ: u64 = 67;
pub const EXIT_ERROR: u64 = 1024;
pub const EXIT_ERROR_BPF: u64 = 1025;
pub const EXIT_ERROR_STALL: u64 = 1026;

/// All known exit kinds with their names.
pub const EXIT_KIND_NAMES: &[(u64, &str)] = &[
    (EXIT_NONE, "NONE"),
    (EXIT_DONE, "DONE"),
    (EXIT_UNREG, "UNREG"),
    (EXIT_UNREG_BPF, "UNREG_BPF"),
    (EXIT_UNREG_KERN, "UNREG_KERN"),
    (EXIT_SYSRQ, "SYSRQ"),
    (EXIT_ERROR, "ERROR"),
    (EXIT_ERROR_BPF, "ERROR_BPF"),
    (EXIT_ERROR_STALL, "ERROR_STALL"),
];

// ---- Task scx flags (task_struct.scx.flags) ----

pub const TASK_QUEUED: u64 = 1 << 0;
pub const TASK_RESET_RUNNABLE_AT: u64 = 1 << 2;
pub const TASK_DEQD_FOR_SLEEP: u64 = 1 << 3;
// State bits [8:9]
pub const TASK_STATE_SHIFT: u32 = 8;
pub const TASK_STATE_MASK: u64 = 3;
pub const TASK_STATE_INIT: u64 = 1;
pub const TASK_STATE_READY: u64 = 2;
pub const TASK_STATE_ENABLED: u64 = 3;
