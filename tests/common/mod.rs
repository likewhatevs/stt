//! Shared helpers for ktstr's integration tests.
//!
//! Living under `tests/common/mod.rs` keeps Rust's integration-test
//! harness from picking the file up as its own test binary (any
//! file directly in `tests/` is compiled as a standalone integration
//! target; files under a subdirectory named `common` are not).
//!
//! This module is itself test-only — nothing here is shipped as part
//! of the `ktstr` crate's public API.

pub mod fixtures;
