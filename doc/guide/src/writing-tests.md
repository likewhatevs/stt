# Writing Tests

stt supports two kinds of tests:

1. **Data-driven scenarios** -- declared as `Scenario` structs in the
   catalog. These run via `cargo stt vm` or `cargo stt vm --gauntlet`.

2. **Integration tests** -- Rust functions annotated with `#[stt_test]`.
   These run via `cargo stt test` or `cargo test`.

Most scheduler testing uses data-driven scenarios. Integration tests
are for cases that need custom host/guest interaction or precise control
over the test lifecycle.
