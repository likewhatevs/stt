# Writing Tests

stt supports two kinds of tests:

1. **Data-driven scenarios** -- declared as `Scenario` structs in the
   catalog. These run via `stt vm`.

2. **Integration tests** -- Rust functions annotated with `#[stt_test]`.
   These run via `cargo nextest run`.

Most scheduler testing uses data-driven scenarios. Integration tests
are for cases that need custom host/guest interaction or precise control
over the test lifecycle.
