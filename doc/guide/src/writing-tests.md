# Writing Tests

Tests are Rust functions annotated with `#[stt_test]`. Each test
boots a KVM VM, runs the scenario inside it, and evaluates results
on the host. Run with `cargo nextest run`.
