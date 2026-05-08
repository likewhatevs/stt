# All commands live here so they are runnable locally and on CI
# identically.  CI YAML handles only checkout, toolchain, cache, and
# tool installation; every `run:` step calls a justfile recipe.

default:
    @just --list

# Format code
fmt:
    cargo fmt --all

# Run all lints (format check + type check + clippy)
lint:
    cargo fmt -- --check
    cargo check --workspace --all-targets
    cargo clippy --workspace --all-targets

# Build a test kernel
kernel-build version="":
    cargo run --bin cargo-ktstr -- ktstr kernel build --skip-sha256 {{version}}

# Run tests against a kernel version
test kernel:
    cargo run --bin cargo-ktstr -- ktstr test --kernel {{kernel}} -- --profile ci --features integration -j $(( $(nproc) * 5 ))

# Run coverage
coverage:
    cargo run --bin cargo-ktstr -- ktstr coverage -- --profile ci --lcov --output-path lcov.info --features integration --exclude-from-report scx-ktstr

# Show sccache statistics
sccache-stats:
    sccache --show-stats

# Show test statistics
stats:
    cargo run --bin cargo-ktstr -- ktstr stats

# Build and link-check the guide book
docs:
    mdbook build doc/guide
    mdbook test doc/guide

# Build API reference
api-docs:
    cargo doc --workspace --no-deps

# Build and serve the guide locally
book-serve:
    mdbook serve doc/guide --open

# Assemble the full documentation site (guide + API docs)
site: docs api-docs
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p _site/guide _site/api
    cp -r doc/guide/book/html/* _site/guide/
    cp -r target/doc/* _site/api/
    cat > _site/index.html <<'HTML'
    <!DOCTYPE html>
    <meta http-equiv="refresh" content="0; url=guide/">
    HTML
