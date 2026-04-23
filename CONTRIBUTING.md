# Contributing to ktstr

Notes for contributors modifying the workspace or its build
configuration. Day-to-day test authoring does not need any of
this.

## liblzma build configuration

ktstr depends on the `xz2` crate with the `static` feature,
which builds `liblzma` from bundled C source during `cargo
build`. The C compiler and autotools listed in the README (see
the "Ubuntu/Debian" / "Fedora" install blocks) are sufficient
for the static build — no separate `liblzma-dev` / `xz-devel`
package is required, and the resulting binary has no runtime
dependency on the host's `liblzma`.

### Switching to the dynamic path

If you modify the workspace to drop the `static` feature on
`xz2`:

1. Install your distro's liblzma development package:
   - Debian / Ubuntu: `liblzma-dev`
   - Fedora: `xz-devel`
2. Ensure `pkg-config` can find it (the package manager's
   install should handle this; if not, inspect
   `PKG_CONFIG_PATH`).

### Why the default is static

The static build keeps CI builds reproducible across host
distros: a `liblzma` ABI bump on one runner no longer silently
shifts tarball-decompression behaviour on another, and the
resulting binary is self-contained enough to copy across
machines without tracking an extra shared-library dependency.
The `ldd` pin test (`tests/ldd_pin.rs`) guards against an
accidental flip away from static by counting dynamic-library
entries — a bump there on any PR needs an explicit
acknowledgement in the commit message.
