#!/usr/bin/env bash
#
# package-linux.sh — build offgrid with dynamically-loaded backends and assemble a
# relocatable Linux x86_64 distribution tarball under dist/.
#
# OFFGRID_TOOLCHAIN=zig (default when available): cargo zigbuild against glibc
# $OFFGRID_GLIBC (default 2.28). OFFGRID_TOOLCHAIN=host: needs the host's glibc.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import sys, json; d = json.load(sys.stdin); print(next(p["version"] for p in d["packages"] if p["name"] == "offgrid-ffi"))')"

# Remove previous output first so a failed build can't leave a stale package behind.
rm -rf "$ROOT/dist/offgrid-${VERSION}-linux-x86_64" "$ROOT/dist/offgrid-${VERSION}-linux-x86_64.tar.gz"

GLIBC="${OFFGRID_GLIBC:-2.28}"
ZIG_VERSION="${OFFGRID_ZIG_VERSION:-0.16.0}"
# A mise shim named `zig` can exist without an active version, so test that it runs.
zig_works() { zig version >/dev/null 2>&1; }
mise_zig_works() { mise exec "zig@${ZIG_VERSION}" -- zig version >/dev/null 2>&1; }
TOOLCHAIN="${OFFGRID_TOOLCHAIN:-}"
if [ -z "$TOOLCHAIN" ]; then
    if command -v cargo-zigbuild >/dev/null 2>&1 && { zig_works || mise_zig_works; }; then
        TOOLCHAIN=zig
    else
        TOOLCHAIN=host
    fi
fi
case "$TOOLCHAIN" in
    zig)
        ZIG_RUN=()
        zig_works || ZIG_RUN=(mise exec "zig@${ZIG_VERSION}" --)
        BUILD=("${ZIG_RUN[@]}" cargo zigbuild --target "x86_64-unknown-linux-gnu.${GLIBC}")
        TARGET_DIR="$ROOT/target/x86_64-unknown-linux-gnu/release"
        # CMake can't infer Debian/Ubuntu's multiarch lib dir from zig cc, so FindVulkan misses libvulkan.so.
        if [ -d /usr/lib/x86_64-linux-gnu ]; then
            export CMAKE_LIBRARY_PATH="${CMAKE_LIBRARY_PATH:+$CMAKE_LIBRARY_PATH:}/usr/lib/x86_64-linux-gnu"
        fi
        ;;
    host)
        BUILD=(cargo build)
        TARGET_DIR="$ROOT/target/release"
        ;;
    *) echo "ERROR: OFFGRID_TOOLCHAIN must be zig or host" >&2; exit 1 ;;
esac

echo ">> building offgrid-ffi ${VERSION} with --features dynamic-backends (toolchain: ${TOOLCHAIN})"
# Ask cargo which offgrid-sys build-script output this build actually used; several
# stale offgrid-sys-* dirs (other feature sets) can exist side by side.
SYS_OUT="$("${BUILD[@]}" --release -p offgrid-ffi --features dynamic-backends --message-format=json-render-diagnostics \
    | python3 -c '
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "build-script-executed" and "offgrid-sys" in m.get("package_id", ""):
        out = m["out_dir"]
print(out)')" || { echo "ERROR: build failed or offgrid-sys out dir not reported" >&2; exit 1; }
OUT_DIR="$(dirname "$SYS_OUT")"
if [ ! -f "$OUT_DIR/out/lib/libllama.so" ]; then
    echo "ERROR: $OUT_DIR/out/lib/libllama.so missing (was offgrid-sys built with dynamic-backends?)" >&2
    exit 1
fi
echo ">> using offgrid-sys output: $OUT_DIR"

echo ">> building offgrid-server"
"${BUILD[@]}" --release -p offgrid-server --features dynamic-backends

PKG_NAME="offgrid-${VERSION}-linux-x86_64"
PKG="$ROOT/dist/$PKG_NAME"
TARBALL="$ROOT/dist/$PKG_NAME.tar.gz"

rm -rf "$PKG"
mkdir -p "$PKG/bin" "$PKG/lib" "$PKG/include"

# Core shared libraries: copy under their SONAME names, dereferencing symlinks.
cp -L "$TARGET_DIR/liboffgrid.so"              "$PKG/lib/liboffgrid.so"
cp -L "$TARGET_DIR/offgrid-server"              "$PKG/bin/offgrid-server"
cp -L "$OUT_DIR/out/lib/libllama.so.0"         "$PKG/lib/libllama.so.0"
cp -L "$OUT_DIR/out/lib/libmtmd.so.0"          "$PKG/lib/libmtmd.so.0"
cp -L "$OUT_DIR/out/lib/libggml.so.0"          "$PKG/lib/libggml.so.0"
cp -L "$OUT_DIR/out/lib/libggml-base.so.0"     "$PKG/lib/libggml-base.so.0"

# Backend modules (CPU variants + Vulkan).
shopt -s nullglob
backends=("$OUT_DIR"/out/bin/*.so)
shopt -u nullglob
if [ "${#backends[@]}" -eq 0 ]; then
    echo "ERROR: no backend modules found in $OUT_DIR/out/bin" >&2
    exit 1
fi
cp -L "${backends[@]}" "$PKG/lib/"

cp "$ROOT/crates/offgrid-ffi/include/offgrid.h" "$PKG/include/offgrid.h"

cat > "$PKG/README.txt" <<EOF
offgrid ${VERSION} — Linux x86_64
================================

Contents
  bin/offgrid-server  OpenAI-compatible HTTP server (sidecar). Example:
                       bin/offgrid-server --model model.gguf --port 8080
                     then point any OpenAI client at http://127.0.0.1:8080/v1
                     (see --help: --embedding-model, --parallel, --api-key, --ctx).
  include/offgrid.h   C API header (JSON in/out, streaming callbacks).
  lib/               liboffgrid.so plus its llama.cpp/ggml dependencies and all
                     backend modules. Keep this directory together.

Linking
  cc your_app.c -I<this dir>/include -L<this dir>/lib -loffgrid \\
     -Wl,-rpath,<this dir>/lib
  The shared library has RUNPATH \$ORIGIN, so at runtime it finds
  libllama.so.0, libmtmd.so.0 (vision), libggml.so.0, libggml-base.so.0 and the backend modules in the
  same directory as liboffgrid.so. Distributing the whole tree is sufficient.

Environment variables
  OFFGRID_BACKEND_DIR  Directory to load backend modules from (default: the
                      directory containing liboffgrid.so).
  OFFGRID_LOG          Log level: off (default), error, warn, info, debug, trace.
                      (offgrid_set_log_callback() takes 0=off .. 5=trace instead.)
  OFFGRID_NATIVE       (build time only) tune the CPU backend for the host CPU.

Notes
  - Backends are loaded at runtime; CPU variants cover multiple x86_64 feature
    levels and the Vulkan backend is optional. If libggml-vulkan.so is absent or
    no GPU is available, it falls back to CPU automatically.
EOF

strip --strip-unneeded "$PKG/bin/offgrid-server"
for so in "$PKG"/lib/*.so "$PKG"/lib/*.so.*; do
    strip --strip-unneeded "$so"
done

tar -czf "$TARBALL" -C "$ROOT/dist" "$PKG_NAME"

echo
echo ">> size summary"
du -sh "$PKG" "$TARBALL"
echo ">> lib/ contents"
ls -la "$PKG/lib"
echo ">> done: $TARBALL"
