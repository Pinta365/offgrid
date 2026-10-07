#!/usr/bin/env bash
#
# package-linux.sh — build aicore with dynamically-loaded backends and assemble a
# relocatable Linux x86_64 distribution tarball under dist/.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import sys, json; d = json.load(sys.stdin); print(next(p["version"] for p in d["packages"] if p["name"] == "aicore-ffi"))')"

echo ">> building aicore-ffi ${VERSION} with --features dynamic-backends"
# Ask cargo which aicore-sys build-script output this build actually used; several
# stale aicore-sys-* dirs (other feature sets) can exist side by side.
SYS_OUT="$(cargo build --release -p aicore-ffi --features dynamic-backends --message-format=json-render-diagnostics \
    | python3 -c '
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "build-script-executed" and "aicore-sys" in m.get("package_id", ""):
        out = m["out_dir"]
print(out)')"
OUT_DIR="$(dirname "$SYS_OUT")"
if [ ! -f "$OUT_DIR/out/lib/libllama.so" ]; then
    echo "ERROR: $OUT_DIR/out/lib/libllama.so missing (was aicore-sys built with dynamic-backends?)" >&2
    exit 1
fi
echo ">> using aicore-sys output: $OUT_DIR"

PKG_NAME="aicore-${VERSION}-linux-x86_64"
PKG="$ROOT/dist/$PKG_NAME"
TARBALL="$ROOT/dist/$PKG_NAME.tar.gz"

rm -rf "$PKG"
mkdir -p "$PKG/lib" "$PKG/include"

# Core shared libraries: copy under their SONAME names, dereferencing symlinks.
cp -L "$ROOT/target/release/libaicore.so"      "$PKG/lib/libaicore.so"
cp -L "$OUT_DIR/out/lib/libllama.so.0"         "$PKG/lib/libllama.so.0"
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

cp "$ROOT/crates/aicore-ffi/include/aicore.h" "$PKG/include/aicore.h"

cat > "$PKG/README.txt" <<EOF
aicore ${VERSION} — Linux x86_64
================================

Contents
  include/aicore.h   C API header (JSON in/out, streaming callbacks).
  lib/               libaicore.so plus its llama.cpp/ggml dependencies and all
                     backend modules. Keep this directory together.

Linking
  cc your_app.c -I<this dir>/include -L<this dir>/lib -laicore \\
     -Wl,-rpath,<this dir>/lib
  The shared library has RUNPATH \$ORIGIN, so at runtime it finds
  libllama.so.0, libggml.so.0, libggml-base.so.0 and the backend modules in the
  same directory as libaicore.so. Distributing the whole tree is sufficient.

Environment variables
  AICORE_BACKEND_DIR  Directory to load backend modules from (default: the
                      directory containing libaicore.so).
  AICORE_LOG          Log level: off (default), error, warn, info, debug, trace.
                      (aicore_set_log_callback() takes 0=off .. 5=trace instead.)
  AICORE_NATIVE       (build time only) tune the CPU backend for the host CPU.

Notes
  - Backends are loaded at runtime; CPU variants cover multiple x86_64 feature
    levels and the Vulkan backend is optional. If libggml-vulkan.so is absent or
    no GPU is available, it falls back to CPU automatically.
EOF

# Strip debug/unneeded symbols.
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
