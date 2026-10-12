#!/usr/bin/env bash
#
# package-windows.sh — cross-compile offgrid for Windows (x86_64-pc-windows-gnullvm)
# with llvm-mingw and assemble a distribution zip under dist/.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=env-llvm-mingw.sh
source "$ROOT/scripts/env-llvm-mingw.sh"

TARGET="x86_64-pc-windows-gnullvm"
TARGET_DIR="$ROOT/target/$TARGET/release"

VERSION="$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import sys, json; d = json.load(sys.stdin); print(next(p["version"] for p in d["packages"] if p["name"] == "offgrid-ffi"))')"

PKG_NAME="offgrid-${VERSION}-windows-x86_64"
PKG="$ROOT/dist/$PKG_NAME"
ZIPFILE="$ROOT/dist/$PKG_NAME.zip"

# Remove previous output first so a failed build can't leave a stale package behind.
rm -rf "$PKG" "$ZIPFILE" "${ZIPFILE%.zip}.tar.gz"

echo ">> building offgrid-ffi ${VERSION} for ${TARGET} with --features dynamic-backends,image"
# Ask cargo which offgrid-sys build-script output this build actually used; several
# stale offgrid-sys-* dirs (other feature sets) can exist side by side.
SYS_OUT="$(cargo build --release -p offgrid-ffi --features dynamic-backends,image --target "$TARGET" \
    --message-format=json-render-diagnostics \
    | python3 -c '
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "build-script-executed" and "offgrid-sys" in m.get("package_id", ""):
        out = m["out_dir"]
print(out)')" || { echo "ERROR: build failed or offgrid-sys out dir not reported" >&2; exit 1; }

OUT_DIR="$(dirname "$SYS_OUT")"
if [ ! -d "$OUT_DIR/out/bin" ]; then
    echo "ERROR: $OUT_DIR/out/bin missing (was offgrid-sys built with dynamic-backends?)" >&2
    exit 1
fi
echo ">> using offgrid-sys output: $OUT_DIR"

echo ">> building offgrid-server.exe"
cargo build --release -p offgrid-server --features dynamic-backends,image --target "$TARGET"

if [ ! -f "$TARGET_DIR/offgrid.dll" ]; then
    echo "ERROR: $TARGET_DIR/offgrid.dll not found" >&2
    exit 1
fi

mkdir -p "$PKG/bin" "$PKG/lib" "$PKG/include"

# offgrid.dll plus every llama/ggml DLL must live in the same folder: offgrid.dll
# delay-loads its dependencies from its own directory.
cp -L "$TARGET_DIR/offgrid.dll" "$PKG/bin/offgrid.dll"
cp -L "$TARGET_DIR/offgrid-server.exe" "$PKG/bin/offgrid-server.exe"
cp -L "$TARGET_DIR/liboffgrid.dll.a" "$PKG/lib/liboffgrid.dll.a"

shopt -s nullglob
dlls=("$OUT_DIR"/out/bin/*.dll)
shopt -u nullglob
if [ "${#dlls[@]}" -eq 0 ]; then
    echo "ERROR: no DLLs found in $OUT_DIR/out/bin" >&2
    exit 1
fi
cp -L "${dlls[@]}" "$PKG/bin/"

cp "$ROOT/crates/offgrid-ffi/include/offgrid.h" "$PKG/include/offgrid.h"

# MSVC import library from the DLL's export table.
echo ">> generating import library from offgrid.dll exports"
llvm-readobj --coff-exports "$PKG/bin/offgrid.dll" \
    | sed -nE 's/^[[:space:]]*Name:[[:space:]]*(.+[^[:space:]])[[:space:]]*$/\1/p' \
    | grep -E '^offgrid_' | sort -u > "$PKG/lib/offgrid.exports.txt"

if [ ! -s "$PKG/lib/offgrid.exports.txt" ]; then
    echo "ERROR: no offgrid_* exports found in offgrid.dll" >&2
    exit 1
fi

{
    echo "LIBRARY offgrid.dll"
    echo "EXPORTS"
    sed 's/^/    /' "$PKG/lib/offgrid.exports.txt"
} > "$PKG/lib/offgrid.def"

if command -v llvm-lib >/dev/null 2>&1; then
    echo ">> using llvm-lib to create offgrid.lib"
    llvm-lib /def:"$PKG/lib/offgrid.def" /machine:x64 /out:"$PKG/lib/offgrid.lib" \
        || llvm-dlltool -m i386:x86-64 -d "$PKG/lib/offgrid.def" -l "$PKG/lib/offgrid.lib"
else
    echo ">> using llvm-dlltool to create offgrid.lib"
    llvm-dlltool -m i386:x86-64 -d "$PKG/lib/offgrid.def" -l "$PKG/lib/offgrid.lib"
fi
rm -f "$PKG/lib/offgrid.exports.txt"  # intermediate; offgrid.def carries the same list

cat > "$PKG/README.txt" <<EOF
offgrid ${VERSION} — Windows x86_64 (llvm-mingw, UCRT)
=====================================================

Contents
  include/offgrid.h     C API header (JSON in/out, streaming callbacks).
  bin/offgrid.dll       The library.
  bin/*.dll            llama.cpp/ggml/stable-diffusion.cpp DLLs and backend modules.
  bin/offgrid-server.exe  OpenAI-compatible HTTP server; see --help (chat, embeddings, and
                       /v1/images/generations with --image-diffusion-model etc.).
  lib/liboffgrid.dll.a  MinGW import library (link with -loffgrid).
  lib/offgrid.lib       MSVC import library (link offgrid.lib).
  lib/offgrid.def       Export definition file.

Keeping the DLLs together
  Keep every DLL from bin/ in the SAME folder as offgrid.dll and put that folder on
  PATH, or load offgrid.dll by full path. offgrid.dll delay-loads its llama/ggml
  dependencies and preloads them from its own directory, so a host can simply
  LoadLibrary("...\\bin\\offgrid.dll") without touching PATH.

Vulkan
  The Vulkan backend (ggml-vulkan.dll) needs a GPU driver that provides
  vulkan-1.dll. If it is missing, offgrid falls back to the CPU backend
  automatically.

Environment variables
  OFFGRID_BACKEND_DIR  Directory to load backend modules from (default: the
                      directory containing offgrid.dll).
  OFFGRID_LOG          Log level: off (default), error, warn, info, debug, trace.
                      (offgrid_set_log_callback() takes 0=off .. 5=trace instead.)
  OFFGRID_NATIVE       (build time only) tune the CPU backend for the host CPU.

Notes
  - Built for x86_64 with the UCRT (Windows 10+).
  - CPU backend variants cover multiple x86_64 feature levels; the matching one
    is selected at runtime.
EOF

for dll in "$PKG"/bin/*.dll "$PKG"/bin/*.exe; do
    llvm-strip --strip-unneeded "$dll"
done

if command -v zip >/dev/null 2>&1; then
    (cd "$ROOT/dist" && zip -r -q "$(basename "$ZIPFILE")" "$PKG_NAME")
else
    (cd "$ROOT/dist" && python3 -m zipfile -c "$(basename "$ZIPFILE")" "$PKG_NAME")
fi
# Same contents as .tar.gz, for the Deno binding's download (Deno can't unpack zip).
tar -C "$ROOT/dist" -czf "${ZIPFILE%.zip}.tar.gz" "$PKG_NAME"

echo
echo ">> size summary"
du -sh "$PKG" "$ZIPFILE"
echo ">> bin/ contents"
ls -la "$PKG/bin"
echo ">> lib/ contents"
ls -la "$PKG/lib"
echo ">> done: $ZIPFILE"
