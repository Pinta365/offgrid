#!/usr/bin/env bash
#
# package-windows.sh — cross-compile aicore for Windows (x86_64-pc-windows-gnullvm)
# with llvm-mingw and assemble a distribution zip under dist/.
#
# Requires scripts/env-llvm-mingw.sh (sourced below) to put the llvm-mingw tools
# (llvm-readobj, llvm-strip, llvm-dlltool, llvm-lib, x86_64-w64-mingw32-clang) on PATH.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=env-llvm-mingw.sh
source "$ROOT/scripts/env-llvm-mingw.sh"

TARGET="x86_64-pc-windows-gnullvm"
TARGET_DIR="$ROOT/target/$TARGET/release"

VERSION="$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import sys, json; d = json.load(sys.stdin); print(next(p["version"] for p in d["packages"] if p["name"] == "aicore-ffi"))')"

PKG_NAME="aicore-${VERSION}-windows-x86_64"
PKG="$ROOT/dist/$PKG_NAME"
ZIPFILE="$ROOT/dist/$PKG_NAME.zip"

# Remove previous output first so a failed build can't leave a stale package behind.
rm -rf "$PKG" "$ZIPFILE"

echo ">> building aicore-ffi ${VERSION} for ${TARGET} with --features dynamic-backends"
# Ask cargo which aicore-sys build-script output this build actually used; several
# stale aicore-sys-* dirs (other feature sets) can exist side by side.
SYS_OUT="$(cargo build --release -p aicore-ffi --features dynamic-backends --target "$TARGET" \
    --message-format=json-render-diagnostics \
    | python3 -c '
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "build-script-executed" and "aicore-sys" in m.get("package_id", ""):
        out = m["out_dir"]
print(out)')" || { echo "ERROR: build failed or aicore-sys out dir not reported" >&2; exit 1; }

OUT_DIR="$(dirname "$SYS_OUT")"
if [ ! -d "$OUT_DIR/out/bin" ]; then
    echo "ERROR: $OUT_DIR/out/bin missing (was aicore-sys built with dynamic-backends?)" >&2
    exit 1
fi
echo ">> using aicore-sys output: $OUT_DIR"

echo ">> building aicore-server.exe"
cargo build --release -p aicore-server --features dynamic-backends --target "$TARGET"

if [ ! -f "$TARGET_DIR/aicore.dll" ]; then
    echo "ERROR: $TARGET_DIR/aicore.dll not found" >&2
    exit 1
fi

mkdir -p "$PKG/bin" "$PKG/lib" "$PKG/include"

# aicore.dll plus every llama/ggml DLL must live in the same folder: aicore.dll
# delay-loads its dependencies from its own directory.
cp -L "$TARGET_DIR/aicore.dll" "$PKG/bin/aicore.dll"
cp -L "$TARGET_DIR/aicore-server.exe" "$PKG/bin/aicore-server.exe"
cp -L "$TARGET_DIR/libaicore.dll.a" "$PKG/lib/libaicore.dll.a"

shopt -s nullglob
dlls=("$OUT_DIR"/out/bin/*.dll)
shopt -u nullglob
if [ "${#dlls[@]}" -eq 0 ]; then
    echo "ERROR: no DLLs found in $OUT_DIR/out/bin" >&2
    exit 1
fi
cp -L "${dlls[@]}" "$PKG/bin/"

cp "$ROOT/crates/aicore-ffi/include/aicore.h" "$PKG/include/aicore.h"

# --- MSVC import library from the DLL's export table -------------------------
echo ">> generating import library from aicore.dll exports"
llvm-readobj --coff-exports "$PKG/bin/aicore.dll" \
    | sed -nE 's/^[[:space:]]*Name:[[:space:]]*(.+[^[:space:]])[[:space:]]*$/\1/p' \
    | grep -E '^aicore_' | sort -u > "$PKG/lib/aicore.exports.txt"

if [ ! -s "$PKG/lib/aicore.exports.txt" ]; then
    echo "ERROR: no aicore_* exports found in aicore.dll" >&2
    exit 1
fi

{
    echo "LIBRARY aicore.dll"
    echo "EXPORTS"
    sed 's/^/    /' "$PKG/lib/aicore.exports.txt"
} > "$PKG/lib/aicore.def"

if command -v llvm-lib >/dev/null 2>&1; then
    echo ">> using llvm-lib to create aicore.lib"
    llvm-lib /def:"$PKG/lib/aicore.def" /machine:x64 /out:"$PKG/lib/aicore.lib" \
        || llvm-dlltool -m i386:x86-64 -d "$PKG/lib/aicore.def" -l "$PKG/lib/aicore.lib"
else
    echo ">> using llvm-dlltool to create aicore.lib"
    llvm-dlltool -m i386:x86-64 -d "$PKG/lib/aicore.def" -l "$PKG/lib/aicore.lib"
fi
rm -f "$PKG/lib/aicore.exports.txt"  # intermediate; aicore.def carries the same list

cat > "$PKG/README.txt" <<EOF
aicore ${VERSION} — Windows x86_64 (llvm-mingw, UCRT)
=====================================================

Contents
  include/aicore.h     C API header (JSON in/out, streaming callbacks).
  bin/aicore.dll       The library.
  bin/*.dll            llama.cpp/ggml DLLs and backend modules.
  lib/libaicore.dll.a  MinGW import library (link with -laicore).
  lib/aicore.lib       MSVC import library (link aicore.lib).
  lib/aicore.def       Export definition file.

Keeping the DLLs together
  Keep every DLL from bin/ in the SAME folder as aicore.dll and put that folder on
  PATH, or load aicore.dll by full path. aicore.dll delay-loads its llama/ggml
  dependencies and preloads them from its own directory, so a host can simply
  LoadLibrary("...\\bin\\aicore.dll") without touching PATH.

Vulkan
  The Vulkan backend (ggml-vulkan.dll) needs a GPU driver that provides
  vulkan-1.dll. If it is missing, aicore falls back to the CPU backend
  automatically.

Environment variables
  AICORE_BACKEND_DIR  Directory to load backend modules from (default: the
                      directory containing aicore.dll).
  AICORE_LOG          Log level: off (default), error, warn, info, debug, trace.
                      (aicore_set_log_callback() takes 0=off .. 5=trace instead.)
  AICORE_NATIVE       (build time only) tune the CPU backend for the host CPU.

Notes
  - Built for x86_64 with the UCRT (Windows 10+).
  - CPU backend variants cover multiple x86_64 feature levels; the matching one
    is selected at runtime.
EOF

# Strip debug/unneeded symbols from the DLLs.
for dll in "$PKG"/bin/*.dll "$PKG"/bin/*.exe; do
    llvm-strip --strip-unneeded "$dll"
done

# --- archive -----------------------------------------------------------------
if command -v zip >/dev/null 2>&1; then
    (cd "$ROOT/dist" && zip -r -q "$(basename "$ZIPFILE")" "$PKG_NAME")
else
    (cd "$ROOT/dist" && python3 -m zipfile -c "$(basename "$ZIPFILE")" "$PKG_NAME")
fi

echo
echo ">> size summary"
du -sh "$PKG" "$ZIPFILE"
echo ">> bin/ contents"
ls -la "$PKG/bin"
echo ">> lib/ contents"
ls -la "$PKG/lib"
echo ">> done: $ZIPFILE"
