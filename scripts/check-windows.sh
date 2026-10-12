#!/usr/bin/env bash
#
# check-windows.sh <dist-dir>
#
set -uo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <dist-dir>" >&2
    exit 2
fi

DIST="$(cd "$1" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=env-llvm-mingw.sh
source "$ROOT/scripts/env-llvm-mingw.sh"

BIN="$DIST/bin"
LIB="$DIST/lib"
HEADER="$DIST/include/offgrid.h"

FAILS=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
info() { echo "INFO: $*"; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== offgrid Windows packaging checks =="
echo "dist: $DIST"
echo

# Lowercased set of DLLs shipped in bin/, for import resolution.
declare -A BIN_DLLS=()
shopt -s nullglob
for f in "$BIN"/*.dll; do
    BIN_DLLS["$(basename "$f" | tr 'A-Z' 'a-z')"]=1
done
shopt -u nullglob

bad_imports=0
while IFS= read -r dll; do
    self="$(basename "$dll" | tr 'A-Z' 'a-z')"
    while IFS= read -r imp; do
        [ -n "$imp" ] || continue
        low="$(echo "$imp" | tr 'A-Z' 'a-z')"
        if [ -n "${BIN_DLLS[$low]:-}" ]; then
            continue
        fi
        case "$low" in
            kernel32.dll|advapi32.dll|ntdll.dll|bcryptprimitives.dll) continue ;;  # bcryptprimitives: Rust std RNG (ProcessPrng), Windows 8+
            api-ms-win-*.dll) continue ;;
            vulkan-1.dll)
                if [ "$self" = "ggml-vulkan.dll" ]; then
                    continue
                fi
                ;;
        esac
        fail "unexpected import: $(basename "$dll") imports $imp"
        bad_imports=$((bad_imports + 1))
    done < <(llvm-readobj --coff-imports "$dll" 2>/dev/null \
        | sed -nE 's/^[[:space:]]*Name:[[:space:]]*(.+[^[:space:]])[[:space:]]*$/\1/p' \
        | sort -u)
done < <(find "$BIN" -maxdepth 1 -iname '*.dll' -type f | sort)

if [ "$bad_imports" -eq 0 ]; then
    pass "all DLL imports resolve to bin/ or allowed system DLLs"
else
    fail "$bad_imports unexpected import(s)"
fi

imports_class="$(llvm-readobj --coff-imports "$BIN/offgrid.dll" 2>/dev/null | awk '
    /^Import \{/       { t = "normal" }
    /^DelayImport \{/  { t = "delay" }
    /^  Name: /        { name = $0; sub(/^  Name: /, "", name); print t, tolower(name) }
')"
delay_bad=0
wanted=(libllama.dll libmtmd.dll ggml.dll ggml-base.dll)
[ -f "$BIN/libstable-diffusion.dll" ] && wanted+=(libstable-diffusion.dll)
for want in "${wanted[@]}"; do
    kind="$(echo "$imports_class" | awk -v n="$want" '$2 == n { print $1 }' | head -1)"
    if [ "$kind" = "delay" ]; then
        info "offgrid.dll delay-loads $want"
    else
        fail "offgrid.dll does not delay-load $want (got: ${kind:-absent})"
        delay_bad=1
    fi
done
if [ "$delay_bad" -eq 0 ]; then
    pass "offgrid.dll delay-loads ${wanted[*]}"
fi

declared="$(python3 - "$HEADER" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
src = re.sub(r'//[^\n]*', '', src)
names = sorted(set(re.findall(r'\b(offgrid_[A-Za-z0-9_]+)\s*\(', src)))
print('\n'.join(names))
PY
)"
exported="$(llvm-readobj --coff-exports "$BIN/offgrid.dll" 2>/dev/null \
    | sed -nE 's/^[[:space:]]*Name:[[:space:]]*(.+[^[:space:]])[[:space:]]*$/\1/p' \
    | grep -E '^offgrid_' | sort -u)"

declared="$(echo "$declared" | LC_ALL=C sort -u)"
exported="$(echo "$exported" | LC_ALL=C sort -u)"

if [ "$declared" = "$exported" ]; then
    pass "offgrid.dll exports exactly the $(echo "$exported" | sed '/^$/d' | wc -l) offgrid_* functions declared in offgrid.h"
else
    fail "export set differs from offgrid.h declarations"
    diff <(echo "$declared") <(echo "$exported") | sed 's/^/      /' || true
fi

compile_ok=1
for ex in chat embed cancel; do
    extra=()
    [ "$ex" = "cancel" ] && extra=(-lpthread)
    if ! x86_64-w64-mingw32-clang -O2 -I"$DIST/include" "$ROOT/examples/c/$ex.c" \
            -o "$TMP/$ex.exe" -L"$DIST/lib" -loffgrid "${extra[@]}" 2>"$TMP/$ex.log"; then
        fail "cross-compile $ex.c"
        sed 's/^/      /' "$TMP/$ex.log" | tail -8
        compile_ok=0
    fi
done
if [ "$compile_ok" -eq 1 ]; then
    pass "examples cross-compile against the package (chat, embed, cancel)"
fi

if [ -f "$TMP/chat.exe" ]; then
    if llvm-readobj --coff-imports "$TMP/chat.exe" 2>/dev/null \
            | sed -nE 's/^[[:space:]]*Name:[[:space:]]*(.+)$/\1/p' \
            | grep -qi '^offgrid\.dll$'; then
        pass "chat.exe imports offgrid.dll"
    else
        fail "chat.exe does not import offgrid.dll"
    fi
fi

info "bin/ size: $(du -sh "$BIN" 2>/dev/null | cut -f1)"
ZIP="${DIST%/}.zip"
if [ -f "$ZIP" ]; then
    info "zip size:  $(du -h "$ZIP" | cut -f1) ($ZIP)"
else
    info "zip size:  <not found at $ZIP>"
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
    exit 0
else
    echo "$FAILS CHECK(S) FAILED"
    exit 1
fi
