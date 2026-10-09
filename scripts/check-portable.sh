#!/usr/bin/env bash
#
# check-portable.sh <dist-dir> <model.gguf>
#
# Portability verification for an offgrid Linux distribution directory (the
# extracted offgrid-<version>-linux-x86_64 directory). Prints PASS/FAIL per check
# and exits nonzero if any check fails.
#
set -uo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <dist-dir> <model.gguf>" >&2
    exit 2
fi

DIST="$(cd "$1" && pwd)"
MODEL="$2"
case "$MODEL" in
    /*) ;;
    *) MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")" ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAT_SRC="$ROOT/examples/c/chat.c"
QUESTION="Say hello in one word."

FAILS=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
info() { echo "INFO: $*"; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== offgrid portability checks =="
echo "dist:  $DIST"
echo "model: $MODEL"
echo

# --- a) ldd: everything resolved from lib/, no libstdc++ ---------------------
# Every shipped ELF: versioned core libs (libllama.so.0, ...), backend modules and the server.
mapfile -t ELFS < <(find "$DIST/lib" "$DIST/bin" -type f \( -name '*.so' -o -name '*.so.*' -o -name 'offgrid-server' \) 2>/dev/null | sort)
info "checking ${#ELFS[@]} binaries"
missing=0
stdcxx=0
while IFS= read -r so; do
    deps="$(env -u LD_LIBRARY_PATH ldd "$so" 2>&1)"
    if echo "$deps" | grep -q 'not found'; then
        missing=1
        echo "      $(basename "$so"): $(echo "$deps" | grep 'not found' | tr '\n' ' ')"
    fi
    if echo "$deps" | grep -q 'libstdc++'; then
        stdcxx=1
        echo "      $(basename "$so"): depends on libstdc++"
    fi
done < <(printf '%s\n' "${ELFS[@]}")

if [ "$missing" -eq 0 ]; then
    pass "ldd: all .so dependencies resolve from lib/ (no LD_LIBRARY_PATH)"
else
    fail "ldd: unresolved dependencies (see above)"
fi

if [ "$stdcxx" -eq 0 ]; then
    pass "ldd: no libstdc++ dependency in any .so"
else
    fail "ldd: libstdc++ dependency present"
fi

# --- b) max required GLIBC version (report only) -----------------------------
glibc_max="$(for so in "${ELFS[@]}"; do
    objdump -T "$so" 2>/dev/null
done | grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' | sort -Vu | tail -1 || true)"
info "max required GLIBC symbol version: ${glibc_max:-<none>}"

# --- c) relocatable compile + run --------------------------------------------
cp -a "$DIST" "$TMP/copy1"
BIN1="$TMP/chat1"
if gcc -Wall -Wextra -O2 -I"$TMP/copy1/include" "$CHAT_SRC" \
        -o "$BIN1" -L"$TMP/copy1/lib" -loffgrid -Wl,-rpath,"$TMP/copy1/lib" 2>"$TMP/gcc1.log"; then
    if timeout 300 "$BIN1" "$MODEL" "$QUESTION" >"$TMP/out1.txt" 2>"$TMP/err1.txt" \
            && grep -q '^assistant:' "$TMP/out1.txt"; then
        pass "relocated build+run: chat answered"
    else
        fail "relocated build+run: chat did not answer"
        tail -5 "$TMP/err1.txt" | sed 's/^/      /'
    fi
else
    fail "relocated build: gcc failed"
    sed 's/^/      /' "$TMP/gcc1.log" | tail -10
fi

# --- d) no-GPU fallback ------------------------------------------------------
if VK_ICD_FILENAMES=/nonexistent VK_DRIVER_FILES=/nonexistent OFFGRID_LOG=info \
        timeout 300 "$BIN1" "$MODEL" "$QUESTION" >"$TMP/out2.txt" 2>"$TMP/err2.txt" \
        && grep -q '^assistant:' "$TMP/out2.txt"; then
    pass "no-GPU fallback: chat answered with Vulkan disabled"
else
    fail "no-GPU fallback: chat did not answer"
    tail -5 "$TMP/err2.txt" | sed 's/^/      /'
fi
devs2="$(grep -m1 '^devices:' "$TMP/out2.txt" || true)"
info "no-GPU devices listed: ${devs2:-<none>}"
if echo "$devs2" | grep -qi 'vulkan'; then
    info "WARNING: a Vulkan device was still listed without a driver"
fi

# --- e) missing Vulkan backend module ---------------------------------------
cp -a "$DIST" "$TMP/copy2"
rm -f "$TMP/copy2/lib/libggml-vulkan.so"
BIN2="$TMP/chat2"
if gcc -Wall -Wextra -O2 -I"$TMP/copy2/include" "$CHAT_SRC" \
        -o "$BIN2" -L"$TMP/copy2/lib" -loffgrid -Wl,-rpath,"$TMP/copy2/lib" 2>"$TMP/gcc2.log"; then
    if timeout 300 "$BIN2" "$MODEL" "$QUESTION" >"$TMP/out3.txt" 2>"$TMP/err3.txt" \
            && grep -q '^assistant:' "$TMP/out3.txt"; then
        devs3="$(grep -m1 '^devices:' "$TMP/out3.txt" || true)"
        if echo "$devs3" | grep -qi 'vulkan'; then
            fail "missing Vulkan module: answered but Vulkan still listed"
        else
            pass "missing Vulkan module: chat answered CPU-only"
        fi
        info "no-Vulkan-module devices listed: ${devs3:-<none>}"
    else
        fail "missing Vulkan module: chat did not answer"
        tail -5 "$TMP/err3.txt" | sed 's/^/      /'
    fi
else
    fail "missing Vulkan module: gcc failed"
fi

# --- f) CPU variant loaded (report only) ------------------------------------
VK_ICD_FILENAMES=/nonexistent VK_DRIVER_FILES=/nonexistent OFFGRID_LOG=debug \
    timeout 300 "$BIN1" "$MODEL" "$QUESTION" >"$TMP/out4.txt" 2>"$TMP/err4.txt" || true
info "CPU variant / backend load evidence:"
grep -iE 'ggml-cpu|loaded.*cpu|cpu backend|variant' "$TMP/err4.txt" | head -8 | sed 's/^/      /' || true
variant_line="$(grep -iE 'ggml-cpu-[a-z0-9]+\.so' "$TMP/err4.txt" | head -1 || true)"
info "selected CPU module: ${variant_line:-<not detected>}"

# h) Packaged server: run bin/offgrid-server from a relocated copy, answer an HTTP request.
mkdir -p "$TMP/srv"
cp -r "$DIST" "$TMP/srv/pkg"
PORT=$(( 20000 + RANDOM % 20000 ))
"$TMP/srv/pkg/bin/offgrid-server" --model "$MODEL" --port "$PORT" >"$TMP/srv.log" 2>&1 &
SRV_PID=$!
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 0.5; done
reply="$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' 2>/dev/null || true)"
kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null || true
if printf '%s' "$reply" | grep -q '"chat.completion"'; then
    pass "server: relocated bin/offgrid-server answered /v1/chat/completions"
else
    fail "server: no valid reply (log below)"
    tail -15 "$TMP/srv.log" | sed 's/^/      /'
fi

# g) Deno: dlopen from a non-C host (different load order/flags than a linked C binary).
#    This caught a static-libstdc++ symbol interposition crash that C checks missed.
if command -v deno >/dev/null 2>&1; then
    mkdir -p "$TMP/deno"
    cp -r "$DIST" "$TMP/deno/pkg"
    if OFFGRID_LIB="$TMP/deno/pkg/lib/liboffgrid.so" OFFGRID_TEST_MODEL="$MODEL" \
        timeout 300 deno test --allow-ffi --allow-env --allow-read --config "$ROOT/bindings/deno/deno.json" \
        "$ROOT/bindings/deno/mod_test.ts" >"$TMP/deno.txt" 2>&1; then
        pass "deno: binding tests pass against relocated package"
    else
        fail "deno: binding tests failed (exit $?), see output below"
        tail -20 "$TMP/deno.txt" | sed 's/^/      /'
    fi
else
    info "deno not installed; skipping Deno check"
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
    exit 0
else
    echo "$FAILS CHECK(S) FAILED"
    exit 1
fi
